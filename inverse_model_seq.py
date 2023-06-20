"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import copy

import numpy as np
import torch

from data import TransitionDatasetForShuffling


def train_inverse_model(
    inverse_models,
    offline_trajs,
    action_available_ind,
    action_unavailable_in_dist,
    action_unavailable_out_dist,
    state_mean,
    state_std,
    max_episode_len,
    cfg,
    logger,
):
    def inverse_model_loss(ahat_dist, a):
        return -ahat_dist.log_likelihood(a).mean()

    def _eval(eval_dataloader, inverse_model):

        inverse_model.eval()
        eval_loss = 0.0

        for total_it, (states_seq, next_states, actions, _, _, _) in enumerate(
            eval_dataloader
        ):
            if cfg.inverse_model.num_past_transitions == 0:
                next_states = next_states.reshape(next_states.shape[0], -1)
                inputs = torch.cat([states_seq, next_states], dim=1)
            else:
                inputs = states_seq.reshape(states_seq.shape[0], -1)

            ahat = inverse_model(inputs.to(cfg.device))
            loss = inverse_model_loss(ahat, actions.to(cfg.device))
            eval_loss += loss.detach().cpu().item()

        eval_loss /= total_it + 1
        return eval_loss

    validation_size = int(len(action_available_ind) * cfg.inverse_model.validation_perc)
    validation_set = np.random.choice(
        action_available_ind, size=validation_size, replace=False
    )
    train_set = np.setdiff1d(action_available_ind, validation_set)

    train_dataset = TransitionDatasetForShuffling(
        trajectories=[offline_trajs[ii] for ii in train_set],
        state_dim=cfg.model.state_dim,
        act_dim=cfg.model.act_dim,
        state_mean=state_mean,
        state_std=state_std,
        max_episode_len=max_episode_len,
        action_range=cfg.model.action_range,
        num_past_transitions=cfg.inverse_model.num_past_transitions,
    )

    validation_dataset = TransitionDatasetForShuffling(
        trajectories=[offline_trajs[ii] for ii in validation_set],
        state_dim=cfg.model.state_dim,
        act_dim=cfg.model.act_dim,
        state_mean=state_mean,
        state_std=state_std,
        action_range=cfg.model.action_range,
        max_episode_len=max_episode_len,
        num_past_transitions=cfg.inverse_model.num_past_transitions,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=cfg.inverse_model.batch_size * 4,
        num_workers=4,
        shuffle=False,
    )

    for nn, inverse_model in enumerate(inverse_models):
        print(f"Training IDM {nn}")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.inverse_model.batch_size,
            num_workers=12,
            shuffle=True,
        )
        inverse_model_optimizer = torch.optim.Adam(
            inverse_model.parameters(),
            lr=cfg.inverse_model.lr,
        )

        iter_num, best_validation_loss = 0, np.Inf
        reached_max_iter = False
        best_inverse_model = copy.deepcopy(inverse_model)

        while not reached_max_iter:

            for jj, (states_seq, next_states, actions, _, _, _) in enumerate(
                train_dataloader
            ):

                inverse_model.train()
                # states_seq = states_seq.reshape(states_seq.shape[0], -1)
                if cfg.inverse_model.num_past_transitions == 0:
                    next_states = next_states.reshape(next_states.shape[0], -1)
                    inputs = torch.cat([states_seq, next_states], dim=1)
                else:
                    inputs = states_seq.reshape(states_seq.shape[0], -1)

                ahat = inverse_model(inputs.to(cfg.device))

                loss = inverse_model_loss(ahat, actions.to(cfg.device))

                inverse_model_optimizer.zero_grad()
                loss.backward()
                inverse_model_optimizer.step()

                iter_num += 1

                if iter_num == 1 or iter_num % 10 == 0:

                    inverse_model.eval()
                    validation_loss = _eval(validation_dataloader, inverse_model)
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_inverse_model = copy.deepcopy(inverse_model)

                    print("=" * 80)
                    print(f"Iteration {iter_num}")
                    logger.log_metrics(
                        {
                            "iter": iter_num,
                            "inverse_model/train_loss": loss.detach().cpu().item(),
                            "inverse_model/validation_loss": validation_loss,
                            "inverse_model/best_validation_loss": best_validation_loss,
                        },
                        iter_num=iter_num,
                        csv_file_name="inverse_model_train.csv",
                        print_output=True,
                    )

                if iter_num >= cfg.inverse_model.total_train_iters:
                    reached_max_iter = True
                    break

        inverse_model = copy.deepcopy(best_inverse_model)


def predict_actions_unlabeled_trajs(
    inverse_models,
    offline_trajs,
    action_available_ind,
    action_unavailable_in_dist,
    action_unavailable_out_dist,
    state_mean,
    state_std,
    action_range,
    cfg,
):
    def inverse_model_loss(ahat_dist, a):
        return -ahat_dist.log_likelihood(a).mean()

    new_trajs = copy.deepcopy(offline_trajs)
    train_loss, uncertainty_train, mse_train = [], [], []
    eval_loss_in_dist, uncertainty_in_dist, mse_in_dist = [], [], []
    eval_loss_out_dist, uncertainty_out_dist, mse_out_dist = [], [], []

    mse_loss = torch.nn.MSELoss(reduction="mean")
    max_len = 2 * cfg.inverse_model.num_past_transitions + 1
    state_dim = cfg.model.state_dim
    num_past_transitions = cfg.inverse_model.num_past_transitions

    for ii, traj in enumerate(new_trajs):

        if num_past_transitions == 0:
            ss = traj["observations"]
            ss = (ss - state_mean) / state_std
            ss = torch.from_numpy(ss).to(dtype=torch.float32)

            next_ss = traj["next_observations"]
            next_ss = (next_ss - state_mean) / state_std
            next_ss = torch.from_numpy(next_ss).to(dtype=torch.float32)

            inputs = torch.cat([ss, next_ss], dim=1).to(device=cfg.device)

        elif num_past_transitions > 0:
            states_seq = []
            traj_len = len(traj["observations"])
            for si in range(traj_len):
                upper = min(traj_len, (si + 1) + 1)
                max_len = num_past_transitions + 2
                lower = max(0, si - num_past_transitions)
                ss = traj["observations"][lower:upper].reshape(-1, state_dim)
                tlen = ss.shape[0]
                ss = np.concatenate([np.zeros((max_len - tlen, state_dim)), ss])
                ss = (ss - state_mean) / state_std
                ss = ss.reshape(-1, state_dim)
                states_seq.append(ss)

            states_seq = torch.from_numpy(np.array(states_seq)).to(dtype=torch.float32)
            inputs = states_seq.reshape(traj_len, -1).to(device=cfg.device)

        aa = traj["actions"]
        aa = (
            torch.from_numpy(aa)
            .to(dtype=torch.float32, device=cfg.device)
            .clamp(*cfg.model.action_range)
        )

        mus, stds = [], []
        traj_loss = 0.0
        for inverse_model in inverse_models:
            inverse_model.eval()
            ahat = inverse_model(inputs)
            traj_loss += inverse_model_loss(ahat, aa).item()
            mus.append(ahat.mean.cpu().detach().numpy())
            stds.append(ahat.std.cpu().detach().numpy())
        traj_loss /= len(inverse_models)

        # Compute Gaussian Mixture Mean and Var
        mus = np.array(mus)  # (ensemble_size, timesteps, action_dim)
        stds = np.array(stds)  # (ensemble_size, timesteps, action_dim)
        mu = np.mean(mus, axis=0)
        mu = np.clip(mu, action_range[0], action_range[1])
        var = np.mean(stds ** 2, axis=0) + np.mean(mus ** 2, axis=0) - mu ** 2

        # average error over all the timesteps for a traj
        err = (
            mse_loss(torch.from_numpy(mu).to(device=cfg.device), aa)
            .detach()
            .cpu()
            .item()
        )

        # fill in predicted actions
        if ii not in action_available_ind:
            traj["actions"] = mu

        # track uncertainty for all the actions (labeled or unlabeled)
        # which is the average var across the action dimensions
        traj["uncertainty"] = var.mean(axis=1)

        if ii in action_available_ind:
            train_loss.append(traj_loss)
            uncertainty_train.append(var.mean(axis=1))
            mse_train.append(err)

        elif ii in action_unavailable_in_dist:
            eval_loss_in_dist.append(traj_loss)
            uncertainty_in_dist.append(var.mean(axis=1))
            mse_in_dist.append(err)
        else:
            eval_loss_out_dist.append(traj_loss)
            uncertainty_out_dist.append(var.mean(axis=1))
            mse_out_dist.append(err)

    return (
        new_trajs,
        train_loss,
        eval_loss_in_dist,
        eval_loss_out_dist,
        uncertainty_train,
        uncertainty_in_dist,
        uncertainty_out_dist,
        mse_train,
        mse_in_dist,
        mse_out_dist,
    )
