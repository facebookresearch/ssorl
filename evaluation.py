"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import time

import numpy as np
import torch


def create_td3bc_eval_fn(env, eval_episodes, state_mean, state_std):
    def evaluate_td3_bc(model):
        eval_start = time.time()
        model.actor.eval()
        # model.critic.eval()

        returns, lengths = [], []
        for _ in range(eval_episodes):
            state, done = env.reset(), False
            ret, steps = 0, 0
            while not done:
                state = (np.array(state).reshape(1, -1) - state_mean) / state_std
                action = model.select_action(state)
                state, reward, done, _ = env.step(action)
                ret += reward
                steps += 1
            returns.append(ret)
            lengths.append(steps)

        outputs = {
            "evaluation/return_mean": np.mean(returns),
            "evaluation/return_std": np.std(returns),
            "evaluation/length_mean": np.mean(lengths),
            "evaluation/length_std": np.std(lengths),
            "time/evaluation": time.time() - eval_start,
        }

        return outputs

    return evaluate_td3_bc


def create_vec_eval_episodes_fn(
    vec_env,
    eval_rtg,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    device,
    max_episode_len,
    use_mean=True,
    reward_scale=0.001,
):
    def eval_episodes_fn(model):
        eval_start = time.time()
        model.eval()

        target_return = [eval_rtg * reward_scale] * vec_env.num_envs
        returns, lengths, state_errs, reward_errs, _ = vec_evaluate_episode_rtg(
            vec_env,
            state_dim,
            act_dim,
            model,
            max_ep_len=max_episode_len,
            reward_scale=reward_scale,
            target_return=target_return,
            mode="normal",
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
        )
        suffix = "_gm" if use_mean else ""
        return {
            f"evaluation/return_mean{suffix}": np.mean(returns),
            f"evaluation/return_std{suffix}": np.std(returns),
            f"evaluation/state_err_mean{suffix}": np.mean(state_errs),
            f"evaluation/state_err_std{suffix}": np.std(state_errs),
            f"evaluation/reward_err_mean{suffix}": np.mean(reward_errs),
            f"evaluation/reward_err_std{suffix}": np.std(reward_errs),
            f"evaluation/length_mean{suffix}": np.mean(lengths),
            f"evaluation/length_std{suffix}": np.std(lengths),
            "time/evaluation": time.time() - eval_start,
        }

    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode_rtg(
    vec_env,
    state_dim,
    act_dim,
    model,
    target_return: list,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
    eps_greedy=0.0,
    use_mean=False,
):
    assert len(target_return) == vec_env.num_envs

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()
    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    reward_err = np.zeros(num_envs)
    state_err = np.zeros(num_envs)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )
        if eps_greedy and np.random.uniform() <= eps_greedy:
            act = []
            for ii in range(num_envs):
                act.append(vec_env.action_space.sample())
            action = torch.tensor(act, device=device, dtype=torch.float32).reshape(
                num_envs, -1
            )
            state_pred, _, reward_pred = model.get_predictions(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                num_envs=num_envs,
            )
            state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
            reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)
        else:
            state_pred, action, reward_pred = model.get_predictions(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                num_envs=num_envs,
            )
            state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
            reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)

            # the return action is a SquashNormal distribution
            if model.stochastic_policy:
                action_dist = action
                action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
                if use_mean:
                    action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
                action = action.clamp(*model.action_range)

        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())

        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)
        reward_err[unfinished] += (reward[unfinished] - reward_pred[unfinished]) ** 2
        state_err[unfinished] += (
            np.linalg.norm(state[unfinished] - state_pred[unfinished], axis=1) ** 2
        )

        actions[:, -1] = action
        state = (
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break

    state_err = state_err / episode_length
    reward_err = reward_err / episode_length

    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            # need to remove the last state
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        state_err,
        reward_err,
        trajectories,
    )
