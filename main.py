"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import time
from pathlib import Path

import hydra
import torch

import utils
import utils_dt
from data import (
    TransitionDataset,
    create_dataloader,
    create_transition_dataloader,
    load_dataset,
)
from evaluation import create_td3bc_eval_fn, create_vec_eval_episodes_fn
from inverse_model_seq import predict_actions_unlabeled_trajs, train_inverse_model
from logger import Logger
from model import mlp
from trainer import CQLTrainer, SequenceTrainer, TD3BCTrainer


@hydra.main(config_path="config", config_name="config")
def main(cfg):

    work_dir = Path.cwd()
    print(f"work_dir: {work_dir}")
    logger = Logger(work_dir, log_to_tb=cfg.log_to_tb)

    utils.set_seed_everywhere(cfg.seed)

    env_spec = utils.get_env_spec(cfg.env)

    (
        offline_trajs,
        state_mean,
        state_std,
        action_available_ind,
        action_unavailable_in_dist,
        action_unavailable_out_dist,
    ) = load_dataset(
        cfg.env,
        action_available_perc=cfg.data.action_available_perc,
        action_available_threshold=cfg.data.action_available_threshold,
    )

    cfg.model.state_dim = env_spec.state_dim
    cfg.model.act_dim = env_spec.act_dim
    cfg.model.action_range = env_spec.action_range
    if cfg.model.name == "td3bc":
        # actor and critic are sent to the device in the initialization function
        model = hydra.utils.instantiate(cfg.model)
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
        # defined inside TD3BC training code
        loss_fn = None
    elif cfg.model.name == "cql":
        model = hydra.utils.instantiate(cfg.model)
        policy_optimizer = torch.optim.Adam(
            model.actor.parameters(),
            1e-4,
        )
        qf_optimizer = torch.optim.Adam(
            list(model.qf1.parameters()) + list(model.qf2.parameters()), 3e-4
        )
        alpha_optimizer = torch.optim.Adam(model.log_alpha.parameters(), 3e-4)
        loss_fn = None
    elif cfg.model.name == "dt":
        cfg.model.n_inner = cfg.model.hidden_size * 4
        cfg.model.target_entropy = env_spec.target_entropy
        model = hydra.utils.instantiate(cfg.model).to(device=cfg.device)
        optimizer, log_temperature_optimizer, scheduler = utils_dt.create_optimizer(
            model, cfg
        )
        loss_fn = utils_dt.create_loss_function(cfg.model.stochastic_policy)
        reward_scale = 1 if "antmaze" in cfg.env else 0.001
    else:
        raise NotImplementedError

    inverse_models = []
    for ii in range(cfg.inverse_model.ensemble_size):
        inverse_models.append(
            mlp.DiagGaussianDistribution(
                input_dim=cfg.model.state_dim
                * (cfg.inverse_model.num_past_transitions + 2),
                output_dim=cfg.model.act_dim,
                hidden_dims=cfg.inverse_model.hidden_dims,
                spectral_norms=cfg.inverse_model.spectral_norms,
                action_range=env_spec.action_range,
                dropout=cfg.inverse_model.dropout,
            ).to(device=cfg.device)
        )

    print("\n\n Make Eval Env\n\n")
    if cfg.model.name == "dt":
        eval_env = utils_dt.make_eval_envs(cfg.env, cfg.num_eval_episodes)
        eval_rtg = utils_dt.get_eval_rtg(cfg.env)
        eval_fn = create_vec_eval_episodes_fn(
            vec_env=eval_env,
            eval_rtg=eval_rtg,
            state_dim=env_spec.state_dim,
            act_dim=env_spec.act_dim,
            state_mean=state_mean,
            state_std=state_std,
            max_episode_len=env_spec.max_episode_len,
            device=cfg.device,
            use_mean=True,
            reward_scale=reward_scale,
        )
    elif cfg.model.name in ["td3bc", "cql"]:
        eval_env = utils.make_eval_env(cfg.env, cfg.seed + 100)
        eval_fn = create_td3bc_eval_fn(
            eval_env, cfg.num_eval_episodes, state_mean, state_std
        )

    snapshot = work_dir / "snapshot.pt"

    iter_n = 0
    total_time = 0.0
    # snapshot saved after training the inverse model and agent
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        payload = utils.load_snapshot(snapshot)

        if cfg.model.name == "td3bc":
            model.actor.load_state_dict(payload["actor"])
            model.actor_target.load_state_dict(payload["actor_target"])
            model.critic.load_state_dict(payload["critic"])
            model.critic_target.load_state_dict(payload["critic_target"])
            actor_optimizer.load_state_dict(payload["actor_optimizer"])
            critic_optimizer.load_state_dict(payload["critic_optimizer"])
        elif cfg.model.name == "dt":
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            log_temperature_optimizer.load_state_dict(
                payload["log_temperature_optimizer"]
            )
            scheduler.load_state_dict(payload["scheduler"])
        elif cfg.model.name == "cql":
            model.actor.load_state_dict(payload["actor"])
            model.qf1.load_state_dict(payload["qf1"])
            model.qf2.load_state_dict(payload["qf2"])
            model.target_qf1.load_state_dict(payload["target_qf1"])
            model.target_qf2.load_state_dict(payload["target_qf2"])
            model.log_alpha.load_state_dict(payload["log_alpha"])
            policy_optimizer.load_state_dict(payload["policy_optimizer"])
            qf_optimizer.load_state_dict(payload["qf_optimizer"])
            alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        else:
            raise NotImplementedError

        iter_n = payload["iter_n"]
        total_time = payload["total_time"]
        for ii in range(len(inverse_models)):
            inverse_models[ii].load_state_dict(payload["inverse_models"][ii])
    elif len(action_available_ind) != len(offline_trajs) and cfg.data.need_pseudo_label:
        print("\n\n Training Inverse Model \n\n")
        print(f"{len(action_available_ind)} labeled trajectories")
        train_inverse_model(
            inverse_models,
            offline_trajs,
            action_available_ind,
            action_unavailable_in_dist,
            action_unavailable_out_dist,
            state_mean,
            state_std,
            env_spec.max_episode_len,
            cfg,
            logger,
        )
    else:
        pass

    if cfg.model.name == "td3bc":
        trainer = TD3BCTrainer(
            model=model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=cfg.device,
        )
    elif cfg.model.name == "cql":
        trainer = CQLTrainer(
            model=model,
            policy_optimizer=policy_optimizer,
            qf_optimizer=qf_optimizer,
            alpha_optimizer=alpha_optimizer,
            device=cfg.device,
        )
    elif cfg.model.name == "dt":
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            log_temperature_optimizer=log_temperature_optimizer,
            scheduler=scheduler,
            device=cfg.device,
        )
    else:
        raise NotImplementedError

    if len(action_available_ind) == len(offline_trajs):
        new_trajs = offline_trajs
    elif cfg.data.need_pseudo_label:
        print("\n\n Predict missing actions for unlabelled trajectories.")
        (
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
        ) = predict_actions_unlabeled_trajs(
            inverse_models,
            offline_trajs,
            action_available_ind,
            action_unavailable_in_dist,
            action_unavailable_out_dist,
            state_mean,
            state_std,
            env_spec.action_range,
            cfg,
        )
        logger.pickle_obj(
            fname="inverse_rst.pkl",
            obj={
                "new_trajs": new_trajs,
                "train_loss": train_loss,
                "eval_loss_in_dist": eval_loss_in_dist,
                "eval_loss_out_dist": eval_loss_out_dist,
                "uncertainty_train": uncertainty_train,
                "uncertainty_in_dist": uncertainty_in_dist,
                "uncertainty_out_dist": uncertainty_out_dist,
                "mse_train": mse_train,
                "mse_in_dist": mse_in_dist,
                "mse_out_dist": mse_out_dist,
            },
        )
    else:
        new_trajs = [
            offline_trajs[ii]
            for ii in range(len(offline_trajs))
            if ii in action_available_ind
        ]

    print(f"\n\n Offline RL Training data contains {len(new_trajs)} trajectories.")

    print("\n\n Training Agent\n\n")
    if cfg.model.name != "dt":
        dataset = TransitionDataset(
            trajectories=new_trajs,
            state_dim=env_spec.state_dim,
            act_dim=env_spec.act_dim,
            state_mean=state_mean,
            state_std=state_std,
            action_range=env_spec.action_range,
            max_episode_len=env_spec.max_episode_len,
            action_available_ind=action_available_ind,
        )

    while iter_n < cfg.train_iters:
        start_time = time.time()
        # in every iteration, prepare the data loader
        if cfg.model.name == "dt":
            dataloader = create_dataloader(
                trajectories=new_trajs,
                num_iters=cfg.num_updates_per_iter,
                batch_size=cfg.batch_size,
                max_len=cfg.model.max_length,
                state_dim=env_spec.state_dim,
                act_dim=env_spec.act_dim,
                state_mean=state_mean,
                state_std=state_std,
                reward_scale=reward_scale,
                action_range=env_spec.action_range,
                max_episode_len=env_spec.max_episode_len,
                action_available_ind=action_available_ind,
            )
        else:
            dataloader = create_transition_dataloader(
                dataset,
                batch_size=cfg.batch_size * 20,
                num_iters=cfg.num_updates_per_iter,
                num_workers=12,
            )

        train_outputs = trainer.train_iteration(
            dataloader=dataloader, loss_fn=loss_fn, num_updates=cfg.num_updates_per_iter
        )

        eval_outputs = eval_fn(model)

        # generate payload
        if cfg.model.name == "td3bc":
            payload = {
                "actor": model.actor.state_dict(),
                "actor_target": model.actor_target.state_dict(),
                "critic": model.critic.state_dict(),
                "critic_target": model.critic_target.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "inverse_models": [im.state_dict() for im in inverse_models],
                "iter_n": iter_n,
                "total_time": total_time,
            }
        elif cfg.model.name == "dt":
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "log_temperature_optimizer": log_temperature_optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "inverse_models": [im.state_dict() for im in inverse_models],
                "iter_n": iter_n,
                "total_time": total_time,
            }
        elif cfg.model.name == "cql":
            payload = {
                "actor": model.actor.state_dict(),
                "qf1": model.qf1.state_dict(),
                "qf2": model.qf2.state_dict(),
                "target_qf1": model.target_qf1.state_dict(),
                "target_qf2": model.target_qf2.state_dict(),
                "log_alpha": model.log_alpha.state_dict(),
                "policy_optimizer": policy_optimizer.state_dict(),
                "qf_optimizer": qf_optimizer.state_dict(),
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "inverse_models": [im.state_dict() for im in inverse_models],
                "iter_n": iter_n,
                "total_time": total_time,
            }
        else:
            raise NotImplementedError

        utils.save_snapshot(work_dir, payload)

        total_time += time.time() - start_time
        outputs = {"time/total": total_time}
        outputs.update(train_outputs)
        outputs.update(eval_outputs)
        print("=" * 80)
        print(f"Iteration {iter_n}")
        logger.log_metrics(
            outputs,
            iter_num=iter_n,
            csv_file_name="results.csv",
            print_output=True,
        )
        iter_n += 1

    eval_env.close()


if __name__ == "__main__":
    main()
