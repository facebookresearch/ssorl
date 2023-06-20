"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import gym
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from lamb import Lamb


def get_eval_rtg(env_name):
    if "hopper" in env_name:
        return 3600.0
    elif "walker" in env_name:
        return 5000.0
    elif "halfcheetah" in env_name:
        return 6000.0
    elif "ant" in env_name:
        return 6000.0
    elif "maze2d-open-dense-v0" in env_name:
        return 60.0
    elif "maze2d" in env_name:
        return 150.0
    else:
        raise NotImplementedError


def create_optimizer(model, cfg):

    optimizer = Lamb(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-3,
        eps=1e-8,
    )
    if cfg.model.stochastic_policy:
        log_temperature_optimizer = torch.optim.Adam(
            [model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )
    else:
        log_temperature_optimizer = None

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / 1e4, 1)
    )

    return optimizer, log_temperature_optimizer, scheduler


def make_eval_envs(env_name, num_eval_episodes):
    def get_env_builder(seed, target_goal=None):
        def make_env_fn():
            import d4rl

            #        if "antmaze" in env_name:
            #            utils.register_antmaze()
            env = gym.make(env_name)
            env.seed(seed)
            if hasattr(env.env, "wrapped_env"):
                env.env.wrapped_env.seed(seed)
            elif hasattr(env.env, "seed"):
                env.env.seed(seed)
            else:
                pass
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            if target_goal:
                env.set_target_goal(target_goal)
                print(f"Set the target goal to be {env.target_goal}")
            return env

        return make_env_fn

    if "antmaze" in env_name:
        env = gym.make(env_name)
        target_goal = env.target_goal
        env.close()
        print(f"Generated fixed target goal: {target_goal}")
    else:
        target_goal = None

    eval_envs = SubprocVecEnv(
        [get_env_builder(i, target_goal=target_goal) for i in range(num_eval_episodes)]
    )
    return eval_envs


def create_loss_function(stochastic_policy):
    def loss_nll(
        s_hat,
        a_hat_dist,
        r_hat,
        s,
        a,
        r,
        attention_mask,
        entropy_reg,
    ):
        # a_hat is a SquashedNormal Distribution
        log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

        entropy = a_hat_dist.entropy().mean()
        a_loss = -(log_likelihood + entropy_reg * entropy)

        s_loss = torch.mean((s_hat - s) ** 2)
        r_loss = torch.mean((r_hat - r) ** 2)
        loss = a_loss

        return (
            loss,
            a_loss,
            s_loss,
            r_loss,
            -log_likelihood,
            entropy,
        )

    def loss_mse(s_hat, a_hat, r_hat, s, a, r):
        a_loss = torch.mean((a_hat - a) ** 2)
        s_loss = torch.mean((s_hat - s) ** 2)
        r_loss = torch.mean((r_hat - r) ** 2)
        loss = a_loss
        return loss, a_loss, s_loss, r_loss

    if stochastic_policy:
        return loss_nll
    else:
        return loss_mse
