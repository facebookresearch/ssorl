"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import random
from collections import namedtuple
from pathlib import Path

import gym
import numpy as np
import torch

EnvSpec = namedtuple(
    "EnvSpec",
    ["state_dim", "act_dim", "action_range", "target_entropy", "max_episode_len"],
)


def get_env_spec(env_name):
    import d4rl

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]
    max_episode_len = env._max_episode_steps
    env.close()
    target_entropy = -act_dim
    return EnvSpec(state_dim, act_dim, action_range, target_entropy, max_episode_len)


def make_eval_env(env_name, seed, target_goal=None):
    import d4rl

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


def save_snapshot(work_dir, payload):
    snapshot = work_dir / "snapshot.pt"
    with open(snapshot, "wb") as f:
        torch.save(payload, f)


def load_snapshot(snapshot):
    with open(snapshot, "rb") as f:
        payload = torch.load(f)
    return payload


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_torch(xs, device, dtype=None):
    if dtype is not None:
        xs = (x.astype(dtype) for x in xs)
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def to_np(t):
    """
    convert a torch tensor to a numpy array
    """
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.detach().cpu().numpy()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
