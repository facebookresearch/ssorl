"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from https://github.com/young-geng/CQL,
which is licensed under the MIT License.
"""

import copy

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            1 - soft_target_update_rate
        ) * target_network_params[k].data + soft_target_update_rate * v.data


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, arch="256-256"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split("-")]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(self, log_std_min=-20.0, log_std_max=2.0, no_tanh=False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        arch="256-256",
        log_std_multiplier=1.0,
        log_std_offset=-1.0,
        no_tanh=False,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(observation_dim, 2 * action_dim, arch)
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)


class FullyConnectedQFunction(nn.Module):
    def __init__(self, observation_dim, action_dim, arch="256-256"):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.network = FullyConnectedNetwork(observation_dim + action_dim, 1, arch)

    def forward(self, observations, actions):
        if actions.ndim == 3 and observations.ndim == 2:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        return torch.squeeze(self.network(input_tensor), dim=-1)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


class CQL:
    def __init__(self, name, state_dim, act_dim, action_range, device="cuda"):

        self.discount = 0.99
        self.reward_scale = 1.0
        self.alpha_multiplier = 1.0
        self.target_entropy = -act_dim
        self.soft_target_update_rate = 5e-3
        self.cql_n_actions = 10
        self.cql_importance_sample = True
        self.cql_target_action_gap = -1.0
        self.cql_temp = 1.0
        self.cql_min_q_weight = 5.0
        self.device = device

        # max_action = action_range[1]
        self.actor = TanhGaussianPolicy(state_dim, act_dim, arch="256-256-256").to(
            device=device
        )

        # Actor(state_dim, act_dim, max_action).to(device=device)
        self.qf1 = FullyConnectedQFunction(state_dim, act_dim, arch="256-256-256").to(
            device
        )
        self.target_qf1 = copy.deepcopy(self.qf1)

        self.qf2 = FullyConnectedQFunction(state_dim, act_dim, arch="256-256-256").to(
            device
        )
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = Scalar(0.0)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _ = self.actor(state, deterministic=True)
        return action.detach().cpu().numpy().flatten()
