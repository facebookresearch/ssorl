"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from
https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py

which is licensed under the MIT License.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.nn.utils import spectral_norm


def mlp(input_dim, output_dim, hidden_dims, spectral_norms, dropout=0.0):
    def maybe_sn(m, use_sn):
        return spectral_norm(m) if use_sn else m

    assert len(hidden_dims) == len(spectral_norms)
    layers = []
    for dim, use_sn in zip(hidden_dims, spectral_norms):
        layers += [
            maybe_sn(nn.Linear(input_dim, dim), use_sn),
            nn.ReLU(inplace=True),
        ]
        input_dim = dim

    if dropout:
        layers += [nn.Dropout(dropout)]
    layers += [nn.Linear(input_dim, output_dim)]

    return layers


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        spectral_norms,
        action_range,
        dropout=0.0,
    ):
        super().__init__()
        self.action_range = action_range

        layers = mlp(input_dim, output_dim, hidden_dims, spectral_norms, dropout)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        action = self.net(x)
        return action.clamp(*self.action_range)


class DiagGaussianDistribution(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    # ignore log_std_bounds for now
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        spectral_norms,
        action_range,
        log_std_bounds=[-5.0, 2.0],
        dropout=0,
    ):
        super().__init__()

        self.mu = nn.Sequential(
            *mlp(input_dim, output_dim, hidden_dims, spectral_norms, dropout)
        )
        self.log_std = nn.Sequential(
            *mlp(input_dim, output_dim, hidden_dims, spectral_norms, dropout)
        )
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        mu.clamp(*self.action_range)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)
    If mu/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, mu, std):
        self.mu = mu
        self.std = std
        self.base_dist = pyd.Normal(mu, std)
        # self.base_dist = pyd.MultivariateNormal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.mu
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # For this transformed distribution, we do not have a close form for entropy
        # that just uses base_dist.entroy(). We need to approximate the entropy
        # E_p[-log p(x)] by N samples.
        #
        # The following code only works for 1d. We need to build grids...
        #     dx = 0.01
        #     x = torch.arange(-0.999, 0.999, dx).repeat((self.N, 1))
        #     log_p = self.log_prob(x)
        #     return (-log_p * log_p.exp() * dx).sum(dim=1).mean()
        #
        # Instead, we just sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p.mean(axis=0) will return the entropy for each squashed normal
        # distribution; log_p.mean() will return mean entropy for all the distributions
        # Besides, since the returned log_prob.mean(axis=0) is of dimension
        # (batch_size, action_dim),
        # we sum up along the action dimensions to obtain the entropy of
        # for each batch in each state
        # Return tensor shape: scalar
        return -log_p.mean(axis=0).sum()

    def log_likelihood(self, x):
        # returned log_prob(x) is of dimension (batch_size,
        # action_dim), sum up along the action dimensions
        # Return tensor shape: (batch_size)
        return self.log_prob(x).sum(axis=1)
