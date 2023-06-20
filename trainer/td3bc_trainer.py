"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from https://github.com/sfujim/TD3_BC,
which is licensed under the MIT License.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
import utils


class TD3BCTrainer:
    def __init__(
        self,
        model,
        actor_optimizer,
        critic_optimizer,
        device="cuda",
    ):
        self.model = model
        self.device = device

        self.model.actor_optimizer = actor_optimizer
        self.model.critic_optimizer = critic_optimizer

    def train_iteration(self, dataloader, loss_fn=None, num_updates=None):
        self.model.actor.train()
        self.model.critic.train()

        train_start = time.time()
        critic_losses, actor_losses = [], []

        total_it, done_training = 0, False
        while not done_training:
            for _, batch in enumerate(dataloader):

                state, next_state, action, reward, avg_rtg, not_done = utils.to_torch(
                    batch, device=self.device
                )
                state = state.reshape(-1, state.shape[-1])
                next_state = next_state.reshape(-1, next_state.shape[-1])
                action = action.reshape(-1, action.shape[-1])
                reward = reward.reshape(-1, 1)
                not_done = not_done.reshape(-1, 1)

                with torch.no_grad():
                    # Select action according to policy and add clipped noise
                    noise = (torch.randn_like(action) * self.model.policy_noise).clamp(
                        -self.model.noise_clip, self.model.noise_clip
                    )

                    next_action = (self.model.actor_target(next_state) + noise).clamp(
                        -self.model.max_action, self.model.max_action
                    )

                    # Compute the target Q value
                    target_Q1, target_Q2 = self.model.critic_target(
                        next_state, next_action
                    )
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + not_done * self.model.discount * target_Q

                # Get current Q estimates
                current_Q1, current_Q2 = self.model.critic(state, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                    current_Q2, target_Q
                )

                # Optimize the critic
                self.model.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.model.critic_optimizer.step()
                critic_losses.append(critic_loss.detach().cpu().item())

                # Delayed policy updates
                if total_it % self.model.policy_freq == 0:

                    # Compute actor loss
                    pi = self.model.actor(state)
                    Q = self.model.critic.Q1(state, pi)
                    lmbda = self.model.alpha / Q.abs().mean().detach()

                    # maximize Q function + BC regularization
                    actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

                    # Optimize the actor
                    self.model.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.model.actor_optimizer.step()
                    actor_losses.append(actor_loss.detach().cpu().item())

                    # Update the frozen target models
                    for param, target_param in zip(
                        self.model.critic.parameters(),
                        self.model.critic_target.parameters(),
                    ):
                        target_param.data.copy_(
                            self.model.tau * param.data
                            + (1 - self.model.tau) * target_param.data
                        )

                    for param, target_param in zip(
                        self.model.actor.parameters(),
                        self.model.actor_target.parameters(),
                    ):
                        target_param.data.copy_(
                            self.model.tau * param.data
                            + (1 - self.model.tau) * target_param.data
                        )

                total_it += 1
                if total_it >= num_updates:
                    done_training = True
                    break

        logs = {}
        logs["time/training"] = time.time() - train_start
        logs["training/critic_loss"] = np.mean(critic_losses)
        logs["training/actor_loss"] = np.mean(actor_losses)

        return logs
