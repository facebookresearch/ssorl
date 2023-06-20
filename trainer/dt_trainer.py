"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import time

import numpy as np
import torch


class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device

    def train_iteration(self, loss_fn, dataloader, num_updates):

        # num_updates is written into  dataloader

        losses, a_losses, s_losses, r_losses, nlls, entropies = [], [], [], [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        if self.model.stochastic_policy:
            for total_it, trajs in enumerate(dataloader):
                loss, a_loss, s_loss, r_loss, nll, entropy = self.train_step_stochastic(
                    loss_fn, trajs
                )
                losses.append(loss)
                a_losses.append(a_loss)
                s_losses.append(s_loss)
                r_losses.append(r_loss)
                nlls.append(nll)
                entropies.append(entropy)

            logs["time/training"] = time.time() - train_start
            logs["training/train_loss_mean"] = np.mean(losses)
            logs["training/train_loss_std"] = np.std(losses)
            logs["training/a_loss"] = a_losses[-1]
            logs["training/s_loss"] = s_losses[-1]
            logs["training/r_loss"] = r_losses[-1]
            logs["training/nll"] = nlls[-1]
            logs["training/entropy"] = entropies[-1]
            logs["training/temp_value"] = self.model.temperature().detach().cpu().item()
        else:
            raise NotImplementedError

        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        # action_target = torch.clone(actions)
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        # target = [0, 0, s0, s1]
        # prediction = [0, 0, shat_1, shat_2]
        # padding_mask = [0, 0, 1, 1]
        # our loss should be defined between s1 and shat_1.
        prediction_mask = padding_mask[:, :-1]

        state_dim = states.shape[2]
        state_preds = state_preds[:, :-1].reshape(-1, state_dim)[
            prediction_mask.reshape(-1) > 0
        ]
        state_target = state_target[:, 1:].reshape(-1, state_dim)[
            prediction_mask.reshape(-1) > 0
        ]

        reward_dim = rewards.shape[2]
        reward_preds = reward_preds[:, :-1].reshape(-1, reward_dim)[
            prediction_mask.reshape(-1) > 0
        ]
        reward_target = reward_target[:, 1:].reshape(-1, reward_dim)[
            prediction_mask.reshape(-1) > 0
        ]

        loss, a_loss, s_loss, r_loss, nll, entropy = loss_fn(
            state_preds,
            action_preds,  # a_hat_dist
            reward_preds,
            state_target,
            action_target,
            reward_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            a_loss.detach().cpu().item(),
            s_loss.detach().cpu().item(),
            r_loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )
