"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from https://github.com/young-geng/CQL,
which is licensed under the MIT License.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
import utils


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            1.0 - soft_target_update_rate
        ) * target_network_params[k].data + soft_target_update_rate * v.data


class CQLTrainer:
    def __init__(
        self,
        model,
        policy_optimizer,
        qf_optimizer,
        alpha_optimizer,
        device="cuda",
    ):
        self.model = model
        self.device = device

        self.policy_optimizer = policy_optimizer
        self.qf_optimizer = qf_optimizer
        self.alpha_optimizer = alpha_optimizer

    def train_iteration(self, dataloader, loss_fn=None, num_updates=None):

        self.model.actor.train()
        self.model.qf1.train()
        self.model.qf2.train()
        self.model.log_alpha.train()

        train_start = time.time()
        qf1_losses, qf2_losses, actor_losses, alpha_losses, alphas = [], [], [], [], []

        total_it, done_training = 0, False
        while not done_training:
            for _, batch in enumerate(dataloader):
                actor_loss, qf1_loss, qf2_loss, alpha_loss, alpha = self.train(batch)
                actor_losses.append(actor_loss)
                qf1_losses.append(qf1_loss)
                qf2_losses.append(qf2_loss)
                alpha_losses.append(alpha_loss)
                alphas.append(alpha)

                total_it += 1
                if total_it >= num_updates:
                    done_training = True
                    break

        logs = {}
        logs["time/training"] = time.time() - train_start
        logs["training/actor_loss"] = np.mean(actor_losses)
        logs["training/qf1_loss"] = np.mean(qf1_losses)
        logs["training/qf2_loss"] = np.mean(qf2_losses)
        logs["training/alpha_loss"] = np.mean(alpha_losses)
        logs["training/alpha"] = np.mean(alphas)

        return logs

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(
            self.model.qf1, self.model.target_qf1, soft_target_update_rate
        )
        soft_target_update(
            self.model.qf2, self.model.target_qf2, soft_target_update_rate
        )

    def train(self, batch):

        (
            observations,
            next_observations,
            actions,
            rewards,
            avg_rtg,
            not_dones,
        ) = utils.to_torch(batch, device=self.device)
        observations = observations.reshape(-1, observations.shape[-1])
        next_observations = next_observations.reshape(-1, next_observations.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        rewards = rewards.reshape(-1)
        not_dones = not_dones.reshape(-1)

        new_actions, log_pi = self.model.actor(observations)

        alpha_loss = -(
            self.model.log_alpha() * (log_pi + self.model.target_entropy).detach()
        ).mean()
        alpha = self.model.log_alpha().exp() * self.model.alpha_multiplier

        """ Policy loss """
        q_new_actions = torch.min(
            self.model.qf1(observations, new_actions),
            self.model.qf2(observations, new_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """ Q function loss """

        q1_pred = self.model.qf1(observations, actions)
        q2_pred = self.model.qf2(observations, actions)

        new_next_actions, next_log_pi = self.model.actor(next_observations)
        target_q_values = torch.min(
            self.model.target_qf1(next_observations, new_next_actions),
            self.model.target_qf2(next_observations, new_next_actions),
        )

        q_target = (
            self.model.reward_scale * rewards + not_dones * self.model.discount * target_q_values
        )
        qf1_loss = F.mse_loss(q1_pred, q_target.detach())
        qf2_loss = F.mse_loss(q2_pred, q_target.detach())

        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.model.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.model.actor(
            observations, repeat=self.model.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.model.actor(
            next_observations, repeat=self.model.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.model.qf1(observations, cql_random_actions)
        cql_q2_rand = self.model.qf2(observations, cql_random_actions)
        cql_q1_current_actions = self.model.qf1(observations, cql_current_actions)
        cql_q2_current_actions = self.model.qf2(observations, cql_current_actions)
        cql_q1_next_actions = self.model.qf1(observations, cql_next_actions)
        cql_q2_next_actions = self.model.qf2(observations, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_pred, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_pred, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        # cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        # cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.model.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_min_qf1_loss = (
            torch.logsumexp(cql_cat_q1 / self.model.cql_temp, dim=1).mean()
            * self.model.cql_min_q_weight
            * self.model.cql_temp
        )
        cql_min_qf2_loss = (
            torch.logsumexp(cql_cat_q2 / self.model.cql_temp, dim=1).mean()
            * self.model.cql_min_q_weight
            * self.model.cql_temp
        )

        """Subtract the log likelihood of data"""
        cql_min_qf1_loss = (
            cql_min_qf1_loss - q1_pred.mean() * self.model.cql_min_q_weight
        )
        cql_min_qf2_loss = (
            cql_min_qf2_loss - q2_pred.mean() * self.model.cql_min_q_weight
        )

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # if self.total_steps % self.target_update_period == 0:
        self.update_target_network(self.model.soft_target_update_rate)

        # cql_metrics = dict(
        #     cql_std_q1=cql_std_q1.mean().item(),
        #     cql_std_q2=cql_std_q2.mean().item(),
        #     cql_q1_rand=cql_q1_rand.mean().item(),
        #     cql_q2_rand=cql_q2_rand.mean().item(),
        #     cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
        #     cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
        #     cql_q1_current_actions=cql_q1_current_actions.mean().item(),
        #     cql_q2_current_actions=cql_q2_current_actions.mean().item(),
        #     cql_q1_next_actions=cql_q1_next_actions.mean().item(),
        #     cql_q2_next_actions=cql_q2_next_actions.mean().item(),
        # )

        # metrics = dict(
        #     log_pi=log_pi.mean().item(),
        #     policy_loss=policy_loss.item(),
        #     qf1_loss=qf1_loss.item(),
        #     qf2_loss=qf2_loss.item(),
        #     alpha_loss=alpha_loss.item(),
        #     alpha=alpha.item(),
        #     average_qf1=q1_pred.mean().item(),
        #     average_qf2=q2_pred.mean().item(),
        #     average_target_q=target_q_values.mean().item(),
        #     total_steps=self.total_steps,
        # )

        # metrics.update(cql_metrics)

        return (
            policy_loss.detach().cpu().item(),
            qf1_loss.detach().cpu().item(),
            qf2_loss.detach().cpu().item(),
            alpha_loss.detach().cpu().item(),
            alpha.detach().cpu().item(),
        )
