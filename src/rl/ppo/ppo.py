import random
from typing import Optional

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from src.rl.ppo.network import ActorCritic
from src.rl.ppo.memory import PPOMemory


class PPO:
    def __init__(self,
                 num_actions: int = 4,
                 lr_actor=1.00e-4,
                 lr_critic=2.00e-4,
                 batch_size=1024,
                 rollout_size=1024,
                 entropy_coefficient=0.001,
                 gamma=0.95,
                 optimizer_epochs=10,
                 epsilon_clip=0.2,
                 gpu=False):

        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.optimizer_epochs = optimizer_epochs
        self.critic_coefficient = 0.5
        self.batch_size = batch_size
        self.rollout_size = rollout_size
        self.entropy_coefficient = entropy_coefficient
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')
        self.loss = nn.MSELoss()
        self.network = ActorCritic(num_actions=num_actions, lr_actor=lr_actor, lr_critic=lr_critic, device=self.device)
        self.network.eval()

    def train(self, ppo_memory: PPOMemory, iteration: int, writer: SummaryWriter | None):
        # ---------------Loss lists--------------- #
        ppo_loss_list = []
        loss_actor_list = []
        loss_critic_list = []
        # ---------------Loss lists--------------- #
        self.network.train()
        obs, action, log_probs, rewards, terminal = ppo_memory.get_all_data()

        returns = torch.tensor(self.generate_returns(rewards, terminal), dtype=torch.float32, device=self.device).unsqueeze(dim=-1)
        returns = ((returns - returns.mean(dim=0)) / (returns.std(dim=0) + 1e-7))

        for _ in range(self.optimizer_epochs):
            for i, (obs_batch, action_batch, log_probs_batch, returns_batch) in enumerate(self.generate_batches(obs, action, log_probs, returns, batch_size=self.batch_size)):
                probs, state_values, dist_entropy = self.network.evaluate(obs_batch, action_batch)
                advantages = returns_batch - state_values.detach()
                # advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).detach()

                ratios = torch.exp(probs - log_probs_batch)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages

                loss_actor = -torch.min(surr1, surr2).mean() - (self.entropy_coefficient * dist_entropy).mean()
                loss_critic = self.loss(returns_batch, state_values) * self.critic_coefficient
                loss = loss_critic + loss_actor
                self.network.optimizer.zero_grad()
                loss.backward()
                self.network.optimizer.step()
                # ---------------Loss lists---------------#
                loss_actor_list.append(loss_actor.detach())
                loss_critic_list.append(loss_critic.detach())
                ppo_loss_list.append(loss.detach())
                # ---------------Loss lists---------------#
        print()
        ppo_memory.clear()
        torch.cuda.empty_cache()
        self.network.eval()

        if writer is not None:
            writer.add_scalar("loss/ppo_loss", torch.mean(torch.stack(ppo_loss_list, dim=0)), iteration)
            tmp = torch.mean(torch.stack(loss_actor_list, dim=0))
            writer.add_scalar("loss/actor_loss", tmp, iteration)
            writer.add_scalar("loss/critic_loss", torch.mean(torch.stack(loss_critic_list, dim=0)), iteration)
            writer.add_scalar("lr/actor_lr", self.network.optimizer.param_groups[1]["lr"], iteration)
            writer.add_scalar("lr/critic_lr", self.network.optimizer.param_groups[2]["lr"], iteration)

    def generate_returns(self, reward_in, terminal_in):
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(reward_in), reversed(terminal_in)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        return returns

    @staticmethod
    def generate_batches(observations: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor, returns: torch.Tensor, batch_size: int, drop_last=True):
        num_samples = len(actions)
        num_batches = num_samples // batch_size

        if not drop_last and num_samples % batch_size != 0:
            num_batches += 1

        # Create a list of indices and shuffle them
        indices = list(range(num_samples))
        random.shuffle(indices)

        for i in range(0, num_batches * batch_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            obs_batch = observations[batch_indices]
            act_batch = actions[batch_indices]
            log_probs_batch = log_probs[batch_indices]
            ret_batch = returns[batch_indices]
            yield obs_batch, act_batch, log_probs_batch, ret_batch

    def load_model(self, model_path):
        if model_path is None:
            return
        try:
            loading_state = torch.load(model_path)
            self.network.load_state_dict(state_dict=loading_state["network"])
            self.network.optimizer.load_state_dict(loading_state["optimizer"])
            print("Loading model was successfully done.")
        except IOError:
            print("\nError while loading model.")

    def store_model(self, model_path):
        torch.save({"network": self.network.state_dict(), "optimizer": self.network.optimizer.state_dict()}, model_path)
