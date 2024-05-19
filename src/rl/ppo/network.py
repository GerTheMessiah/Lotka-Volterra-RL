import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class Function(nn.Module):
    def __init__(self):
        super(Function, self).__init__()

    def forward(self, x):
        return -0.002 * (1 - torch.exp(torch.abs(x)))


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.critic_net = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.Tanh(),
            nn.Linear(8, 8, bias=False),
            nn.Tanh(),
            nn.Linear(8, 1, bias=False),
        )

    def forward(self, matrix) -> torch.Tensor:
        return self.critic_net(matrix)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.actor_net_mean = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.Tanh(),
            nn.Linear(8, 8, bias=False),
            nn.Tanh(),
            nn.Linear(8, 4, bias=False),
            Function()
        )

    def forward(self, obs) -> (torch.Tensor, torch.Tensor):
        return self.actor_net_mean(obs)


class ActorCritic(nn.Module):
    def __init__(self, num_actions: int = 4, lr_actor: float = 1.5e-4, lr_critic: float = 3.0e-4, device="cuda:0"):
        super(ActorCritic, self).__init__()
        torch.set_default_dtype(torch.float32)
        np.random.seed(12)
        torch.manual_seed(12)
        torch.cuda.manual_seed_all(12)
        torch.set_printoptions(profile="full")

        self.actor = Actor()
        self.critic = Critic()
        self.actor_logstd = nn.Parameter(torch.zeros(num_actions, dtype=torch.float32), requires_grad=True)

        self.device = device
        self.to(self.device)

        self.optimizer = Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic},
            {'params': self.actor_logstd, 'lr': lr_actor},
        ], foreach=False, fused=True)

        self.eval()

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        text = f"Number of total trainable parameters: {pytorch_total_params}"
        print(text)
        print("-" * len(text))

    def forward(self, obs: torch.Tensor):
        action_mean = self.actor(obs)
        values = self.critic(obs)
        return action_mean, values

    @torch.no_grad()
    def forward_stochastic(self, obs: np.ndarray):
        obs = self.make_tensor(obs=obs)
        action_mean = self.actor(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action, log_probs = self.determine_action(action_mean, action_std)
        return action, log_probs.sum(-1, keepdim=True)

    @torch.no_grad()
    def forward_deterministic(self, obs: np.ndarray):
        obs = self.make_tensor(obs=obs)

        action_mean = self.actor(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        action, log_probs = self.determine_action(action_mean, action_std)
        return action, log_probs.sum(-1, keepdim=True)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        action_mean, values = self(obs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        action_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_probs.sum(-1, keepdim=True), values, dist_entropy.sum(-1, keepdim=True)

    def make_tensor(self, obs) -> torch.Tensor:
        return torch.from_numpy(obs).to(self.device)

    @staticmethod
    def determine_action(mean: torch.Tensor, var: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        dist = Normal(mean, var)
        action = dist.sample()
        return action, dist.log_prob(action).unsqueeze(0)
