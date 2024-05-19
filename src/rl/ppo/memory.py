from typing import List

import torch


class PPOMemory:
    def __init__(self, device="cuda:0"):
        super(PPOMemory, self).__init__()
        # ------------- PPO -------------#
        self.matrix = []
        self.actions = []
        self.log_probabilities = []
        self.rewards = []
        self.terminals = []
        # ------------- PPO -------------#
        self.device = device

    def add_data(self, matrix, action, log_probs, reward, terminal) -> None:
        self.matrix.append(torch.from_numpy(matrix).to(self.device))
        self.actions.append(action)
        self.log_probabilities.append(log_probs)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def get_all_data(self) -> (dict, torch.Tensor, torch.Tensor, List[float], List[bool]):
        return torch.stack(self.matrix, dim=0), torch.stack(self.actions, dim=0), torch.cat(self.log_probabilities, dim=0), self.rewards, self.terminals

    def clear(self):
        self.matrix.clear()
        self.actions.clear()
        self.log_probabilities.clear()
        self.rewards.clear()
        self.terminals.clear()

    def __len__(self) -> int:
        return len(self.matrix)
