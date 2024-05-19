from typing import Optional

import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Box


class LotkaVolterraEnv(gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super(LotkaVolterraEnv, self).__init__()
        self.action_space = Box(low=-0.002, high=0.0, shape=(4,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=1.5, shape=(8,), dtype=np.float32)

        self.alpha_matrix = np.array(
            [
                [1, 1.09, 1.52, 0],
                [0, 1, 0.44, 1.36],
                [2.33, 0, 1, 0.47],
                [1.21, 0.51, 0.35, 1]
            ])

        self.r = [1, 0.72, 1.53, 1.27]

        self.state: np.ndarray | None = None

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        delta_state = self.lotka_volterra_competition_4d(self.state[:4] + action, self.alpha_matrix, self.r)

        tmp = self.state[:4] + delta_state + action
        tmp = np.array([max(i, 0) for i in tmp])

        t = self.lotka_volterra_competition_4d(tmp, self.alpha_matrix, self.r)

        self.state = np.concatenate([tmp, t], dtype=np.float32)
        x = np.sum(np.absolute(delta_state))

        loss = x < 0.001 or any([self.state[i] == 0.0 for i in range(4)])
        return self.state, -x if not loss else -x - 100, loss, loss, {"abs_sum_of_delta_state": x}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> tuple[ObsType, dict]:
        self.state = np.random.uniform(low=0.006, high=0.9, size=(4,)).astype(np.float32)
        delta_state = self.lotka_volterra_competition_4d(self.state, self.alpha_matrix, self.r)
        self.state = np.concatenate([self.state, delta_state], axis=-1, dtype=np.float32)
        return self.state, {}

    def render(self):
        raise NotImplementedError("render method is not implemented.")

    def close(self):
        pass

    @staticmethod
    def lotka_volterra_competition_4d(y, alpha_matrix, r) -> np.ndarray:
        x1, x2, x3, x4 = y[0], y[1], y[2], y[3]
        dx1dt = r[0] * x1 - r[0] * x1 * (alpha_matrix[0, 0] * x1 + alpha_matrix[0, 1] * x2 + alpha_matrix[0, 2] * x3 + alpha_matrix[0, 3] * x4)
        dx2dt = r[1] * x2 - r[1] * x2 * (alpha_matrix[1, 0] * x1 + alpha_matrix[1, 1] * x2 + alpha_matrix[1, 2] * x3 + alpha_matrix[1, 3] * x4)
        dx3dt = r[2] * x3 - r[2] * x3 * (alpha_matrix[2, 0] * x1 + alpha_matrix[2, 1] * x2 + alpha_matrix[2, 2] * x3 + alpha_matrix[2, 3] * x4)
        dx4dt = r[3] * x4 - r[3] * x4 * (alpha_matrix[3, 0] * x1 + alpha_matrix[3, 1] * x2 + alpha_matrix[3, 2] * x3 + alpha_matrix[3, 3] * x4)
        return np.array([dx1dt, dx2dt, dx3dt, dx4dt], dtype=np.float32)
