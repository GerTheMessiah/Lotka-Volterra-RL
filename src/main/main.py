import os
import pathlib
from datetime import timedelta
from time import time_ns

import gymnasium.wrappers
from torch.utils.tensorboard import SummaryWriter

from src.gymnasium.environment import LotkaVolterraEnv
from src.rl.ppo.memory import PPOMemory

from src.rl.ppo.ppo import PPO
from src.rl.ppo.util import print_progress


def train_ppo(epochs: int = 50000,
              lr_actor: float = 1.5e-4,
              lr_critic: float = 5e-4,
              batch_size: int = 1024,
              rollout_size: int = 1024,
              entropy_coefficient: float = 0.001,
              gamma: float = 0.95,
              optimizer_epochs: int = 10,
              epsilon_clip: float = 0.2,
              gpu: bool = True,
              record: bool = False):

    start_time = time_ns()

    if record:
        src_path = pathlib.Path(__file__).parent.parent.parent
        tensorboard_path = os.path.join(src_path, 'resources', 'tensorboard')
        # writer: SummaryWriter = SummaryWriter(log_dir=tensorboard_path, flush_secs=30)
        # writer.add_custom_scalars(layout=layout)

    game = gymnasium.wrappers.ClipAction(LotkaVolterraEnv())

    ppo = PPO(lr_actor=lr_actor,
              lr_critic=lr_critic,
              batch_size=batch_size,
              rollout_size=rollout_size,
              gamma=gamma,
              optimizer_epochs=optimizer_epochs,
              entropy_coefficient=entropy_coefficient,
              epsilon_clip=epsilon_clip,
              gpu=gpu)

    # ppo.load_model(model_path=model_path)

    ppo_memory = PPOMemory()

    training_game_counter = 0
    for i in range(1, epochs + 1):
        score, terminal, truncated = 0.0, False, False
        obs, info = game.reset()
        while not terminal and not truncated:
            action, log_probs = ppo.network.forward_stochastic(obs=obs)
            obs_, reward, terminal, truncated, info = game.step(action=action.cpu().numpy())
            ppo_memory.add_data(matrix=obs, action=action, log_probs=log_probs, reward=reward, terminal=terminal or truncated)
            score += reward
            obs = obs_

        # ---------------learn--------------- #
        training_game_counter += 1
        if len(ppo_memory) >= ppo.rollout_size:
            ppo.train(ppo_memory, iteration=i, writer=None)
            training_game_counter = 0
        # ---------------learn--------------- #

        t = time_ns()
        passed_time = str(timedelta(microseconds=(t - start_time) / 1_000)).split('.')[0]
        suffix_1 = f"Epoch {i} / {epochs} | Time: {passed_time} | Score: {score:.3f}"
        print_progress(i, epochs, suffix=suffix_1)


if __name__ == '__main__':
    train_ppo(epochs=50000,
              lr_actor=1.5e-4,
              lr_critic=5e-4,
              batch_size=1024,
              rollout_size=1024,
              entropy_coefficient=0.001,
              gamma=0.95,
              optimizer_epochs=10,
              epsilon_clip=0.2,
              gpu=True,
              record=False)
