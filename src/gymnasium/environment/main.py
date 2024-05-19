import warnings
import os

from src.gymnasium.environment.lotka_volterra_env import LotkaVolterraEnv

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"

from ray.rllib.algorithms.ppo import PPOConfig
from ray.air import RunConfig, CheckpointConfig

from ray.tune.stopper import MaximumIterationStopper

import ray
from ray.tune import register_env, Tuner, TuneConfig


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init()
    env_name = "LotkaVolterraEnv-v0"
    register_env(env_name, lambda c: LotkaVolterraEnv(**c))

    config = PPOConfig()
    config = config.training(use_critic=True,
                             use_gae=False,
                             use_kl_loss=False,
                             lr=0.000089266,
                             train_batch_size=4096,
                             mini_batch_size_per_learner=512,
                             num_sgd_iter=4,
                             clip_param=0.148149,
                             vf_loss_coeff=0.805692,
                             entropy_coeff=0.00698166,
                             shuffle_sequences=True,
                             gamma=0.99,
                             model={"fcnet_hiddens": [64, 64, 32], "fcnet_activation": "tanh"})

    config = config.resources(num_cpus_per_worker=1)

    config = config.env_runners(num_env_runners=11,
                                num_envs_per_env_runner=1,
                                validate_env_runners_after_construction=False,
                                explore=True,
                                batch_mode="complete_episodes",
                                exploration_config={"type": "StochasticSampling"},
                                rollout_fragment_length="auto",
                                enable_connectors=False)

    config = config.framework(framework="torch")

    config = config.environment(env=env_name,
                                normalize_actions=False,
                                clip_actions=True)

    config = config.debugging(log_level="ERROR",
                              log_sys_usage=False)

    config = config.reporting(min_sample_timesteps_per_iteration=0, min_time_s_per_iteration=0, metrics_num_episodes_for_smoothing=1)

    config = config.evaluation(evaluation_interval=0, evaluation_duration=100,
                               evaluation_config={"explore": False,})

    config = config.experimental(_enable_new_api_stack=False)

    checkpoint_config = CheckpointConfig(num_to_keep=1200, checkpoint_frequency=10, checkpoint_at_end=True)

    run_config = RunConfig(stop=MaximumIterationStopper(max_iter=1200), checkpoint_config=checkpoint_config)

    tune_config = TuneConfig(num_samples=1, reuse_actors=False)

    Tuner("PPO", param_space=config.to_dict(), tune_config=tune_config, run_config=run_config).fit()

    ray.shutdown()
