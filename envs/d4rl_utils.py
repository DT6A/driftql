import contextlib
import io
import os
import warnings

os.environ.setdefault('D4RL_SUPPRESS_IMPORT_ERROR', '1')
warnings.simplefilter('ignore', DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import d4rl
import gym
import gymnasium
import numpy as np

from envs.env_utils import EpisodeMonitor
from utils.datasets import Dataset


class GymV21EnvCompatibility(gymnasium.Env):
    """Adapt a D4RL/Gym v0.21-style env to the Gymnasium reset/step API."""

    metadata = {}

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = getattr(env, "reward_range", None)
        self.metadata = getattr(env, "metadata", {})
        self.spec = getattr(env, "spec", None)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        warnings.simplefilter("ignore", DeprecationWarning)
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        return self.env.reset(*args, **kwargs), {}

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done and not truncated)
        return observation, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def make_env(env_name):
    """Make D4RL environment."""
    warnings.simplefilter("ignore", DeprecationWarning)
    env = gym.make(env_name)
    env = GymV21EnvCompatibility(env)
    env = EpisodeMonitor(env)
    return env


def get_dataset(
    env,
    env_name,
):
    """Make D4RL dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
    """
    dataset_env = env
    while not hasattr(dataset_env, 'get_dataset') and hasattr(dataset_env, 'env'):
        dataset_env = dataset_env.env
    dataset = d4rl.qlearning_dataset(dataset_env)

    terminals = np.zeros_like(dataset['rewards']) # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards']) # Indicate whether we should bootstrap from the next state.
    rewards = dataset['rewards'].copy().astype(np.float32)
    if 'antmaze' in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
            )
            masks[i] = 1 - dataset['terminals'][i]
        rewards = rewards - 1.0
    else:
        for i in range(len(terminals) - 1):
            if (
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
                or dataset['terminals'][i] == 1.0
            ):
                terminals[i] = 1
            else:
                terminals[i] = 0
            masks[i] = 1 - dataset['terminals'][i]
    masks[-1] = 1 - dataset['terminals'][-1]
    terminals[-1] = 1

    return Dataset.create(
        observations=dataset['observations'].astype(np.float32),
        actions=dataset['actions'].astype(np.float32),
        next_observations=dataset['next_observations'].astype(np.float32),
        terminals=terminals.astype(np.float32),
        rewards=rewards,
        masks=masks,
    )
