import os
import sys
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from utils.flax_utils import nonpytree_field


_UPSTREAM_IDQL_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'external', 'IDQL')
if _UPSTREAM_IDQL_ROOT not in sys.path:
    sys.path.insert(0, _UPSTREAM_IDQL_ROOT)

from jaxrl5.agents.ddpm_iql.ddpm_iql_learner import (  # noqa: E402
    DDPMIQLLearner,
    compute_q,
    expectile_loss,
    exponential_loss,
    quantile_loss,
)
from jaxrl5.networks.diffusion import ddpm_sampler  # noqa: E402


def _box_from_example(example, low, high):
    shape = tuple(example.shape[1:])
    dtype = np.asarray(example).dtype
    return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


def _get_with_alias(config, primary, alias, default):
    if primary in config:
        return config[primary]
    if alias in config:
        return config[alias]
    return default


def _canonicalize_config(config, ex_actions):
    cfg = dict(config)

    lr = cfg.get('lr', 3e-4)
    cfg['actor_lr'] = cfg.get('actor_lr', lr)
    cfg['critic_lr'] = cfg.get('critic_lr', lr)
    cfg['value_lr'] = cfg.get('value_lr', lr)

    cfg['critic_hyperparam'] = _get_with_alias(cfg, 'critic_hyperparam', 'expectile', 0.7)
    cfg['policy_temperature'] = _get_with_alias(cfg, 'policy_temperature', 'alpha', 3.0)
    cfg['T'] = _get_with_alias(cfg, 'T', 'diffusion_steps', 5)
    cfg['N'] = _get_with_alias(cfg, 'N', 'num_samples', 64)
    cfg['M'] = cfg.get('M', cfg.get('repeat_last_step', 0))

    cfg['actor_hidden_dims'] = tuple(cfg.get('actor_hidden_dims', (256, 256, 256)))
    cfg['value_hidden_dims'] = tuple(cfg.get('value_hidden_dims', (256, 256)))

    cfg['discount'] = cfg.get('discount', 0.99)
    cfg['tau'] = cfg.get('tau', 0.005)
    cfg['actor_tau'] = cfg.get('actor_tau', 0.001)
    cfg['ddpm_temperature'] = cfg.get('ddpm_temperature', 1.0)
    cfg['num_qs'] = cfg.get('num_qs', 2)
    cfg['time_dim'] = cfg.get('time_dim', 64)
    cfg['clip_sampler'] = cfg.get('clip_sampler', True)
    cfg['actor_architecture'] = cfg.get('actor_architecture', 'mlp')
    cfg['actor_num_blocks'] = cfg.get('actor_num_blocks', 2)
    cfg['actor_weight_decay'] = cfg.get('actor_weight_decay', None)
    cfg['actor_dropout_rate'] = cfg.get('actor_dropout_rate', None)
    cfg['actor_layer_norm'] = cfg.get('actor_layer_norm', False)
    cfg['actor_objective'] = cfg.get('actor_objective', 'bc')
    cfg['critic_objective'] = cfg.get('critic_objective', 'expectile')
    cfg['beta_schedule'] = cfg.get('beta_schedule', 'vp')
    cfg['decay_steps'] = cfg.get('decay_steps', int(2e6))
    cfg['encoder'] = cfg.get('encoder', None)

    cfg['action_dim'] = ex_actions.shape[-1]
    return cfg


class IDQLAgent(flax.struct.PyTreeNode):
    """Thin repo adapter over the vendored official IDQL learner."""

    learner: Any
    config: Any = nonpytree_field()

    @staticmethod
    def _actor_weights(adv, actor_objective, policy_temperature, critic_hyperparam):
        if actor_objective == 'soft_adv':
            return jnp.where(adv > 0, critic_hyperparam, 1 - critic_hyperparam)
        if actor_objective == 'hard_adv':
            return jnp.where(adv >= -0.01, 1.0, 0.0)
        if actor_objective == 'exp_adv':
            return jnp.minimum(jnp.exp(adv * policy_temperature), 100.0)
        if actor_objective == 'bc':
            return jnp.ones_like(adv)
        raise ValueError(f'Invalid actor objective: {actor_objective}')

    @jax.jit
    def update(self, batch):
        learner, info = self.learner.update(batch)
        return self.replace(learner=learner), info

    @jax.jit
    def total_loss(self, batch, grad_params=None):
        del grad_params

        qs = self.learner.target_critic.apply_fn(
            {'params': self.learner.target_critic.params},
            batch['observations'],
            batch['actions'],
        )
        q = qs.min(axis=0)
        v = self.learner.value.apply_fn({'params': self.learner.value.params}, batch['observations'])

        if self.config['critic_objective'] == 'expectile':
            value_loss = expectile_loss(q - v, self.config['critic_hyperparam']).mean()
        elif self.config['critic_objective'] == 'quantile':
            value_loss = quantile_loss(q - v, self.config['critic_hyperparam']).mean()
        elif self.config['critic_objective'] == 'exponential':
            value_loss = exponential_loss(q - v, self.config['critic_hyperparam']).mean()
        else:
            raise ValueError(f'Invalid critic objective: {self.config["critic_objective"]}')

        next_v = self.learner.value.apply_fn(
            {'params': self.learner.value.params},
            batch['next_observations'],
        )
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        pred_qs = self.learner.critic.apply_fn(
            {'params': self.learner.critic.params},
            batch['observations'],
            batch['actions'],
        )
        critic_loss = ((pred_qs - target_q) ** 2).mean()

        rng = self.learner.rng
        key, rng = jax.random.split(rng)
        times = jax.random.randint(key, (batch['actions'].shape[0],), 0, self.config['T'])
        key, rng = jax.random.split(rng)
        noise = jax.random.normal(key, (batch['actions'].shape[0], self.config['action_dim']))

        alpha_hats = self.learner.alpha_hats[times]
        times = jnp.expand_dims(times, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise

        adv = q - v
        weights = self._actor_weights(
            adv,
            self.config['actor_objective'],
            self.config['policy_temperature'],
            self.config['critic_hyperparam'],
        )
        weights = jax.lax.stop_gradient(weights)

        eps_pred = self.learner.score_model.apply_fn(
            {'params': self.learner.score_model.params},
            batch['observations'],
            noisy_actions,
            times,
            training=False,
        )
        actor_loss = (((eps_pred - noise) ** 2).sum(axis=-1) * weights).mean()

        info = {
            'actor_loss': actor_loss,
            'weights': weights.mean(),
            'value_loss': value_loss,
            'v': v.mean(),
            'critic_loss': critic_loss,
            'q': pred_qs.mean(),
        }
        return actor_loss + value_loss + critic_loss, info

    def sample_actions(self, observations, seed=None, temperature=1.0):
        del temperature

        seed = self.learner.rng if seed is None else seed
        single_input = observations.ndim == len(self.config['ob_dims'])
        if single_input:
            observations = jnp.expand_dims(observations, axis=0)

        batch_size = observations.shape[0]
        tiled_observations = jnp.repeat(observations, self.config['N'], axis=0)
        actions, _ = ddpm_sampler(
            self.learner.score_model.apply_fn,
            self.learner.target_score_model.params,
            self.config['T'],
            seed,
            self.config['action_dim'],
            tiled_observations,
            self.learner.alphas,
            self.learner.alpha_hats,
            self.learner.betas,
            self.config['ddpm_temperature'],
            self.config['M'],
            self.config['clip_sampler'],
        )
        qs = compute_q(
            self.learner.target_critic.apply_fn,
            self.learner.target_critic.params,
            tiled_observations,
            actions,
        )

        actions = actions.reshape(self.config['N'], batch_size, self.config['action_dim'])
        qs = qs.reshape(self.config['N'], batch_size)
        best_idx = jnp.argmax(qs, axis=0)
        batch_idx = jnp.arange(batch_size)
        actions = actions[best_idx, batch_idx]

        if single_input:
            actions = actions[0]
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = _canonicalize_config(config, ex_actions)
        if config['encoder'] is not None:
            raise ValueError('The upstream IDQL adapter currently supports state observations only.')

        obs_space = _box_from_example(ex_observations, low=-1.0, high=1.0)
        act_space = _box_from_example(ex_actions, low=-1.0, high=1.0)
        learner = DDPMIQLLearner.create(
            seed=seed,
            observation_space=obs_space,
            action_space=act_space,
            actor_architecture=config['actor_architecture'],
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            value_lr=config['value_lr'],
            critic_hidden_dims=config['value_hidden_dims'],
            actor_hidden_dims=config['actor_hidden_dims'],
            discount=config['discount'],
            tau=config['tau'],
            critic_hyperparam=config['critic_hyperparam'],
            ddpm_temperature=config['ddpm_temperature'],
            num_qs=config['num_qs'],
            actor_num_blocks=config['actor_num_blocks'],
            actor_weight_decay=config['actor_weight_decay'],
            actor_tau=config['actor_tau'],
            actor_dropout_rate=config['actor_dropout_rate'],
            actor_layer_norm=config['actor_layer_norm'],
            policy_temperature=config['policy_temperature'],
            T=config['T'],
            time_dim=config['time_dim'],
            N=config['N'],
            M=config['M'],
            clip_sampler=config['clip_sampler'],
            actor_objective=config['actor_objective'],
            critic_objective=config['critic_objective'],
            beta_schedule=config['beta_schedule'],
            decay_steps=config['decay_steps'],
        )

        config['ob_dims'] = tuple(ex_observations.shape[1:])
        return cls(learner=learner, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='idql',
            action_dim=ml_collections.config_dict.placeholder(int),
            ob_dims=ml_collections.config_dict.placeholder(tuple),
            batch_size=256,
            lr=3e-4,
            actor_lr=3e-4,
            critic_lr=3e-4,
            value_lr=3e-4,
            actor_hidden_dims=(256, 256, 256),
            value_hidden_dims=(256, 256),
            discount=0.99,
            tau=0.005,
            actor_tau=0.001,
            actor_architecture='mlp',
            actor_num_blocks=2,
            actor_weight_decay=0.0,
            actor_dropout_rate=0.0,
            actor_layer_norm=False,
            policy_temperature=3.0,
            critic_hyperparam=0.7,
            actor_objective='bc',
            critic_objective='expectile',
            diffusion_steps=5,
            T=5,
            time_dim=64,
            num_samples=64,
            N=64,
            repeat_last_step=0,
            M=0,
            ddpm_temperature=1.0,
            clip_sampler=True,
            beta_schedule='vp',
            decay_steps=int(2e6),
            expectile=0.7,
            alpha=3.0,
            num_qs=2,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
