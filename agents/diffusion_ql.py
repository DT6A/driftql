import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.diffusion import cosine_beta_schedule, vp_beta_schedule
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, ensemblize


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return jnp.linspace(beta_start, beta_end, timesteps)


def make_beta_schedule(schedule, timesteps):
    if schedule == 'linear':
        return linear_beta_schedule(timesteps)
    if schedule == 'cosine':
        return cosine_beta_schedule(timesteps)
    if schedule == 'vp':
        return vp_beta_schedule(timesteps)
    raise ValueError(f'Unsupported beta schedule: {schedule}')


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal timestep embedding used by the reference Diffusion-QL actor."""

    output_size: int

    @nn.compact
    def __call__(self, times):
        times = jnp.asarray(times, dtype=jnp.float32)
        if times.ndim == 1:
            times = times[:, None]
        times = times[..., :1]

        half_dim = self.output_size // 2
        if half_dim <= 1:
            raise ValueError('time_dim must be at least 4.')
        scale = jnp.log(10000.0) / (half_dim - 1)
        frequencies = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -scale)
        embeddings = times * frequencies[None, :]
        return jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)


class DiffusionScore(nn.Module):
    """Epsilon-prediction network for a state-conditioned DDPM policy."""

    hidden_dims: tuple
    action_dim: int
    time_dim: int = 16
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self):
        self.time_mlp = MLP((self.time_dim * 2, self.time_dim), activations=mish, activate_final=False)
        self.reverse_mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activations=mish,
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    @nn.compact
    def __call__(self, observations, actions, times, training=False, is_encoded=False):
        del training
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        time_embeddings = SinusoidalTimeEmbedding(self.time_dim)(times)
        time_embeddings = self.time_mlp(time_embeddings)
        inputs = jnp.concatenate([actions, time_embeddings, observations], axis=-1)
        return self.reverse_mlp(inputs)


class DiffusionCritic(nn.Module):
    """Twin-Q critic matching the Diffusion-QL Mish MLP architecture."""

    hidden_dims: tuple
    layer_norm: bool = False
    num_qs: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_qs > 1:
            mlp_class = ensemblize(MLP, self.num_qs)
        self.value_net = mlp_class(
            (*self.hidden_dims, 1),
            activations=mish,
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, actions):
        if self.encoder is not None:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, actions], axis=-1)
        return self.value_net(inputs).squeeze(-1)


class DiffusionQLAgent(flax.struct.PyTreeNode):
    """Diffusion-QL agent implemented natively in JAX/Flax."""

    rng: Any
    network: Any
    betas: Any
    alphas: Any
    alpha_hats: Any
    sqrt_alpha_hats: Any
    sqrt_one_minus_alpha_hats: Any
    sqrt_recip_alpha_hats: Any
    sqrt_recipm1_alpha_hats: Any
    posterior_log_variance_clipped: Any
    posterior_mean_coef1: Any
    posterior_mean_coef2: Any
    config: Any = nonpytree_field()

    def _q_sample(self, actions, times, noise):
        alpha_1 = self.sqrt_alpha_hats[times][:, None]
        alpha_2 = self.sqrt_one_minus_alpha_hats[times][:, None]
        return alpha_1 * actions + alpha_2 * noise

    def _predict_start_from_noise(self, noisy_actions, times, noise):
        x_start = self.sqrt_recip_alpha_hats[times][:, None] * noisy_actions
        x_start = x_start - self.sqrt_recipm1_alpha_hats[times][:, None] * noise
        if self.config['clip_denoised']:
            x_start = jnp.clip(x_start, -1.0, 1.0)
        return x_start

    def _sample_diffusion_actions(self, observations, seed, params, module_name, training=False):
        batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        sample_temperature = self.config['sample_temperature']

        def reverse_step(carry, time):
            current_x, rng = carry
            times = jnp.full((batch_size,), time, dtype=jnp.int32)
            model_times = times[:, None].astype(jnp.float32)
            eps_pred = self.network.select(module_name)(
                observations,
                current_x,
                model_times,
                params=params,
                training=training,
            )
            x_recon = self._predict_start_from_noise(current_x, times, eps_pred)
            model_mean = self.posterior_mean_coef1[time] * x_recon + self.posterior_mean_coef2[time] * current_x

            rng, noise_rng = jax.random.split(rng)
            noise = jax.random.normal(noise_rng, (batch_size, action_dim))
            nonzero_mask = (time > 0).astype(current_x.dtype)
            std = jnp.exp(0.5 * self.posterior_log_variance_clipped[time])
            current_x = model_mean + nonzero_mask * std * sample_temperature * noise

            if self.config['clip_sampler']:
                current_x = jnp.clip(current_x, -1.0, 1.0)

            return (current_x, rng), ()

        seed, init_rng = jax.random.split(seed)
        init_actions = jax.random.normal(init_rng, (batch_size, action_dim))
        timesteps = jnp.arange(self.config['diffusion_steps'] - 1, -1, -1)
        (actions, seed), _ = jax.lax.scan(reverse_step, (init_actions, seed), timesteps)

        for _ in range(self.config['repeat_last_step']):
            (actions, seed), _ = reverse_step((actions, seed), jnp.asarray(0, dtype=jnp.int32))

        return jnp.clip(actions, -1.0, 1.0)

    def critic_loss(self, batch, grad_params, rng):
        batch_size = batch['actions'].shape[0]

        if self.config['max_q_backup']:
            num_backup_samples = self.config['num_backup_samples']
            next_observations = jnp.repeat(batch['next_observations'], repeats=num_backup_samples, axis=0)
            next_actions = self._sample_diffusion_actions(
                next_observations,
                rng,
                params=self.network.params,
                module_name='target_actor',
            )
            target_qs = self.network.select('target_critic')(next_observations, next_actions)
            target_q = target_qs.min(axis=0).reshape(batch_size, num_backup_samples).max(axis=1)
        else:
            next_actions = self._sample_diffusion_actions(
                batch['next_observations'],
                rng,
                params=self.network.params,
                module_name='target_actor',
            )
            target_qs = self.network.select('target_critic')(batch['next_observations'], next_actions)
            if self.config['q_agg'] == 'mean':
                target_q = target_qs.mean(axis=0)
            else:
                target_q = target_qs.min(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * target_q
        target_q = jax.lax.stop_gradient(target_q)

        qs = self.network.select('critic')(
            batch['observations'],
            batch['actions'],
            params=grad_params,
        )
        critic_loss = jnp.square(qs - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'target_q_mean': target_q.mean(),
            'target_q_max': target_q.max(),
            'target_q_min': target_q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_size = batch['actions'].shape[0]
        _, time_rng, noise_rng, sample_rng, q_head_rng = jax.random.split(rng, 5)

        times = jax.random.randint(time_rng, (batch_size,), 0, self.config['diffusion_steps'])
        noise = jax.random.normal(noise_rng, batch['actions'].shape)
        noisy_actions = self._q_sample(batch['actions'], times, noise)

        eps_pred = self.network.select('actor')(
            batch['observations'],
            noisy_actions,
            times[:, None].astype(jnp.float32),
            params=grad_params,
            training=True,
        )
        bc_loss = jnp.mean((eps_pred - noise) ** 2)

        actions = self._sample_diffusion_actions(
            batch['observations'],
            sample_rng,
            params=grad_params,
            module_name='actor',
            training=True,
        )
        qs = self.network.select('critic')(batch['observations'], actions)

        if self.config['q_loss_agg'] == 'mean':
            q = qs.mean(axis=0)
            q_loss = -q.mean()
            normalizer = jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            q_loss = q_loss / normalizer
        elif self.config['q_loss_agg'] == 'min':
            q = qs.min(axis=0)
            q_loss = -q.mean()
            normalizer = jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            q_loss = q_loss / normalizer
        else:
            q1, q2 = qs[0], qs[1]
            q1_loss = -q1.mean() / jax.lax.stop_gradient(jnp.abs(q2).mean() + 1e-6)
            q2_loss = -q2.mean() / jax.lax.stop_gradient(jnp.abs(q1).mean() + 1e-6)
            use_q1 = jax.random.bernoulli(q_head_rng)
            q_loss = jnp.where(use_q1, q1_loss, q2_loss)
            q = jnp.where(use_q1, q1, q2)

        actor_loss = bc_loss + self.config['eta'] * q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'ql_loss': q_loss,
            'q_mean': q.mean(),
            'sample_action_mean': actions.mean(),
            'sample_action_std': actions.std(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name, tau, should_update=True):
        source_key = f'modules_{module_name}'
        target_key = f'modules_target_{module_name}'
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: jnp.where(should_update, tau * p + (1.0 - tau) * tp, tp),
            network.params[source_key],
            network.params[target_key],
        )
        network.params[target_key] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic', self.config['tau'])

        if self.config['step_start_ema'] <= 0:
            should_update_actor = True
        else:
            should_update_actor = new_network.step >= self.config['step_start_ema']
        should_update_actor = should_update_actor & ((new_network.step % self.config['update_ema_every']) == 0)
        self.target_update(new_network, 'actor', self.config['actor_tau'], should_update=should_update_actor)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        del temperature
        seed = self.rng if seed is None else seed
        single_input = observations.ndim == len(self.config['ob_dims'])
        if single_input:
            observations = jnp.expand_dims(observations, axis=0)

        batch_size = observations.shape[0]
        num_samples = self.config['num_action_samples']
        tiled_observations = jnp.repeat(observations, repeats=num_samples, axis=0)
        sample_seed, select_seed = jax.random.split(seed)
        actions = self._sample_diffusion_actions(
            tiled_observations,
            sample_seed,
            params=self.network.params,
            module_name='actor',
        )

        if num_samples > 1:
            qs = self.network.select('target_critic')(tiled_observations, actions).min(axis=0)
            qs = qs.reshape(batch_size, num_samples)
            actions = actions.reshape(batch_size, num_samples, self.config['action_dim'])
            if self.config['sample_action_selection'] == 'argmax':
                action_idx = jnp.argmax(qs, axis=-1)
            else:
                logits = qs / self.config['action_selection_temperature']
                action_idx = jax.random.categorical(select_seed, logits, axis=-1)
            batch_idx = jnp.arange(batch_size)
            actions = actions[batch_idx, action_idx]

        actions = jnp.clip(actions, -1.0, 1.0)
        actions = jnp.where(jnp.isnan(actions), 0.0, actions)
        if single_input:
            actions = actions[0]
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        ex_times = jnp.zeros((*ex_actions.shape[:-1], 1), dtype=ex_actions.dtype)

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        critic_def = DiffusionCritic(
            hidden_dims=tuple(config['critic_hidden_dims']),
            layer_norm=config['critic_layer_norm'],
            num_qs=config['num_qs'],
            encoder=encoders.get('critic'),
        )
        actor_def = DiffusionScore(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=action_dim,
            time_dim=config['time_dim'],
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
        )
        target_actor_def = copy.deepcopy(actor_def)

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, ex_actions, ex_times)),
            target_actor=(target_actor_def, (ex_observations, ex_actions, ex_times)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        tx = optax.adam(learning_rate=config['lr'])
        if config['grad_norm'] > 0:
            tx = optax.chain(optax.clip_by_global_norm(config['grad_norm']), tx)

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=tx)
        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']

        betas = make_beta_schedule(config['beta_schedule'], config['diffusion_steps'])
        alphas = 1.0 - betas
        alpha_hats = jnp.cumprod(alphas, axis=0)
        alpha_hats_prev = jnp.concatenate([jnp.ones((1,), dtype=alpha_hats.dtype), alpha_hats[:-1]], axis=0)
        posterior_variance = betas * (1.0 - alpha_hats_prev) / (1.0 - alpha_hats)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        if config['num_action_samples'] < 1:
            config['num_action_samples'] = 1
        if config['num_backup_samples'] < 1:
            config['num_backup_samples'] = 1

        return cls(
            rng=rng,
            network=network,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hats,
            sqrt_alpha_hats=jnp.sqrt(alpha_hats),
            sqrt_one_minus_alpha_hats=jnp.sqrt(1.0 - alpha_hats),
            sqrt_recip_alpha_hats=jnp.sqrt(1.0 / alpha_hats),
            sqrt_recipm1_alpha_hats=jnp.sqrt(1.0 / alpha_hats - 1.0),
            posterior_log_variance_clipped=jnp.log(jnp.clip(posterior_variance, min=1e-20)),
            posterior_mean_coef1=betas * jnp.sqrt(alpha_hats_prev) / (1.0 - alpha_hats),
            posterior_mean_coef2=(1.0 - alpha_hats_prev) * jnp.sqrt(alphas) / (1.0 - alpha_hats),
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='diffusion_ql',
            ob_dims=ml_collections.config_dict.placeholder(tuple),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(256, 256, 256),
            critic_hidden_dims=(256, 256, 256),
            actor_layer_norm=False,
            critic_layer_norm=False,
            discount=0.99,
            tau=0.005,
            eta=1.0,
            num_qs=2,
            q_agg='min',
            q_loss_agg='random',
            max_q_backup=False,
            num_backup_samples=10,
            beta_schedule='linear',
            diffusion_steps=100,
            time_dim=16,
            sample_temperature=1.0,
            clip_denoised=True,
            clip_sampler=True,
            repeat_last_step=0,
            actor_tau=0.005,
            step_start_ema=1000,
            update_ema_every=5,
            grad_norm=1.0,
            num_action_samples=50,
            sample_action_selection='softmax',
            action_selection_temperature=1.0,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
