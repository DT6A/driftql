import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


def _l2_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _pairwise_l2(a, b, eps=1e-12):
    """Pairwise L2: a [N,F], b [M,F] -> [N,M]."""
    a2 = jnp.sum(a * a, axis=-1, keepdims=True)
    b2 = jnp.sum(b * b, axis=-1, keepdims=True)
    dist2 = a2 + b2.T - 2.0 * (a @ b.T)
    return jnp.sqrt(jnp.maximum(dist2, 0.0) + eps)


def _cfg_get(cfg, key, default):
    try:
        return cfg[key]
    except KeyError:
        return default


def _resolve_noise_dim(cfg, action_dim):
    noise_dim = int(_cfg_get(cfg, 'noise_dim', action_dim))
    if noise_dim <= 0:
        noise_dim = int(action_dim)
    return noise_dim


def _compute_drift(
        gen_a: jnp.ndarray,  # [Ngen, A] generated actions (= q samples)
        pos_a: jnp.ndarray,  # [Npos, A] dataset actions (= p samples)
        temp: float,
        kernel: str = 'laplace',  # "laplace" or "gaussian"
        dim_scale: bool = True,  # divide distances by sqrt(d_a)
        eps: float = 1e-12,
):
    """
    Compute the drift field for one state.

    Returns:
        V:          [Ngen, A] total drift = V_p - V_q
        V_attract:  [Ngen, A] attraction component V_p(x)
        V_repel:    [Ngen, A] repulsion component V_q(x)
    """
    Ngen, action_dim = gen_a.shape

    # Pairwise distances
    dist_pos = _pairwise_l2(gen_a, pos_a, eps=eps)  # [Ngen, Npos]
    dist_neg = _pairwise_l2(gen_a, gen_a, eps=eps)  # [Ngen, Ngen]

    # Self-mask: each sample cannot repel itself
    idx = jnp.arange(Ngen)
    dist_neg = dist_neg.at[idx, idx].set(1e6)

    if dim_scale:
        dim_factor = jnp.sqrt(jnp.asarray(action_dim, dtype=gen_a.dtype))
    else:
        dim_factor = 1.0

    # Kernel logits
    if kernel == 'gaussian':
        # Gaussian: k(x,y) = exp(-||x-y||^2 / 2tau^2)
        dist_pos_scaled = dist_pos / dim_factor
        dist_neg_scaled = dist_neg / dim_factor
        logit_pos = -(dist_pos_scaled ** 2) / (2.0 * temp * temp)
        logit_neg = -(dist_neg_scaled ** 2) / (2.0 * temp * temp)
    else:
        # Laplace: k(x,y) = exp(-||x-y|| / tau)
        logit_pos = -(dist_pos / dim_factor) / temp
        logit_neg = -(dist_neg / dim_factor) / temp

    # Separate softmax. V_p and V_q are independently normalized
    W_pos = jax.nn.softmax(logit_pos, axis=-1)  # [Ngen, Npos]
    W_neg = jax.nn.softmax(logit_neg, axis=-1)  # [Ngen, Ngen]

    disp_to_pos = pos_a[None, :, :] - gen_a[:, None, :]  # [Ngen, Npos, A]
    disp_to_neg = gen_a[None, :, :] - gen_a[:, None, :]  # [Ngen, Ngen, A]

    # V_{pi,k}(x) = sum_j w_j (y_j - x)
    V_attract = jnp.sum(W_pos[:, :, None] * disp_to_pos, axis=1)  # [Ngen, A]
    V_repel = jnp.sum(W_neg[:, :, None] * disp_to_neg, axis=1)  # [Ngen, A]

    V = V_attract - V_repel

    return V, V_attract, V_repel


def _compute_drift_batched(
        gen_a: jnp.ndarray,  # [B, Ngen, A]
        pos_a: jnp.ndarray,  # [B, Npos, A]
        temp: float,
        kernel: str = 'laplace',
        dim_scale: bool = True,
        drift_normalize: bool = False,
        eps: float = 1e-12,
):
    """
    Batched drift field computation.

    Returns:
        V:          [B, Ngen, A]
        V_attract:  [B, Ngen, A]
        V_repel:    [B, Ngen, A]
    """

    def per_item(gen_i, pos_i):
        V_i, Va_i, Vr_i = _compute_drift(
            gen_a=gen_i,
            pos_a=pos_i,
            temp=temp,
            kernel=kernel,
            dim_scale=dim_scale,
            eps=eps,
        )

        if drift_normalize:
            act_dim = gen_i.shape[-1]
            raw_sq = jnp.mean(jnp.sum(V_i * V_i, axis=-1)) / act_dim
            lam = jax.lax.stop_gradient(jnp.sqrt(raw_sq + eps))
            V_i = V_i / lam
            Va_i = Va_i / lam
            Vr_i = Vr_i / lam

        return V_i, Va_i, Vr_i

    return jax.vmap(per_item)(gen_a, pos_a)


class DriftQLAgent(flax.struct.PyTreeNode):
    """
    DriftQL with theoretically-grounded drift field.

    The drift field follows the drifting model framework exactly:
      Delta_{p,q}(x) = V_{p,k}(x) - V_{q,k}(x)

    where V_{pi,k}(x) is the kernel-weighted direction.
    With Gaussian kernel, this is exact score matching on smoothed distributions.
    With Laplace kernel, it approximates score matching with O(1/D) error.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)

        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def drifting_bc_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, sub_rng = jax.random.split(rng, 3)

        # Read config
        kernel = str(_cfg_get(self.config, 'kernel', 'laplace'))
        noise_dim = _resolve_noise_dim(self.config, self.config['action_dim'])
        dim_scale = bool(_cfg_get(self.config, 'dim_scale', True))
        drift_normalize = bool(self.config['drift_normalize'])
        temp = float(self.config['drift_temp'])
        eta = float(self.config['drift_eta'])
        Ngen = int(self.config['drift_ngen'])

        # Optional subsample for drift computation
        drift_bs = int(self.config['drift_batch_size'])
        if drift_bs < batch_size:
            idx = jax.random.choice(sub_rng, batch_size, (drift_bs,), replace=False)
            obs = batch['observations'][idx]
            pos_actions = batch['actions'][idx]
        else:
            obs = batch['observations']
            pos_actions = batch['actions']
            drift_bs = batch_size

        # Positive actions: dataset actions for each state [B, 1, A]
        pos_a = jnp.clip(pos_actions[:, None, :], -1.0, 1.0)

        # Generate Ngen samples per state from current policy
        noises = jax.random.normal(noise_rng, (drift_bs * Ngen, noise_dim))
        obs_rep = jnp.repeat(obs, repeats=Ngen, axis=0)
        bc_raw = self.network.select('actor_bc_drift')(obs_rep, noises, params=grad_params)
        gen_a = jnp.clip(bc_raw.reshape(drift_bs, Ngen, action_dim), -1.0, 1.0)

        # Compute drift field: Delta_{p,q}(x) = V_p(x) - V_q(x)
        V, V_attract, V_repel = _compute_drift_batched(
            gen_a=gen_a,
            pos_a=pos_a,
            temp=temp,
            kernel=kernel,
            dim_scale=dim_scale,
            drift_normalize=drift_normalize,
            eps=float(self.config['drift_eps']),
        )

        target = jax.lax.stop_gradient(jnp.clip(gen_a + eta * V, -1.0, 1.0))
        bc_drift_loss = jnp.mean((gen_a - target) ** 2)
        attract_norm = jnp.mean(_l2_norm(V_attract, axis=-1))
        repel_norm = jnp.mean(_l2_norm(V_repel, axis=-1))
        drift_norm = jnp.mean(_l2_norm(V, axis=-1))

        info = {
            'bc_drift_loss': bc_drift_loss,
            'drift_norm': drift_norm,
            'drift_attract_norm': attract_norm,
            'drift_repel_norm': repel_norm,
        }

        return bc_drift_loss, info

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, bc_rng = jax.random.split(rng, 3)

        bc_drift_loss, bc_info = self.drifting_bc_loss(batch, grad_params, bc_rng)

        # Q-loss: maximize Q(s, f_θ(s,z))
        noise_dim = _resolve_noise_dim(self.config, action_dim)
        noises = jax.random.normal(noise_rng, (batch_size, noise_dim))
        actor_raw = self.network.select('actor_bc_drift')(batch['observations'], noises, params=grad_params)
        actor_raw = jnp.clip(actor_raw, -1.0, 1.0)
        qs = self.network.select('critic')(batch['observations'], actions=actor_raw)

        q_agg = _cfg_get(self.config, 'q_agg_actor', self.config['q_agg'])
        if q_agg == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

        q_loss = -q.mean()

        if self.config['normalize_q_loss']:
            lam_q = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
            q_loss = lam_q * q_loss

        actor_loss = self.config['alpha'] * bc_drift_loss + q_loss

        info = {
            'actor_loss': actor_loss,
            'bc_drift_loss': bc_drift_loss * self.config['alpha'],
            'bc_drift_loss_raw': bc_drift_loss,
            'drift_norm': bc_info['drift_norm'],
            'drift_attract_norm': bc_info['drift_attract_norm'],
            'drift_repel_norm': bc_info['drift_repel_norm'],
            'q_loss': q_loss,
            'q': q.mean(),
        }

        return actor_loss, info

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

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        action_seed, _ = jax.random.split(seed)
        noise_dim = _resolve_noise_dim(self.config, self.config['action_dim'])
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                noise_dim,
            ),
        )
        raw_actions = self.network.select('actor_bc_drift')(observations, noises)
        actions = jnp.clip(raw_actions, -1.0, 1.0)
        actions = jnp.where(jnp.isnan(actions), 0.0, actions)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        noise_dim = _resolve_noise_dim(config, action_dim)
        ex_noises = jnp.zeros((*ex_actions.shape[:-1], noise_dim), dtype=ex_actions.dtype)

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_drift'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )

        actor_bc_drift_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_drift'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_drift=(actor_bc_drift_def, (ex_observations, ex_noises)),
        )

        if encoders.get('actor_bc_drift') is not None:
            network_info['actor_bc_drift_encoder'] = (
                encoders.get('actor_bc_drift'),
                (ex_observations,),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['noise_dim'] = noise_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='driftql',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            noise_dim=0,  # defaults to action_dim when <= 0
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,  # Polyak target update rate
            q_agg='min',  # "min" or "mean" for critic target
            q_agg_actor='mean',  # "min" or "mean" for actor Q-loss
            alpha=10.0,  # drift loss weight
            normalize_q_loss=False,
            # Drift field
            kernel='laplace',  # "laplace" or "gaussian"
            drift_temp=0.2,  # kernel temperature
            dim_scale=True,  # divide distances by sqrt(d_a) before applying temp
            drift_ngen=32,  # number of generated samples per state
            drift_eta=1.0,  # drift step size eta
            drift_normalize=False,  # keep raw magnitude unless explicitly enabled
            drift_batch_size=256,  # subsample size for drift computation
            drift_eps=1e-12,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
