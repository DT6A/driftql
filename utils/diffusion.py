from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0, 0.999)


def vp_beta_schedule(timesteps):
    t = jnp.arange(1, timesteps + 1)
    t_total = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = jnp.exp(-b_min / t_total - 0.5 * (b_max - b_min) * (2 * t - 1) / (t_total**2))
    return 1.0 - alpha


class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            kernel = self.param(
                'kernel',
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            features = 2 * jnp.pi * x @ kernel.T
        else:
            half_dim = self.output_size // 2
            features = jnp.log(10000.0) / (half_dim - 1)
            features = jnp.exp(jnp.arange(half_dim) * -features)
            features = x * features
        return jnp.concatenate([jnp.cos(features), jnp.sin(features)], axis=-1)


@partial(
    jax.jit,
    static_argnames=('actor_apply_fn', 'action_dim', 'num_steps', 'repeat_last_step', 'clip_sampler', 'training'),
)
def ddpm_sampler(
    actor_apply_fn,
    actor_params,
    observations,
    rng,
    action_dim,
    num_steps,
    alphas,
    alpha_hats,
    betas,
    sample_temperature=1.0,
    repeat_last_step=0,
    clip_sampler=True,
    training=False,
):
    batch_size = observations.shape[0]

    def reverse_step(carry, time):
        current_x, rng = carry
        times = jnp.full((batch_size, 1), time, dtype=jnp.float32)
        eps_pred = actor_apply_fn(
            {'params': actor_params},
            observations,
            current_x,
            times,
            training=training,
        )

        alpha_1 = 1.0 / jnp.sqrt(alphas[time])
        alpha_2 = (1.0 - alphas[time]) / jnp.sqrt(1.0 - alpha_hats[time])
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, (batch_size, action_dim))
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * sample_temperature * noise)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1.0, 1.0)

        return (current_x, rng), ()

    rng, init_rng = jax.random.split(rng)
    init_x = jax.random.normal(init_rng, (batch_size, action_dim))
    (actions, rng), _ = jax.lax.scan(reverse_step, (init_x, rng), jnp.arange(num_steps - 1, -1, -1))

    for _ in range(repeat_last_step):
        (actions, rng), _ = reverse_step((actions, rng), 0)

    return jnp.clip(actions, -1.0, 1.0), rng
