import time

import jax
import jax.numpy as jnp

print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')

# 1. Use larger arrays (10,000 x 10,000 is ~400MB per matrix)
size = 10000
print(f'\n1. Creating {size}x{size} matrices...')
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(key1, (size, size))
y = jax.random.normal(key2, (size, size))

x.block_until_ready()
y.block_until_ready()
print('✓ Arrays loaded to GPU.')


@jax.jit
def heavy_matmul(a, b):
    return jnp.dot(a, b)


print('\n2. Compiling (JIT)...')
_ = heavy_matmul(x, y).block_until_ready()
print('✓ Compiled.')

print('\n3. Stressing GPU (check nvidia-smi now!)...')
start_time = time.time()

iterations = 50
for i in range(iterations):
    result = heavy_matmul(x, y)

result.block_until_ready()

end_time = time.time()
print(f'✓ Completed {iterations} heavy operations in {end_time - start_time:.2f} seconds.')
