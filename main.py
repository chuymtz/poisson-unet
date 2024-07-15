import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from yaml import safe_load
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy


with open("config.yaml", "r") as f:
    params = safe_load(f)
    DOMAIN_EXTENT = params['simulation']['DOMAIN_EXTENT']
    NUM_POINTS = params['simulation']['NUM_POINTS']
    NUM_SAMPLES = params['simulation']['NUM_SAMPLES']

def solve_poisson(f):
    return jnp.linalg.solve(A, -f)

def create_discontinuity(key):
    limit_1_key, limit_2_key = jax.random.split(key)
    lower_limit = jax.random.uniform(limit_1_key, (), minval=0.2*DOMAIN_EXTENT, maxval=0.4*DOMAIN_EXTENT)
    upper_limit = jax.random.uniform(limit_2_key, (), minval=0.6*DOMAIN_EXTENT, maxval=0.8*DOMAIN_EXTENT)

    discontinuity = jnp.where((grid >= lower_limit) & (grid <= upper_limit), 1.0, 0.0)

    return discontinuity

grid = jnp.linspace(0, DOMAIN_EXTENT, NUM_POINTS+2)[1:-1]
dx = grid[1] - grid[0]


A = jnp.diag(jnp.ones(NUM_POINTS-1), -1)
A = A- 2*jnp.diag(jnp.ones(NUM_POINTS), 0)
A = A + jnp.diag(jnp.ones(NUM_POINTS-1), 1) 
A = A / dx**2

A.shape
plt.imshow(A)

# random keys
primary_key = jax.random.PRNGKey(0)
keys = jax.random.split(primary_key, NUM_SAMPLES)

# define the RHSs
force_fields = jax.vmap(create_discontinuity)(keys)
force_fields.shape

for i in range(force_fields.shape[0]):
    plt.plot(grid, force_fields[i])
    
    

displacement_fields = jax.vmap(solve_poisson)(force_fields)

for i in range(force_fields.shape[0]):
    plt.plot(grid, displacement_fields[i])