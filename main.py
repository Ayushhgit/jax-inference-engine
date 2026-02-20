import jax
import jax.numpy as jnp

from config import ModelConfig
from model.params import init_params
from model.cache import _init_cache
from inference._generate import generate

cfg = ModelConfig()
params = init_params(jax.random.PRNGKey(0), cfg)
cache = _init_cache(cfg)

start_token = jnp.array(0)

tokens, cache = generate(
    params,
    start_token,
    cache,
    cfg,
    steps=20,
)

print("Generated:", tokens)