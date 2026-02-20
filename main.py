from config import ModelConfig
from model.cache import _init_cache, _update_cache
import jax.numpy as jnp

cfg = ModelConfig()
cache = _init_cache(cfg)

dummy_k = jnp.ones((cfg.num_layers, cfg.num_heads, cfg.head_dim))
dummy_v = jnp.ones((cfg.num_layers, cfg.num_heads, cfg.head_dim))

for i in range(20):
    cache = _update_cache(cache, dummy_k * i, dummy_v * i)

print("Used:", cache.used)
print("Write index:", cache.write_index)
print("Total tokens:", cache.total_tokens)
