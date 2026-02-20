import jax
import jax.numpy as jnp
from typing import NamedTuple
from config import ModelConfig

class KVCache(NamedTuple):
    """
    Multi-layer rolling KV cache.
    k, v shape:
        (num_layers, max_cache_len, num_heads, head_dim)
    """

    k: jnp.ndarray
    v: jnp.ndarray
    write_index: jnp.ndarray
    used: jnp.ndarray
    total_tokens: jnp.ndarray

def _init_cache(config: ModelConfig) -> KVCache:
    """
    Initialize empty KV cache.
    """

    return KVCache(
        k=jnp.zeros(
            (
                config.num_layers,
                config.max_cache_len,
                config.num_heads,
                config.head_dim,
            )
        ),
        v=jnp.zeros(
            (
                config.num_layers,
                config.max_cache_len,
                config.num_heads,
                config.head_dim,
            )
        ),
        write_index=jnp.array(0),
        used=jnp.array(0),
        total_tokens=jnp.array(0),
    )

def get_visible_kv(cache: KVCache):
    """
    Returns visible KV tensors from the physical buffer.
    The attention mask handles hiding unused/invalid slots.

    Output shape:
        (num_layers, max_cache_len, num_heads, head_dim)
    """

    return cache.k, cache.v

def update_cache_layer(cache: KVCache, layer_index: int, new_k_layer: jnp.ndarray, new_v_layer: jnp.ndarray,) -> KVCache:
    """
    Update only a single layer's KV values
    at the current write_index.
    """

    k_updated = cache.k.at[layer_index, cache.write_index].set(new_k_layer)
    v_updated = cache.v.at[layer_index, cache.write_index].set(new_v_layer)

    return KVCache(
        k=k_updated,
        v=v_updated,
        write_index=cache.write_index,
        used=cache.used,
        total_tokens=cache.total_tokens,
    )

def advance_cache_pointer(cache: KVCache) -> KVCache:
    """
    Move circular pointer forward.
    Called once per token after all layers updated.
    """

    max_len = cache.k.shape[1]

    new_write = (cache.write_index + 1) % max_len
    new_used = jnp.minimum(cache.used + 1, max_len)
    new_total = cache.total_tokens + 1

    return KVCache(
        k=cache.k,
        v=cache.v,
        write_index=new_write,
        used=new_used,
        total_tokens=new_total,
    )