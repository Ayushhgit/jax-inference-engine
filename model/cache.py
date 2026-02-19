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


def _update_cache(cache: KVCache, new_k: jnp.ndarray, new_v: jnp.ndarray) -> KVCache:
    """
    Append one token's KV values across all layers.

    new_k/new_v shape:
        (num_layers, num_heads, head_dim)
    """

    # Write new KV into circular buffer position
    k_updated = cache.k.at[:, cache.write_index].set(new_k)
    v_updated = cache.v.at[:, cache.write_index].set(new_v)

    max_len = cache.k.shape[1]

    # Circular Pointer movement
    new_write_index = (cache.write_index + 1) % max_len

    # Increase used count until full
    new_used = jnp.minimum(cache.used + 1, max_len)

    # Absolute token counter (never resets)
    new_total_tokens = cache.total_tokens + 1

    return KVCache(
        k=k_updated,
        v=v_updated,
        write_index=new_write_index,
        used=new_used,
        total_tokens=new_total_tokens,
    )

def get_visible_kv(cache: KVCache):
    """
    Returns logically ordered visible KV tensors.

    Output shape:
        (num_layers, used, num_heads, head_dim)
    """

    max_len = cache.k.shape[1]

    def not_wrapped():
        # Cache not yet full
        return (
            cache.k[:, : cache.used],
            cache.v[:, : cache.used],
        )

    def wrapped():
        # Circular buffer wrapped
        return (
            jnp.concatenate(
                [
                    cache.k[:, cache.write_index :],
                    cache.k[:, : cache.write_index],
                ],
                axis=1,
            ),
            jnp.concatenate(
                [
                    cache.v[:, cache.write_index :],
                    cache.v[:, : cache.write_index],
                ],
                axis=1,
            ),
        )

    return jax.lax.cond(
        cache.used < max_len,
        not_wrapped,
        wrapped,
    )
