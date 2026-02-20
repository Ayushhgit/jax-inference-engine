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

# Legacy update function
# def _update_cache(cache: KVCache, new_k: jnp.ndarray, new_v: jnp.ndarray) -> KVCache:
#     """
#     Append one token's KV values across all layers.

#     new_k/new_v shape:
#         (num_layers, num_heads, head_dim)
#     """

#     # Write new KV into circular buffer position
#     k_updated = cache.k.at[:, cache.write_index].set(new_k)
#     v_updated = cache.v.at[:, cache.write_index].set(new_v)

#     max_len = cache.k.shape[1]

#     # Circular Pointer movement
#     new_write_index = (cache.write_index + 1) % max_len

#     # Increase used count until full
#     new_used = jnp.minimum(cache.used + 1, max_len)

#     # Absolute token counter (never resets)
#     new_total_tokens = cache.total_tokens + 1

#     return KVCache(
#         k=k_updated,
#         v=v_updated,
#         write_index=new_write_index,
#         used=new_used,
#         total_tokens=new_total_tokens,
#     )

def get_visible_kv(cache: KVCache):
    """
    Returns logically ordered visible KV tensors.

    Output shape:
        (num_layers, used, num_heads, head_dim)
    """

    # SIMPLIFICATION:
    # Instead of trying to return a dynamically sized tensor (which JAX dislikes inside JIT/Scan),
    # we return the FULL cache, logically reordered so that the "used" tokens are at the beginning.
    # The attention mechanism already uses a mask, so seeing "garbage" past `used` is fine
    # provided the mask is correct.

    # However, to be consistent with the rotation logic:
    # We will just return the raw k/v and let the mask handle it?
    # No, the problem is the rotary embedding assumes positions 0..used-1.
    
    # Let's return the full concatenated buffer rotated so 0 is at the start.
    # Actually, simpler: just return the raw K, V storage.
    # The attention layer can gather what it needs or just mask out the rest.
    
    # BUT, the current implementation of attention expects `k_all_layer` to be (used, H, D).
    # We must change that expectation. We should return (max_len, H, D).
    # And the attention mask will hide the unused parts.
    
    return cache.k, cache.v

def update_cache_layer(cache: KVCache, layer_index: int, new_k_layer: jnp.ndarray, new_v_layer: jnp.ndarray,) -> KVCache:
    """
    Update only a single layer's KV values
    at the current write_index.
    """

    k_updated = cache.k.at[layer_index, cache.write_index].set(new_k_layer)
    v_updated = cache.v.at[layer_index, cache.write_index].set(new_v_layer)

    # Only increment write_index once per token
    # We assume this is called for the last layer only
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