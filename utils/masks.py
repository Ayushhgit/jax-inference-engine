import jax.numpy as jnp
from model.cache import KVCache

def get_visible_positions(cache: KVCache):
    """
    Returns the absolute positions currently visible in the sliding window.

    Output:
        positions: (used,)
    """
     
    total = cache.total_tokens
    used = cache.used

    start = total - used
    return jnp.arange(start, total)

def sliding_window_mask(cache: KVCache):
    """
    Returns boolean mask for visible tokens
    relative to current decoding step.

    Output:
        mask shape: (used,)
    """

    total = cache.total_tokens
    used = cache.used
    max_len = cache.k.shape[1]

    # Absolute positions in window
    positions = get_visible_positions(cache)

    current_position = total - 1

    # Causal condition
    causal = positions <= current_position

    # Sliding window constraint
    window = positions > current_position - max_len

    return causal & window