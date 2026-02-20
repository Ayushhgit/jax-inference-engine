import jax.numpy as jnp
from model.cache import KVCache

def sliding_window_mask(cache: KVCache):
    """
    Returns boolean mask for valid tokens in the cache buffer.
    Since get_visible_kv now returns the full physical buffer,
    we simply mask out slots that haven't been used yet.
    
    Because the inference is autoregressive (sequential),
    we don't need a causal mask (future masking) for the cache content 
    itself, as the cache only contains past and current tokens.
    
    Output:
        mask shape: (max_len,)
    """
    
    # Static shape from the array
    max_len = cache.k.shape[1]
    
    # Create static indices 0..max_len-1
    idxs = jnp.arange(max_len)
    
    # Valid slots are those < used
    # This works for both filling phase (used < max_len)
    # and wrapped phase (used = max_len)
    mask = idxs < cache.used
    
    return mask

# Deprecated/Unused helper (removed to avoid confusion/errors)
# def get_visible_positions(cache: KVCache): ...