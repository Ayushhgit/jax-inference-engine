import jax
import jax.numpy as jnp

from model.cache import update_cache_layer, get_visible_kv
from utils.masks import sliding_window_mask
from config import ModelConfig

def attention_layer(layer_params, hidden, layer_index, cache, config: ModelConfig):
    """
    Performs attention for one transformer layer.

    hidden: (embed_dim,)
    """

    Wq = layer_params["Wq"]
    Wk = layer_params["Wk"]
    Wv = layer_params["Wv"]
    Wo = layer_params["Wo"]
    inv_freq = layer_params["rope_inv_freq"]

    # Linear projections
    q = hidden @ Wq
    k = hidden @ Wk
    v = hidden @ Wv

    # Reshape into heads
    q = q.reshape(config.num_heads, config.head_dim)
    k = k.reshape(config.num_heads, config.head_dim)
    v = v.reshape(config.num_heads, config.head_dim)

    # Apply RoPE
    # We need the current absolute position for RoPE
    # The cache.total_tokens tracks the number of tokens processed so far
    # The current token being processed is at position `total_tokens`
    position = cache.total_tokens
    
    from model.rope import apply_rope
    q = apply_rope(q, position, inv_freq)
    k = apply_rope(k, position, inv_freq)

    # Update this layer's cache
    cache = update_cache_layer(cache, layer_index, k, v)

    # Retrieve visible KV
    k_all, v_all = get_visible_kv(cache)

    # Extract this layer's KV
    k_all_layer = k_all[layer_index] # (used, H, D)
    v_all_layer = v_all[layer_index] # (used, H, D)

    # Attention scores
    # (H, D) x (T, H, D) -> (H, T)
    scores = jnp.einsum("hd,thd->ht", q, k_all_layer)
    scores = scores / jnp.sqrt(config.head_dim)

    # Sliding window mask
    mask = sliding_window_mask(cache)
    scores = jnp.where(mask, scores, -1e9)

    # Softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)

    # Weighted Sum
    output = jnp.einsum("ht,thd->hd", attn_weights, v_all_layer)

    # Merge Heads
    output = output.reshape(config.embed_dim)

    output = output @ Wo

    return output, cache
