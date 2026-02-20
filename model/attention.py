import jax
import jax.numpy as jnp

from model.cache import _update_cache, get_visible_kv
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

    # Linear projections
    q = hidden @ Wq
    k = hidden @ Wk
    v = hidden @ Wv

    # Reshape into heads
    q = q.reshape(config.num_heads, config.head_dim)
    k = k.reshape(config.num_heads, config.head_dim)
    v = v.reshape(config.num_heads, config.head_dim)


    # Build full-layer tensors for update
    # We create zero tensors for all layers,
    # then fill only this layer's slot.

    new_k = jnp.zeros_like(cache.k[:, 0])
    new_v = jnp.zeros_like(cache.v[:, 0])

    new_k = new_k.at[layer_index].set(k)
    new_v = new_v.at[layer_index].set(v)

    cache = _update_cache(cache, new_k, new_v)

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
