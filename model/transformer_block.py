import jax
import jax.numpy as jnp

from model.attention import attention_layer
from config import ModelConfig

def layer_norm(x, eps=1e-5):
    """
    Simple layer normalization.
    """
    mean = jnp.mean(x)
    var = jnp.mean((x-mean)**2)
    return (x-mean)/jnp.sqrt(var+eps)

# MLP
def mlp(layer_params, x):
    """
    Transformer feedforward network.
    """

    w1 = layer_params["W1"]
    w2 = layer_params["W2"]

    hidden = x @ w1
    hidden = jax.nn.gelu(hidden)
    output = hidden @ w2

    return output

# Transformer Block
def transformer_block(layer_params, hidden, layer_index, cache, config: ModelConfig):
    """ 
    One transformer decoder block
    """
    # Self-Attention (Pre-norm)
    normed = layer_norm(hidden)

    attn_out, cache = attention_layer(
        layer_params,
        normed,
        layer_index,
        cache,
        config
    )

    hidden = hidden + attn_out # Residual connection

    # FeedForward (pre-norm)

    normed = layer_norm(hidden)
    mlp_out = mlp(layer_params, normed)
    hidden = hidden + mlp_out # Residual connection
    return hidden, cache