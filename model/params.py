import jax
import jax.numpy as jnp
from config import ModelConfig
from model.rope import compute_inv_frequencies

def init_layer_params(key, config: ModelConfig):
    """
    Initialize parameters for one transformer block
    """

    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    embed_dim = config.embed_dim
    hidden_dim = 4 * config.embed_dim  # standard transformer MLP expansion

    return {
        "Wq": jax.random.normal(k1, (embed_dim, embed_dim)) * 0.02,
        "Wk": jax.random.normal(k2, (embed_dim, embed_dim)) * 0.02,
        "Wv": jax.random.normal(k3, (embed_dim, embed_dim)) * 0.02,
        "Wo": jax.random.normal(k4, (embed_dim, embed_dim)) * 0.02,
        "W1": jax.random.normal(k5, (embed_dim, hidden_dim)) * 0.02,
        "W2": jax.random.normal(k6, (hidden_dim, embed_dim)) * 0.02,
        "rope_inv_freq": compute_inv_frequencies(config),
        # LayerNorm parameters (initialized to standard values)
        "ln1_gamma": jnp.ones(embed_dim),
        "ln1_beta": jnp.zeros(embed_dim),
        "ln2_gamma": jnp.ones(embed_dim),
        "ln2_beta": jnp.zeros(embed_dim),
    }

def init_params(key, config: ModelConfig):
    """
    Initialize full transformer parameters.
    """

    keys = jax.random.split(key, config.num_layers + 2)

    # Token embedding
    embedding = jax.random.normal(
        keys[0],
        (config.vocab_size, config.embed_dim)
    ) * 0.02

    # Transformer layers
    layers = [
        init_layer_params(keys[i + 1], config)
        for i in range(config.num_layers)
    ]

    # LM head
    lm_head = jax.random.normal(
        keys[-1],
        (config.embed_dim, config.vocab_size)
    ) * 0.02

    return {
        "embedding": embedding,
        "layers": layers,
        "lm_head": lm_head,
    }