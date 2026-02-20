import jax
import jax.numpy as jnp

from model.transformer import transformer_step
from config import ModelConfig


# Single Decode Step (Logits → Next Token)
def decode_step(params, token_id, cache, config: ModelConfig, temperature: float = 1.0, rng_key=None):
    """
    Performs one decoding step.

    Returns:
        next_token (int)
        updated cache
        logits
    """

    logits, cache = transformer_step(params, token_id, cache, config)

    # Temperature scaling
    scaled_logits = logits / temperature

    if rng_key is not None:
        # Stochastic sampling (temperature-aware)
        next_token = jax.random.categorical(rng_key, scaled_logits)
    else:
        # Greedy decoding (argmax)
        next_token = jnp.argmax(scaled_logits)

    return next_token, cache, logits