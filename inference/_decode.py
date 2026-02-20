import jax
import jax.numpy as jnp

from model.transformer import transformer_step
from config import ModelConfig


# Single Decode Step (Logits → Next Token)
def decode_step(params, token_id, cache, config: ModelConfig, temperature: float = 1.0):
    """
    Performs one decoding step.

    Returns:
        next_token (int)
        updated cache
        logits
    """

    logits, cache = transformer_step(params, token_id, cache, config)

    # Temperature scaling
    logits = logits / temperature

    # Convert to probabilities
    probs = jax.nn.softmax(logits)

    # Greedy decoding (argmax)
    next_token = jnp.argmax(probs)

    return next_token, cache, logits