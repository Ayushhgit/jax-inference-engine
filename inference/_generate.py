import jax

from inference._decode import decode_step
from config import ModelConfig

# Autoregressive Generation (Compiled)
def generate(params, start_token, cache, config: ModelConfig, steps: int, temperature: float = 1.0):
    """
    Generate tokens autoregressively using jax.lax.scan.
    Returns:
        generated_tokens: (steps,)
        final_cache
    """

    def step_fn(carry, _):
        """
        carry = (current_token, cache)
        """

        current_token, cache = carry

        next_token, cache, _ = decode_step(params, current_token, cache, config, temperature,
        )

        return (next_token, cache), next_token

    # Initial carry
    init_carry = (start_token, cache)

    # Run scan
    (final_token, final_cache), generated_tokens = jax.lax.scan(
        step_fn,
        init_carry,
        xs=None,
        length=steps,
    )

    return generated_tokens, final_cache