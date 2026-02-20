import jax

from model.transformer import transformer_step
from config import ModelConfig

# Prefill Prompt into KV Cache
def prefill(params, prompt_tokens, cache, config: ModelConfig):
    """
    Runs the transformer over a full prompt sequence
    to populate the KV cache.
    Inputs:
        prompt_tokens: (seq_len,)
        cache: empty or existing KVCache
    Returns:
        last_token
        updated cache
    """

    def step_fn(cache, token):
        logits, cache = transformer_step(
            params,
            token,
            cache,
            config,
        )
        return cache, logits

    # Run through prompt
    final_cache, _ = jax.lax.scan(
        step_fn,
        cache,
        prompt_tokens,
    )

    # Last token becomes starting point for decode
    last_token = prompt_tokens[-1]

    return last_token, final_cache