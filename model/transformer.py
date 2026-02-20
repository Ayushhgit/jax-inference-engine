import jax.numpy as jnp

from model.transformer_block import transformer_block
from config import ModelConfig

# Single Decode Step
def transformer_step(params, token_id, cache, config: ModelConfig ):
    """
    Performs one autoregressive decoding step.
    Inputs:
        token_id: scalar int
        cache: KVCache
    Returns:
        logits: (vocab_size,)
        updated cache
    """

    # Token Embeddings
    embedding_matrix = params["embedding"]
    hidden = embedding_matrix[token_id]  # (embed_dim,)

    # Pass thorugh Transformer layer
    for layer_index in range(config.num_layers):

        layer_params = params["layers"][layer_index]

        hidden, cache = transformer_block(
            layer_params,
            hidden,
            layer_index,
            cache,
            config,
        )

    # Final LM head Projection
    lm_head = params["lm_head"]

    logits = hidden @ lm_head  # (vocab_size,)

    # Advance cache pointer for the next token
    from model.cache import advance_cache_pointer
    cache = advance_cache_pointer(cache)

    return logits, cache
