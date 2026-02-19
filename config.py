from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    configuration for the rolling KV JAX Transformer
    """
    # Vocabulary
    vocab_size: int = 256

    # Transformer Architecture
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2

    # Rolling KV cache window size
    max_cache_len: int = 16

    @property
    def head_dim(self) -> int:
        """
        Dimension per attention head.
        """

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )
        return self.embed_dim // self.num_heads