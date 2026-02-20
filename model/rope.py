import jax.numpy as jnp
from config import ModelConfig


# Precompute RoPE frequencies
def compute_inv_frequencies(config: ModelConfig):
    """
    Compute inverse frequency tensor for RoPE.
    Shape:
        (head_dim // 2,)
    """

    head_dim = config.head_dim
    half_dim = head_dim // 2

    inv_freq = 1.0 / (
        10000 ** (jnp.arange(0, half_dim) / half_dim)
    )

    return inv_freq

# Apply RoPE
def apply_rope(x, position, inv_freq):
    """
    Apply rotary embedding to tensor.
    x shape:
        (num_heads, head_dim)
    position:
        scalar absolute token position
    inv_freq:
        (head_dim // 2,)
    """

    half_dim = x.shape[-1] // 2

    # Split into even/odd parts
    x1 = x[:, :half_dim]
    x2 = x[:, half_dim:]

    # Compute angles
    theta = position * inv_freq  # (half_dim,)

    cos = jnp.cos(theta)
    sin = jnp.sin(theta)

    # Broadcast to heads
    cos = cos[None, :]
    sin = sin[None, :]

    # Apply rotation
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    return jnp.concatenate([rotated_x1, rotated_x2], axis=-1)