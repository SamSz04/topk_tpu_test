"""Bitonic-sort top-k JAX reference.

The original Pallas bitonic implementation was removed from this standalone
directory because it does not reliably compile on TPU Mosaic: its rank-changing
reshape can lower to unsupported casts such as `vector<256> -> vector<128x2x1>`.
Use `fixed_shape_bitonic_v6.py` for directly usable Pallas bitonic variants.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .common import NEG_INF, next_power_of_2, sanitize_scores


@partial(jax.jit, static_argnames=("k",))
def bitonic_sort_jax(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    """Data-oblivious bitonic top-k reference implemented in JAX/XLA."""
    n_orig = x.shape[0]
    n = next_power_of_2(n_orig)
    pad = n - n_orig
    vals = jnp.pad(sanitize_scores(x), (0, pad), constant_values=NEG_INF)
    keys = -vals
    inds = jnp.pad(jnp.arange(n_orig, dtype=jnp.int32), (0, pad), constant_values=n_orig)

    size = 2
    while size <= n:
        stride = size // 2
        while stride >= 1:
            groups = n // (2 * stride)
            key_pairs = keys.reshape((groups, 2, stride))
            idx_pairs = inds.reshape((groups, 2, stride))
            a_key = key_pairs[:, 0, :]
            b_key = key_pairs[:, 1, :]
            a_idx = idx_pairs[:, 0, :]
            b_idx = idx_pairs[:, 1, :]
            a_less = (a_key < b_key) | ((a_key == b_key) & (a_idx < b_idx))
            min_key = jnp.where(a_less, a_key, b_key)
            min_idx = jnp.where(a_less, a_idx, b_idx)
            max_key = jnp.where(a_less, b_key, a_key)
            max_idx = jnp.where(a_less, b_idx, a_idx)
            group_ids = jnp.arange(groups, dtype=jnp.int32)
            asc = ((group_ids * jnp.int32(2 * stride)) & jnp.int32(size)) == 0
            asc = asc[:, None]
            first_key = jnp.where(asc, min_key, max_key)
            first_idx = jnp.where(asc, min_idx, max_idx)
            second_key = jnp.where(asc, max_key, min_key)
            second_idx = jnp.where(asc, max_idx, min_idx)
            keys = jnp.stack([first_key, second_key], axis=1).reshape((n,))
            inds = jnp.stack([first_idx, second_idx], axis=1).reshape((n,))
            stride //= 2
        size *= 2
    return -keys[:k], inds[:k].astype(jnp.int32)
