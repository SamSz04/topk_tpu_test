"""Soft top-k relaxation with Sinkhorn-style normalization.

This is not an exact discrete top-k. It maps most heavy work to matrix-style
operations and is useful only where approximate differentiable selection is
acceptable.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .common import sanitize_scores as _sanitize_scores

@partial(jax.jit, static_argnames=("k", "iters", "temperature"))
def soft_topk_sinkhorn(
    x: jax.Array,
    *,
    k: int,
    iters: int = 8,
    temperature: float = 0.05,
) -> tuple[jax.Array, jax.Array]:
    """Soft top-k relaxation using an N x k Sinkhorn assignment matrix.

    Returns column-wise expected scores and expected indices. This is not a
    discrete exact top-k; it is included to test the MXU-friendly relaxation.
    """
    n = x.shape[0]
    scores = _sanitize_scores(x).astype(jnp.float32)
    ranks = jnp.linspace(1.0, 0.25, k, dtype=jnp.float32).reshape((1, k))
    logits = (scores.reshape((n, 1)) @ ranks) / jnp.float32(temperature)
    log_p = logits
    target_col_mass = jnp.full((k,), n / k, dtype=jnp.float32)

    for _ in range(iters):
        log_p = log_p - jax.nn.logsumexp(log_p, axis=1, keepdims=True)
        col_lse = jax.nn.logsumexp(log_p, axis=0, keepdims=True)
        log_p = log_p - col_lse + jnp.log(target_col_mass.reshape((1, k)))

    p = jnp.exp(log_p)
    denom = jnp.sum(p, axis=0) + 1e-6
    expected_scores = (p.T @ scores.reshape((n, 1))).reshape((k,)) / denom
    expected_idx = (p.T @ jnp.arange(n, dtype=jnp.float32).reshape((n, 1))).reshape((k,)) / denom
    return expected_scores.astype(x.dtype), expected_idx.astype(jnp.float32)
