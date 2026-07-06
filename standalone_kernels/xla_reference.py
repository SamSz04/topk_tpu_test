"""XLA/JAX reference top-k implementation."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .common import sanitize_scores


@partial(jax.jit, static_argnames=("k",))
def xla_topk_reference(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    vals, idx = lax.top_k(sanitize_scores(x), k)
    return vals, idx.astype(jnp.int32)
