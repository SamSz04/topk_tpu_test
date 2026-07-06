"""Iterative max-mask top-k.

Repeat reduce-max, choose the lowest matching index as the deterministic
 tie-breaker, then mask that position to -inf.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .common import NEG_INF, sanitize_scores


@partial(jax.jit, static_argnames=("k",))
def iterative_mask_jax(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    n = x.shape[0]
    idxs = jnp.arange(n, dtype=jnp.int32)
    work = sanitize_scores(x)
    out_vals = jnp.full((k,), NEG_INF, dtype=x.dtype)
    out_idx = jnp.full((k,), -1, dtype=jnp.int32)

    def body(i, state):
        work_i, vals_i, idx_i = state
        m = jnp.max(work_i, axis=0)
        candidates = jnp.where(work_i == m, idxs, jnp.int32(n))
        arg = jnp.min(candidates).astype(jnp.int32)
        vals_i = vals_i.at[i].set(m)
        idx_i = idx_i.at[i].set(arg)
        work_i = jnp.where(idxs == arg, NEG_INF, work_i)
        return work_i, vals_i, idx_i

    _, vals, idx = lax.fori_loop(0, k, body, (work, out_vals, out_idx))
    return vals, idx


def _iterative_mask_kernel(k: int, n: int):
    def kernel(x_ref, vals_ref, idx_ref):
        idxs = jnp.arange(n, dtype=jnp.int32)
        work = sanitize_scores(x_ref[...]).astype(jnp.float32)
        neg_inf = jnp.array(NEG_INF, dtype=jnp.float32)
        vals = []
        idx_out = []
        for _ in range(k):
            m = jnp.max(work, axis=0)
            candidates = jnp.where(work == m, idxs, jnp.int32(n))
            arg = jnp.min(candidates).astype(jnp.int32)
            vals.append(m)
            idx_out.append(arg)
            work = jnp.where(idxs == arg, neg_inf, work)
        vals_ref[...] = jnp.stack(vals)
        idx_ref[...] = jnp.stack(idx_out)
    return kernel


@partial(jax.jit, static_argnames=("k",))
def iterative_mask_pallas(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    n = x.shape[0]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((n,), lambda: (0,))],
        out_specs=(
            pl.BlockSpec((k,), lambda: (0,)),
            pl.BlockSpec((k,), lambda: (0,)),
        ),
        grid=(),
    )
    vals, idx = pl.pallas_call(
        _iterative_mask_kernel(k, n),
        out_shape=(
            jax.ShapeDtypeStruct((k,), jnp.float32),
            jax.ShapeDtypeStruct((k,), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=()),
    )(x)
    return vals.astype(x.dtype), idx
