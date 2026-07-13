"""Exact top-k implementations used by the benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from fast_exact_topk_tpu import top_bounded_k, topk


NEG_INF = -jnp.inf
V6_TILE_ROWS = 16
V6_TILE_COLS = 256
V6_TILE_SIZE = V6_TILE_ROWS * V6_TILE_COLS

IMPL_LAX = "lax_top_k"
IMPL_FAST = "fast_exact"
IMPL_ITERATIVE_JAX = "iterative_jax"
IMPL_ITERATIVE_PALLAS = "iterative_pallas"
IMPL_BUCKET = "bucket_v6_exact"
IMPL_BITONIC_JAX = "bitonic_jax"
IMPL_FIXED_BITONIC_A = "fixed_bitonic_a"
IMPL_FIXED_BITONIC_B = "fixed_bitonic_b"
IMPL_PALLAS_HIGH = "pallas_high"
IMPL_PALLAS_LLO = "pallas_llo"
IMPL_AUTO = "auto"

CORE_IMPLS = (IMPL_LAX, IMPL_FAST)
LOCAL_EXACT_IMPLS = (
    IMPL_ITERATIVE_JAX,
    IMPL_ITERATIVE_PALLAS,
    IMPL_BUCKET,
    IMPL_BITONIC_JAX,
    IMPL_FIXED_BITONIC_A,
    IMPL_FIXED_BITONIC_B,
    IMPL_PALLAS_HIGH,
    IMPL_PALLAS_LLO,
)
ALL_IMPLS = CORE_IMPLS + LOCAL_EXACT_IMPLS
DEFAULT_IMPLS = (IMPL_AUTO,)


def _tpu_compiler_params(**kwargs):
    compiler_params = getattr(pltpu, "TPUCompilerParams", None)
    if compiler_params is None:
        compiler_params = pltpu.CompilerParams
    return compiler_params(**kwargs)


@dataclass(frozen=True)
class FastSchedules:
    stage1: tuple[int, ...]
    stage2: tuple[int, ...]


def sanitize_scores(x: jax.Array) -> jax.Array:
    return jnp.where(jnp.isnan(x), NEG_INF, x)


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def pad_last_dim(x: jax.Array, multiple: int) -> tuple[jax.Array, int]:
    n_orig = x.shape[-1]
    n_pad = round_up(n_orig, multiple)
    pad = n_pad - n_orig
    if pad == 0:
        return x, n_orig
    return jnp.pad(x, ((0, 0), (0, pad)), constant_values=NEG_INF), n_orig


def choose_block_token(rows: int) -> int:
    for candidate in (32, 16, 8, 4, 2, 1):
        if candidate <= rows and rows % candidate == 0:
            return candidate
    return 1


def fast_schedules(k: int) -> FastSchedules:
    # Avoid m=1 as an early-stop stage; k=1 is handled by the final fallback.
    stage1_candidates = (5, 8, 12, 16, 24, 32, 48, 64, 96, 128)
    stage2_candidates = (4, 8, 16, 32, 64, 96, 128)
    return FastSchedules(
        stage1=tuple(m for m in stage1_candidates if m < k),
        stage2=tuple(m for m in stage2_candidates if m < k),
    )


def fast_stage2_upper(max_m: int, k: int, stage2: tuple[int, ...]) -> int:
    schedule = (-1,) + stage2 + (k,)
    for lower, upper in zip(schedule, schedule[1:]):
        if lower < max_m <= upper:
            return upper
    return k


def _dedupe_impls(impls: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for impl in impls:
        if impl not in seen:
            seen.add(impl)
            out.append(impl)
    return tuple(out)


def scenario_default_impls(scenario: str, vocab: int, k: int) -> tuple[str, ...]:
    """Return the exact implementations from the old scenario benchmark plan.

    This intentionally excludes approximate bucket variants and soft Sinkhorn.
    The original rank-changing Pallas bitonic is also excluded because it was a
    compile-failure record, not a performance candidate.
    """
    impls = [IMPL_LAX, IMPL_FAST]
    if scenario == "moe":
        impls.extend(
            [
                IMPL_ITERATIVE_JAX,
                IMPL_ITERATIVE_PALLAS,
                IMPL_BUCKET,
                IMPL_BITONIC_JAX,
                IMPL_FIXED_BITONIC_A,
                IMPL_PALLAS_HIGH,
            ]
        )
        if vocab <= 512:
            impls.append(IMPL_FIXED_BITONIC_B)
            if vocab % 128 == 0:
                impls.append(IMPL_PALLAS_LLO)
    elif scenario == "vocab_decode":
        if k <= 50:
            impls.extend([IMPL_ITERATIVE_PALLAS, IMPL_PALLAS_HIGH])
        if vocab == 32000 and k in (1, 20):
            impls.extend([IMPL_ITERATIVE_JAX, IMPL_BITONIC_JAX])
        if k <= 50 and vocab in (32000, 50304):
            impls.append(IMPL_BUCKET)
        if k in (20, 50) and vocab in (32000, 128256):
            impls.append(IMPL_FIXED_BITONIC_A)
        if k == 20 and vocab == 32000:
            impls.append(IMPL_PALLAS_LLO)
    elif scenario == "dsa_block":
        # Exact DSA block coverage follows the old block-level sparse-attention plan.
        impls.extend([IMPL_ITERATIVE_JAX, IMPL_ITERATIVE_PALLAS, IMPL_BUCKET])
        if vocab <= 2048:
            impls.append(IMPL_BITONIC_JAX)
        if k <= V6_TILE_COLS:
            impls.append(IMPL_FIXED_BITONIC_A)
            if vocab <= 1024 and k <= 64:
                impls.append(IMPL_FIXED_BITONIC_B)
        impls.append(IMPL_PALLAS_HIGH)
        if (vocab, k) in ((256, 64), (1024, 64)) and vocab % 128 == 0:
            impls.append(IMPL_PALLAS_LLO)
    elif scenario == "dsa_token":
        # Token-level DSA has k=2048 and sequence lengths up to 262k in the old
        # plan. The local exact block kernels are not the right default here.
        pass
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return _dedupe_impls(impls)


@partial(jax.jit, static_argnames=("k",))
def lax_top_k_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    vals, idx = lax.top_k(sanitize_scores(x), k)
    return vals, idx.astype(jnp.int32)


@partial(jax.jit, static_argnames=("k",))
def fast_exact_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    x = sanitize_scores(x)
    if k >= 1024:
        vals, idx = topk(x, k=k, num_bins=1024, bins_topm_schedule=(16,))
    else:
        vals, idx = topk(x, k=k)
    return vals, idx.astype(jnp.int32)


@partial(jax.jit, static_argnames=("k",))
def iterative_jax_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    rows, n = x.shape
    idxs = jnp.arange(n, dtype=jnp.int32)[None, :]
    work = sanitize_scores(x)
    out_vals = jnp.full((rows, k), NEG_INF, dtype=x.dtype)
    out_idx = jnp.full((rows, k), -1, dtype=jnp.int32)

    def body(i, state):
        work_i, vals_i, idx_i = state
        best = jnp.max(work_i, axis=1)
        candidates = jnp.where(work_i == best[:, None], idxs, jnp.int32(n))
        arg = jnp.min(candidates, axis=1).astype(jnp.int32)
        vals_i = vals_i.at[:, i].set(best)
        idx_i = idx_i.at[:, i].set(arg)
        work_i = jnp.where(idxs == arg[:, None], NEG_INF, work_i)
        return work_i, vals_i, idx_i

    _, vals, idx = lax.fori_loop(0, k, body, (work, out_vals, out_idx))
    return vals, idx


def _stable_argmax_delete_kernel(k: int, n: int, block_rows: int):
    def kernel(x_ref, vals_ref, idx_ref):
        idxs = jnp.arange(n, dtype=jnp.int32)
        work = sanitize_scores(x_ref[...]).astype(jnp.float32)
        vals = []
        idx_out = []
        for _ in range(k):
            best_val = jnp.max(work, axis=1)
            candidates = jnp.where(work == best_val[:, None], idxs[None, :], jnp.int32(n))
            best_idx = jnp.min(candidates, axis=1).astype(jnp.int32)
            vals.append(best_val.astype(x_ref.dtype))
            idx_out.append(best_idx)
            work = jnp.where(idxs[None, :] == best_idx[:, None], NEG_INF, work)
        vals_ref[...] = jnp.stack(vals, axis=1)
        idx_ref[...] = jnp.stack(idx_out, axis=1)

    return kernel


def _repeated_argmax_2d_impl(x: jax.Array, *, k: int, block_rows: int) -> tuple[jax.Array, jax.Array]:
    n = x.shape[-1]
    leading = x.shape[:-1]
    x2 = sanitize_scores(x).reshape((-1, n))
    rows = x2.shape[0]
    row_groups = (rows + block_rows - 1) // block_rows
    row_pad = row_groups * block_rows
    if row_pad != rows:
        x2 = jnp.pad(x2, ((0, row_pad - rows), (0, 0)), constant_values=NEG_INF)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((block_rows, n), lambda g: (g, 0))],
        out_specs=(
            pl.BlockSpec((block_rows, k), lambda g: (g, 0)),
            pl.BlockSpec((block_rows, k), lambda g: (g, 0)),
        ),
        grid=(row_groups,),
    )
    vals, idx = pl.pallas_call(
        _stable_argmax_delete_kernel(k, n, block_rows),
        out_shape=(
            jax.ShapeDtypeStruct((row_pad, k), x.dtype),
            jax.ShapeDtypeStruct((row_pad, k), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=_tpu_compiler_params(dimension_semantics=("parallel",)),
    )(x2)
    out_shape = leading + (k,)
    return vals[:rows].reshape(out_shape), idx[:rows].reshape(out_shape).astype(jnp.int32)


def _choose_row_block(rows: int) -> int:
    return rows if rows < 8 else 8


@partial(jax.jit, static_argnames=("k",))
def iterative_pallas_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    rows = x.reshape((-1, x.shape[-1])).shape[0]
    return _repeated_argmax_2d_impl(x, k=k, block_rows=_choose_row_block(rows))


@partial(jax.jit, static_argnames=("k",))
def pallas_high_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    rows = x.reshape((-1, x.shape[-1])).shape[0]
    return _repeated_argmax_2d_impl(x, k=k, block_rows=_choose_row_block(rows))


def _select_better_pair(cand_val, cand_idx, best_val, best_idx):
    take = (cand_val > best_val) | ((cand_val == best_val) & (cand_idx < best_idx))
    return jnp.where(take, cand_val, best_val), jnp.where(take, cand_idx, best_idx)


def _reduce_128_pair(vals, idxs):
    best_val = vals
    best_idx = idxs
    width = 128
    while width > 1:
        half = width // 2
        left_val = best_val[:, :half]
        right_val = best_val[:, half:width]
        left_idx = best_idx[:, :half]
        right_idx = best_idx[:, half:width]
        best_val, best_idx = _select_better_pair(right_val, right_idx, left_val, left_idx)
        width = half
    return best_val[:, 0], best_idx[:, 0]


def _llo_style_argmax_delete_kernel(k: int, n: int, block_rows: int):
    def kernel(x_ref, vals_ref, idx_ref):
        work = sanitize_scores(x_ref[...]).astype(jnp.float32)
        full_idx = jnp.arange(n, dtype=jnp.int32)
        for rank in range(k):
            best_val = jnp.full((block_rows,), NEG_INF, dtype=jnp.float32)
            best_idx = jnp.full((block_rows,), jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
            for start in range(0, n, 128):
                chunk_val = work[:, start : start + 128]
                chunk_idx = jnp.broadcast_to(full_idx[start : start + 128][None, :], (block_rows, 128))
                chunk_best_val, chunk_best_idx = _reduce_128_pair(chunk_val, chunk_idx)
                best_val, best_idx = _select_better_pair(chunk_best_val, chunk_best_idx, best_val, best_idx)
            vals_ref[:, rank] = best_val.astype(x_ref.dtype)
            idx_ref[:, rank] = best_idx
            work = jnp.where(full_idx[None, :] == best_idx[:, None], NEG_INF, work)

    return kernel


@partial(jax.jit, static_argnames=("k",))
def pallas_llo_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    n = x.shape[-1]
    if n % 128 != 0:
        raise ValueError("pallas_llo requires the last dimension to be divisible by 128")
    leading = x.shape[:-1]
    x2 = sanitize_scores(x).reshape((-1, n))
    rows = x2.shape[0]
    block_rows = _choose_row_block(rows)
    row_groups = (rows + block_rows - 1) // block_rows
    row_pad = row_groups * block_rows
    if row_pad != rows:
        x2 = jnp.pad(x2, ((0, row_pad - rows), (0, 0)), constant_values=NEG_INF)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((block_rows, n), lambda g: (g, 0))],
        out_specs=(
            pl.BlockSpec((block_rows, k), lambda g: (g, 0)),
            pl.BlockSpec((block_rows, k), lambda g: (g, 0)),
        ),
        grid=(row_groups,),
    )
    vals, idx = pl.pallas_call(
        _llo_style_argmax_delete_kernel(k, n, block_rows),
        out_shape=(
            jax.ShapeDtypeStruct((row_pad, k), x.dtype),
            jax.ShapeDtypeStruct((row_pad, k), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=_tpu_compiler_params(dimension_semantics=("parallel",)),
    )(x2)
    out_shape = leading + (k,)
    return vals[:rows].reshape(out_shape), idx[:rows].reshape(out_shape).astype(jnp.int32)


def _bitonic_sort_1d(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
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


@partial(jax.jit, static_argnames=("k",))
def bitonic_jax_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(lambda row: _bitonic_sort_1d(row, k=k))(x)


def _bucket_v6_2d_kernel(local_k: int, local_k_pad: int, num_tiles: int):
    def kernel(x_ref, vals_ref, idx_ref):
        tile_id = pl.program_id(1)
        base = (tile_id * V6_TILE_SIZE).astype(jnp.int32)
        row = jnp.arange(V6_TILE_ROWS, dtype=jnp.int32)[:, None]
        col = jnp.arange(V6_TILE_COLS, dtype=jnp.int32)[None, :]
        local_pos = row * jnp.int32(V6_TILE_COLS) + col
        global_idx = base + local_pos
        work = sanitize_scores(x_ref[...]).astype(jnp.float32)
        neg_inf = jnp.array(NEG_INF, dtype=jnp.float32)
        vals = []
        idx_out = []
        for _ in range(local_k):
            m = jnp.max(work, axis=(0, 1))
            candidates = jnp.where(work == m, global_idx, jnp.iinfo(jnp.int32).max)
            arg = jnp.min(candidates, axis=(0, 1)).astype(jnp.int32)
            vals.append(m)
            idx_out.append(arg)
            work = jnp.where(global_idx == arg, neg_inf, work)
        vals_ref[...] = jnp.pad(
            jnp.stack(vals),
            (0, local_k_pad - local_k),
            constant_values=neg_inf,
        )
        idx_ref[...] = jnp.pad(
            jnp.stack(idx_out),
            (0, local_k_pad - local_k),
            constant_values=-1,
        )

    return kernel


@partial(jax.jit, static_argnames=("k",))
def bucket_v6_exact_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    """Batched exact bucket select over [rows, vocab]."""
    rows, n_orig = x.shape
    num_tiles = (n_orig + V6_TILE_SIZE - 1) // V6_TILE_SIZE
    n_pad = num_tiles * V6_TILE_SIZE
    local_k = k
    local_k_pad = max(V6_TILE_COLS, round_up(local_k, V6_TILE_COLS))
    x_pad = jnp.pad(sanitize_scores(x), ((0, 0), (0, n_pad - n_orig)), constant_values=NEG_INF)
    x_tiles = x_pad.reshape((rows, num_tiles, V6_TILE_ROWS, V6_TILE_COLS))
    x_tiles = x_tiles.reshape((rows * num_tiles * V6_TILE_ROWS, V6_TILE_COLS))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(
                (V6_TILE_ROWS, V6_TILE_COLS),
                lambda r, t: (r * num_tiles + t, 0),
            )
        ],
        out_specs=(
            pl.BlockSpec((local_k_pad,), lambda r, t: (r * num_tiles + t,)),
            pl.BlockSpec((local_k_pad,), lambda r, t: (r * num_tiles + t,)),
        ),
        grid=(rows, num_tiles),
    )
    cand_vals, cand_idx = pl.pallas_call(
        _bucket_v6_2d_kernel(local_k, local_k_pad, num_tiles),
        out_shape=(
            jax.ShapeDtypeStruct((rows * num_tiles * local_k_pad,), jnp.float32),
            jax.ShapeDtypeStruct((rows * num_tiles * local_k_pad,), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=_tpu_compiler_params(dimension_semantics=("parallel", "parallel")),
    )(x_tiles)
    cand_vals = cand_vals.reshape((rows, num_tiles * local_k_pad))
    cand_idx = cand_idx.reshape((rows, num_tiles * local_k_pad))
    vals, pos = lax.top_k(cand_vals, k)
    idx = jnp.take_along_axis(cand_idx, pos, axis=1).astype(jnp.int32)
    idx = jnp.where(idx < n_orig, idx, -1).astype(jnp.int32)
    return vals.astype(x.dtype), idx


def _row_bitonic_sort_16x256(keys: jax.Array, inds: jax.Array) -> tuple[jax.Array, jax.Array]:
    size = 2
    while size <= V6_TILE_COLS:
        stride = size // 2
        while stride >= 1:
            key_chunks = []
            idx_chunks = []
            for block in range(0, V6_TILE_COLS, 2 * stride):
                a_key = keys[:, block : block + stride]
                b_key = keys[:, block + stride : block + 2 * stride]
                a_idx = inds[:, block : block + stride]
                b_idx = inds[:, block + stride : block + 2 * stride]
                a_less = (a_key < b_key) | ((a_key == b_key) & (a_idx < b_idx))
                min_key = jnp.where(a_less, a_key, b_key)
                min_idx = jnp.where(a_less, a_idx, b_idx)
                max_key = jnp.where(a_less, b_key, a_key)
                max_idx = jnp.where(a_less, b_idx, a_idx)
                asc = (block & size) == 0
                first_key = min_key if asc else max_key
                first_idx = min_idx if asc else max_idx
                second_key = max_key if asc else min_key
                second_idx = max_idx if asc else min_idx
                key_chunks.extend((first_key, second_key))
                idx_chunks.extend((first_idx, second_idx))
            keys = jnp.concatenate(key_chunks, axis=1)
            inds = jnp.concatenate(idx_chunks, axis=1)
            stride //= 2
        size *= 2
    return keys, inds


def _bitonic_row256_2d_kernel(row_k: int, num_tiles: int):
    def kernel(x_ref, vals_ref, idx_ref):
        tile_id = pl.program_id(1)
        base = (tile_id * V6_TILE_SIZE).astype(jnp.int32)
        row = jnp.arange(V6_TILE_ROWS, dtype=jnp.int32)[:, None]
        col = jnp.arange(V6_TILE_COLS, dtype=jnp.int32)[None, :]
        global_idx = base + row * jnp.int32(V6_TILE_COLS) + col
        keys = -sanitize_scores(x_ref[...])
        keys, inds = _row_bitonic_sort_16x256(keys, global_idx)
        vals = -keys
        keep = col < jnp.int32(row_k)
        vals_ref[...] = jnp.where(keep, vals, NEG_INF)
        idx_ref[...] = jnp.where(keep, inds, -1)

    return kernel


@partial(jax.jit, static_argnames=("k",))
def fixed_bitonic_a_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    rows, n_orig = x.shape
    if k > V6_TILE_COLS:
        raise ValueError("fixed_bitonic_a supports k <= 256")
    num_tiles = (n_orig + V6_TILE_SIZE - 1) // V6_TILE_SIZE
    n_pad = num_tiles * V6_TILE_SIZE
    x_pad = jnp.pad(sanitize_scores(x), ((0, 0), (0, n_pad - n_orig)), constant_values=NEG_INF)
    x_tiles = x_pad.reshape((rows, num_tiles, V6_TILE_ROWS, V6_TILE_COLS))
    x_tiles = x_tiles.reshape((rows * num_tiles * V6_TILE_ROWS, V6_TILE_COLS))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda r, t: (r * num_tiles + t, 0))],
        out_specs=(
            pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda r, t: (r * num_tiles + t, 0)),
            pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda r, t: (r * num_tiles + t, 0)),
        ),
        grid=(rows, num_tiles),
    )
    cand_vals, cand_idx = pl.pallas_call(
        _bitonic_row256_2d_kernel(k, num_tiles),
        out_shape=(
            jax.ShapeDtypeStruct((rows * num_tiles * V6_TILE_ROWS, V6_TILE_COLS), x.dtype),
            jax.ShapeDtypeStruct((rows * num_tiles * V6_TILE_ROWS, V6_TILE_COLS), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=_tpu_compiler_params(dimension_semantics=("parallel", "parallel")),
    )(x_tiles)
    cand_vals = cand_vals.reshape((rows, num_tiles * V6_TILE_SIZE))
    cand_idx = cand_idx.reshape((rows, num_tiles * V6_TILE_SIZE))
    vals, pos = lax.top_k(cand_vals, k)
    idx = jnp.take_along_axis(cand_idx, pos, axis=1).astype(jnp.int32)
    idx = jnp.where(idx < n_orig, idx, -1).astype(jnp.int32)
    return vals.astype(x.dtype), idx


def _bitonic_tile_partial_2d_kernel(local_k: int, local_k_pad: int, row_k: int):
    def kernel(x_ref, vals_ref, idx_ref):
        tile_id = pl.program_id(1)
        base = (tile_id * V6_TILE_SIZE).astype(jnp.int32)
        row = jnp.arange(V6_TILE_ROWS, dtype=jnp.int32)[:, None]
        col = jnp.arange(V6_TILE_COLS, dtype=jnp.int32)[None, :]
        global_idx = base + row * jnp.int32(V6_TILE_COLS) + col
        keys = -sanitize_scores(x_ref[...])
        keys, inds = _row_bitonic_sort_16x256(keys, global_idx)
        neg_inf = jnp.array(NEG_INF, dtype=jnp.float32)
        work = jnp.where(col < jnp.int32(row_k), (-keys).astype(jnp.float32), neg_inf)
        vals = []
        idx_out = []
        for _ in range(local_k):
            m = jnp.max(work, axis=(0, 1))
            candidates = jnp.where(work == m, inds, jnp.iinfo(jnp.int32).max)
            arg = jnp.min(candidates, axis=(0, 1)).astype(jnp.int32)
            vals.append(m)
            idx_out.append(arg)
            work = jnp.where(inds == arg, neg_inf, work)
        vals_ref[...] = jnp.pad(jnp.stack(vals), (0, local_k_pad - local_k), constant_values=neg_inf)
        idx_ref[...] = jnp.pad(jnp.stack(idx_out), (0, local_k_pad - local_k), constant_values=-1)

    return kernel


@partial(jax.jit, static_argnames=("k",))
def fixed_bitonic_b_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    rows, n_orig = x.shape
    if k > V6_TILE_COLS:
        raise ValueError("fixed_bitonic_b supports k <= 256")
    num_tiles = (n_orig + V6_TILE_SIZE - 1) // V6_TILE_SIZE
    n_pad = num_tiles * V6_TILE_SIZE
    local_k = k
    row_k = k
    local_k_pad = max(V6_TILE_COLS, round_up(local_k, V6_TILE_COLS))
    x_pad = jnp.pad(sanitize_scores(x), ((0, 0), (0, n_pad - n_orig)), constant_values=NEG_INF)
    x_tiles = x_pad.reshape((rows, num_tiles, V6_TILE_ROWS, V6_TILE_COLS))
    x_tiles = x_tiles.reshape((rows * num_tiles * V6_TILE_ROWS, V6_TILE_COLS))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda r, t: (r * num_tiles + t, 0))],
        out_specs=(
            pl.BlockSpec((local_k_pad,), lambda r, t: (r * num_tiles + t,)),
            pl.BlockSpec((local_k_pad,), lambda r, t: (r * num_tiles + t,)),
        ),
        grid=(rows, num_tiles),
    )
    cand_vals, cand_idx = pl.pallas_call(
        _bitonic_tile_partial_2d_kernel(local_k, local_k_pad, row_k),
        out_shape=(
            jax.ShapeDtypeStruct((rows * num_tiles * local_k_pad,), jnp.float32),
            jax.ShapeDtypeStruct((rows * num_tiles * local_k_pad,), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=_tpu_compiler_params(dimension_semantics=("parallel", "parallel")),
    )(x_tiles)
    cand_vals = cand_vals.reshape((rows, num_tiles * local_k_pad))
    cand_idx = cand_idx.reshape((rows, num_tiles * local_k_pad))
    vals, pos = lax.top_k(cand_vals, k)
    idx = jnp.take_along_axis(cand_idx, pos, axis=1).astype(jnp.int32)
    idx = jnp.where(idx < n_orig, idx, -1).astype(jnp.int32)
    return vals.astype(x.dtype), idx


@partial(jax.jit, static_argnames=("k",))
def _fast_termination_m_impl(x: jax.Array, *, k: int) -> jax.Array:
    kwargs = {}
    if k >= 1024:
        kwargs = {"num_bins": 1024, "bins_topm_schedule": (16,)}
    _, _, _, termination_m, _ = top_bounded_k(
        sanitize_scores(x),
        k=jnp.asarray(k, dtype=jnp.int32),
        max_k=k,
        guarantee_convergence=False,
        **kwargs,
    )
    return termination_m


def block_until_ready(tree):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def fast_termination_m(x: jax.Array, *, k: int) -> np.ndarray:
    return np.asarray(block_until_ready(_fast_termination_m_impl(x, k=k)))


def bucket_candidate_count(vocab: int, k: int) -> int:
    num_tiles = (vocab + V6_TILE_SIZE - 1) // V6_TILE_SIZE
    local_k_pad = max(V6_TILE_COLS, round_up(k, V6_TILE_COLS))
    return num_tiles * local_k_pad


def impl_function(name: str):
    if name == IMPL_LAX:
        return lambda x, k: lax_top_k_impl(x, k=k)
    if name == IMPL_FAST:
        return lambda x, k: fast_exact_impl(x, k=k)
    if name == IMPL_ITERATIVE_JAX:
        return lambda x, k: iterative_jax_impl(x, k=k)
    if name == IMPL_ITERATIVE_PALLAS:
        return lambda x, k: iterative_pallas_impl(x, k=k)
    if name == IMPL_BUCKET:
        return lambda x, k: bucket_v6_exact_impl(x, k=k)
    if name == IMPL_BITONIC_JAX:
        return lambda x, k: bitonic_jax_impl(x, k=k)
    if name == IMPL_FIXED_BITONIC_A:
        return lambda x, k: fixed_bitonic_a_impl(x, k=k)
    if name == IMPL_FIXED_BITONIC_B:
        return lambda x, k: fixed_bitonic_b_impl(x, k=k)
    if name == IMPL_PALLAS_HIGH:
        return lambda x, k: pallas_high_impl(x, k=k)
    if name == IMPL_PALLAS_LLO:
        return lambda x, k: pallas_llo_impl(x, k=k)
    raise ValueError(f"unknown implementation: {name}")
