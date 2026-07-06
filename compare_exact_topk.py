"""Compare exact TPU top-k implementations.

This script compares only exact hard top-k implementations:

1. `jax.lax.top_k`
2. `fast_exact_topk_tpu.topk_optimized`
3. A batched exact version of `standalone_kernels.bucket_select_pallas_v6_tile`
   with `local_k == k`

Run on a TPU VM, for example:

    uv run python compare_exact_topk.py --preset quick
    uv run python compare_exact_topk.py --preset vocab --iters 100
    uv run python compare_exact_topk.py --shape 32,201088,64

Results are written under `results/exact_topk_<timestamp>/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

TRACE_LIBTPU_INIT_ARGS = (
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true"
)
if "--trace" in sys.argv and "LIBTPU_INIT_ARGS" not in os.environ:
    os.environ["LIBTPU_INIT_ARGS"] = TRACE_LIBTPU_INIT_ARGS

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from fast_exact_topk_tpu import topk_blockwise_superset_pallas, topk_optimized
from standalone_kernels.common import (
    NEG_INF,
    V6_TILE_COLS,
    V6_TILE_ROWS,
    V6_TILE_SIZE,
)


IMPL_LAX = "lax_top_k"
IMPL_FAST = "fast_exact"
IMPL_BUCKET = "bucket_v6_exact"
DEFAULT_IMPLS = (IMPL_LAX, IMPL_FAST, IMPL_BUCKET)

DEFAULT_CORRECTNESS_DISTS = (
    "random",
    "repeated",
    "all_equal",
    "concentrated_128",
    "nan_mixed",
)
DEFAULT_PERF_DISTS = ("random", "concentrated_128")


@dataclass(frozen=True)
class ShapeCase:
    rows: int
    vocab: int
    k: int

    @property
    def label(self) -> str:
        return f"r{self.rows}_n{self.vocab}_k{self.k}"


@dataclass(frozen=True)
class FastSchedules:
    stage1: tuple[int, ...]
    stage2: tuple[int, ...]


def parse_shape_case(value: str) -> ShapeCase:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "shape must be ROWS,VOCAB,K, for example 32,201088,64"
        )
    rows, vocab, k = (int(p) for p in parts)
    if rows <= 0 or vocab <= 0 or k <= 0:
        raise argparse.ArgumentTypeError("ROWS, VOCAB, and K must be positive")
    if k > vocab:
        raise argparse.ArgumentTypeError("K must be <= VOCAB")
    return ShapeCase(rows=rows, vocab=vocab, k=k)


def preset_cases(name: str) -> list[ShapeCase]:
    if name == "quick":
        return [
            ShapeCase(1, 32768, 8),
            ShapeCase(8, 32768, 16),
            ShapeCase(32, 65536, 64),
        ]
    if name == "vocab":
        cases: list[ShapeCase] = []
        for rows in (1, 8, 32):
            for vocab in (32768, 65536):
                for k in (8, 16, 64):
                    cases.append(ShapeCase(rows, vocab, k))
            for vocab in (131072,):
                for k in (16, 64, 100):
                    cases.append(ShapeCase(rows, vocab, k))
        cases.extend(
            [
                ShapeCase(32, 201088, 64),
                ShapeCase(32, 204800, 64),
            ]
        )
        return cases
    if name == "smoke":
        return [ShapeCase(1, 4096, 8)]
    raise ValueError(f"unknown preset: {name}")


def dtype_from_name(name: str) -> jnp.dtype:
    if name == "bf16":
        return jnp.bfloat16
    if name == "f32":
        return jnp.float32
    raise ValueError(f"unsupported dtype: {name}")


def sanitize_scores(x: jax.Array) -> jax.Array:
    return jnp.where(jnp.isnan(x), NEG_INF, x)


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


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
    # Avoid m=1 in stage 1: upstream stopping criterion uses topk_vals[:m-1].
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


@partial(jax.jit, static_argnames=("k",))
def lax_top_k_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    vals, idx = lax.top_k(sanitize_scores(x), k)
    return vals, idx.astype(jnp.int32)


@partial(jax.jit, static_argnames=("k",))
def fast_exact_impl(x: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    if k < 2:
        raise ValueError("fast_exact wrapper skips k=1; upstream stage check is not valid")
    schedules = fast_schedules(k)
    x_pad, n_orig = pad_last_dim(sanitize_scores(x), 128)
    rows = x_pad.shape[0]
    vals, idx = topk_optimized(
        x_pad,
        k=k,
        num_blocks=128,
        m_stage1_schedule=schedules.stage1,
        m_stage2_schedule=schedules.stage2,
        block_token=choose_block_token(rows),
    )
    idx = jnp.where(idx < n_orig, idx, -1).astype(jnp.int32)
    return vals, idx


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
    """Batched exact bucket-select over `[rows, vocab]`.

    This is the exact setting of the local bucket algorithm: every 4096-element
    tile emits `local_k == k` candidates, and XLA merges the tile candidates per
    row. It is intended as the fair 2D counterpart to
    `standalone_kernels.bucket_select_pallas_v6_tile`.
    """
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
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x_tiles)
    cand_vals = cand_vals.reshape((rows, num_tiles * local_k_pad))
    cand_idx = cand_idx.reshape((rows, num_tiles * local_k_pad))
    vals, pos = lax.top_k(cand_vals, k)
    idx = jnp.take_along_axis(cand_idx, pos, axis=1).astype(jnp.int32)
    idx = jnp.where(idx < n_orig, idx, -1).astype(jnp.int32)
    return vals.astype(x.dtype), idx


def impl_function(name: str) -> Callable[[jax.Array, int], tuple[jax.Array, jax.Array]]:
    if name == IMPL_LAX:
        return lambda x, k: lax_top_k_impl(x, k=k)
    if name == IMPL_FAST:
        return lambda x, k: fast_exact_impl(x, k=k)
    if name == IMPL_BUCKET:
        return lambda x, k: bucket_v6_exact_impl(x, k=k)
    raise ValueError(f"unknown implementation: {name}")


def make_input(
    key: jax.Array,
    case: ShapeCase,
    dtype: jnp.dtype,
    distribution: str,
) -> jax.Array:
    rows, vocab, k = case.rows, case.vocab, case.k
    if distribution == "random":
        return jax.random.normal(key, (rows, vocab), dtype=jnp.float32).astype(dtype)
    if distribution == "repeated":
        values = jax.random.randint(key, (rows, vocab), minval=-8, maxval=9)
        return values.astype(dtype)
    if distribution == "all_equal":
        return jnp.ones((rows, vocab), dtype=dtype)
    if distribution == "concentrated_128":
        base = jax.random.normal(key, (rows, vocab), dtype=jnp.float32) * 0.01 - 1000.0
        positions = jnp.arange(k, dtype=jnp.int32) * 128
        positions = jnp.where(positions < vocab, positions, jnp.arange(k, dtype=jnp.int32))
        winner_vals = (1000.0 - jnp.arange(k, dtype=jnp.float32)).astype(dtype)
        return base.astype(dtype).at[:, positions].set(winner_vals)
    if distribution == "nan_mixed":
        x = jax.random.normal(key, (rows, vocab), dtype=jnp.float32).astype(dtype)
        nan_count = min(max(1, vocab // 257), 64)
        nan_pos = (jnp.arange(nan_count, dtype=jnp.int32) * 257) % vocab
        x = x.at[:, nan_pos].set(jnp.nan)
        winner_pos = (jnp.arange(k, dtype=jnp.int32) * 128 + 7) % vocab
        winner_vals = (1000.0 - jnp.arange(k, dtype=jnp.float32)).astype(dtype)
        return x.at[:, winner_pos].set(winner_vals)
    raise ValueError(f"unknown distribution: {distribution}")


def block_until_ready(tree):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def as_numpy_pair(result: tuple[jax.Array, jax.Array]) -> tuple[np.ndarray, np.ndarray]:
    vals, idx = block_until_ready(result)
    return np.asarray(vals), np.asarray(idx)


def compare_outputs(
    ref_vals: np.ndarray,
    ref_idx: np.ndarray,
    vals: np.ndarray,
    idx: np.ndarray,
) -> dict[str, object]:
    values_equal = bool(np.array_equal(vals, ref_vals))
    values_allclose = bool(np.allclose(vals, ref_vals, rtol=0.0, atol=0.0))
    indices_equal = bool(np.array_equal(idx, ref_idx))
    if idx.shape == ref_idx.shape:
        set_equal = bool(
            all(set(row.tolist()) == set(ref_row.tolist()) for row, ref_row in zip(idx, ref_idx))
        )
    else:
        set_equal = False
    max_abs_diff = float(np.max(np.abs(vals.astype(np.float32) - ref_vals.astype(np.float32))))
    return {
        "values_equal": values_equal,
        "values_allclose": values_allclose,
        "indices_equal": indices_equal,
        "index_set_equal": set_equal,
        "max_abs_diff": max_abs_diff,
        "exact_ordered_pass": values_equal and indices_equal,
    }


def timing_stats(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "min_ms": float(np.min(arr)),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p5_ms": float(np.percentile(arr, 5)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def bench_impl(
    fn: Callable[[jax.Array, int], tuple[jax.Array, jax.Array]],
    x: jax.Array,
    k: int,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    start = time.perf_counter()
    first = fn(x, k)
    block_until_ready(first)
    first_call_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(warmup):
        block_until_ready(fn(x, k))

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        block_until_ready(fn(x, k))
        samples.append((time.perf_counter() - start) * 1000.0)

    stats = timing_stats(samples)
    stats["first_call_ms"] = first_call_ms
    stats["samples_ms"] = json.dumps(samples)
    return stats


@partial(jax.jit, static_argnames=("k",))
def _fast_termination_m_impl(x: jax.Array, *, k: int) -> jax.Array:
    if k < 2:
        raise ValueError("fast_exact termination stats skip k=1")
    schedules = fast_schedules(k)
    x_pad, _ = pad_last_dim(sanitize_scores(x), 128)
    _, _, termination_m, _ = topk_blockwise_superset_pallas(
        x_pad,
        k=k,
        num_blocks=128,
        block_token=choose_block_token(x_pad.shape[0]),
        m_schedule=schedules.stage1,
    )
    return termination_m


def fast_termination_m(x: jax.Array, *, k: int) -> np.ndarray:
    return np.asarray(block_until_ready(_fast_termination_m_impl(x, k=k)))


def bucket_candidate_count(vocab: int, k: int) -> int:
    num_tiles = (vocab + V6_TILE_SIZE - 1) // V6_TILE_SIZE
    local_k_pad = max(V6_TILE_COLS, round_up(k, V6_TILE_COLS))
    return num_tiles * local_k_pad


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_list(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def run_trace(
    trace_root: Path,
    impl_name: str,
    case: ShapeCase,
    distribution: str,
    fn: Callable[[jax.Array, int], tuple[jax.Array, jax.Array]],
    x: jax.Array,
    warmup: int,
    iters: int,
) -> str:
    trace_dir = trace_root / f"{case.label}_{distribution}_{impl_name}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        block_until_ready(fn(x, case.k))
    jax.profiler.start_trace(str(trace_dir))
    try:
        for _ in range(iters):
            block_until_ready(fn(x, case.k))
    finally:
        jax.profiler.stop_trace()
    return str(trace_dir)


def run(args: argparse.Namespace) -> None:
    dtype = dtype_from_name(args.dtype)
    cases = list(args.shape or preset_cases(args.preset))
    impls = parse_csv_list(args.impls)
    correctness_dists = parse_csv_list(args.correctness_distributions)
    perf_dists = parse_csv_list(args.perf_distributions)

    unknown_impls = sorted(set(impls) - set(DEFAULT_IMPLS))
    if unknown_impls:
        raise ValueError(f"unknown implementations: {unknown_impls}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"results/exact_topk_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "dtype": args.dtype,
        "preset": args.preset,
        "cases": [case.__dict__ for case in cases],
        "impls": impls,
        "correctness_distributions": correctness_dists,
        "perf_distributions": perf_dists,
        "warmup": args.warmup,
        "iters": args.iters,
        "jax_platform": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    correctness_rows: list[dict[str, object]] = []
    perf_rows: list[dict[str, object]] = []
    fast_stage_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []

    key = jax.random.PRNGKey(args.seed)
    print(f"Writing results to {out_dir}")
    print(f"JAX backend: {jax.default_backend()} devices={jax.devices()}")
    if args.trace:
        libtpu_args = os.environ.get("LIBTPU_INIT_ARGS", "")
        for flag in TRACE_LIBTPU_INIT_ARGS.split():
            if flag not in libtpu_args:
                print(f"WARNING: LIBTPU_INIT_ARGS is missing trace flag: {flag}")

    for case_index, case in enumerate(cases):
        print(f"\n=== {case.label} dtype={args.dtype} ===")
        schedules = fast_schedules(case.k)
        for dist_index, distribution in enumerate(correctness_dists):
            key, subkey = jax.random.split(key)
            x = make_input(subkey, case, dtype, distribution)
            print(f"[correctness] distribution={distribution}")

            ref_vals: np.ndarray | None = None
            ref_idx: np.ndarray | None = None
            for impl_name in impls:
                fn = impl_function(impl_name)
                row_base = {
                    "case": case.label,
                    "rows": case.rows,
                    "vocab": case.vocab,
                    "k": case.k,
                    "dtype": args.dtype,
                    "distribution": distribution,
                    "impl": impl_name,
                }
                try:
                    start = time.perf_counter()
                    vals, idx = as_numpy_pair(fn(x, case.k))
                    first_call_ms = (time.perf_counter() - start) * 1000.0
                    if impl_name == IMPL_LAX:
                        ref_vals, ref_idx = vals, idx
                        correctness_rows.append(
                            {
                                **row_base,
                                "status": "ok",
                                "first_call_ms": first_call_ms,
                                "values_equal": True,
                                "values_allclose": True,
                                "indices_equal": True,
                                "index_set_equal": True,
                                "max_abs_diff": 0.0,
                                "exact_ordered_pass": True,
                            }
                        )
                    else:
                        if ref_vals is None or ref_idx is None:
                            ref_vals, ref_idx = as_numpy_pair(impl_function(IMPL_LAX)(x, case.k))
                        cmp = compare_outputs(ref_vals, ref_idx, vals, idx)
                        correctness_rows.append(
                            {
                                **row_base,
                                "status": "ok",
                                "first_call_ms": first_call_ms,
                                **cmp,
                            }
                        )
                except Exception as exc:  # Keep long sweeps alive across compile failures.
                    correctness_rows.append(
                        {
                            **row_base,
                            "status": "error",
                            "error": repr(exc),
                        }
                    )
                    print(f"  {impl_name}: ERROR {exc!r}")

            if IMPL_FAST in impls:
                try:
                    term = fast_termination_m(x, k=case.k)
                    hist = {int(v): int(np.sum(term == v)) for v in np.unique(term)}
                    max_m = int(np.max(term))
                    upper_m = fast_stage2_upper(max_m, case.k, schedules.stage2)
                    fast_stage_rows.append(
                        {
                            "case": case.label,
                            "rows": case.rows,
                            "vocab": case.vocab,
                            "k": case.k,
                            "dtype": args.dtype,
                            "distribution": distribution,
                            "termination_m_hist": json.dumps(hist, sort_keys=True),
                            "max_m": max_m,
                            "stage2_upper_m": upper_m,
                            "fast_candidate_count": upper_m * 128,
                            "bucket_candidate_count": bucket_candidate_count(case.vocab, case.k),
                            "stage1_schedule": json.dumps(schedules.stage1),
                            "stage2_schedule": json.dumps(schedules.stage2),
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    fast_stage_rows.append(
                        {
                            "case": case.label,
                            "rows": case.rows,
                            "vocab": case.vocab,
                            "k": case.k,
                            "dtype": args.dtype,
                            "distribution": distribution,
                            "status": "error",
                            "error": repr(exc),
                        }
                    )

        for perf_dist in perf_dists:
            key, subkey = jax.random.split(key)
            x = make_input(subkey, case, dtype, perf_dist)
            print(f"[benchmark] distribution={perf_dist}")

            ref_median: float | None = None
            impl_stats: dict[str, dict[str, object]] = {}
            for impl_name in impls:
                fn = impl_function(impl_name)
                row_base = {
                    "case": case.label,
                    "rows": case.rows,
                    "vocab": case.vocab,
                    "k": case.k,
                    "dtype": args.dtype,
                    "distribution": perf_dist,
                    "impl": impl_name,
                }
                try:
                    stats = bench_impl(fn, x, case.k, args.warmup, args.iters)
                    impl_stats[impl_name] = stats
                    if impl_name == IMPL_LAX:
                        ref_median = float(stats["median_ms"])
                    perf_rows.append({**row_base, "status": "ok", **stats})
                    print(
                        "  "
                        f"{impl_name}: median={stats['median_ms']:.4f} ms "
                        f"p95={stats['p95_ms']:.4f} ms"
                    )
                except Exception as exc:
                    perf_rows.append({**row_base, "status": "error", "error": repr(exc)})
                    print(f"  {impl_name}: ERROR {exc!r}")

            if ref_median is None and IMPL_LAX in impl_stats:
                ref_median = float(impl_stats[IMPL_LAX]["median_ms"])
            if ref_median is not None:
                for row in perf_rows:
                    if row.get("case") == case.label and row.get("distribution") == perf_dist:
                        if row.get("status") == "ok" and "median_ms" in row:
                            row["speedup_vs_lax_median"] = ref_median / float(row["median_ms"])

            if args.trace:
                trace_root = out_dir / "traces"
                for impl_name in impls:
                    try:
                        trace_dir = run_trace(
                            trace_root,
                            impl_name,
                            case,
                            perf_dist,
                            impl_function(impl_name),
                            x,
                            warmup=max(1, min(args.warmup, 3)),
                            iters=args.trace_iters,
                        )
                        trace_rows.append(
                            {
                                "case": case.label,
                                "distribution": perf_dist,
                                "impl": impl_name,
                                "trace_dir": trace_dir,
                                "status": "ok",
                            }
                        )
                    except Exception as exc:
                        trace_rows.append(
                            {
                                "case": case.label,
                                "distribution": perf_dist,
                                "impl": impl_name,
                                "status": "error",
                                "error": repr(exc),
                            }
                        )

        write_csv(out_dir / "exact_correctness.csv", correctness_rows)
        write_csv(out_dir / "exact_perf.csv", perf_rows)
        write_csv(out_dir / "fast_stage_detail.csv", fast_stage_rows)
        write_csv(out_dir / "trace_dirs.csv", trace_rows)

    print("\nDone.")
    print(f"Correctness: {out_dir / 'exact_correctness.csv'}")
    print(f"Performance: {out_dir / 'exact_perf.csv'}")
    print(f"Fast stage detail: {out_dir / 'fast_stage_detail.csv'}")
    if args.trace:
        print(f"Trace dirs: {out_dir / 'trace_dirs.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "quick", "vocab"),
        default="quick",
        help="Shape preset used when --shape is not provided.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape_case,
        help="Add a shape case ROWS,VOCAB,K. May be passed multiple times.",
    )
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument(
        "--impls",
        default=",".join(DEFAULT_IMPLS),
        help=f"Comma-separated implementations from: {','.join(DEFAULT_IMPLS)}",
    )
    parser.add_argument(
        "--correctness-distributions",
        default=",".join(DEFAULT_CORRECTNESS_DISTS),
        help="Comma-separated correctness distributions.",
    )
    parser.add_argument(
        "--perf-distributions",
        default=",".join(DEFAULT_PERF_DISTS),
        help="Comma-separated benchmark distributions.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Capture JAX profiler traces for benchmark distributions.",
    )
    parser.add_argument("--trace-iters", type=int, default=5)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
