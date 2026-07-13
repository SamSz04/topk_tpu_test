"""Benchmark exact TPU top-k implementations for the target top-k scenarios.

Scenarios:
  - moe
  - vocab_decode
  - dsa_block
  - dsa_token

Implementations:
  - jax.lax.top_k
  - fast_exact_topk_tpu.topk
  - scenario-specific local exact implementations from the earlier top_k work:
    iterative mask, bucket v6, bitonic, fixed-shape bitonic, and repeated argmax
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

TRACE_LIBTPU_INIT_ARGS = (
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true"
)
if "--trace" in sys.argv and "LIBTPU_INIT_ARGS" not in os.environ:
    os.environ["LIBTPU_INIT_ARGS"] = TRACE_LIBTPU_INIT_ARGS

import jax
import jax.numpy as jnp
import numpy as np

from exact_topk_impls import (
    ALL_IMPLS,
    DEFAULT_IMPLS,
    IMPL_AUTO,
    IMPL_FAST,
    IMPL_LAX,
    block_until_ready,
    fast_termination_m,
    impl_function,
    sanitize_scores,
    scenario_default_impls,
)


DEFAULT_CORRECTNESS_DISTS = (
    "random",
    "repeated",
    "all_equal",
    "concentrated_128",
    "concentrated_tile",
    "ascending",
    "descending",
    "nan_mixed",
)
DEFAULT_PERF_DISTS = ("random", "concentrated_128")

SCENARIO_ALIASES = {
    "vocab": "vocab_decode",
    "decoding": "vocab_decode",
    "dsa_sparse_attention": "dsa_block",
}
SCENARIOS = ("moe", "vocab_decode", "dsa_block", "dsa_token")
SPARSE_SCENARIOS = ("dsa_block", "dsa_token")

SCENARIO_NK = {
    "moe": (
        (8, 2),
        (16, 4),
        (64, 8),
        (128, 8),
        (256, 8),
        (512, 10),
        (2048, 1),
        (2048, 2),
    ),
    "vocab_decode": (
        (32000, 1),
        (32000, 20),
        (32000, 50),
        (50304, 1),
        (50304, 20),
        (50304, 50),
        (128256, 1),
        (128256, 20),
        (128256, 50),
        (129280, 1),
        (129280, 20),
        (129280, 50),
        (151936, 1),
        (151936, 20),
        (151936, 50),
        (128256, 100),
        (151936, 100),
    ),
    "dsa_token": (
        (8192, 2048),
        (32768, 2048),
        (65536, 2048),
        (131072, 2048),
        (262144, 2048),
    ),
    "dsa_block": (
        (256, 64),
        (512, 64),
        (1024, 64),
        (1024, 256),
        (64, 16),
        (256, 16),
        (512, 16),
        (2048, 64),
        (2048, 256),
        (2048, 512),
    ),
}

SCENARIO_ROWS = {
    "moe": (1, 32, 128),
    "vocab_decode": (1, 8, 32),
    "dsa_block": (1, 8, 32),
    "dsa_token": (1, 8),
}


@dataclass(frozen=True)
class BenchmarkCase:
    scenario: str
    rows: int
    vocab: int
    k: int

    @property
    def label(self) -> str:
        return f"{self.scenario}_b{self.rows}_n{self.vocab}_k{self.k}"


def normalize_scenario(scenario: str) -> str:
    scenario = SCENARIO_ALIASES.get(scenario, scenario)
    if scenario not in SCENARIOS and scenario != "sparse":
        raise argparse.ArgumentTypeError(f"unknown scenario: {scenario}")
    return scenario


def parse_case(value: str) -> BenchmarkCase:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "case must be SCENARIO,ROWS,N,K, for example vocab_decode,32,151936,50"
        )
    scenario = normalize_scenario(parts[0])
    if scenario == "sparse":
        raise argparse.ArgumentTypeError("use dsa_block or dsa_token for explicit sparse cases")
    rows, vocab, k = (int(p) for p in parts[1:])
    if rows <= 0 or vocab <= 0 or k <= 0:
        raise argparse.ArgumentTypeError("ROWS, N, and K must be positive")
    if k > vocab:
        raise argparse.ArgumentTypeError("K must be <= N")
    return BenchmarkCase(scenario, rows, vocab, k)


def parse_shape(value: str, scenario: str) -> BenchmarkCase:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be ROWS,N,K")
    rows, vocab, k = (int(p) for p in parts)
    scenario = normalize_scenario(scenario)
    if scenario == "sparse":
        raise argparse.ArgumentTypeError("use --scenario dsa_block or --scenario dsa_token with --shape")
    return parse_case(f"{scenario},{rows},{vocab},{k}")


def _dedupe(cases: list[BenchmarkCase]) -> list[BenchmarkCase]:
    seen: set[BenchmarkCase] = set()
    out: list[BenchmarkCase] = []
    for case in cases:
        if case not in seen:
            seen.add(case)
            out.append(case)
    return out


def moe_cases() -> list[BenchmarkCase]:
    return scenario_cases("moe")


def dsa_block_cases() -> list[BenchmarkCase]:
    return scenario_cases("dsa_block")


def dsa_token_cases() -> list[BenchmarkCase]:
    return scenario_cases("dsa_token")


def vocab_decode_cases() -> list[BenchmarkCase]:
    return scenario_cases("vocab_decode")


def scenario_cases(scenario: str) -> list[BenchmarkCase]:
    scenario = normalize_scenario(scenario)
    if scenario == "sparse":
        return _dedupe(dsa_block_cases() + dsa_token_cases())
    return _dedupe(
        [
            BenchmarkCase(scenario, rows, n, k)
            for rows in SCENARIO_ROWS[scenario]
            for n, k in SCENARIO_NK[scenario]
            if k <= n
        ]
    )


def preset_cases(name: str) -> list[BenchmarkCase]:
    if name == "smoke":
        return [
            BenchmarkCase("moe", 8, 128, 8),
            BenchmarkCase("dsa_block", 8, 1024, 64),
            BenchmarkCase("dsa_token", 1, 8192, 2048),
            BenchmarkCase("vocab_decode", 8, 32000, 50),
        ]
    if name == "quick":
        return [
            BenchmarkCase("moe", 32, 128, 8),
            BenchmarkCase("moe", 128, 2048, 2),
            BenchmarkCase("dsa_block", 8, 2048, 64),
            BenchmarkCase("dsa_block", 32, 2048, 512),
            BenchmarkCase("dsa_token", 1, 8192, 2048),
            BenchmarkCase("dsa_token", 8, 32768, 2048),
            BenchmarkCase("vocab_decode", 8, 50304, 50),
            BenchmarkCase("vocab_decode", 32, 151936, 100),
        ]
    if name == "all":
        return _dedupe(moe_cases() + vocab_decode_cases() + dsa_block_cases() + dsa_token_cases())
    name = normalize_scenario(name)
    if name in SCENARIOS:
        return scenario_cases(name)
    if name == "sparse":
        return scenario_cases("sparse")
    raise ValueError(f"unknown preset: {name}")


def dtype_from_name(name: str) -> jnp.dtype:
    if name == "bf16":
        return jnp.bfloat16
    if name == "f32":
        return jnp.float32
    raise ValueError(f"unsupported dtype: {name}")


def make_input(
    key: jax.Array,
    case: BenchmarkCase,
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
    if distribution == "concentrated_tile":
        base = jax.random.normal(key, (rows, vocab), dtype=jnp.float32) * 0.01 - 1000.0
        positions = jnp.arange(k, dtype=jnp.int32)
        winner_vals = (1000.0 - jnp.arange(k, dtype=jnp.float32)).astype(dtype)
        return base.astype(dtype).at[:, positions].set(winner_vals)
    if distribution == "ascending":
        row_offsets = jnp.arange(rows, dtype=jnp.float32)[:, None] * 0.001
        return (jnp.arange(vocab, dtype=jnp.float32)[None, :] + row_offsets).astype(dtype)
    if distribution == "descending":
        row_offsets = jnp.arange(rows, dtype=jnp.float32)[:, None] * 0.001
        return (-jnp.arange(vocab, dtype=jnp.float32)[None, :] + row_offsets).astype(dtype)
    if distribution == "nan_mixed":
        x = jax.random.normal(key, (rows, vocab), dtype=jnp.float32).astype(dtype)
        nan_count = min(max(1, vocab // 257), 64)
        nan_pos = (jnp.arange(nan_count, dtype=jnp.int32) * 257) % vocab
        winner_pos = (jnp.arange(k, dtype=jnp.int32) * 128 + 7) % vocab
        winner_vals = (1000.0 - jnp.arange(k, dtype=jnp.float32)).astype(dtype)
        return x.at[:, nan_pos].set(jnp.nan).at[:, winner_pos].set(winner_vals)
    raise ValueError(f"unknown distribution: {distribution}")


def as_numpy_pair(result: tuple[jax.Array, jax.Array]) -> tuple[np.ndarray, np.ndarray]:
    vals, idx = block_until_ready(result)
    return np.asarray(vals), np.asarray(idx)


def compare_outputs(
    ref_vals: np.ndarray,
    ref_idx: np.ndarray,
    vals: np.ndarray,
    idx: np.ndarray,
    source: np.ndarray,
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
    if idx.shape == vals.shape and source.ndim == 2:
        nonnegative_idx = idx >= 0
        safe_idx = np.where(nonnegative_idx, idx, 0)
        gathered = np.take_along_axis(source, safe_idx, axis=1)
        source_values_match = bool(np.all(nonnegative_idx) and np.array_equal(gathered, vals))
    else:
        source_values_match = False
    return {
        "values_equal": values_equal,
        "values_allclose": values_allclose,
        "indices_equal": indices_equal,
        "index_set_equal": set_equal,
        "source_values_match": source_values_match,
        "max_abs_diff": max_abs_diff,
        "valid_topk_pass": values_equal and source_values_match,
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


def bench_impl(fn, x: jax.Array, k: int, warmup: int, iters: int) -> dict[str, object]:
    start = time.perf_counter()
    block_until_ready(fn(x, k))
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


def _fmt_float(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def write_summary(
    path: Path,
    config: dict[str, object],
    correctness_rows: list[dict[str, object]],
    perf_rows: list[dict[str, object]],
    fast_stage_rows: list[dict[str, object]],
) -> None:
    lines: list[str] = ["# Exact Top-K TPU Comparison", ""]
    lines.append("## Run")
    lines.extend(
        _markdown_table(
            ["field", "value"],
            [
                ["timestamp", config["timestamp"]],
                ["preset", config["preset"]],
                ["dtype", config["dtype"]],
                ["requested implementations", ", ".join(config["requested_impls"])],
                ["correctness distributions", ", ".join(config["correctness_distributions"])],
                ["benchmark distributions", ", ".join(config["perf_distributions"])],
                ["warmup", config["warmup"]],
                ["iters", config["iters"]],
                ["backend", config["jax_platform"]],
                ["devices", "; ".join(config["jax_devices"])],
            ],
        )
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Cases: {len(config['cases'])}")
    lines.append(
        "- Scenario groups: "
        f"`moe`={sum(1 for case in config['cases'] if case['scenario'] == 'moe')}, "
        f"`vocab_decode`={sum(1 for case in config['cases'] if case['scenario'] == 'vocab_decode')}, "
        f"`sparse`={sum(1 for case in config['cases'] if case['scenario'] in SPARSE_SCENARIOS)}"
    )
    for scenario in SCENARIOS:
        count = sum(1 for case in config["cases"] if case["scenario"] == scenario)
        lines.append(f"- `{scenario}` cases: {count}")
    lines.append("- Raw data: `exact_correctness.csv`, `exact_perf.csv`, `fast_stage_detail.csv`")
    if IMPL_AUTO in config["requested_impls"]:
        lines.append("- `auto` expands implementations per case; see `run_config.json` field `impls_by_case`.")
    lines.append("")

    error_rows = [r for r in correctness_rows if r.get("status") != "ok"]
    valid_failures = [
        r
        for r in correctness_rows
        if r.get("status") == "ok" and r.get("valid_topk_pass") not in (True, "True")
    ]
    ordered_mismatches = [
        r
        for r in correctness_rows
        if r.get("status") == "ok" and r.get("exact_ordered_pass") in (False, "False")
    ]
    lines.append("## Correctness")
    lines.extend(
        _markdown_table(
            ["metric", "count"],
            [
                ["rows", len(correctness_rows)],
                ["errors", len(error_rows)],
                ["valid_topk_failures", len(valid_failures)],
                ["jax_order_mismatches", len(ordered_mismatches)],
            ],
        )
    )
    lines.append("")

    lines.append("## Performance")
    ok_perf = [r for r in perf_rows if r.get("status") == "ok"]
    perf_groups: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for row in ok_perf:
        key = (str(row.get("case", "")), str(row.get("distribution", "")))
        perf_groups.setdefault(key, {})[str(row.get("impl", ""))] = row
    perf_table = []
    for (case, dist), group in sorted(perf_groups.items()):
        medians = {impl: float(row["median_ms"]) for impl, row in group.items() if "median_ms" in row}
        fastest = min(medians, key=medians.get) if medians else ""
        for impl, row in sorted(group.items()):
            perf_table.append(
                [
                    case,
                    dist,
                    impl,
                    _fmt_float(row.get("median_ms")),
                    _fmt_float(row.get("p95_ms")),
                    _fmt_float(row.get("speedup_vs_lax_median")),
                    "yes" if impl == fastest else "",
                ]
            )
    lines.extend(
        _markdown_table(
            ["case", "dist", "impl", "median_ms", "p95_ms", "speedup_vs_lax", "fastest"],
            perf_table,
        )
    )
    lines.append("")

    lines.append("## Fast Exact Stage Detail")
    stage_table = []
    perf_dists = set(config["perf_distributions"])
    for row in fast_stage_rows:
        if row.get("status") != "ok" or row.get("distribution") not in perf_dists:
            continue
        stage_table.append(
            [
                row.get("case", ""),
                row.get("distribution", ""),
                row.get("max_depth", ""),
                row.get("depth_hist", ""),
            ]
        )
    lines.extend(
        _markdown_table(
            ["case", "dist", "max_depth", "depth_hist"],
            stage_table,
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def resolve_impls_for_case(requested_impls: tuple[str, ...], case: BenchmarkCase) -> tuple[str, ...]:
    if IMPL_AUTO not in requested_impls:
        return requested_impls
    impls = list(scenario_default_impls(case.scenario, case.vocab, case.k))
    impls.extend(impl for impl in requested_impls if impl != IMPL_AUTO)
    seen: set[str] = set()
    out: list[str] = []
    for impl in impls:
        if impl not in seen:
            seen.add(impl)
            out.append(impl)
    return tuple(out)


def run_trace(trace_root: Path, impl_name: str, case: BenchmarkCase, distribution: str, fn, x: jax.Array, iters: int) -> str:
    trace_dir = trace_root / f"{case.label}_{distribution}_{impl_name}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
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
    if args.case:
        cases = list(args.case)
    elif args.shape:
        cases = [parse_shape(value, args.scenario) for value in args.shape]
    else:
        cases = preset_cases(args.preset)
    requested_impls = parse_csv_list(args.impls)
    correctness_dists = parse_csv_list(args.correctness_distributions)
    perf_dists = parse_csv_list(args.perf_distributions)
    unknown_impls = sorted(set(requested_impls) - set(ALL_IMPLS) - {IMPL_AUTO})
    if unknown_impls:
        raise ValueError(f"unknown implementations: {unknown_impls}")
    impls_by_case = {case.label: list(resolve_impls_for_case(requested_impls, case)) for case in cases}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"results/exact_topk_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "timestamp": timestamp,
        "preset": args.preset,
        "dtype": args.dtype,
        "cases": [asdict(case) for case in cases],
        "requested_impls": list(requested_impls),
        "impls_by_case": impls_by_case,
        "correctness_distributions": list(correctness_dists),
        "perf_distributions": list(perf_dists),
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

    for case in cases:
        print(f"\n=== {case.label} dtype={args.dtype} ===")
        impls = tuple(impls_by_case[case.label])
        print(f"implementations: {', '.join(impls)}")
        for distribution in correctness_dists:
            key, subkey = jax.random.split(key)
            x = make_input(subkey, case, dtype, distribution)
            x_source = np.asarray(block_until_ready(sanitize_scores(x)))
            print(f"[correctness] distribution={distribution}")
            ref_vals: np.ndarray | None = None
            ref_idx: np.ndarray | None = None
            for impl_name in impls:
                fn = impl_function(impl_name)
                row_base = {
                    "scenario": case.scenario,
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
                                "source_values_match": True,
                                "max_abs_diff": 0.0,
                                "valid_topk_pass": True,
                                "exact_ordered_pass": True,
                            }
                        )
                    else:
                        if ref_vals is None or ref_idx is None:
                            ref_vals, ref_idx = as_numpy_pair(impl_function(IMPL_LAX)(x, case.k))
                        correctness_rows.append(
                            {
                                **row_base,
                                "status": "ok",
                                "first_call_ms": first_call_ms,
                                **compare_outputs(ref_vals, ref_idx, vals, idx, x_source),
                            }
                        )
                except Exception as exc:
                    correctness_rows.append({**row_base, "status": "error", "error": repr(exc)})
                    print(f"  {impl_name}: ERROR {exc!r}")

            if IMPL_FAST in impls:
                try:
                    term = fast_termination_m(x, k=case.k)
                    hist = {int(v): int(np.sum(term == v)) for v in np.unique(term)}
                    max_depth = int(np.max(term))
                    fast_stage_rows.append(
                        {
                            "scenario": case.scenario,
                            "case": case.label,
                            "rows": case.rows,
                            "vocab": case.vocab,
                            "k": case.k,
                            "dtype": args.dtype,
                            "distribution": distribution,
                            "depth_hist": json.dumps(hist, sort_keys=True),
                            "max_depth": max_depth,
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    fast_stage_rows.append(
                        {
                            "scenario": case.scenario,
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

        for distribution in perf_dists:
            key, subkey = jax.random.split(key)
            x = make_input(subkey, case, dtype, distribution)
            print(f"[benchmark] distribution={distribution}")
            ref_median: float | None = None
            start_index = len(perf_rows)
            for impl_name in impls:
                row_base = {
                    "scenario": case.scenario,
                    "case": case.label,
                    "rows": case.rows,
                    "vocab": case.vocab,
                    "k": case.k,
                    "dtype": args.dtype,
                    "distribution": distribution,
                    "impl": impl_name,
                }
                try:
                    stats = bench_impl(impl_function(impl_name), x, case.k, args.warmup, args.iters)
                    if impl_name == IMPL_LAX:
                        ref_median = float(stats["median_ms"])
                    perf_rows.append({**row_base, "status": "ok", **stats})
                    print(f"  {impl_name}: median={stats['median_ms']:.4f} ms p95={stats['p95_ms']:.4f} ms")
                except Exception as exc:
                    perf_rows.append({**row_base, "status": "error", "error": repr(exc)})
                    print(f"  {impl_name}: ERROR {exc!r}")
            if ref_median is not None:
                for row in perf_rows[start_index:]:
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
                            distribution,
                            impl_function(impl_name),
                            x,
                            args.trace_iters,
                        )
                        trace_rows.append(
                            {
                                "scenario": case.scenario,
                                "case": case.label,
                                "distribution": distribution,
                                "impl": impl_name,
                                "trace_dir": trace_dir,
                                "status": "ok",
                            }
                        )
                    except Exception as exc:
                        trace_rows.append(
                            {
                                "scenario": case.scenario,
                                "case": case.label,
                                "distribution": distribution,
                                "impl": impl_name,
                                "status": "error",
                                "error": repr(exc),
                            }
                        )

        write_csv(out_dir / "exact_correctness.csv", correctness_rows)
        write_csv(out_dir / "exact_perf.csv", perf_rows)
        write_csv(out_dir / "fast_stage_detail.csv", fast_stage_rows)
        write_csv(out_dir / "trace_dirs.csv", trace_rows)
        write_summary(out_dir / "exact_summary.md", config, correctness_rows, perf_rows, fast_stage_rows)

    print("\nDone.")
    print(f"Correctness: {out_dir / 'exact_correctness.csv'}")
    print(f"Performance: {out_dir / 'exact_perf.csv'}")
    print(f"Fast stage detail: {out_dir / 'fast_stage_detail.csv'}")
    print(f"Summary: {out_dir / 'exact_summary.md'}")
    if args.trace:
        print(f"Trace dirs: {out_dir / 'trace_dirs.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=(
            "smoke",
            "quick",
            "moe",
            "vocab",
            "vocab_decode",
            "sparse",
            "dsa_sparse_attention",
            "dsa_block",
            "dsa_token",
            "all",
        ),
        default="quick",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="Add a case SCENARIO,ROWS,N,K. May be passed multiple times.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        help="Add ROWS,N,K using --scenario. Kept for quick manual runs.",
    )
    parser.add_argument(
        "--scenario",
        choices=("moe", "vocab", "vocab_decode", "dsa_sparse_attention", "dsa_block", "dsa_token"),
        default="vocab_decode",
    )
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--impls", default=",".join(DEFAULT_IMPLS))
    parser.add_argument("--correctness-distributions", default=",".join(DEFAULT_CORRECTNESS_DISTS))
    parser.add_argument("--perf-distributions", default=",".join(DEFAULT_PERF_DISTS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-iters", type=int, default=5)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
