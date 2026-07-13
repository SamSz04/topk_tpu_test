# Exact Top-K TPU Comparison

This repository benchmarks exact top-k implementations on TPU for the three
target use cases in this project:

- MoE routing, where top-k selects experts for each token.
- Vocabulary decoding, where top-k selects candidate tokens from logits.
- DSA sparse attention, where top-k selects either sparse blocks or sparse
  tokens from attention scores.

The comparison focuses on exact top-k only. Approximate top-k and Sinkhorn-style
soft top-k variants are intentionally excluded.

## Implementations

The benchmark always includes these two baselines:

- `lax_top_k`: `jax.lax.top_k`
- `fast_exact`: Tallax divide-and-filter exact top-k, vendored into this repo
  instead of imported from `tallax`

The local exact candidates from the earlier `top_k` work are available through
scenario-specific `auto` selection:

- `iterative_jax`
- `iterative_pallas`
- `bucket_v6_exact`
- `bitonic_jax`
- `fixed_bitonic_a`
- `fixed_bitonic_b`
- `pallas_high`
- `pallas_llo`

`fast_exact` lives in [fast_exact_topk_tpu.py](fast_exact_topk_tpu.py). Its
supporting Tallax modules are vendored under [fast_exact_topk_core/](fast_exact_topk_core/).
The benchmark wrappers and local exact kernels live in
[exact_topk_impls.py](exact_topk_impls.py).

## Repository Layout

- [compare_exact_topk.py](compare_exact_topk.py): CLI runner for correctness,
  latency, optional trace capture, and summary generation.
- [exact_topk_impls.py](exact_topk_impls.py): implementation registry, wrappers,
  local exact kernels, and fast_exact stage-depth helper.
- [fast_exact_topk_tpu.py](fast_exact_topk_tpu.py): vendored Tallax
  divide-and-filter top-k entry points.
- [fast_exact_topk_core/](fast_exact_topk_core/): vendored Tallax helper modules
  for bitonic sorting, symbolic integers, utilities, and convergence schedule
  estimation.
- [pyproject.toml](pyproject.toml): minimal runtime dependencies.

Generated benchmark outputs are written under `results/` and are ignored by git.
Local virtual environments, TPU traces, logs, CSVs, and other run artifacts are
also ignored.

## Setup

On a TPU VM:

```bash
uv sync
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

The project depends on JAX, NumPy, and SciPy. `jax[tpu]` is selected on Linux in
`pyproject.toml`; local non-TPU machines install plain `jax`.

## Scenarios

The `--preset all` sweep is built from the historical benchmark plan:

| Scenario | Meaning | `(N, K)` cases | Batch sizes |
| --- | --- | --- | --- |
| `moe` | expert routing | `(8,2)`, `(16,4)`, `(64,8)`, `(128,8)`, `(256,8)`, `(512,10)`, `(2048,1)`, `(2048,2)` | `1`, `32`, `128` |
| `vocab_decode` | vocabulary decoding logits | `(32000,1/20/50)`, `(50304,1/20/50)`, `(128256,1/20/50/100)`, `(129280,1/20/50)`, `(151936,1/20/50/100)` | `1`, `8`, `32` |
| `dsa_block` | block-level sparse attention | `(256,64)`, `(512,64)`, `(1024,64)`, `(1024,256)`, `(64,16)`, `(256,16)`, `(512,16)`, `(2048,64)`, `(2048,256)`, `(2048,512)` | `1`, `8`, `32` |
| `dsa_token` | token-level sparse attention | `(8192,2048)`, `(32768,2048)`, `(65536,2048)`, `(131072,2048)`, `(262144,2048)` | `1`, `8` |

The CSV/config column named `vocab` is kept for compatibility with older runs.
For non-decoding scenarios it means the number of top-k candidates, not
necessarily a model vocabulary size.

## Implementation Selection

The default `--impls auto` expands implementations by scenario:

- `moe`: `lax_top_k`, `fast_exact`, iterative JAX/Pallas, bucket v6, bitonic
  JAX, fixed bitonic, repeated-argmax Pallas, and selected LLO-style repeated
  argmax.
- `vocab_decode`: `lax_top_k`, `fast_exact`, plus local exact candidates gated
  by the `(N, K)` combinations used in the old benchmark plan.
- `dsa_block`: `lax_top_k`, `fast_exact`, and local exact block-level sparse
  attention candidates.
- `dsa_token`: `lax_top_k` and `fast_exact` only by default, because the old
  local block kernels are not the right default for token-level `K=2048`.

You can override this with a comma-separated list:

```bash
uv run python compare_exact_topk.py \
  --case vocab_decode,8,50304,50 \
  --impls lax_top_k,fast_exact,bucket_v6_exact
```

## Input Distributions

Correctness runs default to:

- `random`
- `repeated`
- `all_equal`
- `concentrated_128`
- `concentrated_tile`
- `ascending`
- `descending`
- `nan_mixed`

Latency runs default to:

- `random`
- `concentrated_128`

NaNs are sanitized to `-inf` before comparison. Ties are validated by top-k value
correctness and source-value consistency; exact JAX index ordering may differ for
valid tied top-k sets.

## Running Benchmarks

Smoke test:

```bash
uv run python compare_exact_topk.py --preset smoke --iters 5
```

Representative quick sweep:

```bash
uv run python compare_exact_topk.py --preset quick --iters 20
```

Full aligned sweep:

```bash
uv run python compare_exact_topk.py --preset all --iters 50
```

One explicit case:

```bash
uv run python compare_exact_topk.py \
  --case dsa_token,8,262144,2048 \
  --iters 100
```

Manual shape with a scenario:

```bash
uv run python compare_exact_topk.py \
  --scenario vocab_decode \
  --shape 32,151936,100 \
  --iters 100
```

Optional TPU trace capture:

```bash
uv run python compare_exact_topk.py \
  --preset smoke \
  --trace \
  --trace-iters 5
```

## Outputs

Each run creates a timestamped directory under `results/` unless `--out-dir` is
provided. The main files are:

- `run_config.json`: benchmark configuration, case list, and implementation
  expansion for each case.
- `exact_correctness.csv`: correctness results against `jax.lax.top_k`.
- `exact_perf.csv`: latency samples and summary statistics.
- `fast_stage_detail.csv`: fast_exact convergence depth histogram per case and
  distribution.
- `exact_summary.md`: generated Markdown summary.
- `trace_dirs.csv`: trace locations when `--trace` is enabled.

These files are run artifacts and are not committed.

## Notes

- Run performance benchmarks on TPU. Several Pallas kernels are TPU-oriented and
  are not intended as CPU benchmarks.
- `fast_exact` is vendored from Tallax so the benchmark does not depend on an
  external `tallax` import at runtime.
- For `K >= 1024`, the local fast_exact wrapper uses
  `num_bins=1024, bins_topm_schedule=(16,)` to cover the large-`K` DSA token
  cases robustly.
