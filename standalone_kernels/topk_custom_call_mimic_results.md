# TopK Custom-call Pallas Mimic Results

Reference document: `Docs/jax_topk_tpu_llo_analysis.md`

## Implemented Pallas variants

File: `top_k/standalone_kernels/topk_custom_call_mimic.py`

Exports:

- `xla_topk_2d_custom_call_hint`: 2D reshape + `jax.lax.top_k`, useful as a baseline and lowering hint.
- `repeated_argmax_topk_pallas_2d`: high-level Pallas repeated stable argmax + delete winner.
- `llo_style_repeated_argmax_pallas_2d`: closer LLO-style Pallas implementation. It explicitly reduces 128-column chunks using pair compare-select, then merges chunk winners, writes the current rank, and masks/deletes the winner for the next rank.

The LLO-style Pallas comparator is:

```python
take = cand_val > best_val or (cand_val == best_val and cand_idx < best_idx)
best = cand if take else best
```

This mirrors the LLO pattern:

```text
vcmp.gt key
vcmp.eq key
vcmp.lt index
vmand / vmor
vsel key / vsel index
```

Limitations: Pallas here cannot directly emit `vunpack`, `vxpose`, or `vrot.slane`. The implementation therefore mimics the algorithmic structure, not the exact vector-register schedule.

## TPU smoke-test results

Device: single TPU v6e device
Dtype: bf16
k: 8

| Test scale | XLA 2D hint ms | high-level Pallas ms | LLO-style Pallas ms | exact |
| --- | ---: | ---: | ---: | --- |
| `bf16[8,2048], k=8` | 0.1739 | 0.1761 | 0.1753 | yes |
| `bf16[16,4096], k=8` | 0.1774 | 0.1801 | 0.1847 | yes |
| `bf16[2,3,2048], k=8` | 0.1717 | 0.1736 | 0.1724 | yes |
| `bf16[2048], k=8` | 0.1721 | 0.1729 | 0.1738 | yes |

These are short smoke-test medians. The important result is that the LLO-style Pallas implementation compiles, is exact, and follows the same repeated stable-argmax/delete-winner structure described in the LLO analysis.

## Practical conclusion

Yes, the LLO analysis can guide a Pallas implementation. The useful Pallas structure is not bitonic sort; it is small-k repeated stable argmax with explicit pair comparison and winner deletion. The current LLO-style Pallas version is a correct and maintainable approximation of that structure. It does not beat XLA custom-call yet, because XLA uses lower-level lane transpose/rotation instructions that Pallas code does not directly control here.
