# TPU Top-K Standalone Kernels

本目录把 `top_k/topk_algorithms.py` 中已经实现和测试过的 top-k 方案按算法族拆成独立文件，方便逐个阅读、benchmark 和替换。

## 文件结构

| 文件 | 对应算法/实现 | 说明 |
| --- | --- | --- |
| `common.py` | 公共常量与工具 | `NEG_INF`、TPU v6 tile `(16,256)`、NaN sanitize、power-of-2 padding。 |
| `xla_reference.py` | XLA baseline | `jax.lax.top_k` reference，用于正确性和性能对照。 |
| `iterative_mask.py` | Iterative Mask | 全局反复 `reduce-max -> tie-break argmin -> mask -inf`。包含 JAX 与 Pallas 版本。 |
| `bitonic_sort.py` | Bitonic Sort JAX reference | data-oblivious bitonic reference；不可编译的原始 Pallas 版本已移除。 |
| `bucket_select.py` | Bucket Select / iterative bucket | 分块后每块用 iterative max-mask 选 `local_k` 候选，再由 XLA merge。包含 flat block 和 TPU v6 `(16,256)` tile 版本。 |
| `fixed_shape_bitonic_v6.py` | Fixed-shape Bitonic A/B/C | 避免 rank-3 reshape 的 TPU v6 bitonic workaround。 |
| `soft_sinkhorn.py` | Soft Top-k / Sinkhorn | 近似、可微、MXU-friendly 的 soft top-k relaxation。 |

## 实现效果概览

### 1. Iterative Mask

实现：`iterative_mask_jax`、`iterative_mask_pallas`

计算流：

1. 将 NaN 替换为 `-inf`。
2. 重复 `k` 次：
   - 对整个输入做 `reduce-max`。
   - 对等于最大值的位置取最小 index，保证 tie-break 稳定。
   - 将该 index 的值 mask 成 `-inf`。
3. 输出按降序排列的 top-k values 和 indices。

优点：

- 逻辑最简单，和硬件 HLO 映射清晰：`reduce-max`、`compare`、`select`、`broadcast`。
- 对小 `k` 很直接，尤其 MoE router 的 `k=1/2/4/8`。
- 不需要动态大小数组。

缺点：

- 时间复杂度约 `O(N*k)`，`k` 增大时线性变慢。
- 每轮都扫完整输入，典型 memory-bound，MXU 基本帮不上忙。
- 当 `k` 接近 `N` 时会出现性能断崖。

适合场景：

- MoE 专家选择：真实专家数通常是几十到几百，典型 `E=8/16/32/64/128/256`，`k=1/2/4/8`。
- 小 `k`、需要精确 top-k、希望实现非常可控的场景。

不适合：

- 词表 top-k：`N=32K/50K/128K` 且 `k` 可到几十或上百时，反复全量扫描开销大。
- 大 `k` sparse attention block selection。

### 2. 原始 Bitonic Sort

实现：`bitonic_sort_jax`

计算流：

1. 将 `N` padding 到 `2^m`。
2. 使用 bitonic sorting network 固定比较交换。
3. 用 `-score` 作为 key，把升序 sort 变成 score 降序。
4. value 相等时用 index 做 tie-break。
5. 取前 `k`。

优点：

- data-oblivious，比较/交换路径与输入数据无关。
- 对中小 `N`、较大 `k` 的完整排序需求概念清晰。
- JAX reference 很适合验证排序网络语义。

缺点：

- 原始 Pallas 版本在 TPU Mosaic 上会遇到 rank-changing reshape lowering 问题，已从本可直接使用目录移除。
- 排序网络比较次数高，`O(N log^2 N)`，只要 top-k 时通常做了过多工作。
- 跨 lane shuffle / reshape / layout 变换容易成为瓶颈。

适合场景：

- 需要 data-oblivious 行为的研究对照。
- `N <= 4096`、`k` 较大、且能接受完整排序开销的场景。

不适合：

- TPU Pallas bitonic 请使用 `fixed_shape_bitonic_v6.py`，不要使用已移除的原始 reshape 版本。
- MoE router 小 `k`。

### 3. Bucket Select / Iterative Bucket

实现：`bucket_select_pallas`、`bucket_select_pallas_v6_tile`

这里的 “iterative bucket” 指的是 Bucket Select 的局部选择阶段：每个 bucket/tile 内用 iterative max-mask 选候选。

计算流：

1. 将输入 padding 并分块。
2. 每个 block/tile 内做 `local_k` 次 iterative max-mask，输出局部候选。
3. 把所有局部候选 flatten。
4. 用 XLA `top_k` 做最终全局 merge。

优点：

- 把全局 top-k 拆成局部候选选择 + 小规模 merge。
- TPU v6 版本固定输入 tile 为 `(16,256)`，避免不友好的 layout。
- 对 `N` 很大但每块只需要少量候选时，candidate buffer 明显小于原始输入。

缺点：

- 若 `local_k < k`，对极端长尾/集中分布可能不精确。
- 若 `local_k >= k` 保证精确，局部 iterative cost 又会上升。
- 实测中对这些单向量 top-k case 仍慢于 XLA top-k。

适合场景：

- Sparse attention / DSA 的块选择：`N` 是 block 数或 score 数，常见 `N=1K-16K` 甚至更大，`k=16-128`。
- 可以容忍或显式控制 bucket approximation 的场景。
- 想把候选生成放入更大 fused kernel 的场景。

不适合：

- 单独调用并期望比 XLA top-k 快的通用 top-k。
- 候选高度集中在少数 block，而 `local_k` 又设置太小的分布。

### 4. Fixed-shape Bitonic v6 A/B/C

实现：`bitonic_row256_pallas_v6`、`bitonic_tile_partial_pallas_v6`、`bitonic_hierarchical_pallas_v6`

背景：原始 bitonic Pallas 失败的核心原因是 Mosaic 对 kernel 内 rank-changing reshape 支持不足。fixed-shape 版本把 kernel-local 数据固定成 TPU v6 友好的 `(16,256)`，用静态 column slice 和 `concatenate` 实现 row-local bitonic compare-exchange。

A：`bitonic_row256_pallas_v6`

- 每个 tile 的每一行 256 元素独立 bitonic sort。
- 每行保留 `row_k` 候选。
- flatten 后用 XLA top-k merge。

B：`bitonic_tile_partial_pallas_v6`

- 先做 A 的 row-local bitonic。
- 再在 tile 内用 iterative max-mask 选 `local_k`。
- 用 XLA top-k merge tile candidates。

C：`bitonic_hierarchical_pallas_v6`

- 先做 B 生成 stage-1 candidates。
- 后续用固定 `(16,256)` Pallas merge stage 逐层归并，避免最终 XLA merge。

实测 median ms：

| N | k | XLA top_k | bucket_v6 | A | B | C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 16 | 0.172 | 0.171 | 0.180 | 0.181 | 0.181 |
| 32,000 | 16 | 0.175 | 0.213 | 0.260 | 0.296 | 0.301 |
| 32,000 | 64 | 0.180 | 0.354 | 0.259 | 0.450 | 0.463 |
| 128,000 | 64 | 0.277 | 0.924 | 0.604 | 1.245 | 1.305 |

优点：

- 已验证能绕开原始 bitonic Pallas 的 reshape compile failure。
- A 在 `N=32000,k=64`、`N=128000,k=64` 上快于 iterative bucket。
- correctness edge cases 通过：全负数、重复最大值、NaN mixed。

缺点：

- 仍慢于 XLA top-k。
- B/C 叠加了 row sort、iterative reduction 和多次 kernel launch，收益不明显。
- A 是 row-local prefilter，若 `row_k < k` 会变成近似；精确使用时通常令 `row_k >= k`。

适合场景：

- 分析和规避 Mosaic reshape limitation。
- 需要 fixed `(16,256)` layout 的 bitonic 实验。
- 作为未来 fused kernel 中的局部候选生成组件。

不适合：

- 当前作为默认生产 top-k。
- 小 `k` MoE router，XLA 或简单 iterative 通常更合理。

### 5. Soft Top-k / Sinkhorn

实现：`soft_topk_sinkhorn`

计算流：

1. 构造 `N x k` assignment logits。
2. 做多轮 row/column log-sum-exp normalization。
3. 用 assignment probability 对 scores 和 indices 求期望。
4. 输出 expected scores / expected indices。

优点：

- 近似可微，适合需要 soft routing / relaxation 的研究。
- 大量工作可表达为矩阵乘、reduce-sum、exp/log，理论上更 MXU-friendly。

缺点：

- 不是精确 top-k，indices 是期望值，不是离散 index。
- `N x k` allocation 很大，内存和数值稳定性压力高。
- 对 autoregressive decoding 的 hard top-k 不能直接替代。

适合场景：

- 可微选择、soft routing、训练期近似。
- 对精确排序不敏感，只关心平滑选择信号。

不适合：

- 推理期词表 hard top-k。
- MoE dispatch 的精确 expert id 选择。
- 需要离散 sparse attention block id 的路径。

### 6. XLA TopK Custom-call Mimic

实现：`xla_topk_2d_custom_call_hint`、`repeated_argmax_topk_pallas_2d`、`llo_style_repeated_argmax_pallas_2d`

来源：参考 `Docs/jax_topk_tpu_llo_analysis.md` 中对 TPU `TopK` custom-call LLO 的分析。该路径的核心不是完整排序，而是：

1. 把输入组织成二维 `[rows, n]`。
2. 对每一行做 `k` 轮 stable argmax。
3. 比较器按 `(value desc, index asc)` 选择 winner。
4. 写出当前 rank 后，把 winner 位置 mask 成 `-inf`。

`xla_topk_2d_custom_call_hint` 是推荐优先尝试的优化：它只做 leading dims flatten，然后调用 `lax.top_k`，让 XLA 更可能选择内建 `TopK` custom-call。

`repeated_argmax_topk_pallas_2d` 是高层 Pallas 模仿实现。`llo_style_repeated_argmax_pallas_2d` 是更接近 LLO 的 Pallas 实现：每个 rank 显式按 128-lane chunk 做 pair compare-select reduction，比较器为 `(value desc, index asc)`，再删除 winner。它遵守 TPU Pallas block 约束，按 1-row/8-row group 处理；但仍不能直接表达 LLO 里的 `vunpack`、`vxpose`、`vrot.slane`。

适合场景：MoE/router 或 decoding 中小 `k`、较大 last dimension、输入可自然 reshape 为 `[rows, n]` 的精确 top-k。

不适合场景：大 `k`，因为外层 rank loop 顺序依赖 winner deletion；此时 XLA 切换到 sort + slice 通常是合理的。

## 场景建议

| 场景 | 真实常见配置 | 推荐实现 | 原因 |
| --- | --- | --- | --- |
| MoE expert top-k | `E=8-256`，`k=1/2/4/8` | XLA top-k 或 `iterative_mask_pallas` | 专家数没有 32K；小 `k` 下 iterative mask 简单可控。若和 router matmul 融合，需重新 benchmark。 |
| Autoregressive vocab top-k | vocab `32K/50K/128K`，`k=1-100` | XLA top-k | 单向量大 vocab top-k 上 XLA 已高度优化，当前 Pallas 版本未跑赢。 |
| Sparse attention / DSA selection | score/block 数 `1K-16K+`，`k=16-128` | `bucket_select_pallas_v6_tile` 或 fixed-shape A | bucket 适合局部候选；fixed-shape A 可作为 compile-safe bitonic prefilter。 |
| Data-oblivious research | `N<=4096`，`k=16-256` | `bitonic_sort_jax` / fixed-shape A | 本目录仅保留 JAX reference；TPU Pallas bitonic 使用 fixed-shape 版本。 |
| Differentiable/soft selection | `N=10K-50K`，精度可放宽 | `soft_topk_sinkhorn` | 只适合近似可微选择，不适合 hard id 输出。 |

## 当前结论

1. XLA top-k 仍是单独 top-k 调用的默认强 baseline。
2. Pallas iterative mask 的价值主要在小 `k`、可融合、可控计算流，而不是替代所有 XLA top-k。
3. Bucket Select 是 DSA/sparse selection 中更有结构意义的方案，但要严肃处理 `local_k` 与分布长尾问题。
4. Fixed-shape bitonic 解决了 compile failure，但性能上更像 workaround 和研究工具，不是默认 production kernel。
5. Soft Sinkhorn 属于另一个问题：近似可微选择，而非精确 top-k。


## 可直接使用性说明

本目录已经移除已知不能通过 TPU Mosaic 编译的原始 `bitonic_sort_pallas`。当前保留的 Pallas 实现均已做过 TPU smoke test：`iterative_mask_pallas`、`bucket_select_pallas_v6_tile`、`bitonic_row256_pallas_v6`、`bitonic_tile_partial_pallas_v6`、`bitonic_hierarchical_pallas_v6`。`bitonic_sort_jax` 是 JAX/XLA reference，不是 Pallas kernel。`soft_topk_sinkhorn` 可编译运行，但语义是近似 soft top-k，不是精确 hard top-k。`repeated_argmax_topk_pallas_2d` 和 `llo_style_repeated_argmax_pallas_2d` 是参考 XLA TopK custom-call 计算流新增的可编译 Pallas 模仿实现，其中后者显式模拟 128-lane pair reduction。
