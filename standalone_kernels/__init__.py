"""Standalone TPU top-k kernel implementations split by algorithm."""

from .xla_reference import xla_topk_reference
from .iterative_mask import iterative_mask_jax, iterative_mask_pallas
from .bitonic_sort import bitonic_sort_jax
from .bucket_select import bucket_select_pallas, bucket_select_pallas_v6_tile
from .soft_sinkhorn import soft_topk_sinkhorn
from .fixed_shape_bitonic_v6 import (
    bitonic_row256_pallas_v6,
    bitonic_tile_partial_pallas_v6,
    bitonic_hierarchical_pallas_v6,
)

from .topk_custom_call_mimic import (
    xla_topk_2d_custom_call_hint,
    repeated_argmax_topk_pallas_2d,
    llo_style_repeated_argmax_pallas_2d,
)
