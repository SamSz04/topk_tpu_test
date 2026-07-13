"""VMEM-based bitonic sort and topk implementations.

This module contains implementations for sorting and top-k operations that fit in VMEM.
"""

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from fast_exact_topk_core.utils import (
  log2 as log2,
  pad as pad,
  float_to_sortable_int,
  sortable_int_to_float,
  pack_bf16_u16_to_i32,
  unpack_bf16_u16_from_i32,
  create_bit_indicator as create_bit_indicator,
  transpose_list_of_lists as transpose_list_of_lists,
  to_32bit_dtype,
  canonicalize_operand,
  is_32bit as is_32bit,
  NUM_LANES,
  NUM_SUBLANES,
  ceil_multiple,
)
from fast_exact_topk_core.bitonic.sort import (
  bitonic_sort_maybe_rolled,
  _compute_padded_shape as _compute_padded_shape_for_sort,
  compute_pair_slice_start_index as compute_pair_slice_start_index,
  compare_and_swap as compare_and_swap,
)
from fast_exact_topk_core.bitonic.topk import (
  bitonic_topk_arrays,
  _compute_padded_shape as _compute_padded_shape_for_topk,
)
from fast_exact_topk_core.symint import SymInt


### VMEM-Based Sort (fits in VMEM)
def bitonic_sort_in_vmem_refs(
  in_refs,
  stage_ref,
  out_refs,
  transpose_refs,  # scratch refs operated on
  tranpose_indices_ref,
  *,
  descending: bool,
  is_stable: bool,
  num_keys: int,
  k: int | None = None,
  stage_unroll: int | bool = True,
  slice_size_unroll: int | bool = True,
  ref_slice_size_unroll: int | bool = True,
  # int key operands may contain INT_MAX (ascending sort) or INT_MIN (descending sort) requiring stable sort to avoid padding leakage
  int_key_operands_may_contain_int_max_or_min: bool = True,
):
  """Pallas kernel for sorting using bitonic sort."""
  operands = [ref[...] for ref in in_refs]
  shape = operands[0].shape
  assert len(shape) == 2
  padded_shape = (
    _compute_padded_shape_for_topk(*shape, k=k)
    if k != shape[1]
    else _compute_padded_shape_for_sort(*shape)
  )

  return_argsort = len(out_refs) > len(in_refs)
  assert len(out_refs) == (len(in_refs) + int(return_argsort))

  if (
    int_key_operands_may_contain_int_max_or_min
    and (shape[1] != padded_shape[1])
    and any(not jnp.issubdtype(x.dtype, jnp.floating) for x in operands)
  ):
    # Floats handled by standardizing 'in bounds' nans to a lower bit value to padding nans in the i32 format theyre sorted in
    # But integer values of INT_MAX/INT_MIN in padding may leak unless comparison is stable
    is_stable = True

  use_indices = is_stable or return_argsort
  indices = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

  if descending and is_stable:
    # Maintain order by sorting indices ascending while keys descending
    # Reverse indices (negate relative to array length), then reverse back before write out
    indices = shape[1] - 1 - indices

  if use_indices:
    operands.insert(num_keys, indices)
    transpose_refs.insert(num_keys, tranpose_indices_ref)
  num_keys += int(is_stable)

  # Optimize bf16 + u16 case by packing into single i32
  use_packed_bf16_u16 = (
    len(operands) == 2
    and operands[0].dtype == jnp.bfloat16
    and (operands[1].dtype == jnp.uint16 or (use_indices and shape[1] <= 2**16))
  )
  if use_packed_bf16_u16:
    operands = [pack_bf16_u16_to_i32(*operands)]
    transpose_refs = transpose_refs[:1]
    num_keys = 1

  for i in range(num_keys):
    if jnp.issubdtype(operands[i].dtype, jnp.floating):
      operands[i] = float_to_sortable_int(operands[i])

  # Most TPU generations only allow 32,32->1 bit comparisons,
  # not bf16,bf16->i1 so we upcast everything to 32bit
  operands = [x.astype(to_32bit_dtype(x.dtype)) for x in operands]

  # If subsorting an input (for hybrid HBM-VMEM sorting) we deal with grid context related offset here
  is_subsort = pl.num_programs(1) != 1
  if is_subsort:
    sort_dim_offset = (
      SymInt(pl.program_id(1), 0, pl.num_programs(1) - 1)
      + int(descending) * pl.num_programs(1)
    ) * padded_shape[1]
  else:
    sort_dim_offset = None
  if k == shape[1]:
    # sort
    operands = bitonic_sort_maybe_rolled(
      operands,
      num_keys=num_keys,
      axis=1,
      descending=descending if not is_subsort else False,
      single_stage=None if stage_ref is None else stage_ref[0],
      stage_unroll=stage_unroll if stage_ref is None else False,
      slice_size_unroll=slice_size_unroll,
      ref_slice_size_unroll=ref_slice_size_unroll,
      # only used when not using pure arrays implementation
      transpose_refs=transpose_refs,
      sort_dim_offset=sort_dim_offset,
    )
  else:
    # top-k
    # this is fully unrolled and supports only certain k
    def _maybe_invert(operands):
      if not descending:
        for i in range(num_keys):
          # all keys should be ints, we flip them
          assert jnp.isdtype(operands[i].dtype, "integral")
          operands[i] = jnp.invert(operands[i])

    _maybe_invert(operands)
    operands = bitonic_topk_arrays(operands, k=k, num_keys=num_keys, axis=1)
    _maybe_invert(operands)

  # Unpack bf16-u16 if used
  if use_packed_bf16_u16:
    operands = list(unpack_bf16_u16_from_i32(operands[0]))
    num_keys = 1 + int(is_stable)

  if use_indices:
    indices = operands.pop(num_keys - int(is_stable))
  if return_argsort:
    if descending and is_stable:
      indices = shape[1] - 1 - indices
    operands.append(indices)

  for i, (out, out_ref) in enumerate(zip(operands, out_refs, strict=True)):
    if jnp.issubdtype(out.dtype, jnp.integer) and jnp.issubdtype(
      out_ref.dtype, jnp.floating
    ):
      # Check if this was a float key that we converted
      out = sortable_int_to_float(out)
    out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(
  jit,
  static_argnames=(
    "num_keys",
    "return_argsort",
    "descending",
    "is_stable",
    "k",
    "interpret",
    "block_token",
    "block_seq",
    "compile_fast",
    "stage_unroll",
    "slice_size_unroll",
    "ref_slice_size_unroll",
  ),
)
def bitonic_sort_in_vmem(
  operand: jax.Array | Sequence[jax.Array],
  # behavior control
  num_keys: int,
  return_argsort: bool = False,
  descending: bool = False,
  is_stable: bool = False,
  k: int | None = None,
  # niche behavior for larger than vmem inputs
  stage: int | jax.Array | None = None,
  interpret: bool = False,
  # implementation details
  block_token: int | None = None,
  block_seq: int | None = None,
  compile_fast: bool | None = None,
  # specialist unroll controls, suggest setting just compile_fast=True if compilation is too slow, it will overwrite and set these other unrolls
  stage_unroll: int | bool = True,
  slice_size_unroll: int | bool = True,
  ref_slice_size_unroll: int | bool = True,
) -> tuple[jax.Array, ...]:
  """Sort arrays that fit in VMEM using bitonic sort.

  Args:
    operand: Input array(s) to sort (2D)
    num_keys: Number of arrays to use as sort keys
    return_argsort: Whether to return argsort indices
    descending: Sort in descending order
    is_stable: Whether to perform stable sort
    stage: Specific stage to run (for multi-stage sorting)
    interpret: Run in interpret mode
    block_token: Token blocking size for memory efficiency
    block_seq: Sequence blocking size for use if subsorting operands
    compile_fast: Use faster compilation settings (reduced unrolling)
    stage_unroll: Number of stages to unroll in bitonic sort
    slice_size_unroll: Slice size unroll parameter for bitonic sort
    ref_slice_size_unroll: Ref slice size unroll parameter for bitonic sort
    unroll_stages: Whether to unroll stages in bitonic sort

  Returns:
    Tuple of sorted arrays (and optionally argsort indices)
  """
  operands, shape = canonicalize_operand(operand)
  if block_token is None:
    # heuristic for fitting in VMEM
    # multiple of NUM_SUBLANES, between NUM_SUBLANES and NUM_LANES
    block_token = min(
      ceil_multiple(min((2**14) // shape[0], shape[0]), NUM_SUBLANES), NUM_LANES
    )
  if block_seq is None:
    block_seq = shape[1]
  if shape[1] % block_seq != 0:
    raise ValueError
  if k is None:
    k = shape[1]
  if k != shape[1] and block_seq != shape[1]:
    raise ValueError("k is not compatible with subsorting")

  if compile_fast is None:
    # if projected compilation time expected to be more than a minute, compile fast
    compile_fast = (
      block_token
      * block_seq
      * (len(operands) + int(return_argsort or is_stable))
    ) > 2**19
  if compile_fast:
    # reduces compilation time scaling to linear
    stage_unroll, slice_size_unroll, ref_slice_size_unroll = (6, 7, 8)

  unconverted_operands = tuple(operands)
  # On CPU (interpret mode), convert floats to sortable ints outside Pallas to avoid ref bitcast lowering issues. On TPU, keep conversion inside Pallas kernel for efficiency (and it can allow bf16-u16 packing)
  if interpret:
    for i in range(num_keys):
      if jnp.issubdtype(operands[i].dtype, jnp.floating):
        operands[i] = float_to_sortable_int(operands[i])

  block_shape = (block_token, block_seq)
  out_shapes = jax.tree.map(
    lambda v: jax.ShapeDtypeStruct((shape[0], k), v.dtype), unconverted_operands
  )
  if return_argsort:
    out_shapes += (jax.ShapeDtypeStruct((shape[0], k), jnp.int32),)

  in_specs = (
    [pl.BlockSpec(block_shape, lambda i, j: (i, j)) for _ in operands],
    pl.BlockSpec(memory_space=pltpu.SMEM) if stage is not None else None,
  )
  out_specs = tuple(
    pl.BlockSpec((block_token, min(k, block_seq)), lambda i, j: (i, j))
    for _ in out_shapes
  )

  # Create transpose scratch refs for bitonic sort
  dim0, dim1 = _compute_padded_shape_for_sort(*block_shape)
  transpose_block_shape = (dim1 // pl.cdiv(NUM_LANES, dim0), NUM_LANES)
  scratch_shapes = (
    [
      pltpu.VMEM(
        transpose_block_shape,
        jnp.int32 if i < num_keys else to_32bit_dtype(ref.dtype),
      )
      for i, ref in enumerate(operands)
    ],
    pltpu.VMEM(transpose_block_shape, jnp.int32),
  )

  if stage is not None:
    stage = stage[None]

  return pl.pallas_call(
    functools.partial(
      bitonic_sort_in_vmem_refs,
      descending=descending,
      num_keys=num_keys,
      is_stable=is_stable,
      k=k,
      stage_unroll=stage_unroll,
      slice_size_unroll=slice_size_unroll,
      ref_slice_size_unroll=ref_slice_size_unroll,
    ),
    out_shape=(out_shapes,),
    in_specs=in_specs,
    out_specs=(out_specs,),
    scratch_shapes=scratch_shapes,
    grid=(pl.cdiv(shape[0], block_token), shape[1] // block_seq),
    compiler_params=pltpu.CompilerParams(
      vmem_limit_bytes=int(0.9 * 2**27),
    ),
    interpret=interpret,
  )(operands, stage)[0]


@functools.partial(
  jit,
  static_argnames=(
    "k",
    "num_keys",
    "return_argsort",
    "is_stable",
    "interpret",
    "block_token",
  ),
)
def bitonic_topk_in_vmem(
  operand: jax.Array | Sequence[jax.Array],
  k: int,
  num_keys: int = 1,
  return_argsort: bool = True,
  is_stable: bool = False,
  block_token: int | None = None,
  interpret: bool = False,
) -> tuple[jax.Array, ...]:
  """Top-K selection using bitonic sort."""
  return bitonic_sort_in_vmem(
    operand,
    num_keys=num_keys,
    return_argsort=return_argsort,
    descending=True,
    is_stable=is_stable,
    k=k,
    block_token=block_token,
    interpret=interpret,
  )
