"""
Bitonic sort using compressed transpose format.
"""

import functools
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
# from jax.experimental.pallas import tpu as pltpu

from fast_exact_topk_core.utils import (
  NUM_LANES,
  NUM_SUBLANES,
  log2,
  flatten,
  ceil_multiple,
  iota_tile,
  pad,
  transpose_list_of_lists,
  to_compressed_transpose_format,
  from_compressed_transpose_format,
  to_32bit_dtype,
  create_bit_indicator,
  set_cummax,
  map_batch_dim_to_smaller_than_hardware_tile_size,
  join_tiles_to_array,
  split_array_to_tiles,
  maybe_static_jit,
)
from fast_exact_topk_core.symint import SymInt, unwrap


@functools.partial(
  maybe_static_jit,
  static_argnames=("num_keys", "has_unique_key"),
  maybe_static_argnames=("is_descending",),
)
def compare_and_swap(
  lefts,
  rights,
  *,
  num_keys: int,
  is_descending: jax.Array | None,
  is_right_half=None,
  has_unique_key=False,
):
  """Compare and conditionally swap array pairs.

  Args:
    lefts: Tuple of left arrays to compare
    rights: Tuple of right arrays to compare
    num_keys: Number of arrays to use as sort keys
    is_descending: Boolean mask for sort direction (None implies ascending)
    is_right_half: Mask for subtile comparisons. Needed for handling ties in values correctly.
    has_unique_key: Whether first key is guaranteed unique (optimizes sort)

  Returns:
    Tuple of (sorted_lefts, sorted_rights) or sorted values for subtile.
  """
  num_arrs = len(lefts)

  def _compare_pair(i, left, right):
    handle_subtile_ties = (
      is_right_half is not None
      and not has_unique_key
      and num_arrs != num_keys
      and i == num_keys - 1
    )

    if handle_subtile_ties:
      left, right = (
        jnp.where(is_right_half, right, left),
        jnp.where(is_right_half, left, right),
      )

    mask = (
      left > right
      if type(is_descending) == bool and is_descending
      else right > left
    )
    mask = mask.astype(jnp.int32)

    if is_right_half is not None and not handle_subtile_ties:
      mask = jnp.bitwise_xor(mask, is_right_half.astype(jnp.int32))
    return mask

  masks = tuple(
    _compare_pair(i, left, right)
    for i, (left, right) in enumerate(zip(lefts, rights, strict=True))
  )

  ties = [(left == right) for left, right in zip(lefts, rights, strict=True)]

  mask = masks[0]
  for k in range(1, num_keys):
    # Break ties in primary key with secondary key comparison
    mask = jnp.where(ties[k - 1], masks[k], mask)
    ties[k] &= ties[k - 1]

  if is_descending is not None and type(is_descending) != bool:
    # Dynamic descending mask
    mask = mask.astype(bool)
    is_descending = is_descending.astype(bool)
    mask = mask ^ is_descending

  return jax.tree.map(
    lambda left, right: (
      (jnp.where(mask, left, right), jnp.where(mask, right, left))
      if is_right_half is None
      else jnp.where(mask, left, right)
    ),
    lefts,
    rights,
  )


@lru_cache
def compute_pair_slice_start_index(i, separation, slice_length=1):
  """Compute start index for pair-wise array slicing."""
  if slice_length > separation:
    raise ValueError(
      f"Separation must be at least slice length, {separation=} {slice_length=}"
    )
  slices_per_pair = separation // slice_length
  pair_idx = i // slices_per_pair
  slice_idx = i % slices_per_pair
  return pair_idx * 2 * separation + slice_idx * slice_length


def _compute_padded_shape(
  unpadded_dim0: int, unpadded_dim1: int
) -> tuple[int, int]:
  """Compute padded shape compatible with compressed transpose format requirements.

  This function finds the minimal padded shape that satisfies the constraints:
  - dim0 is a power of 2 between NUM_SUBLANES and NUM_LANES (inclusive)
  - dim1 must satisfy divisibility requirements
  - For dim0 < NUM_LANES: num_elems must be divisible by NUM_LANES^2 so mosaic
    lowers the split and concat on full tiles, subtile concat not supported
  - For dim0 == NUM_LANES (small inputs): dim1 just needs to be a power of 2 >= NUM_SUBLANES

  Args:
    unpadded_dim0: Original first dimension size
    unpadded_dim1: Original second dimension size

  Returns:
    Tuple of (padded_dim0, padded_dim1) compatible with compressed transpose format
  """
  if unpadded_dim0 >= NUM_LANES:
    dim0 = ceil_multiple(unpadded_dim0, NUM_LANES)
    dim1 = 2 ** log2(ceil_multiple(unpadded_dim1, NUM_SUBLANES))
    return (dim0, dim1)

  dim0s = [
    2**i
    for i in range(log2(NUM_SUBLANES), log2(NUM_LANES) + 1)
    if 2**i >= unpadded_dim0
  ]
  shapes = []
  for dim0 in dim0s:
    if dim0 == NUM_LANES:
      # For very small inputs where dim0 pads to NUM_LANES, just pad dim1 to next power of 2 >= NUM_SUBLANES
      dim1 = 2 ** log2(ceil_multiple(unpadded_dim1, NUM_SUBLANES))
    else:
      # For dim0 < NUM_LANES, maintain NUM_LANES^2 divisibility constraint
      dim1 = 2 ** log2(
        ceil_multiple(unpadded_dim1, NUM_LANES * NUM_LANES // dim0)
      )
    shapes.append((dim0, dim1))
  # take minimal num elements, larger dim0 on ties as cross tile ops are faster than cross lane
  return sorted(shapes, key=lambda x: (x[0] * x[1], -x[0]))[0]


def _resplit(operands, target_tile_dim0: int):
  def _resplit_inner(operand):
    tiles = jax.tree.leaves(operand)
    dim0 = tiles[0].shape[0]
    if dim0 == target_tile_dim0:
      return tiles
    elif dim0 > target_tile_dim0:
      return flatten([
        jnp.split(tile, dim0 // target_tile_dim0, axis=0) for tile in tiles
      ])
    else:
      l = target_tile_dim0 // dim0
      return [
        jnp.concatenate(operand[i * l : (i + 1) * l], axis=0)
        for i in range(len(tiles) // l)
      ]

  return [_resplit_inner(x) for x in operands]


def _rejoin(operands):
  def _inner(operand):
    tiles = jax.tree.leaves(operand)
    return jnp.concatenate(tiles, axis=0)

  return [_inner(x) for x in operands]


def reverse_tiles(arr):
  return jnp.concatenate(
    jnp.split(arr, arr.shape[0] // NUM_SUBLANES, 0)[::-1], axis=0
  )


def _get_full_size(inputs):
  """Compute full size (dim0 in compressed format) from inputs.

  Args:
    inputs: Can be a single array, list of arrays, or nested list structure

  Returns:
    Full size of the first dimension in compressed transpose format
  """
  leaves = jax.tree.leaves(inputs[0])
  return len(leaves) * leaves[0].shape[0]


def concrete_and_true(b):
  return type(b) == bool and b


def _compute_is_descending(
  stage: SymInt | int,
  tile_start_offset: SymInt | int,
  tile_local_offset: jax.Array,
  sort_dim_offset: SymInt | int,
  full_size: int,
  substage: int | None = None,
):
  # Check if we can optimize based on stage comparisons
  if concrete_and_true(stage < log2(NUM_SUBLANES)) or concrete_and_true(
    stage >= log2(full_size)
  ):
    # Same pattern for all tiles
    return create_bit_indicator(
      unwrap(stage), tile_local_offset + unwrap(sort_dim_offset)
    )

  if concrete_and_true(stage >= log2(NUM_SUBLANES)) and concrete_and_true(
    stage < log2(full_size)
  ):
    # Bit set by tile_offset, constant within tile, differs across tiles
    return create_bit_indicator(
      unwrap(stage), tile_start_offset + unwrap(sort_dim_offset)
    )

  # Can't optimize - use full computation
  return create_bit_indicator(
    unwrap(stage),
    tile_start_offset + tile_local_offset + unwrap(sort_dim_offset),
  )


@functools.partial(
  maybe_static_jit,
  static_argnames=(
    "substage",
    "num_keys",
    "batch_size",
    "full_size",
    "concat_threshold",
    "max_reduce",
  ),
  maybe_static_argnames=("stage", "sort_dim_offset"),
)
def bitonic_sort_substage(
  arrs_tiles,
  *,
  substage,
  num_keys: int,
  batch_size: int,
  stage: SymInt | int | None = None,
  sort_dim_offset: int = 0,
  full_size: int = None,
  concat_threshold: int | None = None,
  max_reduce: bool = False,
):
  """Perform intra-tile bitonic comparison for sort.

  Args:
    arrs_tiles: Tuple of lists of tile arrays
    axis: Axis along which to apply permutation (0 or 1)
    separation: Distance between elements to compare within tile
    stage: Current sorting stage
    num_keys: Number of sort keys
    sort_dim_offset: Offset for bitonic order calculation
    batch_size: Batch size for computing tile offsets

  Returns:
    Tuple of lists of tiles with updated values
  """
  assert max_reduce or stage is not None
  separation = 2**substage
  # if still arrays, we make it into one big tile so its sanitized to list[list[jax.ndarray]]
  arrs_tiles = list(map(jax.tree.leaves, arrs_tiles))
  if full_size is None:
    full_size = len(arrs_tiles[0]) * arrs_tiles[0][0].shape[0]
  if separation < NUM_SUBLANES or separation >= full_size:
    # we need to permute within tiles
    axis = int(separation >= full_size)
    intra_tile_separation = (
      separation if axis == 0 else ((separation * batch_size) // full_size)
    )

    # we need hardware tiles to lower the permute
    arrs_tiles = _resplit(arrs_tiles, NUM_SUBLANES)
    # Compute is_descending for each tile based on bitonic pattern
    tile_local_offset = iota_tile(0) + (iota_tile(1) // batch_size) * full_size
    is_right_half = create_bit_indicator(
      log2(intra_tile_separation), iota_tile(axis)
    )
    permutation = jnp.bitwise_xor(iota_tile(axis), intra_tile_separation)
    # Apply permutation to all tiles
    arrs_tiles_permuted = jax.tree.map(
      lambda tile: jnp.take_along_axis(tile, permutation, axis=axis), arrs_tiles
    )

    # Compare and merge with permuted values
    outs_tiles = [[None for _ in t] for t in arrs_tiles]
    for idx, (lefts, rights) in enumerate(
      zip(
        *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
        strict=True,
      )
    ):
      for arr_idx, out in enumerate(
        compare_and_swap(
          lefts,
          rights,
          is_descending=_compute_is_descending(
            stage=stage,
            tile_start_offset=idx * NUM_SUBLANES,
            tile_local_offset=tile_local_offset,
            sort_dim_offset=sort_dim_offset,
            full_size=full_size,
            substage=substage,
          )
          if not max_reduce
          else True,
          is_right_half=is_right_half,
          num_keys=num_keys,
        )
      ):
        outs_tiles[arr_idx][idx] = out
  else:
    # Comparison between tiles

    # concatting tiles simplifies the code, but hides optimizationsy from the compiler. So until tiles are large (the concat_threshold) we keep them as hardware tile size
    tile_size = (
      separation
      if ((concat_threshold is not None) and (separation >= concat_threshold))
      else NUM_SUBLANES
    )

    arrs_tiles = _resplit(arrs_tiles, tile_size)
    tile_shape = arrs_tiles[0][0].shape
    num_tiles = len(arrs_tiles[0])
    tile_separation = separation // tile_shape[0]

    tile_local_offset = (
      iota_tile(0, tile_shape)
      + (iota_tile(1, tile_shape) // batch_size) * full_size
    )

    outs_tiles = [[None for _ in t] for t in arrs_tiles]
    for i in range(num_tiles // 2):
      idx = compute_pair_slice_start_index(i, separation=tile_separation)
      lefts, rights = (
        transpose_list_of_lists(arrs_tiles)[j]
        for j in (idx, idx + tile_separation)
      )
      for arr_idx, (out_left, out_right) in enumerate(
        compare_and_swap(
          lefts,
          rights,
          is_descending=_compute_is_descending(
            stage=stage,
            tile_start_offset=idx * tile_shape[0],
            tile_local_offset=tile_local_offset,
            sort_dim_offset=sort_dim_offset,
            full_size=full_size,
            substage=substage,
          )
          if not max_reduce
          else True,
          num_keys=num_keys,
        )
      ):
        outs_tiles[arr_idx][idx] = out_left
        if not max_reduce:
          outs_tiles[arr_idx][idx + tile_separation] = out_right
  if max_reduce:
    # remove the Nones, the lower half we discard for top-k usage
    outs_tiles = [
      [v for v in out_tiles if v is not None] for out_tiles in outs_tiles
    ]
  assert all(not any(v is None for v in out_tiles) for out_tiles in outs_tiles)
  return outs_tiles


def _bitonic_sort_substages_array_or_refs(
  inputs,
  substage_and_stage_schedule: list[tuple[int, int]],
  *,
  is_ref: bool,
  num_keys: int,
  batch_size: int,
  sort_dim_offset: int = 0,
  inner_size=None,
  outer_size=None,
  concat_threshold=None,
):
  """Apply bitonic sort substages to inputs. Applies inner and outer unrolls. inner_unroll can reduce register pressure. If input is arrays the outer unroll is meaningless, but it controls unroll for refs.

  Args:
    inputs: list[pl.MemoryRef | jax.Array] to sort
    is_ref: True if inputs are MemoryRefs, False if JAX arrays
  """
  # full_size is dim0 of the input in compressed transpose format
  full_size = _get_full_size(inputs)
  if outer_size is None:
    outer_size = full_size
  if inner_size is None:
    inner_size = outer_size
  if concat_threshold is None:
    concat_threshold = inner_size
  inner_size = min(inner_size, outer_size)  # guarding

  # checks if the outer chunking of input is compatible with the substage comparison separation, splitting it up into parts if not
  inner_size_compatible = tuple(
    2**substage < inner_size for (substage, _) in substage_and_stage_schedule
  )
  if all(inner_size_compatible):
    pass
  elif all((not b for b in inner_size_compatible)):
    outer_size = full_size
    inner_size = full_size
  else:
    # will switch between running on chunks and full input. We do the longest run we can of same inner_size
    split_i = next(
      i
      for i, v in enumerate(inner_size_compatible)
      if v != inner_size_compatible[0]
    )
    for sch in [
      substage_and_stage_schedule[:split_i],
      substage_and_stage_schedule[split_i:],
    ]:
      inputs = _bitonic_sort_substages_array_or_refs(
        inputs,
        substage_and_stage_schedule=sch,
        num_keys=num_keys,
        batch_size=batch_size,
        sort_dim_offset=sort_dim_offset,
        outer_size=outer_size,
        inner_size=inner_size,
        concat_threshold=inner_size,
        is_ref=is_ref,
      )
    return inputs

  grid_size = full_size // outer_size
  assert full_size % outer_size == 0

  def process_block(outer_i):
    outer_tiles = [
      input_[outer_i * outer_size : (outer_i + 1) * outer_size]
      if not is_ref
      else input_[pl.dslice(outer_i * outer_size, outer_size)]
      for input_ in inputs
    ]
    # Standardize to list-of-lists format
    outer_tiles = list(map(jax.tree.leaves, outer_tiles))

    outer_out_tiles = []
    for inner_i, inner_tiles in enumerate(
      transpose_list_of_lists(_resplit(outer_tiles, inner_size))
    ):
      tile_offset = (
        sort_dim_offset + outer_i * outer_size + inner_i * inner_size
      )
      for substage, stage in substage_and_stage_schedule:
        inner_tiles = bitonic_sort_substage(
          inner_tiles,
          substage=substage,
          stage=stage,
          num_keys=num_keys,
          batch_size=batch_size,
          sort_dim_offset=tile_offset,
          full_size=full_size,
          concat_threshold=concat_threshold,
        )
      outer_out_tiles.append([jnp.concat(x, axis=0) for x in inner_tiles])
    outer_out_tiles = transpose_list_of_lists(outer_out_tiles)

    # Write back to refs
    if is_ref:
      for ref, arr in zip(inputs, _rejoin(outer_out_tiles), strict=True):
        ref[pl.dslice(outer_i * outer_size, outer_size)] = arr
      return None
    else:
      return outer_out_tiles

  if is_ref:
    pl.loop(0, grid_size)(process_block)
  else:
    return transpose_list_of_lists([
      process_block(outer_i) for outer_i in range(grid_size)
    ])


# Partial functions for cleaner usage
_bitonic_sort_substages_arrays = functools.partial(
  _bitonic_sort_substages_array_or_refs, is_ref=False
)
_bitonic_sort_substages_refs = functools.partial(
  _bitonic_sort_substages_array_or_refs, is_ref=True
)


def _bitonic_sort_arrays(
  arrs_tiles,
  stage_unroll,
  num_stages,
  sort_dim_offset,
  slice_size,
  num_keys,
  batch_size,
):
  schedule = [
    (substage, stage)
    for stage in range(1, num_stages + 1)
    for substage in range(stage)[::-1]
  ]
  return _bitonic_sort_substages_arrays(
    arrs_tiles,
    schedule,
    num_keys=num_keys,
    batch_size=batch_size,
    sort_dim_offset=sort_dim_offset,
    inner_size=slice_size,
  )


@functools.partial(
  map_batch_dim_to_smaller_than_hardware_tile_size, max_batch_size=NUM_LANES
)
def bitonic_sort_maybe_rolled(
  operands: list[jax.Array],
  num_keys: int = 1,
  axis: int = 1,
  descending: bool = False,
  stage_unroll: int | bool = True,
  slice_size_unroll: int | bool = True,
  ref_slice_size_unroll: int | bool = True,
  transpose_refs=None,
  num_stages: int | None = None,
  single_stage: jax.Array | None = None,
  sort_dim_offset: SymInt | int | None = None,
):
  """
  Bitonic sort using compressed transpose format, , offers both rolled and
  fully unrolled implementation.

  Similar to bitonic_topk_arrays but performs full sort without reduction.
  Uses the same tiling strategy and format conversion for efficient TPU execution.

  Handles arbitrary sort dimensions efficiently:
  - Dimensions ≤ NUM_LANES (128): Uses compressed transpose format substages
  - Dimensions > NUM_LANES: Extends with cross-lane permutation substages
  - Example: (8, 2048) sorted using stage-based bitonic reduce with full tile unrolling

  Args:
      operands: List of JAX arrays of shape (dim0, dim1)
      num_keys: Number of sort keys (default: 1)
      axis: Axis along which to perform sort (0 or 1)
      descending: If True, sort in descending order
      stage_unroll: Stage unrolling control (int or bool)
          - True: fully unrolled stages (pure arrays implementation if ref_slice_size_unroll is also full)
          - False: rolled stages (uses refs)
          - int: specific stage unroll value
      slice_size_unroll: Slice size control (int or bool)
          - True: use full_size
          - False: use 0
          - int: specific slice size
      ref_slice_size_unroll: Reference slice size control (int or bool)
          - True: use full_size
          - False: use 0
          - int: specific ref slice size

  Returns:
      List of JAX arrays of same shape as input, sorted along specified axis
  """
  sort_axis = axis
  batch_axis = 1 - sort_axis
  shape = operands[0].shape

  assert shape[batch_axis] <= NUM_LANES, (
    f"Batch size {shape[batch_axis]} must be <= NUM_LANES ({NUM_LANES})"
  )
  if sort_axis == 1:
    padded_shape = _compute_padded_shape(*shape)
  elif sort_axis == 0:
    padded_shape = (max(2 ** log2(shape[0]), NUM_SUBLANES), NUM_LANES)
  else:
    raise ValueError

  # Pad both dimensions if needed
  # Always append padding after the array:
  # - For ascending sort: pad with 'max' so padding values sort to the end
  # - For descending sort: pad with 'min' so padding values sort to the end
  arrs = [
    pad(op, block_shape=padded_shape, val="min" if descending else "max")
    for op in operands
  ]
  arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]

  num_stages = log2(shape[sort_axis]) if num_stages is None else num_stages

  batch_size = arrs[0].shape[batch_axis]
  assert batch_size <= NUM_LANES
  # Convert to compressed transpose format
  arrs_tiles = jax.tree.map(
    (
      to_compressed_transpose_format if sort_axis == 1 else split_array_to_tiles
    ),
    arrs,
  )

  # full_size is dim0 of the input in compressed transpose format
  full_size = _get_full_size(arrs_tiles)

  # Standardize unrolls
  if type(slice_size_unroll) == bool:
    slice_size_unroll = log2(full_size) if slice_size_unroll else 0
  if type(ref_slice_size_unroll) == bool:
    ref_slice_size_unroll = log2(full_size) if ref_slice_size_unroll else 0
  if type(stage_unroll) == bool:
    stage_unroll = num_stages if stage_unroll else 6
  # guard values to their max/mins
  stage_unroll = min(stage_unroll, num_stages)
  slice_size = max(2**stage_unroll, 2**slice_size_unroll)
  ref_slice_size = max(slice_size, 2**ref_slice_size_unroll)
  slice_size, ref_slice_size = (
    min(max(size, NUM_SUBLANES), full_size)
    for size in (slice_size, ref_slice_size)
  )

  # Offset to control ascending vs descending final order
  if sort_dim_offset is None:
    sort_dim_offset = int(descending) * (2**num_stages)

  if shape[sort_axis] <= NUM_LANES:
    # forcibly unroll
    stage_unroll = num_stages
    slice_size = ref_slice_size = full_size

  sort_kwargs = dict(
    num_keys=num_keys,
    batch_size=batch_size,
    sort_dim_offset=sort_dim_offset,
    inner_size=slice_size,
    outer_size=ref_slice_size,
  )

  if ref_slice_size == full_size and stage_unroll == num_stages:
    # fully unrolled, pure jax array implementation. best runtime, slow to compile for large shapes
    schedule = [
      (substage, stage)
      for stage in range(1, num_stages + 1)
      for substage in range(stage)[::-1]
    ]
    if single_stage:
      # special code branch for sorting things which dont fit in HBM
      schedule = [
        (substage, single_stage) for substage in range(num_stages)[::-1]
      ]

    arrs_tiles = _bitonic_sort_substages_arrays(
      arrs_tiles, schedule, **sort_kwargs
    )
  else:
    # use the transpose refs
    assert transpose_refs is not None, (
      "transpose_refs required when not fully unrolling"
    )

    for i, arr in enumerate(_rejoin(arrs_tiles)):
      # cut transpose refs if too large
      transpose_refs[i] = transpose_refs[i].at[: arr.shape[0]]
      transpose_refs[i][...] = arr

    num_crosslane_stages = log2(NUM_LANES // batch_size)
    stage_sections = set_cummax((
      stage_unroll,
      # two sections added to allow for is_descending optimization
      # specializing for constant intra-tile from constant across tiles patterns
      num_stages - num_crosslane_stages - 1,
      num_stages,
    ))
    stage_sections = tuple(
      i + 1 for i in stage_sections
    )  # stages are 1-indexed

    schedule = [
      (substage, stage)
      for stage in range(1, stage_sections[0])
      for substage in range(stage)[::-1]
    ]

    if single_stage is not None:
      # special code branch for sorting things which dont fit in HBM
      schedule = [
        (substage, single_stage) for substage in range(num_stages)[::-1]
      ]
      stage_sections = (0,)

    _bitonic_sort_substages_refs(transpose_refs, schedule, **sort_kwargs)

    for stage_lb, stage_ub in zip(stage_sections, stage_sections[1:]):
      # run the cross tile and cross lane fori_loops separately so we can make optimizations on is_descending
      @pl.loop(stage_lb, stage_ub)
      def run_dynamic_stage(stage):
        # bounds are inclusive on both ends
        # this is used to make optimizations on is_descending inside the code
        stage = SymInt(stage, lower_bound=stage_lb, upper_bound=stage_ub - 1)

        for substage in range(stage_lb, stage_ub)[::-1]:

          @pl.when(stage > substage)
          def run_substage():
            _bitonic_sort_substages_refs(
              transpose_refs, [(substage, stage)], **sort_kwargs
            )

        _bitonic_sort_substages_refs(
          transpose_refs,
          [(substage, stage) for substage in range(stage_lb)[::-1]],
          **sort_kwargs,
        )

    # back in array flow
    arrs_tiles = [[ref[...]] for ref in transpose_refs]

  # Convert back from compressed transpose format
  if sort_axis == 1:
    arrs = [
      from_compressed_transpose_format(_rejoin(tiles), dim0=batch_size)
      for tiles in arrs_tiles
    ]
  else:
    arrs = [
      join_tiles_to_array(_rejoin(tiles), dim0=2**num_stages)
      for tiles in arrs_tiles
    ]

  # Unpad to original shape
  return [arr[: shape[0], : shape[1]] for arr in arrs]
