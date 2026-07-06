"""Shared helpers and TPU-v6 tile constants for top-k kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

NEG_INF = -jnp.inf
V6_TILE_ROWS = 16
V6_TILE_COLS = 256
V6_TILE_SIZE = V6_TILE_ROWS * V6_TILE_COLS


def next_power_of_2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def sanitize_scores(x: jax.Array) -> jax.Array:
    return jnp.where(jnp.isnan(x), NEG_INF, x)
