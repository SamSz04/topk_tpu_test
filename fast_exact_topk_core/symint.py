"""SymInt: A wrapper around JAX tracers with static bound and divisibility metadata.

This class enables static evaluation of comparisons and operations when bounds
allow concrete determination, reducing compilation overhead in Pallas kernels.
"""

import math
import jax.numpy as jnp


# Helper to unwrap SymInt to value for use in expressions
def unwrap(x):
  return x.value if isinstance(x, SymInt) else x


class SymInt:
  """Integer wrapper with lower and upper bound and divisibility metadata.

  Wraps a JAX tracer (typically from pl.loop) with compile-time known bounds
  and divisibility constraints, enabling static evaluation of comparisons and
  optimized code generation.

  Args:
      value: The actual JAX tracer or integer value
      lower_bound: Inclusive lower bound (compile-time constant)
      upper_bound: Inclusive upper bound (compile-time constant)
      multiple_of: Integer that this value is guaranteed to be a multiple of

  Example:
      >>> @pl.loop(0, 10)
      >>> def loop_body(i):
      >>>     i_bounded = SymInt(i, lower_bound=0, upper_bound=9)
      >>>     # Can statically determine: i_bounded < 100 -> True
      >>>     if i_bounded < 100:  # Returns True, no dynamic check
      >>>         ...
      >>>     # Divisibility tracking:
      >>>     j = i_bounded * 16
      >>>     # j.multiple_of == 16, so j % 16 returns concrete 0
  """

  def __init__(
    self,
    value,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
    multiple_of: int = 1,
  ):
    if jnp.ndim(value) != 0:
      raise ValueError
    self.value = value
    if type(value) == int:
      lower_bound, upper_bound, multiple_of = value, value, value
    if lower_bound is None:
      raise ValueError
    if upper_bound is None:
      raise ValueError
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.multiple_of = abs(multiple_of) if multiple_of != 0 else 1

  def __repr__(self):
    mult_str = f", ×{self.multiple_of}" if self.multiple_of > 1 else ""
    return f"SymInt({self.value}, [{self.lower_bound}, {self.upper_bound}]{mult_str})"

  # Arithmetic operations - only what's needed

  def __add__(self, other):
    if isinstance(other, int):
      return SymInt(
        self.value + other,
        self.lower_bound + other,
        self.upper_bound + other,
        math.gcd(self.multiple_of, other),
      )
    elif isinstance(other, SymInt):
      # Divisibility: gcd of the two multiples
      new_mult = math.gcd(self.multiple_of, other.multiple_of)
      return SymInt(
        self.value + other.value,
        self.lower_bound + other.lower_bound,
        self.upper_bound + other.upper_bound,
        new_mult,
      )
    return NotImplemented

  def __radd__(self, other):
    return self.__add__(other)

  def __mul__(self, other):
    if isinstance(other, int):
      # Divisibility: (x * n) is multiple of (multiple_of * n)
      new_mult = abs(self.multiple_of * other) if other != 0 else 1

      if other >= 0:
        return SymInt(
          self.value * other,
          self.lower_bound * other,
          self.upper_bound * other,
          new_mult,
        )
      else:
        # Negative multiplier reverses bounds
        return SymInt(
          self.value * other,
          self.upper_bound * other,
          self.lower_bound * other,
          new_mult,
        )
    return NotImplemented

  def __rmul__(self, other):
    return self.__mul__(other)

  def __mod__(self, other):
    if isinstance(other, int) and other > 0:
      # Check if value is a multiple of other
      if self.multiple_of % other == 0:
        return 0
      if self.lower_bound >= 0 and self.upper_bound < other:
        # mod has no effect
        return self
      # General case: could be any value in [0, other-1]
      # Result is multiple of gcd(multiple_of, other)
      new_mult = math.gcd(self.multiple_of, other)
      return SymInt(self.value % other, 0, other - 1, new_mult)
    return NotImplemented

  # Comparison operations - return concrete values when bounds allow

  def __lt__(self, other):
    if isinstance(other, int):
      if self.upper_bound < other:
        # All possible values are less than other
        return True
      elif self.lower_bound >= other:
        # All possible values are >= other
        return False
      else:
        # Dynamic check needed
        return self.value < other
    return NotImplemented

  def __le__(self, other):
    if isinstance(other, int):
      if self.upper_bound <= other:
        return True
      elif self.lower_bound > other:
        return False
      else:
        return self.value <= other
    return NotImplemented

  def __gt__(self, other):
    if isinstance(other, int):
      if self.lower_bound > other:
        return True
      elif self.upper_bound <= other:
        return False
      else:
        return self.value > other
    return NotImplemented

  def __ge__(self, other):
    if isinstance(other, int):
      if self.lower_bound >= other:
        return True
      elif self.upper_bound < other:
        return False
      else:
        return self.value >= other
    return NotImplemented

  def __int__(self):
    """Returns tracer value, used as a way to exit SymInt"""
    return self.value

  def __hash__(self):
    """Make hashable for use in dicts/sets."""
    # We can't hash the JAX tracer, so we use bounds
    return hash((self.lower_bound, self.upper_bound, self.multiple_of))
