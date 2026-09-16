#!/usr/bin/env python3
"""Is a difference between two buffers significant, and which of two arms is wrong.

This is the judging tool. It takes a difference that something else has already
measured and answers the two questions a magnitude cannot: does this difference
survive into a stored bfloat16 word at all, and when two arms disagree, which of
them does an independent reference side with.

WHICH OF THE THREE TOOLS IN THIS FOLDER TO REACH FOR
----------------------------------------------------
  compare.py              a threshold, a pass and a fail. Reach for it when you
                          already know what tolerance you accept.
  ulp_magnitudes.py       how far apart two bfloat16 buffers are, in ULPs,
                          absolutely and relatively. Issues no verdict at all,
                          on purpose.
  this module             whether a difference MATTERS, and whose fault it is.
                          It classifies, splits and arbitrates.

WHAT IT REFUSES TO DO
---------------------
It does not launch anything. There is no device call, no kernel, no accelerator
import and no lock acquisition anywhere in this file; every function operates on
whatever buffers it is handed, wherever those buffers already live. That
separation is not tidiness. An analysis module that also drives a device can
only be tested by running the device, which means the arithmetic that decides
your verdict is exercised for the first time on the machine you were trying to
make a claim about. Keeping the analysis here makes its self-test a CPU program
that runs in a second, and makes "this ran off hardware" an observation rather
than a promise -- see `_selftest`, which asserts at the end that no accelerator
was ever initialized.

It also refuses to arbitrate when the reference it was given disagrees with both
arms almost everywhere. See MIN_PLAUSIBLE_AGREEMENT below.

THE MISMATCH UNIT: ONE BFLOAT16 ULP, AND WHY
--------------------------------------------
An element counts as a mismatch when `abs(reference - arm) > ulp_bf16(reference)`.
One ULP of bfloat16 at the reference's own magnitude, because the arms STORE
bfloat16: a difference smaller than one ULP is not a disagreement the storage
format is capable of representing, so calling it one would be counting noise
that no consumer of the buffer can observe. The tolerance is exposed as a
parameter, but 1.0 is the principled value and it is derived from the storage
format rather than chosen to make a number look good.

Subnormals and zero collapse to the smallest normal ULP. That is deliberately
the conservative direction: it makes the test HARDER to trip near zero rather
than easier, so the rule cannot manufacture mismatches out of tiny values.

THE PERTURBATION SIZE: DERIVED, NOT PICKED
------------------------------------------
`delta_accum_from_sigma(sigma, k)` returns `sigma * sqrt(k) * unit_roundoff`.
This is the scale of the floating-point rounding that a k-long accumulation of
terms with spread `sigma` cannot avoid, so it is the SMALLEST perturbation that
a legitimate difference in reduction order could produce. Deriving it beats
picking one, because a threshold chosen after the numbers are visible is a
threshold chosen to reach a conclusion.

The default unit roundoff is 2**-24 and not 2**-23. 2**-24 is half a ULP, the
largest relative error a single correctly-rounded float32 operation can
introduce; 2**-23 is the gap between 1.0 and the next float32. Using the second
where the first is meant silently DOUBLES every threshold derived from it, which
is the kind of error that never announces itself.

SENSITIVITY IS MEASURED, NOT INFERRED
-------------------------------------
An element is SENSITIVE when perturbing the inputs it is computed from, by
+/- delta, changes its stored bfloat16 word. Everything else is SATURATED: the
activation and the output rounding between them destroy a perturbation of that
size before it reaches memory.

Defining this operationally rather than analytically is the single most
important piece of reasoning in this file, and it was learned the expensive way.
The tempting analytic shortcut is to test whether the pre-activation value lies
inside the clamped region -- `abs(value) < limit` -- and call those elements
sensitive. In one study that test was wrong by about 40x, for two reasons that
compound:

  * the clamp was ONE-SIDED. A large negative input was formally unclamped and
    so counted as sensitive by the analytic test, while in fact its derivative
    had already vanished because the sigmoid had underflowed.
  * output rounding kills small responses that the clamp would have let through.
    An element can be well inside the live region and still not move a bfloat16
    word.

Perturbing and looking costs 2**n evaluations for n input buffers. It is worth
it: the analytic test is a statement about where the function is live, and the
question is whether a difference reaches memory, which is not the same question.

DEFAULTS THAT CAME FROM ONE STUDY AND ARE NOT UNIVERSAL
-------------------------------------------------------
`MIN_PLAUSIBLE_AGREEMENT = 0.5` and the quantile points are the values used by
the investigation this module was extracted from. They are defaults so that the
extracted behaviour is reproducible, not because 0.5 is a law. Every one of them
is a keyword argument; if your buffers are not like that study's, pass your own.
The unit roundoff and the bfloat16 format constants are different in kind --
those follow from IEEE-754 and the storage format, not from anyone's study.

BACKENDS
--------
Everything here is written once, against a very small array interface, and runs
on torch tensors, on numpy arrays, or on plain Python sequences. The last of
those exists so that the self-test can run on a host with neither library
installed -- an analysis module whose correctness can only be checked where
torch happens to be installed is back to being untestable for the wrong reason.

Elementwise arithmetic is performed in float32 on every backend, including the
pure-Python one, which rounds each operation to float32 explicitly. The one
quantity that is NOT guaranteed bit-identical across backends is the quantile
interpolation, because each library computes it in its own way; every count,
fraction, maximum, mask and verdict is exact and backend-independent.
"""

from __future__ import annotations

import itertools
import math
import struct
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# bfloat16, from the format: 1 sign bit, 8 exponent bits, 7 explicit mantissa bits.
BF16_MANTISSA_BITS = 7
BF16_MIN_NORMAL_EXPONENT = -126

# The float32 UNIT ROUNDOFF: half a ULP, the largest relative error a single
# correctly-rounded float32 operation can introduce. Not 2**-23, which is the
# gap between 1.0 and its successor and would double every derived threshold.
FP32_UNIT_ROUNDOFF = 2.0**-24

# One bfloat16 ULP at the reference's magnitude. A difference below this is not
# representable in the format the arms store, so it is not a disagreement.
DEFAULT_ULP_TOLERANCE = 1.0

# Below this whole-buffer agreement between the reference and BOTH arms, the
# likeliest explanation is that the reference is pointed at the wrong elements --
# an indexing or permutation error makes a reference disagree with both arms
# everywhere -- and the first hypothesis must then be that the reference is
# wrong, not that both kernels are. A verdict is refused rather than reported.
# 0.5 is the value the originating study used; it is not a universal constant.
DEFAULT_MIN_PLAUSIBLE_AGREEMENT = 0.5

# Report the whole distribution rather than the flattering end of it.
DEFAULT_QUANTILE_POINTS: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 1.0)

CLASSIFICATION_RULE = (
    "operational: an element is SENSITIVE iff perturbing every input it is "
    "computed from by +/-delta changes its stored bfloat16 word; every other "
    "element is SATURATED. The analytic |value| < limit test is not used, "
    "because a one-sided clamp plus output rounding made that test wrong by "
    "about 40x in the study this rule came from."
)


class ReferenceImplausible(RuntimeError):
    """The reference disagrees with both arms so widely that it arbitrates nothing."""


# ---------------------------------------------------------------------------
# Backends.
#
# The algorithms below are written ONCE, against the handful of operations named
# here. torch and numpy spell almost all of them identically as module-level
# functions, so both share one adapter shape; the pure-Python backend implements
# the same names over a minimal array type.
# ---------------------------------------------------------------------------


def _to_float32(value: float) -> float:
    """Round a Python float to float32, saturating to infinity on overflow.

    `struct` raises rather than saturating, but IEEE-754 round-to-nearest turns
    an out-of-range magnitude into an infinity, so that is what is returned.
    """
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _bf16_round_bits(bits: int) -> int:
    """float32 bit pattern to the float32 bit pattern of its bfloat16 rounding.

    Round-to-nearest-even on the 16 bits being discarded, which is the rounding
    every float32-to-bfloat16 conversion in common use performs. NaN is handled
    by the callers, because the carry out of a saturated mantissa can turn a
    NaN payload into a finite number.
    """
    bias = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + bias) >> 16) << 16


class _Vec:
    """A flat float32-or-bool buffer with just enough array behaviour.

    This is not a general array library and is not trying to be one. It exists
    so the pure-Python backend can execute the same code the torch and numpy
    backends execute, which is what lets the self-test run where neither library
    is installed. Every float operation rounds its result to float32, so the
    arithmetic matches the other backends rather than silently running wider.
    """

    __slots__ = ("data", "shape", "is_bool")

    def __init__(self, data: list, shape: tuple[int, ...], is_bool: bool = False):
        self.data = data
        self.shape = shape
        self.is_bool = is_bool

    # -- construction -------------------------------------------------------

    @classmethod
    def from_nested(cls, values: Any) -> "_Vec":
        flat: list[float] = []
        shape = _nested_shape(values)
        _flatten_into(values, flat)
        return cls([_to_float32(float(v)) for v in flat], shape)

    def _float(self, data: list[float]) -> "_Vec":
        return _Vec(data, self.shape, False)

    def _mask(self, data: list[bool]) -> "_Vec":
        return _Vec(data, self.shape, True)

    # -- arithmetic, each operation rounded to float32 ----------------------

    def _pair(self, other: Any):
        if isinstance(other, _Vec):
            if len(other.data) != len(self.data):
                raise ValueError(
                    f"length mismatch: {len(self.data)} against {len(other.data)}"
                )
            return other.data
        return [other] * len(self.data)

    def __add__(self, other):
        rhs = self._pair(other)
        return self._float([_to_float32(a + b) for a, b in zip(self.data, rhs)])

    def __sub__(self, other):
        rhs = self._pair(other)
        return self._float([_to_float32(a - b) for a, b in zip(self.data, rhs)])

    def __mul__(self, other):
        rhs = self._pair(other)
        return self._float([_to_float32(a * b) for a, b in zip(self.data, rhs)])

    __rmul__ = __mul__

    def __truediv__(self, other):
        rhs = self._pair(other)
        return self._float([_to_float32(a / b) for a, b in zip(self.data, rhs)])

    def __neg__(self):
        return self._float([-a for a in self.data])

    def __abs__(self):
        return self._float([abs(a) for a in self.data])

    # -- comparison ---------------------------------------------------------

    def __gt__(self, other):
        rhs = self._pair(other)
        return self._mask([a > b for a, b in zip(self.data, rhs)])

    def __le__(self, other):
        rhs = self._pair(other)
        return self._mask([a <= b for a, b in zip(self.data, rhs)])

    def __ne__(self, other):  # type: ignore[override]
        rhs = self._pair(other)
        return self._mask([a != b for a, b in zip(self.data, rhs)])

    def __eq__(self, other):  # type: ignore[override]
        rhs = self._pair(other)
        return self._mask([a == b for a, b in zip(self.data, rhs)])

    __hash__ = None  # type: ignore[assignment]

    # -- boolean ------------------------------------------------------------

    def __and__(self, other):
        rhs = self._pair(other)
        return self._mask([bool(a) and bool(b) for a, b in zip(self.data, rhs)])

    def __or__(self, other):
        rhs = self._pair(other)
        return self._mask([bool(a) or bool(b) for a, b in zip(self.data, rhs)])

    def __invert__(self):
        return self._mask([not bool(a) for a in self.data])

    # -- reduction and selection -------------------------------------------

    def __getitem__(self, mask: "_Vec") -> "_Vec":
        chosen = [v for v, keep in zip(self.data, mask.data) if keep]
        return _Vec(chosen, (len(chosen),), self.is_bool)

    def sum(self):
        return sum(1 for v in self.data if v) if self.is_bool else sum(self.data)

    def max(self):
        # NaN propagates, because torch's and numpy's `.max()` propagate it and
        # this backend must not report a quietly more flattering maximum than
        # the other two. Python's builtin `max` would step over a NaN instead.
        best = None
        for value in self.data:
            if value != value:
                return value
            if best is None or value > best:
                best = value
        return best

    def __len__(self):
        return len(self.data)


def _nested_shape(values: Any) -> tuple[int, ...]:
    shape: list[int] = []
    node = values
    while isinstance(node, (list, tuple)):
        shape.append(len(node))
        if not node:
            break
        node = node[0]
    return tuple(shape)


def _flatten_into(values: Any, out: list) -> None:
    if isinstance(values, (list, tuple)):
        for item in values:
            _flatten_into(item, out)
    else:
        out.append(values)


class _PythonNamespace:
    """The module-level functions the algorithms call, over `_Vec`."""

    @staticmethod
    def abs(x: _Vec) -> _Vec:
        return abs(x)

    @staticmethod
    def floor(x: _Vec) -> _Vec:
        # NaN and infinity pass through, as they do in torch and numpy.
        # `math.floor` raises on both, which would turn a NaN element into a
        # crash on this backend alone.
        return _Vec(
            [v if not math.isfinite(v) else float(math.floor(v)) for v in x.data],
            x.shape,
        )

    @staticmethod
    def log2(x: _Vec) -> _Vec:
        return _Vec(
            [v if not math.isfinite(v) else _to_float32(math.log2(v)) for v in x.data],
            x.shape,
        )

    @staticmethod
    def exp2(x: _Vec) -> _Vec:
        return _Vec([_to_float32(2.0**v) for v in x.data], x.shape)

    @staticmethod
    def clip(x: _Vec, low, high) -> _Vec:
        data = []
        for v in x.data:
            if low is not None and v < low:
                v = low
            if high is not None and v > high:
                v = high
            data.append(v)
        return _Vec(data, x.shape, x.is_bool)

    @staticmethod
    def isfinite(x: _Vec) -> _Vec:
        return _Vec([math.isfinite(v) for v in x.data], x.shape, True)

    @staticmethod
    def quantile(x: _Vec, q: float) -> float:
        return _linear_quantile(sorted(x.data), q)


def _linear_quantile(ordered: Sequence[float], q: float) -> float:
    """The linear-interpolation quantile, the default definition in both libraries.

    Position `q * (n - 1)` in the sorted values, interpolating linearly between
    the two neighbouring samples. Stated explicitly because "the 99th
    percentile" names several different numbers depending on the convention.
    """
    if not ordered:
        raise ValueError("no values to take a quantile of")
    if len(ordered) == 1:
        return _to_float32(ordered[0])
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return _to_float32(ordered[low])
    fraction = position - low
    return _to_float32(ordered[low] + (ordered[high] - ordered[low]) * fraction)


class _Backend:
    """The operations that genuinely differ between torch, numpy and plain Python."""

    name = "unknown"
    xp: Any = None
    tiny = float(struct.unpack("<f", struct.pack("<I", 0x00800000))[0])

    def as_f32(self, x):
        raise NotImplementedError

    def numel(self, x) -> int:
        raise NotImplementedError

    def shape(self, x) -> tuple:
        raise NotImplementedError

    def round_to_bf16(self, x):
        """Round to bfloat16 and widen back to float32.

        Widening back loses nothing -- every bfloat16 value is exactly a float32
        value -- and it keeps one dtype flowing through the comparisons, so the
        three backends agree on what is being compared. Comparison stays by
        VALUE and not by bit pattern, because NaN must compare unequal to NaN;
        an element whose output is NaN under perturbation has certainly moved.
        """
        raise NotImplementedError

    def bool_zeros_like(self, x):
        raise NotImplementedError


class _TorchBackend(_Backend):
    name = "torch"

    def __init__(self, module):
        self.xp = module
        self.tiny = float(module.finfo(module.float32).tiny)

    def as_f32(self, x):
        return x.to(self.xp.float32)

    def numel(self, x) -> int:
        return int(x.numel())

    def shape(self, x) -> tuple:
        return tuple(x.shape)

    def round_to_bf16(self, x):
        return self.as_f32(self.as_f32(x).to(self.xp.bfloat16))

    def bool_zeros_like(self, x):
        return self.xp.zeros_like(x, dtype=self.xp.bool)


class _NumpyBackend(_Backend):
    name = "numpy"

    def __init__(self, module):
        self.xp = module
        self.tiny = float(module.finfo(module.float32).tiny)

    def as_f32(self, x):
        return self.xp.asarray(x).astype(self.xp.float32)

    def numel(self, x) -> int:
        return int(self.xp.asarray(x).size)

    def shape(self, x) -> tuple:
        return tuple(self.xp.asarray(x).shape)

    def round_to_bf16(self, x):
        numpy = self.xp
        wide = self.as_f32(x)
        bits = wide.view(numpy.uint32)
        bias = ((bits >> numpy.uint32(16)) & numpy.uint32(1)) + numpy.uint32(0x7FFF)
        rounded = ((bits + bias) >> numpy.uint32(16)) << numpy.uint32(16)
        # A NaN whose mantissa is all ones carries out of 32 bits and comes back
        # as a small finite number, so NaN is passed through untouched instead.
        return numpy.where(numpy.isnan(wide), wide, rounded.view(numpy.float32))

    def bool_zeros_like(self, x):
        return self.xp.zeros_like(self.as_f32(x), dtype=bool)


class _PurePythonBackend(_Backend):
    name = "python"

    def __init__(self):
        self.xp = _PythonNamespace()

    def as_f32(self, x):
        return x if isinstance(x, _Vec) else _Vec.from_nested(x)

    def numel(self, x) -> int:
        return len(self.as_f32(x))

    def shape(self, x) -> tuple:
        return self.as_f32(x).shape

    def round_to_bf16(self, x):
        wide = self.as_f32(x)
        data = []
        for value in wide.data:
            if math.isnan(value):
                data.append(value)
                continue
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            data.append(struct.unpack("<f", struct.pack("<I", _bf16_round_bits(bits)))[0])
        return _Vec(data, wide.shape)

    def bool_zeros_like(self, x):
        wide = self.as_f32(x)
        return _Vec([False] * len(wide), wide.shape, True)


def backend_for(*buffers) -> _Backend:
    """Pick the backend from what was actually handed in.

    Dispatch is by module name rather than by importing torch or numpy to test
    against them, so that asking this module to compare two Python lists does
    not drag a deep-learning framework into the process.
    """
    for buffer in buffers:
        module = type(buffer).__module__.split(".")[0]
        if module == "torch":
            return _TorchBackend(sys.modules["torch"])
        if module == "numpy":
            return _NumpyBackend(sys.modules["numpy"])
    return _PurePythonBackend()


# ---------------------------------------------------------------------------
# The bfloat16 grid.
# ---------------------------------------------------------------------------


def ulp_bf16(values, *, backend: _Backend | None = None):
    """The spacing between adjacent bfloat16 values at each element's magnitude.

    Subnormals and zero collapse to the smallest normal ULP, which is the
    conservative direction: it makes the mismatch test HARDER to trip near zero
    rather than easier, so it cannot manufacture mismatches.

    The exponent is taken as `floor(log2(|x|))`, which relies on log2 being
    exact at exact powers of two -- if it were to return 1.9999999 for 4.0 the
    ULP would come out a factor of two small. Every implementation in common use
    is exact there, and the self-test checks the powers of two directly.

    A NaN gets a NaN ULP rather than a fabricated one, which is why a NaN
    reference element never registers as a mismatch downstream.
    """
    b = backend or backend_for(values)
    xp = b.xp
    magnitude = xp.abs(b.as_f32(values))
    exponent = xp.floor(xp.log2(xp.clip(magnitude, b.tiny, None)))
    exponent = xp.clip(exponent, float(BF16_MIN_NORMAL_EXPONENT), None)
    return xp.exp2(exponent - float(BF16_MANTISSA_BITS))


def mismatch_mask(
    reference,
    arm,
    *,
    ulp_tolerance: float = DEFAULT_ULP_TOLERANCE,
    backend: _Backend | None = None,
):
    """Elements where the arm differs from the reference by more than the tolerance.

    `abs(reference - arm) > ulp_tolerance * ulp_bf16(reference)`. The default
    tolerance of one ULP is the principled one and comes from the storage
    format, not from a study; raising it is a deliberate statement that you
    accept differences the format CAN represent.
    """
    b = backend or backend_for(reference, arm)
    if b.shape(reference) != b.shape(arm):
        raise ValueError(
            f"shape mismatch: reference {b.shape(reference)} vs arm {b.shape(arm)}"
        )
    wide = b.as_f32(reference)
    absolute = b.xp.abs(wide - b.as_f32(arm))
    threshold = ulp_bf16(wide, backend=b) * float(ulp_tolerance)
    return absolute > threshold


# ---------------------------------------------------------------------------
# Magnitude.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MagnitudeStats:
    """Every declared quantity together, rather than the flattering one.

    A maximum without a distribution behind it invites "the worst element is
    huge" and "almost every element is fine" to be argued from the same data,
    so both are always present.
    """

    count: int
    max_abs_diff: float
    max_rel_diff: float
    mismatch_count: int
    mismatch_fraction: float
    quantiles_abs_diff: dict[str, float]
    ulp_tolerance: float


def magnitude_stats(
    reference,
    arm,
    *,
    ulp_tolerance: float = DEFAULT_ULP_TOLERANCE,
    quantile_points: Sequence[float] = DEFAULT_QUANTILE_POINTS,
    backend: _Backend | None = None,
) -> MagnitudeStats:
    """Compare one arm against the reference under the ULP mismatch rule.

    Quantiles are taken over the FINITE absolute differences only, so that one
    NaN cannot present itself as the shape of the distribution. They cover every
    element, not only the mismatching ones: a mismatch fraction is only readable
    beside the spread of the differences that did not mismatch.

    The MAXIMA do not do that -- a NaN anywhere makes `max_abs_diff` and
    `max_rel_diff` come back NaN. That is worth knowing because the neighbouring
    `ulp_magnitudes.py` makes the opposite choice and takes its maxima over
    finite positions only. Here a NaN is loud, and a NaN maximum beside a finite
    quantile set is a reliable signal that the buffer contains one.

    A NaN on the REFERENCE side is never counted as a mismatch, whichever way
    it goes: its ULP threshold is NaN, its absolute difference is NaN, and NaN
    is greater than neither. Mismatches are undercounted rather than invented
    when a reference contains NaN, so check `max_abs_diff` before reading the
    mismatch count.

    On the torch backend the quantile call has an input-size limit of roughly 16
    million elements; numpy and the pure-Python backend have none. If you hit it,
    pass `quantile_points=()`.
    """
    b = backend or backend_for(reference, arm)
    xp = b.xp
    if b.shape(reference) != b.shape(arm):
        raise ValueError(
            f"shape mismatch: reference {b.shape(reference)} vs arm {b.shape(arm)}"
        )
    wide = b.as_f32(reference)
    absolute = xp.abs(wide - b.as_f32(arm))
    mismatching = absolute > (ulp_bf16(wide, backend=b) * float(ulp_tolerance))
    denominator = xp.clip(xp.abs(wide), b.tiny, None)
    count = b.numel(absolute)

    finite = absolute[xp.isfinite(absolute)]
    quantiles = {
        f"q{point:g}": (float(xp.quantile(finite, point)) if b.numel(finite) else 0.0)
        for point in quantile_points
    }
    return MagnitudeStats(
        count=count,
        max_abs_diff=float(absolute.max()) if count else 0.0,
        max_rel_diff=float((absolute / denominator).max()) if count else 0.0,
        mismatch_count=int(mismatching.sum()),
        mismatch_fraction=float(mismatching.sum()) / max(count, 1),
        quantiles_abs_diff=quantiles,
        ulp_tolerance=float(ulp_tolerance),
    )


# ---------------------------------------------------------------------------
# Sensitivity.
# ---------------------------------------------------------------------------


def delta_accum_from_sigma(
    sigma: float,
    k: int,
    *,
    unit_roundoff: float = FP32_UNIT_ROUNDOFF,
) -> float:
    """The perturbation size for a k-long accumulation: sigma * sqrt(k) * unit_roundoff.

    Derived rather than chosen. It is the scale of the floating-point rounding
    that a k-long accumulation of terms with spread `sigma` cannot avoid, so it
    is the smallest perturbation a legitimate difference in reduction ORDER
    could produce. Anything smaller would classify rounding noise as signal;
    anything larger would ask whether the kernel survives a perturbation it was
    never going to receive.

    `unit_roundoff` defaults to the float32 value, 2**-24. Pass 2**-53 for a
    float64 accumulator. Do not pass 2**-23 for float32 by reaching for "machine
    epsilon": that is the gap between 1.0 and its successor, twice the largest
    error a single rounded operation makes, and it doubles every threshold
    derived from it without saying so.
    """
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError(f"sigma must be finite and positive, got {sigma}")
    if k <= 0:
        raise ValueError(f"the accumulation length k must be positive, got {k}")
    if unit_roundoff <= 0.0 or not math.isfinite(unit_roundoff):
        raise ValueError(f"unit_roundoff must be finite and positive, got {unit_roundoff}")
    return sigma * math.sqrt(k) * unit_roundoff


def classify_sensitivity(
    inputs: Sequence[Any],
    recompute: Callable[..., Any],
    delta: float,
    *,
    backend: _Backend | None = None,
):
    """Mark the elements where a perturbation of size `delta` reaches a stored word.

    `inputs` are the buffers the output is computed FROM -- the accumulators, the
    gate and linear branches, whatever feeds the final activation. `recompute`
    takes exactly those buffers, in that order, and returns the output before it
    is rounded for storage; this function does the bfloat16 rounding itself, so
    the caller cannot accidentally compare at a wider precision than the one the
    result actually lands in.

    An element is SENSITIVE if ANY of the 2**len(inputs) corner perturbations --
    every input offset by -delta or +delta, in every combination -- changes its
    stored bfloat16 word. Corners only, and never a zero offset: the question is
    whether a perturbation of this SIZE is observable, and the corners bound
    that. Cost is 2**len(inputs) calls to `recompute`.

    Why measured and not inferred: the analytic alternative, testing whether the
    value lies in the clamped region, was wrong by about 40x in the study this
    came from, because a one-sided clamp leaves a large negative input formally
    unclamped while its derivative has already vanished, and because output
    rounding destroys small responses the clamp would have let through.
    """
    if not inputs:
        raise ValueError("classify_sensitivity needs at least one input buffer")
    if delta <= 0.0 or not math.isfinite(delta):
        raise ValueError(f"delta must be finite and positive, got {delta}")
    b = backend or backend_for(*inputs)
    base = b.round_to_bf16(recompute(*inputs))
    observable = b.bool_zeros_like(base)
    for signs in itertools.product((-delta, delta), repeat=len(inputs)):
        perturbed = [
            b.as_f32(buffer) + float(step) for buffer, step in zip(inputs, signs)
        ]
        probe = b.round_to_bf16(recompute(*perturbed))
        observable = observable | (probe != base)
    return observable


@dataclass(frozen=True)
class SensitivitySplit:
    """Every mismatching element classified, by measurement rather than inference."""

    mismatch_count: int
    sensitive_count: int
    saturated_count: int
    sensitive_fraction_of_mismatches: float
    sensitive_population_count: int
    sensitive_population_fraction: float
    delta_measured: float
    delta_reference: float | None
    input_sigma: float | None
    classification_rule: str


def sensitivity_split(
    observable,
    mismatching,
    *,
    delta_measured: float,
    input_sigma: float | None = None,
    delta_reference: float | None = None,
    backend: _Backend | None = None,
) -> SensitivitySplit:
    """Cross the measured sensitivity mask with the mismatch mask.

    The sensitive POPULATION is reported beside the sensitive mismatch count on
    purpose. "Forty mismatches were sensitive" means one thing if a tenth of the
    buffer is sensitive and something else entirely if a thousandth is, and
    neither figure stands in for the other.

    `delta_reference` is an optional second estimate of the same perturbation
    size, obtained independently -- typically computed off-hardware from a
    matched distribution before the real measurement exists. It is recorded
    BESIDE the measured value and must never be substituted for it; it defaults
    to None precisely so that a number from somebody else's run cannot arrive in
    your record by default.
    """
    b = backend or backend_for(observable, mismatching)
    if b.shape(observable) != b.shape(mismatching):
        raise ValueError(
            f"mask shape mismatch: observable {b.shape(observable)} vs "
            f"mismatching {b.shape(mismatching)}"
        )
    mismatch_count = int(mismatching.sum())
    sensitive = int((mismatching & observable).sum())
    population = int(observable.sum())
    return SensitivitySplit(
        mismatch_count=mismatch_count,
        sensitive_count=sensitive,
        saturated_count=mismatch_count - sensitive,
        sensitive_fraction_of_mismatches=sensitive / max(mismatch_count, 1),
        sensitive_population_count=population,
        sensitive_population_fraction=population / max(b.numel(observable), 1),
        delta_measured=float(delta_measured),
        delta_reference=None if delta_reference is None else float(delta_reference),
        input_sigma=None if input_sigma is None else float(input_sigma),
        classification_rule=CLASSIFICATION_RULE,
    )


# ---------------------------------------------------------------------------
# Arbitration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arbitration:
    """When two arms disagree, which one does an independent reference side with?

    The four counts are exhaustive and disjoint, which is the point: "the
    reference disagrees with exactly one arm" is only meaningful if the elements
    where it disagrees with both, or with neither, are on the page beside it.
    """

    arms_differ_count: int
    agrees_with_left_only: int
    agrees_with_right_only: int
    agrees_with_both: int
    agrees_with_neither: int
    whole_buffer_agreement_left: float
    whole_buffer_agreement_right: float
    min_plausible_agreement: float
    verdict: str


def _agreement(reference, arm, ulp_tolerance: float, b: _Backend):
    wide = b.as_f32(reference)
    absolute = b.xp.abs(wide - b.as_f32(arm))
    return absolute <= (ulp_bf16(wide, backend=b) * float(ulp_tolerance))


def arbitrate(
    reference,
    left,
    right,
    *,
    ulp_tolerance: float = DEFAULT_ULP_TOLERANCE,
    min_plausible_agreement: float = DEFAULT_MIN_PLAUSIBLE_AGREEMENT,
    backend: _Backend | None = None,
) -> Arbitration:
    """Attribute each disagreement between two arms to one of them, or refuse to.

    Raises `ReferenceImplausible` when the reference's whole-buffer agreement
    with BOTH arms is below `min_plausible_agreement`. That is not a defensive
    formality. An indexing or permutation error in the reference -- reading the
    wrong rows, a transposed layout, an off-by-one in a gather -- produces
    exactly the signature "the reference disagrees with both arms almost
    everywhere", and the first hypothesis for that signature must be that the
    reference is wrong, not that two independent kernels are both wrong in
    different ways. Returning a confident verdict there would be reporting an
    artefact of your own indexing as a finding about someone's kernel.

    The 0.5 default came from one study. If your reference is legitimately
    lower-precision than the arms, raise or lower it deliberately.
    """
    b = backend or backend_for(reference, left, right)
    for name, arm in (("left", left), ("right", right)):
        if b.shape(reference) != b.shape(arm):
            raise ValueError(
                f"shape mismatch: reference {b.shape(reference)} vs {name} {b.shape(arm)}"
            )
    agrees_left = _agreement(reference, left, ulp_tolerance, b)
    agrees_right = _agreement(reference, right, ulp_tolerance, b)
    elements = max(b.numel(agrees_left), 1)
    whole_left = float(agrees_left.sum()) / elements
    whole_right = float(agrees_right.sum()) / elements
    if max(whole_left, whole_right) < min_plausible_agreement:
        raise ReferenceImplausible(
            f"whole-buffer agreement is {whole_left:.4f}/{whole_right:.4f}, below "
            f"{min_plausible_agreement}; the reference arbitrates nothing. The first "
            "hypothesis for disagreeing with both arms everywhere is that the "
            "reference is indexed wrong."
        )
    differ = b.as_f32(left) != b.as_f32(right)
    only_left = int((differ & agrees_left & ~agrees_right).sum())
    only_right = int((differ & agrees_right & ~agrees_left).sum())
    both = int((differ & agrees_left & agrees_right).sum())
    neither = int((differ & ~agrees_left & ~agrees_right).sum())
    differ_count = int(differ.sum())
    return Arbitration(
        arms_differ_count=differ_count,
        agrees_with_left_only=only_left,
        agrees_with_right_only=only_right,
        agrees_with_both=both,
        agrees_with_neither=neither,
        whole_buffer_agreement_left=whole_left,
        whole_buffer_agreement_right=whole_right,
        min_plausible_agreement=float(min_plausible_agreement),
        verdict=_verdict(only_left, only_right, differ_count),
    )


def _verdict(only_left: int, only_right: int, differ: int) -> str:
    """Name the outcome without softening it. Ties and empties are not verdicts."""
    if differ == 0:
        return "NO_DISAGREEMENT_BETWEEN_THE_ARMS_TO_ARBITRATE"
    if only_left == 0 and only_right == 0:
        return "REFERENCE_ATTRIBUTES_NO_DISAGREEMENT_TO_EITHER_ARM"
    if only_right > only_left:
        return "REFERENCE_SIDES_WITH_RIGHT_AGAINST_LEFT"
    if only_left > only_right:
        return "REFERENCE_SIDES_WITH_LEFT_AGAINST_RIGHT"
    return "REFERENCE_SPLITS_EVENLY_BETWEEN_THE_ARMS"


# ---------------------------------------------------------------------------
# Self-test.
#
# Runs on plain Python lists, so it needs no torch and no numpy. It drives the
# real functions, not a replica of them, and the last assertion is that no
# accelerator was initialized -- which makes "this ran off hardware" an
# observation rather than a promise.
# ---------------------------------------------------------------------------

PASSES: list[str] = []


def _ok(message: str) -> None:
    PASSES.append(message)
    print(f"  ok   {message}")


def _skip(message: str) -> None:
    print(f"  skip {message}")


def _check_ulp() -> None:
    values = [1.0, 2.0, 256.0, -3.0]
    expected = [2.0**-7, 2.0**-6, 2.0**1, 2.0**-6]
    observed = ulp_bf16(values).data
    if observed != expected:
        raise AssertionError(f"bf16 ULP is wrong: {observed} != {expected}")
    # Zero and a subnormal collapse to the smallest normal ULP, not to zero.
    small = ulp_bf16([0.0, 5e-40]).data
    if small != [2.0 ** (BF16_MIN_NORMAL_EXPONENT - BF16_MANTISSA_BITS)] * 2:
        raise AssertionError(f"ULP near zero collapsed wrongly: {small}")
    # A sub-ULP difference is not a mismatch; one just above it is.
    below = mismatch_mask([1.0], [1.0 + 2.0**-8]).data
    above = mismatch_mask([1.0], [1.0 + 2.0**-6]).data
    if below != [False] or above != [True]:
        raise AssertionError(f"the ULP mismatch rule is wrong: {below} {above}")
    _ok("bf16 ULP is exact, collapses safely at zero, and a sub-ULP diff is no mismatch")


def _check_delta_derivation() -> None:
    sigma, k = 209.28, 5120
    expected = sigma * math.sqrt(k) * 2.0**-24
    if delta_accum_from_sigma(sigma, k) != expected:
        raise AssertionError("delta derivation does not match sigma*sqrt(k)*2**-24")
    # The unit roundoff is a parameter, and 2**-23 really does double it. This is
    # the error the default exists to prevent, so it is checked rather than
    # asserted in a comment.
    doubled = delta_accum_from_sigma(sigma, k, unit_roundoff=2.0**-23)
    if doubled != 2.0 * expected:
        raise AssertionError("the unit roundoff parameter is not doing what it says")
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            delta_accum_from_sigma(bad, k)
        except ValueError:
            continue
        raise AssertionError(f"delta derivation accepted sigma={bad}")
    try:
        delta_accum_from_sigma(sigma, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("delta derivation accepted k=0")
    _ok("delta derivation is sigma*sqrt(k)*unit_roundoff and rejects bad arguments")


def _clamped_product(limit: float):
    """A stand-in for a clamped gated activation.

    The gate branch is clamped ABOVE only and the linear branch on both sides,
    which is the shape of the activation whose one-sidedness broke the analytic
    sensitivity test. An input far outside its clamp contributes nothing when
    perturbed, which is what makes a saturated population exist at all.
    """

    def recompute(gate, up):
        return _PythonNamespace.clip(gate, None, limit) * _PythonNamespace.clip(
            up, -limit, limit
        )

    return recompute


def _fixture():
    """A buffer that contains BOTH populations, or the checks below are vacuous."""
    limit = 1.0
    # The first four are inside both clamps and respond to a perturbation. The
    # last four are pinned by both clamps and cannot.
    gate = _Vec.from_nested([0.10, 0.20, 0.30, 0.40, 900.0, 900.0, 900.0, 900.0])
    up = _Vec.from_nested([0.50, 0.60, 0.70, 0.80, 900.0, 900.0, 900.0, 900.0])
    recompute = _clamped_product(limit)
    return gate, up, recompute


def _check_sensitivity_has_both_populations(gate, up, recompute, delta):
    observable = classify_sensitivity([gate, up], recompute, delta)
    sensitive = int(observable.sum())
    total = len(observable)
    if sensitive == 0:
        raise AssertionError("no sensitive elements in the fixture; the checks are vacuous")
    if sensitive == total:
        raise AssertionError(
            "every element is sensitive in the fixture; the saturated control cannot fail"
        )
    _ok(f"fixture carries both populations: {sensitive} sensitive of {total}")
    return observable


def _check_sensitivity_is_monotone_in_delta(gate, up, recompute, delta) -> None:
    """A bigger perturbation cannot make an observable element unobservable."""
    small = classify_sensitivity([gate, up], recompute, delta)
    large = classify_sensitivity([gate, up], recompute, delta * 100.0)
    if int(small.sum()) == 0:
        raise AssertionError(
            "nothing is observable at the smaller delta, so monotonicity is vacuous"
        )
    if int((small & ~large).sum()) != 0:
        raise AssertionError(
            "sensitivity is not monotone in delta: an element observable at the "
            "smaller delta is not observable at a larger one"
        )
    _ok("sensitivity is monotone in delta")


def _plant(reference: _Vec, where: _Vec, magnitude: float) -> _Vec:
    """A defective arm carrying a planted difference only where `where` says."""
    return _Vec(
        [
            _to_float32(value + magnitude) if flag else value
            for value, flag in zip(reference.data, where.data)
        ],
        reference.shape,
    )


def _check_sensitive_band(reference, observable, delta) -> None:
    left = _plant(reference, observable, 1.0)
    right = _Vec(list(reference.data), reference.shape)
    stats = magnitude_stats(reference, left)
    if stats.mismatch_count == 0:
        raise AssertionError("a planted sensitive-band difference produced no mismatch")
    if stats.max_abs_diff <= 0.0:
        raise AssertionError(f"the planted difference has no magnitude: {stats}")
    split = sensitivity_split(
        observable, mismatch_mask(reference, left), delta_measured=delta
    )
    if split.saturated_count != 0:
        raise AssertionError(
            f"{split.saturated_count} sensitive-band mismatches were classified saturated"
        )
    if split.sensitive_fraction_of_mismatches != 1.0:
        raise AssertionError(f"the sensitive fraction is wrong: {split}")
    ruling = arbitrate(reference, left, right)
    if ruling.verdict != "REFERENCE_SIDES_WITH_RIGHT_AGAINST_LEFT":
        raise AssertionError(f"arbitration named the wrong arm: {ruling.verdict}")
    if ruling.agrees_with_left_only != 0:
        raise AssertionError("arbitration credited the defective arm")
    if (
        ruling.agrees_with_left_only
        + ruling.agrees_with_right_only
        + ruling.agrees_with_both
        + ruling.agrees_with_neither
        != ruling.arms_differ_count
    ):
        raise AssertionError(f"the four arbitration counts do not partition: {ruling}")
    _ok("planted sensitive-band difference is seen, classified sensitive, attributed right")


def _check_saturated_band(observable, delta) -> None:
    """The control that can fail.

    If everything came back sensitive the split would be decoration, so a check
    that only ever passes is not a check. Elements outside the sensitive
    population must be counted saturated even when they are mismatching.
    """
    saturated = ~observable
    if int(saturated.sum()) == 0:
        raise AssertionError("no saturated elements in the fixture; the control is vacuous")
    split = sensitivity_split(observable, saturated, delta_measured=delta)
    if split.sensitive_count != 0:
        raise AssertionError(
            f"{split.sensitive_count} saturated-band elements were classified sensitive"
        )
    if split.saturated_count != int(saturated.sum()):
        raise AssertionError("the saturated count does not account for every element")
    if split.delta_reference is not None:
        raise AssertionError("a delta_reference appeared without the caller supplying one")
    _ok("saturated-band elements are classified saturated, not sensitive")


def _check_implausible_reference_refuses(reference) -> None:
    permuted = _Vec(list(reversed(reference.data)), reference.shape)
    try:
        arbitrate(permuted, reference, reference)
    except ReferenceImplausible as error:
        if "indexed wrong" not in str(error):
            raise AssertionError(f"refused without naming the likely cause: {error}")
        _ok("a permuted reference refuses to arbitrate instead of naming an arm")
        return
    raise AssertionError("a permuted reference produced a confident verdict")


def _check_verdicts_do_not_soften() -> None:
    """Ties and empties are named as such rather than resolved."""
    if _verdict(0, 0, 0) != "NO_DISAGREEMENT_BETWEEN_THE_ARMS_TO_ARBITRATE":
        raise AssertionError("an empty comparison produced a verdict")
    if _verdict(0, 0, 5) != "REFERENCE_ATTRIBUTES_NO_DISAGREEMENT_TO_EITHER_ARM":
        raise AssertionError("unattributed disagreement was attributed anyway")
    if _verdict(3, 3, 6) != "REFERENCE_SPLITS_EVENLY_BETWEEN_THE_ARMS":
        raise AssertionError("a tie was broken")
    if _verdict(4, 3, 7) != "REFERENCE_SIDES_WITH_LEFT_AGAINST_RIGHT":
        raise AssertionError("the majority side was misnamed")
    _ok("ties and empty comparisons are named, not resolved into a winner")


def _check_stats_report_the_distribution() -> None:
    reference = [1.0] * 100
    arm = [1.0] * 99 + [2.0]
    stats = magnitude_stats(reference, arm)
    if stats.count != 100 or stats.mismatch_count != 1:
        raise AssertionError(f"stats miscounted: {stats}")
    if stats.mismatch_fraction != 0.01:
        raise AssertionError(f"mismatch fraction is wrong: {stats}")
    if stats.max_abs_diff != 1.0 or stats.max_rel_diff != 1.0:
        raise AssertionError(f"maxima are wrong: {stats}")
    # The median of the differences is zero while the maximum is one. Both are
    # reported, which is the reason the quantiles are here at all.
    if stats.quantiles_abs_diff["q0.5"] != 0.0 or stats.quantiles_abs_diff["q1"] != 1.0:
        raise AssertionError(f"quantiles are wrong: {stats.quantiles_abs_diff}")
    # A NaN must not be allowed to present itself as the shape of the
    # distribution; it is excluded from the quantiles rather than sorted.
    with_nan = magnitude_stats([1.0] * 99 + [float("nan")], [1.0] * 100)
    if not math.isfinite(with_nan.quantiles_abs_diff["q1"]):
        raise AssertionError("a NaN reached the quantiles")
    # The maxima do the opposite and stay loud, and a NaN reference element is
    # not counted as a mismatch. Both are pinned here because both are easy to
    # change by accident and neither is what the neighbouring tool does.
    if math.isfinite(with_nan.max_abs_diff):
        raise AssertionError(f"a NaN was swallowed by the maximum: {with_nan}")
    if with_nan.mismatch_count != 0:
        raise AssertionError(f"a NaN reference element was counted a mismatch: {with_nan}")
    _ok("NaN is excluded from the quantiles, kept in the maxima, and never a mismatch")


def _check_shape_mismatch_refuses() -> None:
    for call in (
        lambda: magnitude_stats([1.0, 2.0], [1.0]),
        lambda: mismatch_mask([1.0, 2.0], [1.0]),
        lambda: arbitrate([1.0, 2.0], [1.0], [1.0, 2.0]),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("a shape mismatch was compared instead of refused")
    _ok("comparing different shapes is refused rather than broadcast")


def _check_other_backends() -> None:
    """Run the ULP rule on torch and numpy too, when they are installed.

    Skipped rather than failed when they are not. The point of the pure-Python
    backend is that the checks above already ran; this only confirms that the
    other two spell the same arithmetic.
    """
    expected = [2.0**-7, 2.0**-6, 2.0**1, 2.0**-6]
    for name in ("numpy", "torch"):
        try:
            module = __import__(name)
        except ImportError:
            _skip(
                f"{name} backend not checked: {name} is not importable here. "
                f"The checks above already ran on the pure-Python backend; to cover "
                f"this one, install {name} and re-run `python3 sensitivity_classify.py`."
            )
            continue
        values = module.asarray([1.0, 2.0, 256.0, -3.0]) if name == "numpy" else (
            module.tensor([1.0, 2.0, 256.0, -3.0])
        )
        observed = [float(v) for v in ulp_bf16(values)]
        if observed != expected:
            raise AssertionError(f"{name} backend disagrees on bf16 ULP: {observed}")
        _ok(f"{name} backend computes the same bf16 ULP")


def _selftest() -> int:
    print("sensitivity_classify --self-test (pure Python, no accelerator)")
    _check_ulp()
    _check_delta_derivation()
    _check_stats_report_the_distribution()
    _check_shape_mismatch_refuses()
    _check_verdicts_do_not_soften()

    gate, up, recompute = _fixture()
    sigma = 1.0
    delta = delta_accum_from_sigma(sigma, 64)
    # The derived delta leaves very few sensitive elements in a small fixture,
    # which is the right physics and a poor test: an attribution check over one
    # element proves little. The band checks therefore run at a DECLARED
    # multiplier, which widens the sensitive population without touching the
    # rule. This multiplier is a self-test device only.
    probe_delta = delta * 1.0e5
    observable = _check_sensitivity_has_both_populations(gate, up, recompute, probe_delta)
    _check_sensitivity_is_monotone_in_delta(gate, up, recompute, probe_delta)

    reference = _PurePythonBackend().round_to_bf16(recompute(gate, up))
    _check_sensitive_band(reference, observable, probe_delta)
    _check_saturated_band(observable, probe_delta)
    _check_implausible_reference_refuses(reference)

    _check_other_backends()

    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        raise AssertionError("the self-test initialized CUDA; it must run off hardware")
    _ok("no accelerator was initialized: zero device launches")
    print(f"sensitivity_classify selftest PASS ({len(PASSES)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
