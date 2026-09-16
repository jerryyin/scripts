#!/usr/bin/env python3
"""How big is a bfloat16 difference, measured in units that need no threshold.

Compares two buffers of bfloat16 words and reports the MAGNITUDE of their
disagreement. It deliberately makes no pass/fail judgement: every function here
answers "how far apart are these two words" and none answers "is that too far".
There is no threshold, no pass, no fail, and no criterion under which a
difference stops counting. Whoever reads the output decides what is acceptable
for their kernel; this module only measures. If you are tempted to add a
tolerance here, put it in the caller instead -- the same numbers mean different
things for different kernels, and baking one limit in would hide that.

Two comparison modes, because the right unit depends on what is being compared:

  compare_words()             two bfloat16 buffers (e.g. two variants of the
                              same kernel). Primary unit: ULP.
  compare_against_reference() one bfloat16 buffer against a higher-precision
                              reference. Primary unit: all three.

WHY ULP IS THE PRIMARY UNIT WHEN BOTH SIDES ARE BFLOAT16
--------------------------------------------------------
Two bfloat16 words that differ at all differ by at least one unit in the last
place, by construction, because both sides are bfloat16. There is no sub-ULP
case when comparing bf16 against bf16. So an absolute or relative difference
there is a magnitude-scaled restatement of something the bit patterns already
say exactly, and it makes a rounding-boundary difference at a large value look
bigger than a structural error at a small one. The ULP distance does not do
that.

Against the reference the situation is the opposite: the reference is computed
at a different precision, the sub-ULP case is real, and absolute and relative
differences carry information that a bit-pattern distance cannot. So that
comparison reports all three.

THE ORDINAL MAPPING, WHICH IS THE ONE NON-OBVIOUS PART
-------------------------------------------------------
bfloat16 is a sign-magnitude format, so subtracting raw bit patterns gives a
nonsense answer for any pair that straddles zero: 0x8001 (the smallest negative)
and 0x0001 (the smallest positive) are two representable steps apart, but their
raw patterns differ by 32768. Mapping each pattern to a signed ordinal --
positives to +bits, negatives to -(bits without the sign bit) -- makes the
distance the number of representable values between them, which is what "how
many ulps" means. Positive and negative zero both map to 0, which is correct:
they are numerically equal and their distance should be zero, not one.

NaN and infinity have no position in that ordering. They are counted
structurally and EXCLUDED from the ULP distribution rather than given a
fabricated distance, and every record says how many positions were excluded --
a field that is absent must never look like a field that was measured and came
out empty.
"""

from __future__ import annotations

from typing import Any

import numpy

# bfloat16, from the format: 1 sign bit, 8 exponent bits, 7 mantissa bits.
BF16_SIGN = numpy.uint16(0x8000)
BF16_MAGNITUDE = numpy.uint16(0x7FFF)
BF16_EXPONENT = numpy.uint16(0x7F80)
BF16_MANTISSA = numpy.uint16(0x007F)

# Per-position detail is capped so one comparison stays one readable record rather than
# growing with the size of the buffer. The counts above it are never capped.
SAMPLE_CAP = 64

PRIMARY_UNIT_NOTE = (
    "When both sides are bfloat16 the ULP distance is the primary figure and the "
    "absolute and relative differences are secondary. Two bfloat16 words that differ at "
    "all differ by at least one unit in the last place, so there is no sub-ULP case here "
    "and a magnitude-scaled difference would make a rounding-boundary difference at a "
    "large value look bigger than a structural error at a small one. Against the "
    "reference all three are primary, because the reference is computed at a different "
    "precision and the sub-ULP case is real."
)

NOT_A_TOLERANCE_RULING = (
    "This is a magnitude, not a verdict. A small maximum difference does not make a "
    "disagreement benign and a large one does not make it a miscompile. Nothing in this "
    "record applies a threshold; interpreting it is the caller's job."
)

RELATIVE_DIFFERENCE_RULE = (
    "relative difference = |left - right| / max(|left|, |right|), and 0.0 where both "
    "sides are zero. Stated because a relative difference has no meaning without its "
    "denominator, and dividing by one arbitrary side would make the number depend on "
    "which side was called left."
)

ULP_RULE = (
    "ULP distance = |ordinal(left) - ordinal(right)|, where ordinal maps a bfloat16 bit "
    "pattern to its position in the ordered list of representable values: +bits for a "
    "positive, -(bits without the sign bit) for a negative. Positive and negative zero "
    "both map to 0, so their distance is 0 and not 1. A distance of 1 means the two "
    "words are adjacent representable values with nothing between them. NaN and infinity "
    "have no position in that ordering and are excluded from the distribution."
)


def decode_bf16(words: numpy.ndarray) -> numpy.ndarray:
    """bfloat16 bit patterns to float32, exactly and without rounding."""
    return (words.astype(numpy.uint32) << 16).view(numpy.float32)


def encode_bf16_round_to_nearest_even(values: numpy.ndarray) -> numpy.ndarray:
    """float32 to bfloat16 bit patterns, round-to-nearest-even.

    Used only to put a reference computed at a higher precision onto the same
    grid as an arm's output, so that a ULP distance between them is defined.
    Rounding the reference is a change of representation for the purpose of
    measuring a distance; the absolute and relative differences reported beside
    it are computed from the UNROUNDED reference, so no information is lost by
    this step -- it only adds a third unit.
    """
    bits = values.astype(numpy.float32).view(numpy.uint32)
    # Round half to even on the 16 bits being discarded.
    rounding_bias = ((bits >> numpy.uint32(16)) & numpy.uint32(1)) + numpy.uint32(0x7FFF)
    return ((bits + rounding_bias) >> numpy.uint32(16)).astype(numpy.uint16)


def is_nan(words: numpy.ndarray) -> numpy.ndarray:
    return ((words & BF16_EXPONENT) == BF16_EXPONENT) & ((words & BF16_MANTISSA) != 0)


def is_infinity(words: numpy.ndarray) -> numpy.ndarray:
    return (words & BF16_MAGNITUDE) == BF16_EXPONENT


def ordinal(words: numpy.ndarray) -> numpy.ndarray:
    """Position of each bit pattern in the ordered list of representable values."""
    magnitude = (words & BF16_MAGNITUDE).astype(numpy.int64)
    negative = (words & BF16_SIGN) != 0
    return numpy.where(negative, -magnitude, magnitude)


def ulp_distance(left: numpy.ndarray, right: numpy.ndarray) -> numpy.ndarray:
    """Representable values between two bit patterns. Meaningless at NaN/infinity."""
    return numpy.abs(ordinal(left) - ordinal(right))


def _histogram(distances: numpy.ndarray) -> dict[str, Any]:
    """The ULP distribution, as counts in bands plus the exact quantiles.

    Bands rather than a mean: the question is whether the differing positions are
    adjacent representable values or far apart, and a mean over a distribution
    with a long tail answers neither. The bands are powers of two, which is a
    presentation choice and not a threshold -- no band boundary decides anything,
    and the exact maximum and quantiles are reported beside them.
    """

    if distances.size == 0:
        return {"status": "NO_POSITIONS_WITH_A_DEFINED_ULP_DISTANCE", "bands": {}}
    # A distance of zero between two DIFFERING bit patterns is possible and means
    # exactly one thing: positive zero opposite negative zero. It gets its own band
    # rather than being folded in with the adjacent-value case, because "the arms
    # disagree on the sign of a zero" and "the arms are one representable value
    # apart" are different findings.
    bands: dict[str, int] = {
        "exactly_0_signed_zero_only": int((distances == 0).sum()),
        "exactly_1": int((distances == 1).sum()),
    }
    edges = [2, 4, 8, 16, 64, 256, 1024, 4096, 16384]
    previous = 1
    for edge in edges:
        bands[f"{previous + 1}_to_{edge}"] = int(
            ((distances > previous) & (distances <= edge)).sum()
        )
        previous = edge
    bands[f"above_{previous}"] = int((distances > previous).sum())
    if sum(bands.values()) != int(distances.size):
        raise AssertionError("the ULP bands do not partition the positions")
    quantiles = numpy.percentile(distances, [50, 90, 99, 100]).tolist()
    return {
        "status": "RECORDED",
        "positions": int(distances.size),
        "bands": bands,
        "bands_are_a_presentation_not_a_threshold": (
            "No band boundary decides anything. The exact maximum and quantiles are "
            "beside them and the raw distances of the sampled positions are below."
        ),
        "minimum": int(distances.min()),
        "median": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "maximum": int(distances.max()),
        "positions_at_exactly_1_ulp": bands["exactly_1"],
        "fraction_at_exactly_1_ulp": float(bands["exactly_1"] / distances.size),
    }


def _structural(left: numpy.ndarray, right: numpy.ndarray) -> dict[str, Any]:
    left_nan, right_nan = is_nan(left), is_nan(right)
    left_inf, right_inf = is_infinity(left), is_infinity(right)
    left_zero = (left & BF16_MAGNITUDE) == 0
    right_zero = (right & BF16_MAGNITUDE) == 0
    opposite_sign = ((left & BF16_SIGN) != (right & BF16_SIGN)) & ~(left_zero & right_zero)
    return {
        "positions_with_a_nan_on_either_side": int((left_nan | right_nan).sum()),
        "positions_with_an_infinity_on_either_side": int((left_inf | right_inf).sum()),
        "positions_with_a_zero_opposite_a_nonzero": int(
            ((left_zero & ~right_zero) | (right_zero & ~left_zero)).sum()
        ),
        "positions_with_opposite_signs": int(opposite_sign.sum()),
        "counted_over": "all differing positions, not only the sampled ones",
    }


def _position_records(
    indices: numpy.ndarray,
    left: numpy.ndarray,
    right: numpy.ndarray,
    left_value: numpy.ndarray,
    right_value: numpy.ndarray,
    distances: numpy.ndarray,
    defined: numpy.ndarray,
    columns: int,
) -> list[dict[str, Any]]:
    records = []
    for position, flat in enumerate(indices.tolist()):
        record: dict[str, Any] = {
            "flat_index": int(flat),
            "row": int(flat // columns),
            "output_column": int(flat % columns),
            "left_raw_hex": f"0x{int(left[position]):04x}",
            "right_raw_hex": f"0x{int(right[position]):04x}",
            "left_value": float(left_value[position]),
            "right_value": float(right_value[position]),
            "absolute_difference": float(
                abs(float(left_value[position]) - float(right_value[position]))
            ),
        }
        denominator = max(abs(float(left_value[position])), abs(float(right_value[position])))
        record["relative_difference"] = (
            0.0 if denominator == 0.0 else record["absolute_difference"] / denominator
        )
        record["ulp_distance"] = (
            int(distances[position]) if bool(defined[position]) else None
        )
        if record["ulp_distance"] is None:
            record["ulp_distance_undefined_because"] = (
                "a NaN or an infinity has no position in the ordering of representable "
                "values"
            )
        records.append(record)
    return records


def compare_words(
    left_words: numpy.ndarray,
    right_words: numpy.ndarray,
    *,
    columns: int,
    where: str,
    primary_unit: str,
    sample_cap: int = SAMPLE_CAP,
) -> dict[str, Any]:
    """Every magnitude of the difference between two bfloat16 payloads.

    `primary_unit` is "ulp" for a bf16-against-bf16 comparison and "all" for a
    comparison against the reference. It changes only which figures the record
    labels primary; every figure is computed and present either way, so a reader
    who disagrees with the labelling still has the numbers.
    """

    if primary_unit not in ("ulp", "all"):
        raise ValueError(f"unknown primary unit {primary_unit!r}")
    if left_words.shape != right_words.shape:
        return {
            "status": "NOT_COMPARED_PAYLOAD_LENGTHS_DIFFER",
            "where": where,
            "left_elements": int(left_words.size),
            "right_elements": int(right_words.size),
        }

    differing = numpy.flatnonzero(left_words != right_words)
    record: dict[str, Any] = {
        "where": where,
        "primary_unit": "ULP distance" if primary_unit == "ulp" else (
            "absolute, relative and ULP together"
        ),
        "primary_unit_note": PRIMARY_UNIT_NOTE,
        "is_not_a_tolerance_ruling": NOT_A_TOLERANCE_RULING,
        "ulp_rule": ULP_RULE,
        "relative_difference_rule": RELATIVE_DIFFERENCE_RULE,
        "denominator_elements": int(left_words.size),
        "differing_positions": int(differing.size),
        "sample_cap": int(sample_cap),
    }
    if differing.size == 0:
        record["status"] = "NO_DIFFERING_POSITIONS"
        record["values"] = []
        record["sampled_positions"] = 0
        record["positions_not_sampled"] = 0
        return record

    record["status"] = "RECORDED"
    left = left_words[differing]
    right = right_words[differing]
    left_value = decode_bf16(left).astype(numpy.float64)
    right_value = decode_bf16(right).astype(numpy.float64)

    defined = ~(is_nan(left) | is_nan(right) | is_infinity(left) | is_infinity(right))
    distances = numpy.zeros(differing.size, dtype=numpy.int64)
    distances[defined] = ulp_distance(left[defined], right[defined])
    record["ulp"] = _histogram(distances[defined])
    record["ulp"]["positions_excluded_as_nan_or_infinity"] = int((~defined).sum())
    record["ulp"]["excluded_positions_are_counted_structurally_instead"] = True

    finite = numpy.isfinite(left_value) & numpy.isfinite(right_value)
    absolute = numpy.abs(left_value - right_value)
    denominator = numpy.maximum(numpy.abs(left_value), numpy.abs(right_value))
    with numpy.errstate(invalid="ignore"):
        # A NaN divided by a NaN is expected here and is not a problem: those
        # positions are excluded from the maxima by the finite mask below and
        # counted structurally instead.
        relative = numpy.divide(
            absolute, denominator, out=numpy.zeros_like(absolute), where=denominator != 0
        )
    # Maxima over finite positions only, so one NaN cannot swallow the reading.
    record["maxima_taken_over"] = (
        "positions where both sides are finite, so that a single NaN cannot swallow the "
        "magnitude maxima. NaN and infinity positions are reported in the structural "
        "counts instead."
    )
    record["finite_positions"] = int(finite.sum())
    if finite.any():
        record["largest_absolute_difference"] = float(absolute[finite].max())
        record["largest_relative_difference"] = float(relative[finite].max())
    else:
        record["largest_absolute_difference"] = None
        record["largest_relative_difference"] = None
    record["structural"] = _structural(left, right)

    sampled = min(int(sample_cap), int(differing.size))
    record["sampled_positions"] = sampled
    record["positions_not_sampled"] = int(differing.size) - sampled
    record["sample_order"] = "the first differing positions in flat-index order"
    record["values"] = _position_records(
        differing[:sampled], left[:sampled], right[:sampled],
        left_value[:sampled], right_value[:sampled],
        distances[:sampled], defined[:sampled], columns,
    )
    return record


def compare_against_reference(
    arm_words: numpy.ndarray,
    reference_values: numpy.ndarray,
    *,
    columns: int,
    where: str,
    sample_cap: int = SAMPLE_CAP,
) -> dict[str, Any]:
    """An arm's output against a reference computed at a higher precision.

    The ULP distance is taken against the reference ROUNDED to bfloat16, which is
    the only way a bit-pattern distance is defined at all. The absolute and
    relative differences are taken against the UNROUNDED reference, so the
    rounding cannot flatter or inflate them.
    """

    reference_words = encode_bf16_round_to_nearest_even(reference_values)
    record = compare_words(
        arm_words, reference_words,
        columns=columns, where=where, primary_unit="all", sample_cap=sample_cap,
    )
    record["reference_rounding"] = (
        "The ULP distance is measured against the reference rounded to bfloat16 "
        "(round-to-nearest-even), because a bit-pattern distance is undefined otherwise. "
        "The absolute and relative differences below are measured against the UNROUNDED "
        "reference, so rounding it cannot change them."
    )
    if record.get("status") == "NOT_COMPARED_PAYLOAD_LENGTHS_DIFFER":
        return record

    # Computed even when NO bit pattern differs, and that is the whole point: the
    # reference can sit strictly between two representable values, round onto the
    # arm's word, and still be a nonzero distance away. A row that reported only
    # differing patterns here would report zero difference for a case where the
    # difference is real and simply smaller than one representable step.
    arm_value = decode_bf16(arm_words).astype(numpy.float64)
    exact = reference_values.astype(numpy.float64)
    absolute = numpy.abs(arm_value - exact)
    denominator = numpy.maximum(numpy.abs(arm_value), numpy.abs(exact))
    with numpy.errstate(invalid="ignore"):
        # A NaN divided by a NaN is expected here and is not a problem: those
        # positions are excluded from the maxima by the finite mask below and
        # counted structurally instead.
        relative = numpy.divide(
            absolute, denominator, out=numpy.zeros_like(absolute), where=denominator != 0
        )
    finite = numpy.isfinite(arm_value) & numpy.isfinite(exact)
    record["against_the_unrounded_reference"] = {
        "positions_compared": int(arm_words.size),
        "note": (
            "Taken over EVERY position, not only the ones whose rounded bit patterns "
            "differ. An arm can sit close to the reference everywhere and still land on "
            "a different bit pattern at a few positions, and the opposite is also "
            "possible; a count of differing patterns does not bound this."
        ),
        "finite_positions": int(finite.sum()),
        "largest_absolute_difference": (
            float(absolute[finite].max()) if finite.any() else None
        ),
        "largest_relative_difference": (
            float(relative[finite].max()) if finite.any() else None
        ),
        "mean_absolute_difference": (
            float(absolute[finite].mean()) if finite.any() else None
        ),
    }
    return record


def _selftest() -> None:
    """Planted cases, in both worlds, plus every structural case and the cap."""

    def payload(overrides: dict[int, int], size: int = 512, fill: int = 0x3F80):
        buffer = numpy.full(size, fill, dtype=numpy.uint16)
        for index, bits in overrides.items():
            buffer[index] = numpy.uint16(bits)
        return buffer

    checks = 0

    # 0x3F80 is 1.0; 0x3F81 is the next representable value above it.
    adjacent = compare_words(
        payload({}), payload({5: 0x3F81}), columns=64, where="adjacent", primary_unit="ulp"
    )
    assert adjacent["differing_positions"] == 1, adjacent
    assert adjacent["ulp"]["maximum"] == 1, adjacent["ulp"]
    assert adjacent["ulp"]["positions_at_exactly_1_ulp"] == 1
    assert adjacent["values"][0]["ulp_distance"] == 1
    assert adjacent["values"][0]["left_value"] == 1.0
    checks += 1

    # 1.0 against 2.0: a whole binade, 128 representable values apart in bf16.
    binade = compare_words(
        payload({}), payload({5: 0x4000}), columns=64, where="binade", primary_unit="ulp"
    )
    assert binade["ulp"]["maximum"] == 128, binade["ulp"]
    assert abs(binade["largest_relative_difference"] - 0.5) < 1e-12
    assert binade["ulp"]["positions_at_exactly_1_ulp"] == 0
    checks += 1

    # The mapping's whole reason for existing: across zero, the raw patterns differ
    # by 32768 and the true distance is 2.
    across_zero = compare_words(
        payload({}, size=8, fill=0x0001), payload({0: 0x8001}, size=8, fill=0x0001),
        columns=8, where="across zero", primary_unit="ulp",
    )
    assert across_zero["ulp"]["maximum"] == 2, across_zero["ulp"]
    checks += 1

    # Positive and negative zero are numerically equal; their distance is 0, not 1.
    # They are different bit patterns, so they DO count as a differing position.
    signed_zero = compare_words(
        payload({}, size=8, fill=0x0000), payload({0: 0x8000}, size=8, fill=0x0000),
        columns=8, where="signed zero", primary_unit="ulp",
    )
    assert signed_zero["differing_positions"] == 1
    assert signed_zero["ulp"]["maximum"] == 0, signed_zero["ulp"]
    assert signed_zero["structural"]["positions_with_opposite_signs"] == 0
    assert signed_zero["structural"]["positions_with_a_zero_opposite_a_nonzero"] == 0
    checks += 1

    # 0x7FC0 NaN, 0x7F80 +infinity, 0x0000 zero, 0xBF80 -1.0 against +1.0.
    structural = compare_words(
        payload({}), payload({1: 0x7FC0, 2: 0x7F80, 3: 0x0000, 4: 0xBF80}),
        columns=64, where="structural", primary_unit="ulp",
    )
    counts = structural["structural"]
    assert structural["differing_positions"] == 4, structural
    assert counts["positions_with_a_nan_on_either_side"] == 1, counts
    assert counts["positions_with_an_infinity_on_either_side"] == 1, counts
    assert counts["positions_with_a_zero_opposite_a_nonzero"] == 1, counts
    assert counts["positions_with_opposite_signs"] == 1, counts
    assert structural["ulp"]["positions_excluded_as_nan_or_infinity"] == 2, structural["ulp"]
    assert structural["ulp"]["positions"] == 2, structural["ulp"]
    # A NaN must not be allowed to present itself as the largest magnitude.
    assert numpy.isfinite(structural["largest_absolute_difference"]), structural
    # The NaN and infinity positions carry no fabricated ULP distance.
    undefined = [v for v in structural["values"] if v["ulp_distance"] is None]
    assert len(undefined) == 2, undefined
    checks += 1

    # Identical payloads report nothing rather than an empty reading.
    quiet = compare_words(
        payload({}), payload({}), columns=64, where="equal", primary_unit="ulp"
    )
    assert quiet["status"] == "NO_DIFFERING_POSITIONS" and quiet["differing_positions"] == 0
    checks += 1

    # The cap stops the sample and does not stop the summary.
    planted = {index: 0x4000 for index in range(SAMPLE_CAP + 17)}
    capped = compare_words(
        payload(planted, size=512), payload({}, size=512),
        columns=64, where="cap", primary_unit="ulp",
    )
    assert capped["differing_positions"] == SAMPLE_CAP + 17
    assert capped["sampled_positions"] == SAMPLE_CAP
    assert capped["positions_not_sampled"] == 17
    assert len(capped["values"]) == SAMPLE_CAP
    assert capped["ulp"]["positions"] == SAMPLE_CAP + 17, capped["ulp"]
    checks += 1

    # Against a higher-precision reference the sub-ULP case is real: a reference
    # that sits between two representable values rounds onto one of them, so the
    # bit patterns can agree while the absolute difference is nonzero.
    reference = numpy.full(8, 1.0 + 2.0 ** -9, dtype=numpy.float32)
    arm = payload({}, size=8)
    against = compare_against_reference(arm, reference, columns=8, where="sub-ulp")
    assert against["against_the_unrounded_reference"]["largest_absolute_difference"] > 0.0
    assert against["against_the_unrounded_reference"]["positions_compared"] == 8
    checks += 1

    # Round-to-nearest-even is the rounding claimed, so check the tie.
    # One bf16 step at 1.0 is 2**-7, so an exact tie sits at 2**-8 past a
    # representable value. Both ties below land on the EVEN neighbour, which is
    # down in the first case and up in the second -- a round-half-up
    # implementation would take both upward and fail the first.
    ties = numpy.array([1.0 + 2.0 ** -8, 1.0 + 3.0 * 2.0 ** -8], dtype=numpy.float32)
    rounded = encode_bf16_round_to_nearest_even(ties)
    assert int(rounded[0]) == 0x3F80, hex(int(rounded[0]))   # tie, even is below
    assert int(rounded[1]) == 0x3F82, hex(int(rounded[1]))   # tie, even is above
    # And a plain three-quarter-step rounds up, tie rule not involved.
    assert int(encode_bf16_round_to_nearest_even(
        numpy.array([1.0 + 3.0 * 2.0 ** -9], dtype=numpy.float32)
    )[0]) == 0x3F81
    checks += 1

    # Nothing here decides anything: no function returns a verdict field.
    for record in (adjacent, binade, structural, against):
        for key in record:
            assert "matches" not in key and "verdict" not in key and "pass" not in key, key
    checks += 1

    print(f"ulp_magnitudes selftest PASS ({checks} checks, 0 device launches)")


if __name__ == "__main__":
    _selftest()
