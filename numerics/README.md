# numerics/ — numeric comparison and test-data generation

Comparing two numeric buffers and manufacturing inputs to feed them. These tools report
**magnitudes**; deciding whether a difference is acceptable is the caller's job, because
the same number means different things for different kernels.

```
numerics/
├── README.md           # this file
├── compare.py          # compare two binary buffers, configurable dtype and threshold
├── genRandInput.py     # generate random binary inputs of a given shape/dtype; also dumps
└── ulp_magnitudes.py   # bfloat16 difference in ULPs, absolute and relative; no verdict
```

Requires `numpy`.

## Files

- `genRandInput.py` — writes a random binary buffer of a given shape and dtype, and with
  `--dump` prints an existing one in readable form.

  ```bash
  python3 genRandInput.py 2x235x363x224xbf16.bin --shape 2 235 363 224 --dtype bf16
  python3 genRandInput.py input.bin --shape 1 2 3 --dtype bf16 --dump
  ```

- `compare.py` — compares two binary outputs with a configurable dtype and threshold.
  The quick pass/fail for "did this change break the kernel".

- `ulp_magnitudes.py` — the careful one. Reports how far apart two bfloat16 buffers are and
  **deliberately makes no pass/fail judgement**: no threshold, no criterion under which a
  difference stops counting. Run it directly for its self-test.

## Why ULP is the right primary unit for bf16 against bf16

Two bfloat16 words that differ at all differ by **at least one unit in the last place**, by
construction, because both sides are bfloat16. There is no sub-ULP case. So an absolute or
relative difference between two bf16 buffers is a magnitude-scaled restatement of something
the bit patterns already say exactly — and it distorts, because it makes a rounding-boundary
difference at a large value look bigger than a structural error at a small one. The ULP
distance does not do that.

Against a **higher-precision reference** the situation reverses: the reference is computed
on a finer grid, the sub-ULP case is real, and absolute and relative differences carry
information a bit-pattern distance cannot. So `compare_against_reference()` reports all
three and `compare_words()` treats ULP as primary.

The one non-obvious mechanic: bfloat16 is sign-magnitude, so subtracting raw bit patterns is
nonsense across zero — `0x8001` and `0x0001` are two representable steps apart but their raw
patterns differ by 32768. Each pattern is mapped to a signed **ordinal** first (positives to
`+bits`, negatives to `-(bits without the sign bit)`), which makes the distance the number
of representable values between them. Both zeros map to 0, so their distance is 0 and not 1.

NaN and infinity have no position in that ordering. They are counted structurally and
**excluded** from the ULP distribution rather than given a fabricated distance, and every
record states how many positions were excluded — a field that is absent must never look like
a field that was measured and came out empty.
