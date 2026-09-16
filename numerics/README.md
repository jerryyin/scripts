# numerics/ — numeric comparison and test-data generation

Comparing two numeric buffers and manufacturing inputs to feed them. These tools report
**magnitudes**; deciding whether a difference is acceptable is the caller's job, because
the same number means different things for different kernels.

```
numerics/
├── README.md           # this file
├── compare.py          # compare two binary buffers, configurable dtype and threshold
├── distinct_hashes.py  # count distinct hashes over N runs: did every run produce the
│                       #   same bytes? Pure standard library, no device, no numpy
├── genRandInput.py     # generate random binary inputs of a given shape/dtype, or a
│                       #   buffer of special values; also dumps
├── sensitivity_classify.py  # does a difference matter, and which of two arms is wrong
└── ulp_magnitudes.py   # bfloat16 difference in ULPs, absolute and relative; no verdict
```

`compare.py`, `genRandInput.py` and `ulp_magnitudes.py` require `numpy`.
`distinct_hashes.py` and `sensitivity_classify.py` are standard library only, so they run
wherever the outputs landed — the latter will use torch or numpy arrays if you hand them
to it, but needs neither.

**Which one to reach for.** `compare.py` when you already know the tolerance you accept.
`ulp_magnitudes.py` when you want the magnitude of a difference with no verdict attached.
`sensitivity_classify.py` when you need to know whether a difference matters.
`distinct_hashes.py` when the question is not "is it right" but "is it the same every
time".

## Files

- `genRandInput.py` — writes a random binary buffer of a given shape and dtype, and with
  `--dump` prints an existing one in readable form. With `--special` it writes the values
  that uniform random data will never produce: both zeros, NaN, both infinities, the
  smallest denormal, the largest finite value, and the two neighbours of 1.0. Either a
  buffer made entirely of them, or a random buffer with them sprinkled in.

  ```bash
  python3 genRandInput.py 2x235x363x224xbf16.bin --shape 2x235x363x224 --dtype bf16
  python3 genRandInput.py input.bin --shape 1x2x3 --dtype bf16 --dump
  python3 genRandInput.py --shape 1x256 --dtype bf16 --special
  python3 genRandInput.py --shape 1x4096 --dtype f32 --special --special-fraction 0.01
  ```

- `distinct_hashes.py` — SHA-256 over the outputs of N runs of the same computation, and
  a count of how many distinct hashes came back. One means every run produced the same
  bytes; more than one means the computation returned different bytes from identical
  bytes, and the grouping shows which runs agreed with which. It reports counts and never
  a rate, and the count means nothing unless identical input was re-supplied before every
  run. Run it with `--selftest` for its self-test.

  ```bash
  python3 distinct_hashes.py outputs/ --pattern 'run*.bin' --expect-runs 20
  ```

  The same judgement is also implemented in C++ by
  `../llvm/amdgpu-repro/driver/replay.cpp`, which hashes a device buffer in-process
  because writing ~10 MB per run to disk for 120 runs just to hash it is not worth doing.
  Neither copy can be deleted, so `test_determinism_rule_agreement.py` checks that the
  two have not drifted apart in meaning — which a pre-commit hook cannot see. Its hash
  lines are byte-identical to the driver's, and `--driver-format LABEL` adds the driver's
  summary line so `summarize_replay_ab.py` reduces a file-based run and a device run
  through one parser.

- `sensitivity_classify.py` — the arbiter, and the one file here that issues a verdict.
  Where `ulp_magnitudes.py` reports how far apart two buffers are and refuses to judge,
  this one judges: it classifies each differing element as **sensitive** (a perturbation
  of the derived size reaches the stored bfloat16 word) or **saturated** (clamping and
  output rounding destroy it first), and when two arms disagree it names which one an
  independent reference sides with — or refuses to name one, when the reference disagrees
  with both arms widely enough that the likeliest explanation is that the reference itself
  is indexed wrong. Contains no device code by design. Run it directly for its self-test.

  Its defaults are documented but not universal: the minimum plausible reference agreement
  (0.5) came from one study rather than from a law, and the perturbation size is derived
  from the reduction length rather than chosen, so that no threshold is picked after the
  numbers are visible.

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

## Measure the instrument's floor before you believe a disagreement

Before trusting a comparison that reports a disagreement, find out what it reports **when
there is nothing for it to find**. A floor measurement is a run in which a disagreement
*cannot be produced by the program under test*. Without one, a repeat of the same build that
disagrees is ambiguous: either the comparison invented the difference or the program genuinely
returned different bytes, and nothing in the result distinguishes those.

The instrument is a chain, and each link is bounded by the cheapest method that can bound it:

1. **The comparison itself.** Deterministic code over two byte strings cannot report a
   disagreement between identical bytes. Settle that by reading the code — it costs nothing
   and needs no run. It is an *inspection*, not a floor, and it must never be cited as one: it
   says nothing about the links on either side of it.
2. **The capture path — this is the floor proper.** Run the program **exactly once**, then read
   its output buffer back repeatedly without launching anything in between, saving each read.
   Hash the saved reads with `distinct_hashes.py`. Nothing wrote to that buffer between the
   first read and the last, so a second distinct hash has no source but the copy and the
   comparison. Record the evidence that nothing wrote to it rather than asserting it.
3. **The full loop, which is not a floor.** Run the program N times with byte-identical input
   re-supplied before each one, and count distinct hashes again. That count bounds the *sum* of
   what the capture path contributes and what the program varies by on its own. It cannot
   separate them, so do not describe it as a property of the program alone.

Hold everything identical on both sides except the one thing under test — same build, same
input bytes, same buffer pre-fill, same machine since the same boot. "The detector read zero,
so it cannot have mattered" is an argument, not a control.

Only once step 2 comes back with one distinct hash does a disagreement from `compare.py`, or a
nonzero distance from `ulp_magnitudes.py`, belong to the program. If it comes back with more
than one, the harness is what you are measuring and fixing it comes first. Report the floor
**beside** the finding and in the same unit — a count of distinct hashes over a stated number
of captures. Do not subtract it from the finding, and do not turn either into a rate.
