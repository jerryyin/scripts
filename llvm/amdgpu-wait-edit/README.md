# llvm/amdgpu-wait-edit/ — narrow a suspected missing wait to a site, a class, a distance

Five assembly transforms for gfx1250 that answer, step by step, *which wait is missing* when
a kernel produces nondeterministic results. Each takes a control assembly and a drained
assembly and rewrites **only `s_wait_alu depctr_*` lines**, so every variant has the same
instruction count, the same order and the same addresses as its neighbours. Two variants
that differ in one immediate are comparable; two that differ in their instruction stream are
not, and that property is what the whole directory is built to preserve.

```
amdgpu-wait-edit/
├── README.md                    # this file
├── make-drained-everywhere.py   # maximal drain before EVERY instruction  (the transform)
├── make-null-drain.py           # same insertions, every added wait waits for NOTHING (control)
├── make-half-drain.py           # drain a SELECTED subset of sites: --sites LO:HI[,LO:HI]
├── make-class-drain.py          # + --wait-form all|none|only:CLASS|allbut:CLASS
└── make-distance-drain.py       # + --wait-form dist:N   (va_vdst distance 0-15)
```

## The chain, and why each step exists

Read them in this order; each answers the objection the previous one leaves open.

1. **`make-drained-everywhere.py`** — put a maximal drain before every instruction. This is a
   superset of every site any definition of "needed" could pick out, so if a missing wait is
   the cause, this clears it. Merely *strengthening* the waits the pass already emitted could
   not answer the question, because it says nothing about a site where the pass emits no wait
   at all — which is exactly where a missing wait would be.

2. **`make-null-drain.py`** — the negative control, and without it step 1 proves nothing. A
   drain before every instruction plausibly makes a rare timing window *rarer* without
   closing it, and a metric that saturates at "1 distinct result in 60 runs" cannot tell that
   from a fix. So this emits the same ~833 insertions with every added wait set to **wait for
   nothing**: same instruction count, same addresses, same size, differing only in immediates.
   It perturbs identically and waits for nothing. If perturbation caused the clearing, this
   clears too.

3. **`make-half-drain.py`** — bisection. Drain one half of the sites, leave the other half at
   wait-for-nothing, measure, recurse. Narrows the answer from "a wait is missing somewhere"
   to a **position**. `--sites` takes a comma-separated list of half-open ranges, because a
   converging bisection lands on cuts that are not contiguous.

4. **`make-class-drain.py`** — a position is not something a compiler maintainer can act on; a
   **dependency class** is. A maximal drain waits on all six classes at once, so every clean
   result so far is silent about which one is carrying it. This emits, at selected sites, a
   wait that stops for one named class and nothing else.

5. **`make-distance-drain.py`** — the last step, and the one that produces something
   actionable. No pass emits a maximal drain; a pass emits a **distance**. `va_vdst` is the
   one field of the six that takes sixteen values (0–15), so this walks that axis to find the
   boundary between sufficient and not sufficient.

## The import chain is real, not conceptual

`distance` imports `class`, which imports `half`. Each extension adds **one degree of
freedom** and reuses the walk, the classifier, the site numbering and (for `distance`)
`main()` itself, so a site index means the same thing in every script by construction and
there is no second copy to drift.

Each is a separate file rather than an edit to the previous one for a specific reason: the
earlier scripts have already produced measured objects, and those must stay reproducible from
the script exactly as it stood.

The sibling is located from `__file__`, so the chain works from any working directory and
survives the whole directory being moved or copied. If a sibling is missing you get a named
error, not an opaque `AttributeError`.

## Check the endpoints whenever you change any of these

The extensions overlap the scripts they extend, which gives free byte-for-byte gates. These
are worth running rather than assuming:

| Invocation | Must reproduce |
| --- | --- |
| `half --sites 0:0` | the null-drain object, byte for byte |
| `half --sites 0:N` | the drained-everywhere object (with the pass's own waits restored) |
| `class --wait-form none --sites 0:0` | the null-drain object |
| `class --wait-form all` over a range | the all-drain object for that range |
| `distance --wait-form dist:0` over a range | the `class --wait-form only:va_vdst` object |
| `distance --wait-form dist:15` over a range | the null-drain object |

If `dist:0` and `only:va_vdst` ever diverge, the distance axis is no longer the same axis as
the class axis and nothing measured along it is comparable.

Independently of these, verify the core invariant directly: **strip every wait line from both
assemblies and require the remainder to be identical.** That check is the reason any of these
transforms can be trusted, so run it rather than assuming it.

## Encoding notes, established rather than assumed

The six fields and their maxima come from the target's own table (`AMDGPUAsmUtils.cpp:66-74`
in the build under test), and every immediate is **round-tripped through the assembler and
read back out of the object's `.text`**, not predicted:

- For every wait field the **maximum is also the default and means "wait for nothing"**, so
  **zero is the strongest** setting each field can express.
- All six at zero (maximal drain) assembles to `0x0080`; all six at maximum (wait for
  nothing) assembles to `0xff9f`. Both are one 32-bit SOPP instruction, which is why
  substituting one for the other moves no address and changes no size.
- `depctr_hold_cnt` is **deliberately left alone**. It is not a wait — it is a separate
  control over counter behaviour — and driving a non-wait field to a non-default value would
  be a second edit riding along with the first.

Two places where a drain is put before a **group** rather than inside it, and both are real
limits of the transform rather than caveats:

- `s_clause 0x1` declares that the next two instructions issue as one clause. A wait between
  them would break the clause and thus change the non-wait instruction stream. The drain goes
  before the `s_clause`.
- `s_delay_alu` is a hint about the instruction that follows it. A drain between the hint and
  its target would retarget the hint at the drain. The drain goes before the hint.

Both put the drain **earlier**, never omit one, so the wait set stays a superset of anything
the pass could have emitted.

## Usage

```bash
python3 make-drained-everywhere.py <control.s> <output.s>
python3 make-null-drain.py         <control.s> <drained.s> <output.s>
python3 make-half-drain.py         <control.s> <drained.s> <output.s> --sites 521:524,532:536
python3 make-class-drain.py        <control.s> <drained.s> <output.s> --sites 514:518 --wait-form only:va_vdst
python3 make-distance-drain.py     <control.s> <drained.s> <output.s> --sites 524:532 --wait-form dist:7
```

Every script raises rather than writing out an object it cannot account for. A transform that
silently produced a *slightly* different instruction stream would invalidate the comparison
without telling you, so anything unexpected is a hard failure.
