# llvm/amdgpu-repro/ — does an LLVM codegen flag change what a kernel COMPUTES?

A reusable A/B template for the question "I think this backend flag is not just a scheduling
knob". Hold the LLVM IR byte-identical, compile with and without the flag, classify every
instruction that changed, then run both code objects on identical input bytes and count
distinct output hashes. One hash means deterministic. More than one means the kernel returned
different bytes from the same input.

The defaults are the case it was built for: `-amdgpu-expert-scheduling-mode` on gfx1250.
Everything is overridable.

```
amdgpu-repro/
├── README.md                     # this file
├── reproduce.sh                  # the driver: codegen | build | run | all
└── driver/
    ├── replay.cpp                # loads a code object, replays recorded kernel args, hashes output
    ├── check-manifest-fixture.sh # proves replay's manifest validation actually refuses (no GPU)
    └── summarize_replay_ab.py    # turn a run's logs into one JSON record
```

You supply the IR: `$IR_DIR/<kernel>.ll` for each name in `$KERNELS`.

## The three legs

```bash
./reproduce.sh codegen   # compile both ways, diff, classify.   NO GPU REQUIRED
./reproduce.sh build     # assemble both into code objects.     NO GPU REQUIRED
./reproduce.sh run       # A/B on device, count output hashes.  NEEDS THE GPU + captured args
./reproduce.sh all       # all three
```

**`codegen` is the whole off-hardware argument** and the part someone else can check without
owning any of your data. Run it first.

## Reading the codegen output

`classify_diff` sorts the A/B assembly diff into four buckets and prints both a raw and a
*cancelled* changed-line count. The gap between them matters: a line that appears on both
sides with byte-identical text was not changed, it was **re-aligned**, because insertions
elsewhere pushed the surrounding context around. Counting those as changes inflates the
residual — on one revision it turned three changed hint instructions into a reported eight.
So removed lines are cancelled against identical added lines as a multiset, and only the
remainder is classified.

The stated limit of that, rather than a hidden one: cancellation cannot distinguish a
re-aligned line from a genuinely **relocated** one, so this does not detect pure reordering.
The residual-zero test is a test for instructions appearing, vanishing or changing text — not
for scheduling.

Bucket **(iv), "everything else", must be zero.** That is the bucket that says nothing
*other* than the expected effect happened. Buckets (i)–(iii) name the instructions the
**default** flag is expected to move (a wave-scheduling mode set, `s_wait_alu depctr_*`
waits, `s_delay_alu` hints — which are counted separately and never folded into the wait
count, because inserting waits necessarily perturbs them). **If you point this at a different
flag, (i)–(iii) will read zero and everything will land in (iv).** That is not a malfunction;
it means `classify_diff` needs retuning for the instructions your flag touches.

## What you need for the device half

`replay.cpp` **replays recorded kernel arguments; it does not synthesise them.** `run` needs
a capture directory containing a `manifest.txt` with a SHA-256 per buffer.

Each module needs **its own** capture. Two modules can share a kernel name and an argument
layout and still differ in work-group size, and one capture replayed into both is a launch
that is wrong in a way the output hash cannot show. Resolution order per module:

| Source | Notes |
| --- | --- |
| `CAPTURE_<stem>` | explicit, wins |
| `$CAPTURE/<stem>/` | a per-module subdirectory |
| `$CAPTURE/<stem>-<suffix>/` | exactly one match, or it refuses rather than picking |
| `$CAPTURE` | flat single-capture layout |

More than one suffixed match is a **refusal**, not a sort-order pick.

`check-manifest-fixture.sh` demonstrates that the manifest validation genuinely refuses all
three ways a manifest can be wrong. It compiles `replay.cpp` and drives its loader over
synthetic captures in a temporary directory: **no device, no lock, no ROCm call**, nothing
written outside `/tmp`. Run it during someone else's session safely.

## Toolchain discovery

`llc` is looked up rather than hardcoded, because the only builds that know a new target are
recent and everybody keeps them somewhere different. Order: `$LLC`, `PATH`, `$LLC_SEARCH`
globs, `/opt/rocm/llvm/bin/llc`. A candidate must both know `$MCPU` and accept `$FLAG`.

```bash
LLC_SEARCH="$HOME/llvm-build/*/bin/llc:/opt/other-llvm/bin/llc" ./reproduce.sh codegen
```

`clang` and `ld.lld` are required only by `build`, not by `codegen` — some LLVM packages ship
`llc` and `clang` but no `ld.lld` driver, and demanding them up front would withhold the one
result that needs no device. The linker can be pinned separately with `LD`; when it is not
the one beside `llc`, the difference is **printed**, because the instructions come from `llc`
and the assembler and a borrowed linker belongs in the record.

**Do not pipe `llc` into `grep -q` under `set -o pipefail`.** `grep -q` exits at its first
match, killing `llc` with SIGPIPE; the pipeline reports 141 and a usable `llc` is rejected.
This bit the `--help-list-hidden` probe every time and the `-mcpu=help` probe never, purely
because of output length, which made it look like the flag was absent. `llc_is_usable` reads
each output in full and then matches.

## Configuration

| Variable | Default |
| --- | --- |
| `MCPU` | `gfx1250` |
| `FLAG` | `-amdgpu-expert-scheduling-mode` |
| `KERNELS` | `subject8 reference4` (space-separated stems) |
| `IR_DIR` | `<here>/ir` |
| `OUT` | `<here>/out` |
| `RUNS` | `120` |
| `CAPTURE`, `CAPTURE_<stem>` | `<here>/args` |
| `LLC`, `CLANG`, `LD`, `HIPCC`, `LLC_SEARCH` | discovered |
