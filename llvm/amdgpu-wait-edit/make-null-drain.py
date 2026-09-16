#!/usr/bin/env python3
"""The negative control for make-drained-everywhere.py: same perturbation, no waiting.

THE OBJECTION THIS ANSWERS, and it is the first thing a reviewer will say: a drain before every
instruction plausibly makes a rare timing window RARER without closing it, and a metric that
saturates at "1 distinct result in 60 runs" cannot tell that from a genuine fix. So an apparent
fix might be caused by the ~833 added instructions perturbing the kernel rather than by those
instructions actually waiting. Without this control, the drained-everywhere result proves
nothing.

THE CONTROL: the drained-everywhere variant with every ADDED wait set to WAIT FOR NOTHING, and
the waits the pass originally emitted put back to their ORIGINAL values.
Same instruction count, same addresses, same size, differing only in wait immediates. It perturbs
identically and it waits for nothing. If the perturbation cleared it, this clears too. If the
waits cleared it, this is affected.

THE NO-WAIT VALUE, established the way the maximal drain was established -- from the target's own
field table and confirmed by round-tripping the encoding, not assumed. AMDGPUAsmUtils.cpp:66-74
in the build under test gives, as {name, max, default, offset, width}:

    depctr_hold_cnt   max 1   default 1   offset 7    width 1
    depctr_sa_sdst    max 1   default 1   offset 0    width 1
    depctr_va_vdst    max 15  default 15  offset 12   width 4
    depctr_va_sdst    max 7   default 7   offset 9    width 3
    depctr_va_ssrc    max 1   default 1   offset 8    width 1
    depctr_va_vcc     max 1   default 1   offset 1    width 1
    depctr_vm_vsrc    max 7   default 7   offset 2    width 3

For every wait field the maximum is the "wait for nothing" value, so every field at its maximum
is a wait for nothing in all six dependency classes at once. Assembling it and reading the
encoding back gives 0xff9f, which is exactly what those offsets and widths predict; the maximal
drain used by make-drained-everywhere.py reads back as 0x0080 at the same anchor. Both are one 32-bit
SOPP instruction, so substituting one for the other moves no address and changes no size.

HOW THE FILE IS DERIVED, and this is deliberately not a re-run of the placement logic in
make-drained-everywhere.py. Re-deriving where a drain goes would leave the null object's
perturbation only as identical as two independent implementations happen to agree, and the whole
value of this control is that it perturbs IDENTICALLY. So this walks the control assembly and the
drained assembly together and rewrites only their wait lines:

  - a drained line equal to the control line is copied and both advance;
  - a drained wait line whose control counterpart is also a wait is a wait the pass emitted, which
    the drained-everywhere transform rewrote to be maximal, and it is restored to its original text;
  - a drained wait line with no control counterpart is one that transform inserted, and it
    becomes the no-wait form;
  - anything else differing is not a wait, which cannot happen if the drained-everywhere output
    was well formed, and it raises rather than being written out.

The result is the drained file with wait immediates substituted line for line, so the instruction
count, the order and every address follow by construction rather than by agreement.
"""

import re
import sys

# Every wait field at its MAXIMUM, which the field table also gives as its default: wait for
# nothing, in all six dependency classes. Encodes to 0xff9f.
NOWAIT = (
    "\ts_wait_alu depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) "
    "depctr_va_vcc(1) depctr_vm_vsrc(7) depctr_sa_sdst(1)\n"
)

WAIT_ALU = re.compile(r"^\s*s_wait_alu\b")
DELAY_ALU = re.compile(r"^\s*s_delay_alu\b")


def derive(control: list[str], drained: list[str]) -> tuple[list[str], dict[str, int]]:
    out: list[str] = []
    stats = {"copied": 0, "restored": 0, "nulled": 0}
    j = 0
    for i, line in enumerate(drained):
        if j < len(control) and line == control[j]:
            out.append(line)
            j += 1
            stats["copied"] += 1
            continue
        if not WAIT_ALU.match(line):
            raise RuntimeError(
                f"drained line {i + 1} differs from the control and is not a wait: {line!r}; "
                "the drained-everywhere transform only moves waits, so this is a contradiction "
                "and nothing may be written out"
            )
        if j < len(control) and WAIT_ALU.match(control[j]):
            # A wait the pass emitted, which the drained-everywhere transform rewrote to be
            # maximal. This control puts it back to its ORIGINAL value.
            out.append(control[j])
            j += 1
            stats["restored"] += 1
        else:
            out.append(NOWAIT)
            stats["nulled"] += 1
    if j != len(control):
        raise RuntimeError(
            f"only {j} of {len(control)} control lines were consumed; the two files do not align "
            "and the derivation is not trustworthy"
        )
    return out, stats


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} <control.s> <drained.s> <output.s>")
    control_path, drained_path, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(control_path) as handle:
        control = handle.readlines()
    with open(drained_path) as handle:
        drained = handle.readlines()

    out, stats = derive(control, drained)

    # Fail loudly rather than emitting a control that quietly is not one.
    if stats["nulled"] == 0:
        raise RuntimeError(f"{drained_path}: no added wait was found to null; this is not the drained variant")
    if stats["restored"] == 0:
        raise RuntimeError(f"{control_path}: no emitted wait was found to restore; this is not the expected assembly")

    def normalise(stream: list[str]) -> list[str]:
        return ["\tWAIT\n" if WAIT_ALU.match(l) else l for l in stream]

    # THE PERTURBATION IS IDENTICAL, checked rather than argued from the construction. With every
    # wait line replaced by the same placeholder, the null file and the drained file must be the
    # same file: same instruction count, same order, same registers, waits in the same places.
    if normalise(out) != normalise(drained):
        raise RuntimeError(
            f"{dst}: with wait lines normalised the null file differs from the drained file; the "
            "two do not perturb identically and the output must not be used"
        )
    # And stripped of waits entirely it must still be the control, which is the gate the previous
    # check passed -- carried forward so this object stands on its own rather than inheriting it.
    stripped_control = [l for l in control if not WAIT_ALU.match(l)]
    stripped_out = [l for l in out if not WAIT_ALU.match(l)]
    if stripped_control != stripped_out:
        raise RuntimeError(f"{dst}: stripped of waits the null file is not the control assembly")
    if not stripped_out:
        raise RuntimeError(
            f"{dst}: the stripped stream is empty, so the comparison above passed on two empty "
            "lists; that is a tool failure and not a result"
        )

    # THE FOURTEEN ARE THE ORIGINALS, compared against a populated list rather than by absence.
    original_waits = [l for l in control if WAIT_ALU.match(l)]
    restored_waits = [l for l in out if WAIT_ALU.match(l) and l != NOWAIT]
    if not original_waits:
        raise RuntimeError(f"{control_path}: the control carries no waits at all; nothing to restore")
    if original_waits != restored_waits:
        raise RuntimeError(
            f"{dst}: the restored waits are not the control's waits, in order; the waits the "
            "pass emitted must be left at their original values"
        )
    # Paired with its opposite: the nulled lines must exist and must all be the no-wait form.
    nulled = [l for l in out if l == NOWAIT]
    if len(nulled) != stats["nulled"] or not nulled:
        raise RuntimeError(f"{dst}: the no-wait line count does not agree with the derivation")

    # The hints are not disturbed, checked positively in both streams.
    hints_control = sum(1 for l in control if DELAY_ALU.match(l))
    hints_out = sum(1 for l in out if DELAY_ALU.match(l))
    if hints_control == 0 or hints_control != hints_out:
        raise RuntimeError(
            f"{dst}: s_delay_alu hints went from {hints_control} to {hints_out}; either they moved "
            "or there were none to compare, and neither is acceptable"
        )

    with open(dst, "w") as handle:
        handle.writelines(out)

    print(f"lines copied unchanged: {stats['copied']}")
    print(f"emitted waits restored to their original values: {stats['restored']}")
    print(f"added waits set to wait-for-nothing: {stats['nulled']}")
    print(f"wait lines in output: {sum(1 for l in out if WAIT_ALU.match(l))}")
    print(f"s_delay_alu hints: {hints_out}, unchanged from the control")
    print("normalised against the drained variant: identical, so the perturbation is the same")
    return 0


if __name__ == "__main__":
    sys.exit(main())
