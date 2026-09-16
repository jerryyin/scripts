#!/usr/bin/env python3
"""Drain a SELECTED SUBSET of wait sites, to bisect which sites actually matter.

Extends make-drained-everywhere.py / make-null-drain.py; see README.md for the chain.

THE QUESTION. Once make-null-drain.py has established that a fix comes from the added waits
WAITING rather than from the perturbation of adding ~833 instructions, some SUBSET of those
sites is what does the work. Naming that subset is the difference between telling a compiler
maintainer "a wait is missing somewhere" and telling them where. The route is bisection: drain
one half of the sites, leave the other half at wait-for-nothing so the instruction stream and
every address stay put, and measure. Repeat on whichever half keeps the behaviour.

WHAT THIS SCRIPT IS. The null-drain transform with a selection. Every added site is either the
six-field maximal drain (selected) or the wait-for-nothing form (not selected); the waits the
pass originally emitted keep their ORIGINAL values in every case, exactly as in the null-drain
object. The two ENDPOINTS of the selection reproduce objects the other two scripts produce,
which is the cheapest available check that this is the same instrument and not a new one:

    --sites 0:N     the DRAINED-EVERYWHERE object   (with one difference, see below)
    --sites 0:0     the NULL-DRAIN object            byte for byte

The parenthesis matters and is not a caveat about this script: the drained-everywhere transform
also rewrites the waits the pass emitted to be maximal, whereas from the null-drain control
onward those are left alone. So the full selection here is "drain every ADDED site", which is
the drained-everywhere object with the pass's own waits restored -- deliberately, because
holding them fixed across the whole bisection is what keeps every step differing from its
neighbours in added sites ALONE. A bisection whose steps differ in two things at once cannot
attribute the result to either.

WHY EVERY OBJECT IN THE NARROWING IS THE SAME SIZE. Both forms are one 32-bit SOPP instruction:
the maximal drain, every wait field at zero, assembles to 0x0080, and wait-for-nothing, every field
at its maximum, assembles to 0xff9f. Both were established from the target's own field table
(AMDGPUAsmUtils.cpp:66-74 in the build under test) and round-tripped through the assembler rather
than assumed. Substituting one for the other moves no address and changes no size, which is what
lets a narrowing step differ from its neighbour in immediates alone.

HOW THE FILE IS DERIVED, and this is the null-drain script's walk, unchanged in kind. It walks the
control assembly and the drained assembly together and rewrites only wait lines: a drained line
equal to its control line is copied; a drained wait whose control counterpart is also a wait is one
the pass emitted, restored to its original text; a drained wait with no control counterpart is one
the drained-everywhere transform inserted, and it becomes the drain or the no-wait form according
to the selection; anything else differing raises rather than being written out. Instruction count,
order and every address follow by construction.

SITE NUMBERING is by position in the instruction stream -- added site 0 is the first one in address
order, the last site is N-1. That is the bisection axis and the one any finding should be quoted at.

--SITES TAKES MORE THAN ONE RANGE, because a bisection converges on cuts that are not contiguous.
A real example: fifteen surviving sites split into seven AT the matrix instructions and eight at
the LDS loads between them; the eight were contiguous (524:532) and the seven were not (521:524
and 532:536). Rather than a second selection mechanism beside this one, --sites accepts a
comma-separated list of half-open ranges: `--sites 521:524,532:536`. A single
range behaves exactly as before -- proven by regenerating an already-measured object and comparing it
to the archived file byte for byte, not argued from the diff. Ranges must be ascending and disjoint;
overlapping or out-of-order ranges refuse rather than being normalised, because a selection that
quietly means something other than what it says is the failure this whole route cannot afford.
"""

import argparse
import re
import sys

# Every wait field at ZERO: wait for everything, in all six dependency classes. Encodes to 0x0080.
DRAIN = (
    "\ts_wait_alu depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) "
    "depctr_va_vcc(0) depctr_vm_vsrc(0) depctr_sa_sdst(0)\n"
)

# Every wait field at its MAXIMUM, which the field table also gives as its default: wait for
# nothing. Encodes to 0xff9f.
NOWAIT = (
    "\ts_wait_alu depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) "
    "depctr_va_vcc(1) depctr_vm_vsrc(7) depctr_sa_sdst(1)\n"
)

WAIT_ALU = re.compile(r"^\s*s_wait_alu\b")
DELAY_ALU = re.compile(r"^\s*s_delay_alu\b")


def classify(control: list[str], drained: list[str]) -> list[tuple[str, str]]:
    """Walk the two assemblies together and label every drained line.

    Returns one (kind, text) per line of the drained file, where kind is "copy" for a line the two
    files share, "restore" for one of the waits the pass emitted, and "added" for a wait the
    drained-everywhere transform inserted. Nothing else can occur: a drained line that differs from
    its control counterpart and is not a wait would contradict that transform, and raises.
    """
    labelled: list[tuple[str, str]] = []
    j = 0
    for i, line in enumerate(drained):
        if j < len(control) and line == control[j]:
            labelled.append(("copy", line))
            j += 1
            continue
        if not WAIT_ALU.match(line):
            raise RuntimeError(
                f"drained line {i + 1} differs from the control and is not a wait: {line!r}; "
                "the drained-everywhere gate said only waits moved, so this is a contradiction "
                "and nothing may be written out"
            )
        if j < len(control) and WAIT_ALU.match(control[j]):
            labelled.append(("restore", control[j]))
            j += 1
        else:
            labelled.append(("added", line))
    if j != len(control):
        raise RuntimeError(
            f"only {j} of {len(control)} control lines were consumed; the two files do not align "
            "and the derivation is not trustworthy"
        )
    return labelled


def normalise(stream: list[str]) -> list[str]:
    return ["\tWAIT\n" if WAIT_ALU.match(line) else line for line in stream]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", help="the unmodified assembly")
    parser.add_argument("drained", help="the drained-everywhere assembly")
    parser.add_argument("output", help="where to write the partial-drain assembly")
    parser.add_argument(
        "--sites",
        required=True,
        help="half-open range LO:HI over the added sites, in address order, that gets the maximal "
        "drain; every other added site gets wait-for-nothing. A comma-separated list of ranges is "
        "accepted for a selection that is not contiguous, e.g. 521:524,532:536; the ranges must be "
        "ascending and disjoint",
    )
    args = parser.parse_args()

    ranges: list[tuple[int, int]] = []
    for piece in args.sites.split(","):
        match = re.fullmatch(r"(\d+):(\d+)", piece)
        if not match:
            raise SystemExit(
                f"--sites must be LO:HI, or a comma-separated list of them, got {args.sites!r}"
            )
        lo, hi = int(match.group(1)), int(match.group(2))
        if lo > hi:
            raise SystemExit(f"--sites LO must not exceed HI, got {piece!r}")
        if ranges and lo < ranges[-1][1]:
            raise SystemExit(
                f"--sites ranges must be ascending and disjoint; {piece!r} starts before the "
                f"previous range ended at {ranges[-1][1]}.  Refusing rather than normalising: a "
                "selection that quietly means something other than what it says cannot be gated"
            )
        ranges.append((lo, hi))
    selected_width = sum(hi - lo for lo, hi in ranges)
    hi_max = max(hi for _, hi in ranges)

    with open(args.control) as handle:
        control = handle.readlines()
    with open(args.drained) as handle:
        drained = handle.readlines()

    labelled = classify(control, drained)
    total_added = sum(1 for kind, _ in labelled if kind == "added")
    if total_added == 0:
        raise RuntimeError(f"{args.drained}: no added wait was found; this is not the drained variant")
    if hi_max > total_added:
        raise SystemExit(
            f"--sites HI is {hi_max} but there are only {total_added} added sites; refusing rather "
            "than silently clamping a range that does not mean what it says"
        )

    out: list[str] = []
    drained_sites: list[int] = []
    nowait_sites: list[int] = []
    site = 0
    restored = 0
    for kind, text in labelled:
        if kind == "added":
            if any(lo <= site < hi for lo, hi in ranges):
                out.append(DRAIN)
                drained_sites.append(site)
            else:
                out.append(NOWAIT)
                nowait_sites.append(site)
            site += 1
        else:
            out.append(text)
            if kind == "restore":
                restored += 1

    # THE PERTURBATION IS IDENTICAL, checked rather than argued from the construction: with every
    # wait line replaced by one placeholder this file must be the drained file, so every object in
    # the narrowing has the same instruction count, the same order and the same addresses.
    if normalise(out) != normalise(drained):
        raise RuntimeError(
            f"{args.output}: with wait lines normalised this file differs from the drained file; "
            "the narrowing step does not perturb identically and must not be used"
        )
    # And stripped of waits entirely it must still be the control.
    stripped_control = [line for line in control if not WAIT_ALU.match(line)]
    stripped_out = [line for line in out if not WAIT_ALU.match(line)]
    if stripped_control != stripped_out:
        raise RuntimeError(f"{args.output}: stripped of waits this file is not the control assembly")
    if not stripped_out:
        raise RuntimeError(
            f"{args.output}: the stripped stream is empty, so the comparison above passed on two "
            "empty lists; that is a tool failure and not a result"
        )

    # THE FOURTEEN ARE THE ORIGINALS, compared against a populated list rather than by absence.
    original_waits = [line for line in control if WAIT_ALU.match(line)]
    restored_waits = [
        line for line in out if WAIT_ALU.match(line) and line not in (DRAIN, NOWAIT)
    ]
    if not original_waits:
        raise RuntimeError(f"{args.control}: the control carries no waits at all; nothing to restore")
    if original_waits != restored_waits:
        raise RuntimeError(
            f"{args.output}: the restored waits are not the control's waits, in order; the "
            "fourteen the pass emitted are held at their original values at every step"
        )
    if restored != len(original_waits):
        raise RuntimeError(f"{args.output}: the restored count does not agree with the walk")

    # The selection is the selection, counted in the output rather than assumed from the range.
    if sum(1 for line in out if line == DRAIN) != len(drained_sites):
        raise RuntimeError(f"{args.output}: the drained-site count does not agree with the selection")
    if sum(1 for line in out if line == NOWAIT) != len(nowait_sites):
        raise RuntimeError(f"{args.output}: the no-wait-site count does not agree with the selection")
    if len(drained_sites) + len(nowait_sites) != total_added:
        raise RuntimeError(f"{args.output}: the two site groups do not account for every added site")
    # Paired with its opposite so neither check can pass by matching nothing: at a genuine halving
    # step both groups are populated, and an empty one means the range missed.
    if selected_width > 0 and not drained_sites:
        raise RuntimeError(f"{args.output}: --sites {args.sites} selected no site at all")
    if selected_width < total_added and not nowait_sites:
        raise RuntimeError(f"{args.output}: --sites {args.sites} left no site at wait-for-nothing")

    # The hints are not disturbed, checked positively in both streams.
    hints_control = sum(1 for line in control if DELAY_ALU.match(line))
    hints_out = sum(1 for line in out if DELAY_ALU.match(line))
    if hints_control == 0 or hints_control != hints_out:
        raise RuntimeError(
            f"{args.output}: s_delay_alu hints went from {hints_control} to {hints_out}; either "
            "they moved or there were none to compare, and neither is acceptable"
        )

    with open(args.output, "w") as handle:
        handle.writelines(out)

    print(f"added sites in total: {total_added}")
    print(f"sites given the maximal drain: {len(drained_sites)}  (--sites {args.sites})")
    print(f"sites left at wait-for-nothing: {len(nowait_sites)}")
    print(f"emitted waits held at their original values: {restored}")
    print(f"wait lines in output: {sum(1 for line in out if WAIT_ALU.match(line))}")
    print(f"s_delay_alu hints: {hints_out}, unchanged from the control")
    print("normalised against the drained variant: identical, so the perturbation is the same")
    if drained_sites:
        print(f"first and last drained site index: {drained_sites[0]}, {drained_sites[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
