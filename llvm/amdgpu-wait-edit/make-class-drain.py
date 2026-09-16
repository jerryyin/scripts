#!/usr/bin/env python3
"""Drain ONE named dependency class at selected sites: which of the six is doing the work?

Extends make-half-drain.py; see README.md for the chain.

THE QUESTION. Bisecting with make-half-drain.py narrows a missing wait down to a POSITION in the
assembly stream. A compiler maintainer cannot act on a position -- they can act on a dependency
CLASS. A maximal drain waits on all SIX classes at once, so every clean result from the earlier
scripts is silent about which of the six is carrying it. This transform makes that question
askable: at a selected site it emits a wait that stops for one named class and for nothing else.

WHY THIS IS A SEPARATE FILE AND NOT AN EDIT, and why it is not a new instrument.
make-half-drain.py answers "which SITES" and cannot answer "which CLASS" by construction: it
holds exactly two wait forms as literal strings, all-six-at-zero and all-six-at-maximum, so every
object it can emit drains all six classes or none. Widening it would also change a script that
has already produced measured objects, and those must stay reproducible from the script as it
stood. So the walk, the classifier and the site numbering are IMPORTED from it rather than
reimplemented: this file adds one degree of freedom to an existing instrument and nothing else.

The real proof that it is the same instrument is not the shared import but reproduction at the
endpoints -- `--wait-form all` over a site range must reproduce the all-drain object for that
range byte for byte, and `--wait-form none --sites 0:0` must reproduce the null-drain object.
Check that whenever you change any of these scripts.

THE FORMS, AND THE FACT THAT THEY ARE ROUND-TRIPPED RATHER THAN ASSUMED. The six fields and their
maxima come from the target's own table, AMDGPUAsmUtils.cpp:66-74 in the build under test, and the
assembled immediate of every form this script can emit is read back out of the object's .text by
reading the assembled object rather than predicted. Assembled with clang for gfx1250 the fourteen
forms come out as, all one opcode 0xbf88 and all one 32-bit word:

    all                  0x0080      none                 0xff9f
    only:va_vdst         0x0f9f      allbut:va_vdst       0xf080
    only:va_sdst         0xf19f      allbut:va_sdst       0x0e80
    only:va_ssrc         0xfe9f      allbut:va_ssrc       0x0180
    only:va_vcc          0xff9d      allbut:va_vcc        0x0082
    only:vm_vsrc         0xff83      allbut:vm_vsrc       0x009c
    only:sa_sdst         0xff9e      allbut:sa_sdst       0x0081

Two properties of that table are checks and not decoration. The six per-class bit masks are
disjoint and together cover every bit that varies -- 0xf000, 0x0e00, 0x0100, 0x001c, 0x0002, 0x0001,
union 0xff1f -- while bit 7 is set in all fourteen and belongs to no field. And for every class X,
only:X XOR allbut:X equals all XOR none, which is what "the singleton and the complement are
opposite halves of the same six" means at the encoding level. Neither property was assumed; both
are computed from the immediates read back out of the assembled object.

"ONLY X" MEANS X AT ZERO AND THE OTHER FIVE AT THEIR MAXIMUM, where the maximum is also the field's
default and means do not wait for that class. "ALLBUT X" is the reverse. Both are still one SOPP
instruction, so substituting any form for any other moves no address and changes no size, exactly
as in the site narrowing.

WHAT THIS SCRIPT DOES NOT DO. It does not touch the waits the pass emitted: those keep their
original values at every site, including any relaxed wait at the head of the selected group. A
class-confined variant is the pass's own wait left exactly as it is, with class-confined waits
ADDED below it. Nothing here modifies, replaces or reasons about what the pass chose to emit,
and sufficiency of an added wait is never evidence of necessity.
"""

import argparse
import importlib.util
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

def _load_sibling(module_name: str, filename: str):
    """Load a sibling script as a module, by path, from wherever this file lives.

    These scripts have hyphens in their names, so they cannot be plain imports. The path is
    derived from __file__ rather than from the working directory or an installed location,
    so the chain keeps working wherever the directory is checked out or copied to.
    """
    path = HERE / filename
    if not path.is_file():
        raise SystemExit(
            f"{pathlib.Path(__file__).name} extends {filename}, which must sit beside it. "
            f"Expected it at {path} and it is not there."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path} as a module")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so that anything inside it which looks itself up by name -- a
    # dataclass resolving annotations, for instance -- finds a module rather than None.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# The site numbering, the walk and the classifier are the bisection transform's, imported rather
# than reimplemented, so a site index means the same thing in both scripts by construction.
_mhd = _load_sibling("make_half_drain", "make-half-drain.py")
classify = _mhd.classify
normalise = _mhd.normalise
WAIT_ALU = _mhd.WAIT_ALU
DELAY_ALU = _mhd.DELAY_ALU

# Field order and maxima from AMDGPUAsmUtils.cpp:66-74 in the build under test. The maximum is also
# the default, and means "do not wait for this class".
FIELDS: list[tuple[str, int]] = [
    ("va_vdst", 15),
    ("va_sdst", 7),
    ("va_ssrc", 1),
    ("va_vcc", 1),
    ("vm_vsrc", 7),
    ("sa_sdst", 1),
]
CLASSES = [name for name, _ in FIELDS]


def wait_text(values: list[int]) -> str:
    if len(values) != len(FIELDS):
        raise RuntimeError("a wait form must give a value for every one of the six classes")
    fields = " ".join(f"depctr_{name}({v})" for (name, _), v in zip(FIELDS, values))
    return f"\ts_wait_alu {fields}\n"


def form_values(form: str) -> list[int]:
    """Turn a form name into its six field values. Unknown names refuse rather than defaulting."""
    maxima = [m for _, m in FIELDS]
    if form == "all":
        return [0] * len(FIELDS)
    if form == "none":
        return list(maxima)
    match = re.fullmatch(r"(only|allbut):(\w+)", form)
    if not match:
        raise SystemExit(
            f"--wait-form must be 'all', 'none', 'only:CLASS' or 'allbut:CLASS', got {form!r}; "
            f"the classes are {', '.join(CLASSES)}"
        )
    kind, name = match.group(1), match.group(2)
    if name not in CLASSES:
        raise SystemExit(
            f"{name!r} is not one of the six dependency classes ({', '.join(CLASSES)}); refusing "
            "rather than guessing which was meant"
        )
    index = CLASSES.index(name)
    if kind == "only":
        values = list(maxima)
        values[index] = 0
    else:
        values = [0] * len(FIELDS)
        values[index] = maxima[index]
    return values


def parse_ranges(text: str) -> list[tuple[int, int]]:
    """Half-open ranges over the added sites, ascending and disjoint, refusing anything else.

    Same rule as the narrowing transform's --sites: a selection that quietly means something other
    than what it says cannot be gated.
    """
    ranges: list[tuple[int, int]] = []
    for piece in text.split(","):
        match = re.fullmatch(r"(\d+):(\d+)", piece)
        if not match:
            raise SystemExit(f"--sites must be LO:HI, or a comma-separated list of them, got {text!r}")
        lo, hi = int(match.group(1)), int(match.group(2))
        if lo > hi:
            raise SystemExit(f"--sites LO must not exceed HI, got {piece!r}")
        if ranges and lo < ranges[-1][1]:
            raise SystemExit(
                f"--sites ranges must be ascending and disjoint; {piece!r} starts before the "
                f"previous range ended at {ranges[-1][1]}"
            )
        ranges.append((lo, hi))
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", help="the unmodified assembly")
    parser.add_argument("drained", help="the drained-everywhere assembly")
    parser.add_argument("output", help="where to write the class-confined-drain assembly")
    parser.add_argument(
        "--sites",
        required=True,
        help="half-open range LO:HI over the added sites, in address order, that gets the selected "
        "wait form; every other added site gets wait-for-nothing",
    )
    parser.add_argument(
        "--wait-form",
        required=True,
        help="what goes at the selected sites: 'all' (every class, the maximal drain), 'none' "
        "(no class, the wait-for-nothing form), 'only:CLASS' (that class and no other), or "
        "'allbut:CLASS' (the other five and not that one)",
    )
    args = parser.parse_args()

    ranges = parse_ranges(args.sites)
    selected_width = sum(hi - lo for lo, hi in ranges)
    hi_max = max(hi for _, hi in ranges)
    selected_text = wait_text(form_values(args.wait_form))
    nowait_text = wait_text([m for _, m in FIELDS])

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
    selected_sites: list[int] = []
    nowait_sites: list[int] = []
    site = 0
    restored = 0
    for kind, text in labelled:
        if kind == "added":
            if any(lo <= site < hi for lo, hi in ranges):
                out.append(selected_text)
                selected_sites.append(site)
            else:
                out.append(nowait_text)
                nowait_sites.append(site)
            site += 1
        else:
            out.append(text)
            if kind == "restore":
                restored += 1

    # THE PERTURBATION IS IDENTICAL, checked rather than argued: with every wait line replaced by
    # one placeholder this file must be the drained file, so every variant has the same
    # instruction count, the same order and the same addresses as every other and as the control.
    if normalise(out) != normalise(drained):
        raise RuntimeError(
            f"{args.output}: with wait lines normalised this file differs from the drained file; "
            "the arm does not perturb identically and must not be used"
        )
    stripped_control = [line for line in control if not WAIT_ALU.match(line)]
    stripped_out = [line for line in out if not WAIT_ALU.match(line)]
    if stripped_control != stripped_out:
        raise RuntimeError(f"{args.output}: stripped of waits this file is not the control assembly")
    if not stripped_out:
        raise RuntimeError(
            f"{args.output}: the stripped stream is empty, so the comparison above passed on two "
            "empty lists; that is a tool failure and not a result"
        )

    # THE WAITS THE PASS EMITTED ARE THE ORIGINALS, compared against a populated list rather than
    # by absence. This is what makes every arm "the pass's own wait, with something added below it".
    original_waits = [line for line in control if WAIT_ALU.match(line)]
    restored_waits = [
        line for line in out if WAIT_ALU.match(line) and line not in (selected_text, nowait_text)
    ]
    if not original_waits:
        raise RuntimeError(f"{args.control}: the control carries no waits at all; nothing to restore")
    if original_waits != restored_waits:
        raise RuntimeError(
            f"{args.output}: the restored waits are not the control's waits, in order; the waits "
            "the pass emitted are held at their original values in every arm"
        )
    if restored != len(original_waits):
        raise RuntimeError(f"{args.output}: the restored count does not agree with the walk")

    # The selection is counted in the output rather than assumed from the range. When the selected
    # form IS the wait-for-nothing form these two counts cannot be separated, and the script says so
    # instead of reporting a number it cannot stand behind.
    forms_coincide = selected_text == nowait_text
    if not forms_coincide:
        if sum(1 for line in out if line == selected_text) != len(selected_sites):
            raise RuntimeError(f"{args.output}: the selected-site count does not agree with the selection")
        if sum(1 for line in out if line == nowait_text) != len(nowait_sites):
            raise RuntimeError(f"{args.output}: the no-wait-site count does not agree with the selection")
    elif sum(1 for line in out if line == nowait_text) != total_added:
        raise RuntimeError(f"{args.output}: with a 'none' selection every added site must be wait-for-nothing")
    if len(selected_sites) + len(nowait_sites) != total_added:
        raise RuntimeError(f"{args.output}: the two site groups do not account for every added site")
    # Paired with its opposite so neither check can pass by matching nothing.
    if selected_width > 0 and not selected_sites:
        raise RuntimeError(f"{args.output}: --sites {args.sites} selected no site at all")
    if selected_width < total_added and not nowait_sites:
        raise RuntimeError(f"{args.output}: --sites {args.sites} left no site at wait-for-nothing")

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
    print(f"wait form at the selected sites: {args.wait_form}  ->  {selected_text.strip()}")
    print(f"sites given that form: {len(selected_sites)}  (--sites {args.sites})")
    print(f"sites left at wait-for-nothing: {len(nowait_sites)}"
          + ("  (the selected form IS wait-for-nothing, so the two groups are not distinguishable "
             "in the output and were not counted separately)" if forms_coincide else ""))
    print(f"waits the pass emitted, held at their original values: {restored}")
    print(f"wait lines in output: {sum(1 for line in out if WAIT_ALU.match(line))}")
    print(f"s_delay_alu hints: {hints_out}, unchanged from the control")
    print("normalised against the drained variant: identical, so the perturbation is the same")
    if selected_sites:
        print(f"first and last selected site index: {selected_sites[0]}, {selected_sites[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
