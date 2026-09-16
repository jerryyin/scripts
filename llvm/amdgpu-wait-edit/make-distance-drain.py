#!/usr/bin/env python3
"""Drain va_vdst to a given DISTANCE (0-15): how little of that class is enough?

Extends make-class-drain.py; see README.md for the chain. This is the last step, and the one
that produces something a compiler maintainer can act on directly.

THE QUESTION. Once make-class-drain.py has established that a wait stopping for va_vdst and no
other class is sufficient, and that no set omitting va_vdst is, the result is still not
actionable: those variants stop for va_vdst COMPLETELY, the field at zero, its strongest
setting. No compiler pass emits a maximal drain -- a pass emits a DISTANCE. So the question is
whether sufficiency needs the field at zero, or whether some larger value (stop for fewer
outstanding results, not for all of them) also suffices, and where the boundary lies.

THE AXIS IS BOUNDED BY THE ENCODING ITSELF AND NOT BY A JUDGEMENT. va_vdst is the one field of the
six that takes sixteen values, 0 through 15, and its maximum is also its default and means "do not
wait for this class at all". So the axis has exactly sixteen points and both of its ENDPOINTS ARE
ALREADY MEASURED OBJECTS:

    distance 0   IS the only:va_vdst form     (make-class-drain.py --wait-form only:va_vdst)
    distance 15  IS the wait-for-nothing form (make-class-drain.py --wait-form none)

That gives a free correctness gate, and it is worth running rather than assuming: over the same
site range, `--wait-form dist:0` must reproduce the only:va_vdst object byte for byte, and
`--wait-form dist:15` must reproduce the null-drain object byte for byte. If either drifts, the
distance axis is no longer the same axis as the class axis and nothing along it is comparable.

WHY THIS IS A SEPARATE FILE AND NOT AN EDIT, and why it is not a new instrument.
make-class-drain.py has already produced measured objects, and those must stay reproducible from
that script exactly as it stands -- the same reason make-class-drain.py sits beside
make-half-drain.py rather than widening it.
Its form vocabulary refuses anything but 'all', 'none', 'only:CLASS' and 'allbut:CLASS', and it
refuses by raising rather than by defaulting, which is the behaviour one wants and is not a
behaviour to soften.

So this file ADDS ONE DEGREE OF FREEDOM TO AN EXISTING INSTRUMENT AND NOTHING ELSE, by the narrowest
route available: it imports make-class-drain.py, extends its form vocabulary with 'dist:N', and then
runs ITS main() unchanged. Every check in that file -- the perturbation is identical under
normalisation, the stripped stream is the control, the waits the pass emitted are held at their
original values and in order, the site groups account for every added site, the s_delay_alu hints
are unmoved -- runs here exactly as written, on the same code, not on a copy of it. Nothing is
reimplemented, so nothing can drift from what it reimplements.

"DISTANCE d" MEANS va_vdst AT d AND THE OTHER FIVE CLASSES AT THEIR MAXIMUM, which is their default
and means do not wait for them. It is one SOPP instruction, the same opcode and the same single
32-bit word as every other form these scripts emit, so substituting any distance for any other moves
no address and changes no size. Assembled for gfx1250 the immediate is the only:va_vdst immediate
with the field written into its top four bits; the launcher reads all sixteen back out of the
object's .text rather than predicting them, and derives the field's mask from the class forms rather
than hard-coding 0xf000.

WHAT THIS SCRIPT DOES NOT DO. It does not touch the waits the pass emitted: those keep their
original values at every site. A distance variant is the pass's own output with class-confined,
distance-confined waits ADDED -- never the pass's output tightened. Nothing here modifies,
replaces or reasons about what the pass chose to emit, and sufficiency is never necessity.
"""

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
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# The walk, the classifier, the site numbering, the form vocabulary, every check and main() itself
# are the class transform's, imported rather than reimplemented, so a site index and a variant's
# structural guarantees mean the same thing in both scripts by construction.
_mcd = _load_sibling("make_class_drain", "make-class-drain.py")

# Every public name of the class transform is RE-EXPORTED, not reimplemented, so a launcher that
# imports this module reads the same table, the same classifier and the same renderer that emits
# the object -- one behaviour, no second copy to drift. form_values is overridden below and is the
# only name whose behaviour differs; it is re-installed into the imported module afterwards so that
# its main() sees the extended vocabulary too.
for _name in dir(_mcd):
    if not _name.startswith("_"):
        globals().setdefault(_name, getattr(_mcd, _name))

FIELDS = _mcd.FIELDS
CLASSES = _mcd.CLASSES
VA_VDST_INDEX = CLASSES.index("va_vdst")
VA_VDST_MAX = FIELDS[VA_VDST_INDEX][1]

_base_form_values = _mcd.form_values


def form_values(form: str) -> list[int]:
    """'dist:N' for N in 0..15; everything else is handed to the class transform UNCHANGED.

    Out-of-range and malformed distances refuse rather than clamping, for the reason the site
    ranges refuse rather than clamping: a selection that quietly means something other than what it
    says cannot be gated.
    """
    match = re.fullmatch(r"dist:(\d+)", form)
    if not match:
        return _base_form_values(form)
    distance = int(match.group(1))
    if distance > VA_VDST_MAX:
        raise SystemExit(
            f"--wait-form dist:{distance} is outside the field: va_vdst takes {VA_VDST_MAX + 1} "
            f"values, 0 through {VA_VDST_MAX}, and {VA_VDST_MAX} is its maximum and its default. "
            "Refusing rather than clamping to a distance you did not ask for"
        )
    values = [maximum for _, maximum in FIELDS]
    values[VA_VDST_INDEX] = distance
    return values


# The one degree of freedom, installed into the imported module so that ITS main() -- and therefore
# every check in it, unmodified -- sees the extended vocabulary. main() resolves form_values as a
# module global, so this is the whole of the change and there is no second copy of anything.
_mcd.form_values = form_values

if __name__ == "__main__":
    sys.exit(_mcd.main())
