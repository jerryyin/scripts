#!/usr/bin/env python3
"""Insert a maximal dependency drain before EVERY instruction in a gfx1250 function.

The base transform of this directory. See README.md for how the five scripts relate.

THE QUESTION IT ANSWERS: can a wait set that is maximally conservative AND placed wherever it
might be needed clear a suspected missing-wait bug? Merely strengthening the waits a compiler
pass already chose to emit cannot answer that -- it says nothing about a site where the pass
emits no wait at all, which is exactly where a missing wait would be.

So this does not guess where a wait is needed. It puts a maximal drain before EVERY instruction
in the function, which is a superset of every site any definition of "needed" could pick out,
and it rewrites the fourteen waits the pass emitted to be maximal too.

THE MAXIMAL DRAIN, established from the target definition rather than assumed.
AMDGPUAsmUtils.cpp:66-74 gives each depctr field's default, maximum, shift and width:

    depctr_va_vdst   default 15  max 15  shift 12  width 4
    depctr_va_sdst   default  7  max  7  shift  9  width 3
    depctr_va_ssrc   default  1  max  1  shift  8  width 1
    depctr_va_vcc    default  1  max  1  shift  1  width 1
    depctr_vm_vsrc   default  7  max  7  shift  2  width 3
    depctr_sa_sdst   default  1  max  1  shift  0  width 1
    depctr_hold_cnt  default  1  max  1  shift  7  width 1

For every WAIT field the default equals the maximum, which is the "wait for nothing" value, so
zero is the strongest wait each field can express. Setting all six to zero encodes to 0x0080.

`depctr_hold_cnt` is deliberately left alone. It is not a wait -- it is a separate control over
counter behaviour -- and driving a non-wait field to a non-default value would be a second edit
riding along with this one, which is exactly what the gate exists to prevent. Zeroing it as well
would encode 0x0000; this variant does not do that.

TWO PLACES WHERE A DRAIN IS PUT BEFORE A GROUP RATHER THAN INSIDE IT. These are real limits of
the transform, stated rather than hidden:

  - `s_clause 0x1` declares that the next two instructions issue as one clause. Inserting a wait
    between them would break the clause, and breaking it would change the non-wait instruction
    stream -- which would void the whole comparison. The drain goes before the `s_clause`.
  - `s_delay_alu` is a hint about the instruction that follows it. Inserting a drain between the
    hint and its target would retarget the hint at the drain. The drain goes before the hint, so
    the hint still reaches the instruction it was emitted for.

Both are cases of putting the drain EARLIER, never of omitting one, so the wait set stays a
superset of anything the pass could have emitted.

Nothing but wait lines is added, removed, reordered or rewritten. Verify that independently by
stripping every wait line from both assemblies and requiring the remainder to be identical --
that check is the whole reason this transform can be trusted, so run it, do not assume it.
"""

import re
import sys

# Every wait field at zero; hold_cnt left at its default. Encodes to 0x0080.
DRAIN = (
    "\ts_wait_alu depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) "
    "depctr_va_vcc(0) depctr_vm_vsrc(0) depctr_sa_sdst(0)\n"
)

INSTRUCTION = re.compile(r"^\t[a-z]")
CLAUSE = re.compile(r"^\ts_clause\s+(0x[0-9a-fA-F]+|\d+)\s*$")
DELAY_ALU = re.compile(r"^\ts_delay_alu\b")
WAIT_ALU = re.compile(r"^\ts_wait_alu\b")
LABEL = re.compile(r"^([^\s:]+):")
BRANCH = re.compile(r"^\t(s_c?branch[a-z0-9_]*|s_call[a-z0-9_]*|s_setpc[a-z0-9_]*)\s+(.*)$")


def branch_targets(lines: list[str]) -> set[str]:
    """Labels that control flow can actually arrive at, read from the branch operands.

    Not every label is a branch target. This assembly is full of `.Ltmp<N>:` markers emitted for
    the DWARF line table, and nothing jumps to them. The distinction matters because a drain
    placed before a hint sits before any label between them: if that label is a real branch
    target, an arriving branch would skip the drain, so the group may not be extended across it.
    If it is a line-table marker, nothing arrives there and the drain is unconditionally reached.
    Derived from the file rather than from a naming convention.
    """
    targets: set[str] = set()
    for line in lines:
        if not INSTRUCTION.match(line):
            continue
        match = BRANCH.match(line)
        if match:
            targets.update(re.findall(r"\.L[A-Za-z0-9_.$]+", match.group(2)))
    return targets


def transform(lines: list[str], targets: set[str]) -> tuple[list[str], dict[str, int]]:
    out: list[str] = []
    stats = {"inserted": 0, "rewritten": 0, "clause_members_skipped": 0, "instructions": 0}
    clause_remaining = 0
    # Index into `out` at which a drain may be placed to precede a pending s_delay_alu group.
    pending_group_start: int | None = None

    for line in lines:
        if not INSTRUCTION.match(line):
            out.append(line)
            # A LABEL ends the adjacency: a branch landing there would jump past a drain placed
            # before the hint, so the group cannot be extended across it. Directives that emit no
            # instruction -- .loc, .cfi_*, line comments -- do NOT end it. This distinction is
            # load-bearing: `.loc` lines sit between almost every hint and its target in this
            # assembly, and treating them as separators put a drain between the two.
            label = LABEL.match(line)
            if label and label.group(1) in targets:
                pending_group_start = None
            continue

        stats["instructions"] += 1

        if WAIT_ALU.match(line):
            # A wait the pass emitted. Make it maximal; do not stack another in front of it.
            out.append(DRAIN)
            stats["rewritten"] += 1
            pending_group_start = None
            continue

        if clause_remaining > 0:
            # Inside a declared clause: the drain already went in before the s_clause.
            out.append(line)
            clause_remaining -= 1
            stats["clause_members_skipped"] += 1
            pending_group_start = None
            continue

        insert_at = len(out) if pending_group_start is None else pending_group_start
        out.insert(insert_at, DRAIN)
        stats["inserted"] += 1
        out.append(line)

        clause = CLAUSE.match(line)
        if clause:
            clause_remaining = int(clause.group(1), 0) + 1
            pending_group_start = None
        elif DELAY_ALU.match(line):
            # The next instruction belongs with this hint; a drain for it goes before the hint.
            pending_group_start = len(out) - 1
        else:
            pending_group_start = None

    return out, stats


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <input.s> <output.s>")
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as handle:
        lines = handle.readlines()

    targets = branch_targets(lines)
    if not targets:
        raise RuntimeError(
            f"{src}: no branch target was found anywhere, so every label would be treated as "
            "unreachable; that is a parse failure and not a property of this kernel"
        )
    out, stats = transform(lines, targets)

    with open(dst, "w") as handle:
        handle.writelines(out)

    # Fail loudly rather than producing a variant that quietly did nothing useful.
    if stats["inserted"] == 0:
        raise RuntimeError(f"{src}: no drain was inserted anywhere; the transform did nothing")
    if stats["rewritten"] == 0:
        raise RuntimeError(f"{src}: no existing wait was rewritten; the input is not the expected assembly")

    stripped_src = [l for l in lines if not WAIT_ALU.match(l)]
    stripped_out = [l for l in out if not WAIT_ALU.match(l)]
    if stripped_src != stripped_out:
        raise RuntimeError(
            f"{src}: with every s_wait_alu line removed the two streams still differ; "
            "the transform moved something other than waits and its output must not be used"
        )

    # WHETHER A HINT REACHES ITS TARGET IS COMPARED AGAINST THE CONTROL, not assumed.
    # The pass itself emits `s_delay_alu` immediately followed by `s_wait_alu` in several
    # places, so "a hint followed by a wait" is a shape the control already has and is not
    # evidence that this transform introduced one. What must hold is that the relationship is
    # UNCHANGED hint by hint: a hint the pass left touching its target still touches it, and a
    # hint the pass had already separated is separated by the same kind of line.
    def hint_followed_by_wait(stream: list[str]) -> list[bool]:
        flags = []
        for i, text in enumerate(stream):
            if not DELAY_ALU.match(text):
                continue
            for nxt in stream[i + 1:]:
                if INSTRUCTION.match(nxt):
                    flags.append(bool(WAIT_ALU.match(nxt)))
                    break
            else:
                flags.append(False)
        return flags

    before = hint_followed_by_wait(lines)
    after = hint_followed_by_wait(out)
    if len(before) != len(after):
        raise RuntimeError(f"{src}: the hint count changed, {len(before)} to {len(after)}")
    newly_separated = sum(1 for b, a in zip(before, after) if a and not b)
    if newly_separated:
        raise RuntimeError(
            f"{src}: {newly_separated} s_delay_alu hint(s) that reached their target in the "
            "control are separated from it by an inserted drain; that retargets the hint and "
            "the output must not be used"
        )
    # The check above is satisfied by an empty hint list, so pair it with one whose correct
    # answer is the opposite: hints that do reach their target must exist, in both streams.
    reaching_before = sum(1 for b in before if not b)
    reaching_after = sum(1 for a in after if not a)
    if reaching_before == 0 or reaching_after == 0:
        raise RuntimeError(
            f"{src}: no hint reaches its target in one of the streams (control {reaching_before}, "
            "variant {reaching_after}); the comparison above would pass vacuously, which is a "
            "tool failure and not a result"
        )
    print(f"delay-alu hints: {len(after)} total, {reaching_after} reaching their target, "
          f"unchanged from the control")
    print(f"branch targets found: {len(targets)}")

    for key in ("instructions", "inserted", "rewritten", "clause_members_skipped"):
        print(f"{key}: {stats[key]}")
    print(f"wait lines in output: {sum(1 for l in out if WAIT_ALU.match(l))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
