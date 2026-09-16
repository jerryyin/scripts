#!/usr/bin/env python3
"""Two tools answer "did every run produce the same bytes". This checks they still agree.

The same judgement is implemented twice in this repo, and for a good reason:

  - ../llvm/amdgpu-repro/driver/replay.cpp hashes a device buffer in-process. It has to.
    The output is roughly 10 MB per run, so hashing 120 runs through a file-based tool
    would mean writing over a gigabyte of blobs to disk for no other purpose.
  - numerics/distinct_hashes.py hashes files that already landed on disk, needs no device
    and no numpy, and runs wherever the outputs are.

Neither can be deleted, so the risk is not duplicated code -- it is the two drifting apart
in MEANING. That already happened once: replay.cpp stated its rule unconditionally and had
no guard on the run count, so `--runs 1` would print "1 distinct output hash" and, by its
own stated criterion, read as proof of determinism from a single launch. distinct_hashes.py
refused that case by name. The two tools disagreed about what the measurement means, which
is worse than either being wrong alone, because a reader would trust whichever they ran.

A pre-commit hook can refuse an EDIT to a duplicated file. It cannot see two
implementations of one rule diverge in meaning. That is what this test is for.

Standard library only, and it launches nothing: it drives distinct_hashes.py on synthesised
files and reads replay.cpp and summarize_replay_ab.py as text. Checking a C++ driver by
reading it is weaker than running it, and is what is available without hipcc or a GPU --
each such check says so where it is asserted.

Run:  python3 test_determinism_rule_agreement.py
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
COUNTER = HERE / "distinct_hashes.py"
DRIVER = HERE.parent / "llvm" / "amdgpu-repro" / "driver" / "replay.cpp"
SUMMARIZER = HERE.parent / "llvm" / "amdgpu-repro" / "driver" / "summarize_replay_ab.py"

CHECKS = 0
FAILURES: list[str] = []


def ok(label: str) -> None:
    global CHECKS
    CHECKS += 1
    print(f"  ok   {label}")


def bad(label: str, detail: str) -> None:
    global CHECKS
    CHECKS += 1
    FAILURES.append(f"{label}: {detail}")
    print(f"  FAIL {label}\n         {detail}")


def expect(condition: bool, label: str, detail: str) -> None:
    ok(label) if condition else bad(label, detail)


def run_counter(*argv: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(COUNTER), *argv],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def write_runs(directory: pathlib.Path, payloads: list[bytes]) -> None:
    for index, payload in enumerate(payloads):
        (directory / f"run{index:03d}.bin").write_bytes(payload)


# ---------------------------------------------------------------------------
# 1. The rule itself, on the cases the two tools could disagree about.
# ---------------------------------------------------------------------------

def check_rule() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)

        agree = root / "agree"
        agree.mkdir()
        write_runs(agree, [b"identical"] * 4)
        code, out = run_counter(str(agree))
        expect(code == 0, "agreement exits 0", f"got {code}")
        expect(
            "SAME BYTES EVERY RUN" in out,
            "agreement is reported as agreement",
            out.splitlines()[0] if out else "<no output>",
        )

        differ = root / "differ"
        differ.mkdir()
        write_runs(differ, [b"a", b"a", b"b"])
        code, out = run_counter(str(differ))
        expect(code == 1, "disagreement exits 1", f"got {code}")

        # The case the two tools disagreed about. One run is not a comparison.
        single = root / "single"
        single.mkdir()
        write_runs(single, [b"only"])
        code, out = run_counter(str(single))
        expect(code == 2, "a single run yields no verdict, exit 2", f"got {code}")
        expect(
            "NO VERDICT" in out,
            "a single run is not called deterministic",
            out.splitlines()[0] if out else "<no output>",
        )
        record = json.loads(run_counter(str(single), "--json")[1])
        expect(
            record.get("deterministic") is None,
            "a single run leaves `deterministic` unset rather than true",
            f"deterministic={record.get('deterministic')!r}",
        )

        empty = root / "empty"
        empty.mkdir()
        code, out = run_counter(str(empty))
        expect(code == 2, "an empty set is refused, exit 2", f"got {code}")


def check_driver_states_the_same_rule() -> None:
    """Source check, not a run: no hipcc and no GPU here, so this reads the driver."""
    if not DRIVER.exists():
        bad("driver is present", f"{DRIVER} not found")
        return
    text = DRIVER.read_text(encoding="utf-8", errors="replace")

    guard = re.search(r"runs\s*<\s*2", text)
    expect(
        guard is not None,
        "the driver refuses fewer than two runs (read from source, not run)",
        "no `runs < 2` guard in replay.cpp: it would report determinism from one launch, "
        "which distinct_hashes.py refuses. The two rules have drifted.",
    )
    expect(
        "check_only && runs < 2" in text.replace("!", ""),
        "the driver's guard exempts the no-launch manifest check",
        "the `runs < 2` guard is not scoped to launching runs, so --check-manifest "
        "may now be refused for having no runs to compare",
    )


# ---------------------------------------------------------------------------
# 2. The output format both sides agreed to speak.
# ---------------------------------------------------------------------------

def check_format_interop() -> None:
    """The counter's hash line must stay byte-compatible with the driver's printf.

    This is the failure this test exists to prevent a second time: the counter used to
    append a size to its hash line, which made the line LOOK like the driver's while
    failing the summarizer's end-anchored regex. The result was an empty record rather
    than an error, which is the worst way for two tools to be incompatible.
    """
    if not DRIVER.exists() or not SUMMARIZER.exists():
        bad("driver and summarizer are present", "one of them is missing")
        return

    driver_text = DRIVER.read_text(encoding="utf-8", errors="replace")
    expect(
        '"  %6d x  %s\\n"' in driver_text,
        "the driver still prints the hash line this test knows about",
        "replay.cpp's hash-line printf changed; re-derive the counter's format from it",
    )

    # Take the regexes from the summarizer itself rather than restating them, so that
    # editing the summarizer cannot silently invalidate this check.
    summarizer_text = SUMMARIZER.read_text(encoding="utf-8", errors="replace")
    hash_src = re.search(r"HASH_LINE\s*=\s*re\.compile\(\s*r?(['\"])(.+?)\1", summarizer_text)
    summary_src = re.search(r"SUMMARY_LINE\s*=\s*re\.compile\(\s*r?(['\"])(.+?)\1", summarizer_text)
    if not hash_src or not summary_src:
        bad("the summarizer's regexes are readable", "could not find HASH_LINE/SUMMARY_LINE")
        return
    hash_line = re.compile(hash_src.group(2))
    summary_line = re.compile(summary_src.group(2))

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "runs"
        root.mkdir()
        write_runs(root, [b"same", b"same", b"same", b"odd"])
        _, out = run_counter(str(root), "--driver-format", "armA")

    hits = [hash_line.match(line) for line in out.splitlines()]
    matched = [m for m in hits if m]
    expect(
        len(matched) == 2,
        "the summarizer's HASH_LINE parses the counter's hash lines",
        f"matched {len(matched)} of an expected 2 -- the formats have drifted apart",
    )
    expect(
        sum(int(m.group(1)) for m in matched) == 4,
        "the parsed multiplicities account for every run",
        f"parsed {[int(m.group(1)) for m in matched]}, expected to sum to 4",
    )
    summaries = [summary_line.match(line) for line in out.splitlines()]
    matched_summary = [m for m in summaries if m]
    expect(
        len(matched_summary) == 1,
        "the summarizer's SUMMARY_LINE parses the counter's --driver-format line",
        f"matched {len(matched_summary)} of an expected 1",
    )
    if matched_summary:
        label, runs, distinct = matched_summary[0].groups()
        expect(
            (label, runs, distinct) == ("armA", "4", "2"),
            "the parsed summary carries the label, run count and distinct count",
            f"got {(label, runs, distinct)}",
        )


# ---------------------------------------------------------------------------
# 3. Neither tool may turn a count into a rate.
# ---------------------------------------------------------------------------

def check_no_rates() -> None:
    """Both sides classify; neither reports a frequency. The denominator is a loop count
    someone chose, so a rate would describe the harness and read as a property of the
    computation."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "runs"
        root.mkdir()
        write_runs(root, [b"a", b"a", b"b", b"c"])
        _, text = run_counter(str(root))
        record = json.loads(run_counter(str(root), "--json")[1])

    expect("%" not in text, "the counter's report contains no percentage", "found '%'")
    banned = ("rate", "percent", "fraction", "frequency", "ratio", "share")
    offenders = [k for k in _all_keys(record) if any(b in k.lower() for b in banned)]
    expect(
        not offenders,
        "the counter's record has no rate-shaped field",
        f"found {offenders}",
    )


def _all_keys(node: object) -> list[str]:
    if isinstance(node, dict):
        out = list(node.keys())
        for value in node.values():
            out.extend(_all_keys(value))
        return out
    if isinstance(node, list):
        out: list[str] = []
        for value in node:
            out.extend(_all_keys(value))
        return out
    return []


def main() -> int:
    print("determinism rule agreement: distinct_hashes.py vs replay.cpp")
    print()
    check_rule()
    check_driver_states_the_same_rule()
    check_format_interop()
    check_no_rates()
    print()
    if FAILURES:
        print(f"FAIL ({len(FAILURES)} of {CHECKS} checks)")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"determinism rule agreement PASS ({CHECKS} checks, 0 device launches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
