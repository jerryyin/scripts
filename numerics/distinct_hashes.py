#!/usr/bin/env python3
"""Did every run produce the same bytes? A count of distinct hashes, and nothing else.

Given the output files of N runs of the same computation on identical input, this
hashes each file with SHA-256 and reports how many DISTINCT hashes came back:

  N runs, byte-identical inputs re-supplied before each one
    -> 1 distinct hash   : every run produced the same bytes
    -> more than 1       : the computation returned different bytes from identical bytes

That is the whole reading. There is no reference implementation here, no tolerance, and
no arithmetic over the contents of the files: a file is a bag of bytes, and the only
question asked of it is which other files it is byte-identical to. A set of outputs that
fails this test cannot be blamed on a wrong reference or a badly chosen threshold,
because nothing is compared against anything except itself.

This is the third of the three questions this folder answers. `ulp_magnitudes.py` asks
how far an output is from a reference, `compare.py` asks whether it is within a
threshold, and this asks whether the computation returns the same answer twice. The
first two need a second buffer to compare against; this one needs the same computation
run more than once.

THE PRECONDITION, WHICH IS NOT OPTIONAL
---------------------------------------
The count means nothing unless the identical input bytes were RE-SUPPLIED before every
run. A harness that uploads the input once and then launches N times is not running the
same computation N times: the first run may have written over its own input, or over a
buffer the next run reads, and then the second hash is telling you about a corrupted
input rather than a nondeterministic computation. Re-supply the input bytes before each
run, and confirm they are the bytes you think they are, before reading anything into a
count above 1. This tool cannot check that for you -- it sees only the outputs -- so it
states the precondition in every record instead of implying it was met.

WHAT IT REFUSES TO DO, and these are deliberate rather than unfinished
----------------------------------------------------------------------
It does not pool files that do not belong together. One invocation is one set of runs of
one build on one machine since one boot. Naming two sets beside each other, with their
provenance attached, is fine; hashing them into one count is not, because the second
hash then means "these two sets differ" and not "this computation is nondeterministic".

It does not convert counts into rates, percentages, or frequencies, anywhere. Nine of
ten files agreeing is not "10% nondeterminism". The denominator is a loop count you
chose, so dividing by it produces a number that looks like a property of the computation
and is really a property of your harness; the next ten runs can split differently. The
finding is established the moment a second distinct hash exists, and a rate adds no
information to it while inviting comparison against other rates that were measured over
different loop counts. The multiplicity of each hash is reported as a count, because
"one odd run out of ten" and "five against five" are different findings -- but a
multiplicity is a classification of which runs agreed, never a frequency of failure.

It does not compare across a boot or a provenance boundary. Two sets separated by a
reboot, a driver reload, a rebuild, or a different machine are two records that may be
named together and may not be differenced.

It issues no verdict beyond the count. One distinct hash means the runs agreed; it does
not mean the answer is correct, and it does not mean the next run will agree. For
correctness, use the other two tools in this folder.

WHY SHA-256
-----------
So the hashes here can be compared directly against hashes printed by other tooling
rather than being a private checksum of this file's invention. The GPU-side version of
this same measurement -- llvm/amdgpu-repro/driver/replay.cpp, which needs a device and a
captured argument manifest -- prints SHA-256 of the output buffer after every launch, and
its hashes and these are the same strings for the same bytes.

EXIT CODES
----------
  0  exactly one distinct hash over two or more files: the runs agreed
  1  more than one distinct hash: identical input bytes produced different bytes
  2  no verdict was possible (no files, a single file, a path that is not there, or a
     file count that does not match --expect-runs)

Exit 0 is reserved for an observed agreement between at least two runs, so a caller that
branches on the exit status cannot read "nothing was measured" as "everything agreed".

USAGE
    python3 distinct_hashes.py run00.bin run01.bin run02.bin
    python3 distinct_hashes.py outputs/ --expect-runs 20
    python3 distinct_hashes.py 'outputs/run*.bin' --json
    python3 distinct_hashes.py --selftest
"""

from __future__ import annotations

import argparse
import fnmatch
import glob as globbing
import hashlib
import json
import os
import sys
import tempfile
from typing import Any

# Files are hashed in chunks so that an output buffer larger than memory is still
# hashable, and so the size reported beside a hash is the number of bytes actually fed
# to the digest rather than a separately queried length that could disagree with it.
CHUNK_BYTES = 8 * 1024 * 1024

GLOB_METACHARACTERS = "*?["

ONE_HASH = "ONE_DISTINCT_HASH"
MANY_HASHES = "MORE_THAN_ONE_DISTINCT_HASH"
ONE_FILE = "ONE_FILE_ONLY_SO_NOT_EVIDENCE_EITHER_WAY"
REFUSED_NO_FILES = "REFUSED_NO_FILES_TO_HASH"
REFUSED_MISSING = "REFUSED_A_NAMED_INPUT_IS_NOT_THERE"
REFUSED_COUNT = "REFUSED_FILE_COUNT_DOES_NOT_MATCH_EXPECTED_RUNS"

REFUSALS = [
    "This record covers one set of runs of one build on one machine since one boot. It "
    "may be named beside another such record, with provenance attached to both, and it "
    "may not be pooled with one or differenced against one.",
    "A count of distinct hashes is a classification and never a frequency. Nothing here "
    "is divided by the number of runs, because that denominator is a loop count chosen "
    "by the harness and the resulting number would look like a property of the "
    "computation.",
    "The multiplicity beside each hash says which runs agreed with which. It is not a "
    "failure count and must not be presented as one.",
]

PRECONDITION = (
    "The count is meaningful only if byte-identical input was re-supplied before every "
    "run. If the input was uploaded once and reused, a second distinct hash may be "
    "reporting an input that one run overwrote rather than a nondeterministic "
    "computation. This tool sees only outputs and cannot check the precondition."
)

NOT_A_CORRECTNESS_CLAIM = (
    "One distinct hash means the runs agreed with each other. It does not mean the "
    "answer is right -- a computation can be reliably wrong -- and it does not predict "
    "the next run. Use ulp_magnitudes.py or compare.py for correctness."
)


def sha256_of_file(path: str) -> tuple[str, int]:
    """SHA-256 of a file's bytes, with the number of bytes that went into it."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _scan_directory(
    directory: str, *, recursive: bool, pattern: str
) -> tuple[list[str], list[dict[str, str]]]:
    """Regular files in a directory, plus every entry that was not taken and why.

    Not recursive unless asked. A directory of run outputs that has grown a
    subdirectory has grown it for a reason -- a nested log dump, a saved-off earlier
    batch -- and silently hashing what is inside it would put files from another set
    into this count. So a subdirectory is skipped and NAMED, which is the part that
    matters: the reader sees that something was there and was not taken.
    """
    taken: list[str] = []
    skipped: list[dict[str, str]] = []
    if recursive:
        walked: list[str] = []
        for root, directory_names, file_names in os.walk(directory):
            directory_names.sort()
            for name in sorted(file_names):
                walked.append(os.path.join(root, name))
        entries = walked
    else:
        entries = [
            os.path.join(directory, name) for name in sorted(os.listdir(directory))
        ]
    for entry in entries:
        name = os.path.basename(entry)
        if os.path.isdir(entry):
            skipped.append({
                "path": entry,
                "why": "a subdirectory, not descended into because --recursive was not given",
            })
        elif not os.path.isfile(entry):
            skipped.append({"path": entry, "why": "not a regular file"})
        elif not fnmatch.fnmatch(name, pattern):
            skipped.append({"path": entry, "why": f"does not match --pattern {pattern!r}"})
        else:
            taken.append(entry)
    return taken, skipped


def resolve_inputs(
    arguments: list[str], *, recursive: bool = False, pattern: str = "*"
) -> dict[str, Any]:
    """Turn files, directories and globs into the exact list of files to hash.

    --pattern filters what is picked up from a directory or a glob, not a file named
    explicitly on the command line: naming a file and then being told it was filtered
    out would be a surprise, whereas a directory listing is a guess about what belongs
    and deserves one.
    """
    taken: list[str] = []
    skipped: list[dict[str, str]] = []
    missing: list[str] = []
    for argument in arguments:
        if os.path.isdir(argument):
            found, not_taken = _scan_directory(
                argument, recursive=recursive, pattern=pattern
            )
            taken.extend(found)
            skipped.extend(not_taken)
        elif os.path.isfile(argument):
            taken.append(argument)
        elif any(character in argument for character in GLOB_METACHARACTERS):
            matches = sorted(globbing.glob(argument, recursive=recursive))
            if not matches:
                missing.append(argument)
            for match in matches:
                if os.path.isdir(match):
                    found, not_taken = _scan_directory(
                        match, recursive=recursive, pattern=pattern
                    )
                    taken.extend(found)
                    skipped.extend(not_taken)
                elif not os.path.isfile(match):
                    skipped.append({"path": match, "why": "not a regular file"})
                elif not fnmatch.fnmatch(os.path.basename(match), pattern):
                    skipped.append({
                        "path": match, "why": f"does not match --pattern {pattern!r}"
                    })
                else:
                    taken.append(match)
        else:
            missing.append(argument)

    # The same file reached twice -- by a glob that overlaps an explicit name, say --
    # would inflate the multiplicity of its hash and make one run look like two.
    # Resolved through symlinks, which is as far as a path comparison can go; two hard
    # links to one file are indistinguishable from two files here and will be counted
    # as two.
    unique: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for path in taken:
        real = os.path.realpath(path)
        if real in seen:
            duplicates.append(path)
        else:
            seen.add(real)
            unique.append(path)
    return {"paths": unique, "skipped": skipped, "missing": missing, "duplicates": duplicates}


def build_record(
    paths: list[str],
    *,
    skipped: list[dict[str, str]] | None = None,
    missing: list[str] | None = None,
    duplicates: list[str] | None = None,
    expect_runs: int | None = None,
) -> dict[str, Any]:
    """Hash every file and assemble the one record this tool produces."""
    skipped = list(skipped or [])
    missing = list(missing or [])
    duplicates = list(duplicates or [])

    groups: dict[str, list[str]] = {}
    sizes: dict[str, int] = {}
    for path in paths:
        digest, size = sha256_of_file(path)
        groups.setdefault(digest, []).append(path)
        sizes[path] = size

    # Largest group first, so an odd run out lands at the bottom of the listing where it
    # reads as the exception it is. The hash string breaks ties, so the same set of files
    # always produces the same record.
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))

    record: dict[str, Any] = {
        "record_type": "distinct_output_hash_count",
        "schema_version": 1,
        "algorithm": "sha256",
        "files_hashed": len(paths),
        "distinct_hashes": len(groups),
        "hashes": [
            {
                "sha256": digest,
                "files": len(members),
                "bytes": sizes[members[0]],
                "paths": list(members),
            }
            for digest, members in ordered
        ],
        "precondition": PRECONDITION,
        "not_a_correctness_claim": NOT_A_CORRECTNESS_CLAIM,
        "what_this_record_refuses_to_do": REFUSALS,
    }

    if missing:
        record["status"] = REFUSED_MISSING
        record["deterministic"] = None
        record["missing_inputs"] = missing
        record["headline"] = (
            f"REFUSED: {len(missing)} named input(s) could not be resolved to any file, "
            "so the set of runs is not the set that was asked for: "
            + ", ".join(missing)
        )
    elif not paths:
        # Zero files is not zero distinct hashes. Reporting a count over an empty set
        # would put a number where no measurement happened, and a reader scanning for
        # "1" versus "more than 1" would find neither and might read the 0 as agreement.
        record["status"] = REFUSED_NO_FILES
        record["deterministic"] = None
        record["headline"] = (
            "REFUSED: no files to hash. A count of distinct hashes over an empty set is "
            "not 0, it is undefined -- nothing was measured."
        )
    elif expect_runs is not None and len(paths) != expect_runs:
        record["status"] = REFUSED_COUNT
        record["deterministic"] = None
        record["expected_runs"] = expect_runs
        record["headline"] = (
            f"REFUSED: {len(paths)} file(s) found but --expect-runs {expect_runs} was "
            "given. A run that produced no output file cannot disagree with anything, "
            "so the missing outputs have to be accounted for before the count is read."
        )
    elif len(paths) == 1:
        # One file is one run, and one run cannot disagree with itself. The count really
        # is 1, and it is not evidence of anything, so `deterministic` is left null
        # rather than set true: an absent measurement must not look like a measurement
        # that came out well.
        record["status"] = ONE_FILE
        record["deterministic"] = None
        record["headline"] = (
            "NO VERDICT: 1 file, 1 distinct hash. A single run cannot agree or disagree "
            "with another run, so this is not evidence of determinism. Supply the "
            "outputs of at least two runs."
        )
    elif len(groups) == 1:
        record["status"] = ONE_HASH
        record["deterministic"] = True
        record["headline"] = (
            f"SAME BYTES EVERY RUN: {len(paths)} files, 1 distinct hash. Every run "
            "produced byte-identical output."
        )
    else:
        record["status"] = MANY_HASHES
        record["deterministic"] = False
        record["headline"] = (
            f"DIFFERENT BYTES: {len(paths)} files, {len(groups)} distinct hashes. If "
            "identical input was re-supplied before every run, the computation returned "
            "different bytes from identical bytes."
        )

    if skipped:
        record["skipped_entries"] = skipped
    if duplicates:
        record["duplicate_paths_dropped"] = duplicates

    # Two conditions that would otherwise let a false agreement through quietly.
    empty_files = [path for path in paths if sizes[path] == 0]
    if empty_files:
        record["zero_byte_files"] = empty_files
        record["zero_byte_warning"] = (
            "Some files hashed are empty, and the SHA-256 of nothing is a perfectly "
            "stable hash. Runs that all wrote nothing will agree here. Check that the "
            "outputs were written before reading the count."
        )
    distinct_sizes = sorted(set(sizes.values()))
    if len(distinct_sizes) > 1:
        record["byte_sizes_present"] = distinct_sizes
        record["size_warning"] = (
            "The files are not all the same length, so at least one run produced a "
            "differently sized output. That is a truncated or failed run rather than a "
            "numeric disagreement, and it should be resolved before the count is read."
        )
    return record


def format_report(record: dict[str, Any]) -> str:
    """The text form: headline first, then the grouping, then the caveats."""
    lines = [record["headline"], ""]
    if record["hashes"]:
        for group in record["hashes"]:
            # Byte-identical to the format ../llvm/amdgpu-repro/driver/replay.cpp prints
            # ("  %6d x  %s"), so that driver's summarizer can read this tool's output
            # unchanged. The size used to be appended here, which made the line LOOK like
            # the driver's while failing its end-anchored regex -- an incompatibility that
            # produced an empty record instead of an error. It sits on its own line now.
            lines.append(f"  {group['files']:6d} x  {group['sha256']}")
            lines.append(f"             ({group['bytes']} bytes each)")
            for path in group["paths"]:
                lines.append(f"             {path}")
        lines.append("")
    for key in ("zero_byte_warning", "size_warning"):
        if key in record:
            lines.append(record[key])
            lines.append("")
    if "skipped_entries" in record:
        lines.append("Not hashed:")
        for entry in record["skipped_entries"]:
            lines.append(f"  {entry['path']}  --  {entry['why']}")
        lines.append("")
    if "duplicate_paths_dropped" in record:
        lines.append("Named more than once and counted once:")
        for path in record["duplicate_paths_dropped"]:
            lines.append(f"  {path}")
        lines.append("")
    lines.append(record["precondition"])
    lines.append("")
    lines.append(record["what_this_record_refuses_to_do"][1])
    return "\n".join(lines)


def exit_code_for(record: dict[str, Any]) -> int:
    if record["status"] == ONE_HASH:
        return 0
    if record["status"] == MANY_HASHES:
        return 1
    return 2


def _selftest() -> int:
    """Synthesised file sets covering every case the reading can land in."""

    checks = 0

    def write(directory: str, name: str, payload: bytes) -> str:
        path = os.path.join(directory, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def record_for(arguments: list[str], **kwargs: Any) -> dict[str, Any]:
        resolved = resolve_inputs(
            arguments,
            recursive=kwargs.pop("recursive", False),
            pattern=kwargs.pop("pattern", "*"),
        )
        return build_record(
            resolved["paths"],
            skipped=resolved["skipped"],
            missing=resolved["missing"],
            duplicates=resolved["duplicates"],
            **kwargs,
        )

    with tempfile.TemporaryDirectory() as root:
        # Every run produced the same bytes: one hash, multiplicity 5, exit 0.
        same = os.path.join(root, "same")
        os.mkdir(same)
        for index in range(5):
            write(same, f"run{index:02d}.bin", b"\x01\x02\x03\x04" * 64)
        identical = record_for([same])
        assert identical["status"] == ONE_HASH, identical
        assert identical["distinct_hashes"] == 1, identical
        assert identical["files_hashed"] == 5, identical
        assert identical["deterministic"] is True, identical
        assert identical["hashes"][0]["files"] == 5, identical["hashes"]
        assert exit_code_for(identical) == 0
        checks += 1

        # Every run produced different bytes: N hashes, each multiplicity 1.
        differing = os.path.join(root, "differing")
        os.mkdir(differing)
        for index in range(4):
            write(differing, f"run{index:02d}.bin", bytes([index]) * 64)
        all_different = record_for([differing])
        assert all_different["status"] == MANY_HASHES, all_different
        assert all_different["distinct_hashes"] == 4, all_different
        assert all_different["deterministic"] is False, all_different
        assert [group["files"] for group in all_different["hashes"]] == [1, 1, 1, 1]
        assert exit_code_for(all_different) == 1
        checks += 1

        # One odd run out of ten, which is the case the grouping exists for: the report
        # has to show that nine runs agreed and name the one that did not.
        odd = os.path.join(root, "odd")
        os.mkdir(odd)
        for index in range(10):
            payload = b"\xaa" * 64 if index != 7 else b"\xaa" * 63 + b"\xab"
            write(odd, f"run{index:02d}.bin", payload)
        one_odd = record_for([odd])
        assert one_odd["status"] == MANY_HASHES, one_odd
        assert one_odd["distinct_hashes"] == 2, one_odd
        assert [group["files"] for group in one_odd["hashes"]] == [9, 1], one_odd["hashes"]
        assert one_odd["hashes"][1]["paths"] == [os.path.join(odd, "run07.bin")], one_odd
        assert "run07.bin" in format_report(one_odd)
        checks += 1

        # A single file: the count is 1 and it is not evidence, so no verdict and exit 2.
        single = record_for([os.path.join(same, "run00.bin")])
        assert single["status"] == ONE_FILE, single
        assert single["distinct_hashes"] == 1, single
        assert single["deterministic"] is None, single
        assert exit_code_for(single) == 2
        checks += 1

        # An empty set refuses rather than reporting 0 distinct hashes.
        vacant = os.path.join(root, "vacant")
        os.mkdir(vacant)
        empty_set = record_for([vacant])
        assert empty_set["status"] == REFUSED_NO_FILES, empty_set
        assert empty_set["deterministic"] is None, empty_set
        assert empty_set["hashes"] == [], empty_set
        assert "not 0" in empty_set["headline"], empty_set["headline"]
        assert exit_code_for(empty_set) == 2
        # And a glob that matches nothing is a named input that is not there, which is a
        # different refusal from an empty directory: the caller asked for files by name.
        nothing_matched = record_for([os.path.join(root, "nowhere", "run*.bin")])
        assert nothing_matched["status"] == REFUSED_MISSING, nothing_matched
        assert exit_code_for(nothing_matched) == 2
        checks += 1

        # A directory containing a subdirectory. Default: the subdirectory is skipped and
        # named, and nothing inside it enters the count. With --recursive it is taken.
        nested = os.path.join(root, "nested")
        os.mkdir(nested)
        write(nested, "run00.bin", b"\x11" * 32)
        write(nested, "run01.bin", b"\x11" * 32)
        inner = os.path.join(nested, "earlier-batch")
        os.mkdir(inner)
        write(inner, "run00.bin", b"\x22" * 32)
        shallow = record_for([nested])
        assert shallow["files_hashed"] == 2, shallow
        assert shallow["status"] == ONE_HASH, shallow
        assert [entry["path"] for entry in shallow["skipped_entries"]] == [inner], shallow
        assert "subdirectory" in shallow["skipped_entries"][0]["why"]
        assert inner in format_report(shallow)
        deep = record_for([nested], recursive=True)
        assert deep["files_hashed"] == 3, deep
        assert deep["status"] == MANY_HASHES, deep
        assert deep["distinct_hashes"] == 2, deep
        checks += 1

        # Zero-byte outputs agree with each other, and that is exactly the false
        # agreement worth shouting about: three runs that wrote nothing are not three
        # runs that agreed on an answer.
        hollow = os.path.join(root, "hollow")
        os.mkdir(hollow)
        for index in range(3):
            write(hollow, f"run{index:02d}.bin", b"")
        empty_files = record_for([hollow])
        assert empty_files["status"] == ONE_HASH, empty_files
        assert len(empty_files["zero_byte_files"]) == 3, empty_files
        assert empty_files["hashes"][0]["sha256"] == hashlib.sha256(b"").hexdigest()
        assert "zero_byte_warning" in empty_files
        checks += 1

        # Differently sized outputs are a failed run, not a numeric disagreement.
        ragged = os.path.join(root, "ragged")
        os.mkdir(ragged)
        write(ragged, "run00.bin", b"\x33" * 64)
        write(ragged, "run01.bin", b"\x33" * 32)
        sizes = record_for([ragged])
        assert sizes["byte_sizes_present"] == [32, 64], sizes
        assert "size_warning" in sizes, sizes
        checks += 1

        # Explicit file list, glob, and --pattern reach the same files as the directory.
        listed = record_for(sorted(
            os.path.join(same, f"run{index:02d}.bin") for index in range(5)
        ))
        globbed = record_for([os.path.join(same, "run*.bin")])
        patterned = record_for([nested], pattern="*.bin")
        assert listed["hashes"] == identical["hashes"], listed
        assert globbed["hashes"] == identical["hashes"], globbed
        assert patterned["files_hashed"] == 2, patterned
        filtered = record_for([nested], pattern="*.txt")
        assert filtered["status"] == REFUSED_NO_FILES, filtered
        checks += 1

        # The same file named twice is one run, not two.
        doubled = record_for([
            os.path.join(same, "run00.bin"),
            os.path.join(same, "run01.bin"),
            os.path.join(same, "run00.bin"),
        ])
        assert doubled["files_hashed"] == 2, doubled
        assert doubled["hashes"][0]["files"] == 2, doubled
        assert doubled["duplicate_paths_dropped"] == [os.path.join(same, "run00.bin")]
        checks += 1

        # --expect-runs is the only protection against a run that produced no file at
        # all: nine files that agree look like agreement, and the tenth run may have
        # crashed. A mismatch is a refusal, not a count.
        short = record_for([same], expect_runs=6)
        assert short["status"] == REFUSED_COUNT, short
        assert exit_code_for(short) == 2
        assert record_for([same], expect_runs=5)["status"] == ONE_HASH
        checks += 1

        # The hashes are SHA-256 of the file bytes and nothing else, so they can be
        # checked against any other tool's hash of the same bytes.
        payload = b"\x01\x02\x03\x04" * 64
        assert identical["hashes"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
        checks += 1

        # No rate anywhere. Every number in every record is an integer count, no key
        # names one, and the JSON form is what a machine reads, so checking it there
        # covers what a caller can consume.
        forbidden = ("rate", "percent", "fraction", "frequency", "ratio", "share")
        for candidate in (identical, all_different, one_odd, single, empty_set, deep, sizes):
            text = json.dumps(candidate)
            for key, value in candidate.items():
                assert not any(word in key.lower() for word in forbidden), key
                assert not isinstance(value, float), (key, value)
            for group in candidate["hashes"]:
                for key, value in group.items():
                    assert not isinstance(value, float), (key, value)
            assert "%" not in text.replace("%s", ""), text
            json.loads(text)  # the record round-trips, so --json is machine-readable
        checks += 1

    print(f"distinct_hashes selftest PASS ({checks} checks, 0 device launches)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Count the distinct SHA-256 hashes over the outputs of N runs of the same "
            "computation on identical input. One hash means every run produced the same "
            "bytes; more than one means it did not."
        ),
        epilog=(
            "The count is a classification and never a rate. Nothing here is divided by "
            "the number of runs."
        ),
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="files, directories, or globs. A glob should be quoted if the shell would "
             "expand it differently from this tool.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="descend into subdirectories. Off by default: a subdirectory is skipped and "
             "named, so files from another batch cannot join this count unasked.",
    )
    parser.add_argument(
        "--pattern", default="*",
        help="filter entries picked up from a directory or a glob (default '*'). Does "
             "not filter a file named explicitly.",
    )
    parser.add_argument(
        "--expect-runs", type=int, default=None,
        help="the number of files that should be present. A mismatch is refused rather "
             "than counted, because a run that produced no output cannot disagree.",
    )
    parser.add_argument("--json", action="store_true", help="emit the record as JSON.")
    parser.add_argument(
        "--driver-format", metavar="LABEL",
        help="also print the summary line ../llvm/amdgpu-repro/driver/replay.cpp prints, "
             "under this label, so that driver's summarize_replay_ab.py can reduce a "
             "file-based run and a device run through one parser. The hash lines already "
             "match it; this adds the summary line it anchors on.",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="run the built-in tests on synthesised files and exit. No inputs needed.",
    )
    args = parser.parse_args()

    if args.selftest:
        return _selftest()
    if not args.inputs:
        parser.error("no inputs given. Pass files, a directory, a glob, or --selftest.")

    resolved = resolve_inputs(args.inputs, recursive=args.recursive, pattern=args.pattern)
    record = build_record(
        resolved["paths"],
        skipped=resolved["skipped"],
        missing=resolved["missing"],
        duplicates=resolved["duplicates"],
        expect_runs=args.expect_runs,
    )
    if args.json:
        print(json.dumps(record, indent=2))
    else:
        if args.driver_format:
            # Exactly replay.cpp's wording, so its summarizer's end-anchored regex
            # matches. "runs" here is files hashed, which is the same quantity only if
            # every run wrote exactly one file -- which is why --expect-runs exists.
            print(
                f"{args.driver_format}: {record['files_hashed']} runs, "
                f"{record['distinct_hashes']} distinct output hash(es)"
            )
        print(format_report(record))
    return exit_code_for(record)


if __name__ == "__main__":
    sys.exit(main())
