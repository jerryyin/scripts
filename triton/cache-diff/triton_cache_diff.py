#!/usr/bin/env python3
"""Compare the compiled kernels of two Triton/Gluon cache directories (TRITON_CACHE_DIR).

usage: triton_cache_diff.py BEFORE_DIR AFTER_DIR [--ext hsaco,amdgcn] [--json OUT]

Groups every compiled file by kernel name (the file stem) and extension and compares, per
name, the multiset of SHA-256 digests on each side. For `.amdgcn` it also compares a
normalized digest that drops debug-location and comment lines (`.loc`, `.file`, `;`-comments,
`.Ltmp`/`.Lfunc_end`/`.Linfo` labels and DWARF sections), so a source-line shift without a code change
shows as `same-code` instead of `differs`. For `.hsaco` the normalized digest covers only the
`.text` and `.rodata` sections (code and kernel descriptors; no debug info or source paths).
Prints one line per (kernel, ext):
  identical | same-code (normalized digests equal) | differs | only-before | only-after
Exit status 0 when nothing `differs` or is one-sided, else 1.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

_DROP = re.compile(r"^\s*(\.loc\b|\.file\b|;|\.Ltmp|\.Lfunc_end|\.Linfo|\.Ldebug|\.Lsection|\.Lcu_|\.section\s+\.debug)")


def normalized_amdgcn(text: str) -> str:
    out, in_debug = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(".section"):
            in_debug = ".debug" in s
        if in_debug or _DROP.match(line):
            continue
        out.append(re.sub(r"\s*;.*$", "", line).rstrip())
    return "\n".join(out)


def elf_code_sections(data: bytes, names=(".text", ".rodata")) -> bytes:
    """The bytes of the named sections of an ELF64 little-endian object (a `.hsaco`): the
    machine code and kernel descriptors, without debug sections or embedded source paths."""
    if data[:4] != b"\x7fELF" or data[4] != 2:
        return data
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sh = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    stroff = sh[shstrndx][4]
    out = b""
    for name_off, _, _, _, off, size, *_ in sh:
        name = data[stroff + name_off:data.index(b"\0", stroff + name_off)].decode()
        if name in names:
            out += name.encode() + b"\0" + data[off:off + size]
    return out


def scan(root: Path, exts: set[str]) -> dict:
    found = collections.defaultdict(list)
    for p in sorted(root.rglob("*")):
        ext = p.suffix.lstrip(".")
        if not p.is_file() or ext not in exts:
            continue
        data = p.read_bytes()
        entry = {"sha256": hashlib.sha256(data).hexdigest(), "path": str(p.relative_to(root))}
        if ext == "hsaco":
            entry["normalized"] = hashlib.sha256(elf_code_sections(data)).hexdigest()
        if ext == "amdgcn":
            entry["normalized"] = hashlib.sha256(
                normalized_amdgcn(data.decode(errors="replace")).encode()).hexdigest()
        found[(p.stem, ext)].append(entry)
    return found


def compare(before: dict, after: dict) -> list[dict]:
    rows = []
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key, []), after.get(key, [])
        if not b or not a:
            verdict = "only-after" if not b else "only-before"
        elif sorted(e["sha256"] for e in b) == sorted(e["sha256"] for e in a):
            verdict = "identical"
        elif "normalized" in b[0] and sorted(e["normalized"] for e in b) == sorted(e["normalized"] for e in a):
            verdict = "same-code"
        else:
            verdict = "differs"
        rows.append({"kernel": key[0], "ext": key[1], "before": len(b), "after": len(a), "verdict": verdict,
                     "before_sha256": sorted(e["sha256"] for e in b), "after_sha256": sorted(e["sha256"] for e in a)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--ext", default="hsaco,amdgcn")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    exts = set(a.ext.split(","))
    rows = compare(scan(a.before, exts), scan(a.after, exts))
    for r in rows:
        print(f"{r['verdict']:<12} {r['kernel']}.{r['ext']} before={r['before']} after={r['after']} "
              f"{','.join(h[:16] for h in r['before_sha256'])} -> {','.join(h[:16] for h in r['after_sha256'])}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    return 0 if all(r["verdict"] in ("identical", "same-code") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
