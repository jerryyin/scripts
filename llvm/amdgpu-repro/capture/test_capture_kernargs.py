#!/usr/bin/env python3
"""CPU tests for capture_kernargs.py: the argument table read from llc's code-object metadata
and the packing of by-value arguments. Run: python3 -m pytest -q test_capture_kernargs.py"""
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_kernargs as ck  # noqa: E402

# One kernel: a named pointer (record led by .address_space), an unnamed scalar (led by .offset),
# and a hidden argument the launcher supplies.
METADATA = """\t.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .name:           x
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           8
        .value_kind:     hidden_global_offset_x
    .group_segment_fixed_size: 1024
    .kernarg_segment_size: 272
    .name:           k
    .symbol:         k.kd
...
\t.end_amdgpu_metadata
"""


def write(tmp_path, text):
    path = tmp_path / "kernel.s"
    path.write_text(text)
    return path


def test_reads_named_unnamed_and_hidden_arguments(tmp_path):
    layout = ck.read_layout(write(tmp_path, METADATA))
    assert (layout.name, layout.kernarg_size, layout.group_segment_size) == ("k", 272, 1024)
    assert [(a.index, a.offset, a.size, a.kind) for a in layout.args] == [
        (0, 0, 8, "global_buffer"), (1, 8, 4, "by_value")]
    assert [a.kind for a in layout.hidden] == ["hidden_global_offset_x"]
    assert [a.index for a in layout.pointers] == [0]


@pytest.mark.parametrize("text, message", [
    # a last record missing a key (a partial record in the middle is not detected: the next
    # record's .offset overwrites it)
    (METADATA.replace("        .value_kind:     hidden_global_offset_x\n", ""), "did not close"),
    (METADATA.replace("    .symbol:         k.kd\n", "    .symbol:         k.kd\n    .symbol:         j.kd\n"),
     "more than one kernel"),
    ("\t.text\n", "no amdhsa.kernels"),
])
def test_refuses_metadata_it_cannot_trust(tmp_path, text, message):
    with pytest.raises(RuntimeError, match=message):
        ck.read_layout(write(tmp_path, text))


def test_packs_by_value_arguments_little_endian():
    arg = lambda size: ck.Arg(index=0, offset=0, size=size, kind="by_value")
    assert ck._scalar_bytes(arg(4), -1) == struct.pack("<i", -1)
    assert ck._scalar_bytes(arg(4), 0xFFFFFFFF) == struct.pack("<I", 0xFFFFFFFF)
    assert ck._scalar_bytes(arg(8), 1.5) == struct.pack("<d", 1.5)
    with pytest.raises(RuntimeError):
        ck._scalar_bytes(arg(4), True)
