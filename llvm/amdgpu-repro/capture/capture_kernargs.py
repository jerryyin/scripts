#!/usr/bin/env python3
"""Freeze one real launch of the kernel so it can be replayed without Triton.

The driver in ../driver/replay.cpp understands nothing about the kernel. It needs a
directory holding the recorded bytes of every buffer, a kernarg blob with the scalar
arguments already in place, and the launch geometry. This script produces that
directory.

Two entry points:

    python capture_kernargs.py offsets --asm kernel.s

        Print the kernel's argument table, read from the code object metadata that
        llc emitted. This is the authoritative layout -- offsets are not guessed
        and not recomputed from the IR. Run this first: if the explicit argument
        count is not what you expect, stop, because everything downstream silently
        assumes it.

    from capture_kernargs import Recorder

        Recorder(asm=..., outdir=...).record(kernel_args, grid, block, lds)

        Called at the launch site, with the same argument tuple the kernel is
        launched with. Tensors are recorded as raw device bytes; ints and floats
        are packed into the kernarg blob.

Fails loudly on anything it cannot account for. A capture that silently drops or
misplaces an argument produces a replay that is wrong in a way nobody can see, and
the whole point of the harness is that there is nothing in it to get wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# The metadata note llc emits is YAML embedded in the .s. We read the handful of
# fields that matter with a line scanner rather than taking a YAML dependency,
# because this has to run inside whatever container the board happens to have.
# The first key of each YAML record carries a "- " list marker, and which key comes
# first varies: a named argument leads with .address_space or .name, while a
# compiler-added one has no name and leads with .offset. Every key pattern therefore
# has to tolerate the marker, or those records are dropped without a word.
_P = r"^\s*(?:-\s+)?\."

# Not `.name:` -- that appears inside every argument record as well, so matching it
# picks up parameter names and survives only by the kernel's own .name happening to
# come last. `.symbol:` appears exactly once per kernel.
_KERNEL_RE = re.compile(_P + r"symbol:\s*(\S+?)\.kd\s*$")
_OFFSET_RE = re.compile(_P + r"offset:\s*(\d+)")
_SIZE_RE = re.compile(_P + r"size:\s*(\d+)")
_KIND_RE = re.compile(_P + r"value_kind:\s*(\S+)")
_KERNARG_SEGMENT_RE = re.compile(_P + r"kernarg_segment_size:\s*(\d+)")
_GROUP_SEGMENT_RE = re.compile(_P + r"group_segment_fixed_size:\s*(\d+)")

# Buffer arguments carry one of these kinds. Anything else that is not hidden_* is
# a by-value scalar.
_POINTER_KINDS = frozenset({"global_buffer", "dynamic_shared_pointer"})


@dataclass
class Arg:
    index: int
    offset: int
    size: int
    kind: str

    @property
    def is_pointer(self) -> bool:
        return self.kind in _POINTER_KINDS

    @property
    def is_hidden(self) -> bool:
        # The compiler appends its own arguments (block counts, grid dims, the
        # implicit-arg pointer). They are set up by the launch mechanism, not by
        # the caller, so they are not part of the kernel's parameter list.
        return self.kind.startswith("hidden_")


@dataclass
class KernelLayout:
    name: str
    kernarg_size: int
    group_segment_size: int
    all_args: list[Arg]

    @property
    def args(self) -> list[Arg]:
        """The kernel's own parameters, in declaration order."""
        return [a for a in self.all_args if not a.is_hidden]

    @property
    def hidden(self) -> list[Arg]:
        return [a for a in self.all_args if a.is_hidden]

    @property
    def pointers(self) -> list[Arg]:
        return [a for a in self.args if a.is_pointer]


def read_layout(asm_path: Path) -> KernelLayout:
    """Read the argument table out of the assembly's own metadata."""
    text = asm_path.read_text()
    if "amdhsa.kernels" not in text:
        raise RuntimeError(
            f"{asm_path} carries no amdhsa.kernels metadata; it is not a code-object "
            "assembly emitted by llc for amdgcn-amd-amdhsa"
        )

    name: str | None = None
    kernarg_size: int | None = None
    group_size: int | None = None
    records: list[Arg] = []
    offset = size = kind = None

    # Each argument is a block of .offset, .size and .value_kind lines, in
    # declaration order. A record closes when all three have been seen.
    for line in text.splitlines():
        if m := _KERNEL_RE.match(line):
            if name is not None:
                # Every field below is accumulated flat, so a second kernel would
                # silently overwrite the first one's sizes and append its arguments
                # to the first one's list.
                raise RuntimeError(
                    f"{asm_path} defines more than one kernel ({name}, {m.group(1)}); "
                    "this reader handles a single-kernel code object only"
                )
            name = m.group(1)
        elif m := _KERNARG_SEGMENT_RE.match(line):
            kernarg_size = int(m.group(1))
        elif m := _GROUP_SEGMENT_RE.match(line):
            group_size = int(m.group(1))
        elif m := _OFFSET_RE.match(line):
            offset = int(m.group(1))
        elif m := _SIZE_RE.match(line):
            size = int(m.group(1))
        elif m := _KIND_RE.match(line):
            kind = m.group(1)

        if offset is not None and size is not None and kind is not None:
            records.append(
                Arg(index=len(records), offset=offset, size=size, kind=kind)
            )
            offset = size = kind = None

    if (offset, size, kind) != (None, None, None):
        # A partly-read record means a key pattern stopped matching -- a metadata
        # format change, most likely. Silently dropping it would shift every
        # argument after it, so this is fatal.
        raise RuntimeError(
            f"{asm_path}: an argument record did not close "
            f"(offset={offset}, size={size}, value_kind={kind}). The metadata layout "
            "is not what this reader expects; do not trust the offsets."
        )
    if name is None or kernarg_size is None:
        raise RuntimeError(f"{asm_path}: metadata has no kernel name or kernarg size")
    if not records:
        raise RuntimeError(f"{asm_path}: metadata lists no kernel arguments")

    layout = KernelLayout(
        name=name,
        kernarg_size=kernarg_size,
        group_segment_size=group_size or 0,
        all_args=records,
    )
    # Renumber the caller-visible arguments from zero so the indices in the
    # manifest line up with the tuple the caller passes, not with the metadata's
    # record order (which includes the hidden ones).
    for i, arg in enumerate(layout.args):
        arg.index = i
    return layout


class Recorder:
    """Freeze one launch into a directory the C++ driver can replay."""

    def __init__(self, asm: str | Path, outdir: str | Path) -> None:
        self.asm_path = Path(asm)
        self.layout = read_layout(self.asm_path)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        kernel_args: Sequence[Any],
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        lds: int,
        output_arg_index: int = 0,
    ) -> Path:
        """Record `kernel_args` -- the same tuple the kernel is launched with.

        CALL THIS BEFORE THE LAUNCH, NOT AFTER. The output buffer is recorded like
        any other, and the driver re-uploads it before every replay. Recording it
        after the launch would freeze that launch's results as the replay's starting
        state -- which still detects run-to-run differences, but replays a state the
        original launch never started from, and can therefore mask a failure that
        depends on what the buffer held going in. Nothing here can tell which side
        of the launch it was called on, so this is the caller's to get right.

        Tensors are written out as raw device bytes. Scalars are packed into the
        kernarg blob at their recorded offsets. Pointer slots are left zero; the
        driver patches in the addresses of its own allocations at replay time, so
        no device address from this machine is baked into the capture.
        """
        layout = self.layout

        if len(kernel_args) != len(layout.args):
            raise RuntimeError(
                f"{layout.name} takes {len(layout.args)} arguments per its code-object "
                f"metadata but {len(kernel_args)} were passed. Every offset below this "
                "point would be misassigned, so this is fatal rather than a warning."
            )
        if output_arg_index not in {a.index for a in layout.pointers}:
            raise RuntimeError(
                f"output_arg_index {output_arg_index} is not a pointer argument; "
                f"pointer arguments are {[a.index for a in layout.pointers]}"
            )

        kernarg = bytearray(layout.kernarg_size)
        # Provenance goes in as COMMENTS. A capture directory is otherwise anonymous:
        # both modules in this package compile to a kernel named _matmul_decode with
        # an identical 176-byte argument layout, so nothing inside a manifest says
        # which one it came from. The driver skips '#' lines and fails on unknown
        # keys, so a comment records this without changing the format it parses.
        manifest: list[str] = [
            f"# captured from {self.asm_path}",
            f"# kernarg_segment_size {layout.kernarg_size}, "
            f"{len(layout.args)} caller arguments, {len(layout.pointers)} pointers",
            f"# threads per workgroup {block[0] * block[1] * block[2]} -- the code "
            f"object this was captured from must accept at least this many, and so "
            f"must any object it is replayed into",
            f"kernel {layout.name}",
            f"grid {grid[0]} {grid[1]} {grid[2]}",
            f"block {block[0]} {block[1]} {block[2]}",
            f"lds {lds}",
            f"output_buffer {output_arg_index}",
        ]

        for arg, value in zip(layout.args, kernel_args):
            if arg.is_pointer:
                data = _tensor_bytes(arg, value)
                fname = f"buf_{arg.index:02d}.bin"
                (self.outdir / fname).write_bytes(data)
                manifest.append(
                    f"buffer {arg.index} {arg.offset} {len(data)} {fname} "
                    f"{hashlib.sha256(data).hexdigest()}"
                )
                # The pointer slot stays zero. Replay fills it in.
            else:
                kernarg[arg.offset : arg.offset + arg.size] = _scalar_bytes(arg, value)

        (self.outdir / "kernarg.bin").write_bytes(bytes(kernarg))
        (self.outdir / "manifest.txt").write_text("\n".join(manifest) + "\n")
        return self.outdir


def _tensor_bytes(arg: Arg, value: Any) -> bytes:
    """Raw device bytes behind a pointer argument."""
    import torch  # only reached where torch is already loaded, at a launch site

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"argument {arg.index} is a pointer in the code-object metadata but "
            f"{type(value)!r} was passed; a pointer argument must be recorded as "
            "real bytes, not as a number"
        )

    storage = value.untyped_storage()
    element_size = value.element_size()

    # The kernel was handed value.data_ptr(), which is the storage base plus the
    # tensor's offset into it. The driver allocates a fresh buffer and patches in
    # its base address, so a nonzero offset would shift every access. Refuse
    # rather than record something that replays subtly wrong.
    if value.storage_offset() != 0:
        raise RuntimeError(
            f"argument {arg.index} is a view starting "
            f"{value.storage_offset() * element_size} bytes into its storage. The "
            "replay driver patches in the base of its own allocation, so this "
            "capture would be shifted. Report this rather than working around it: "
            "it means the launch site hands the kernel a view, and the manifest "
            "format needs a pointer offset field."
        )

    # A uint8 view over the whole storage, so the bytes are taken verbatim
    # regardless of the tensor's dtype, shape or stride.
    flat = torch.empty(0, dtype=torch.uint8, device=value.device)
    flat.set_(storage)
    data = bytes(flat.cpu().numpy())

    if len(data) != storage.nbytes():
        raise RuntimeError(
            f"argument {arg.index}: read {len(data)} bytes from a "
            f"{storage.nbytes()}-byte storage"
        )
    return data


def _scalar_bytes(arg: Arg, value: Any) -> bytes:
    """Pack a by-value argument into its kernarg slot."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"argument {arg.index} is a {arg.size}-byte by-value argument "
            f"({arg.kind}) but {type(value)!r} was passed"
        )
    if isinstance(value, int):
        fmt = {1: "<b", 2: "<h", 4: "<i", 8: "<q"}.get(arg.size)
        if fmt is None:
            raise RuntimeError(f"argument {arg.index}: odd integer size {arg.size}")
        # The kernel reinterprets the bits, so either signedness is acceptable as
        # long as the value fits one of them.
        try:
            return struct.pack(fmt, value)
        except struct.error:
            return struct.pack(fmt.upper(), value)
    fmt = {4: "<f", 8: "<d"}.get(arg.size)
    if fmt is None:
        raise RuntimeError(f"argument {arg.index}: odd float size {arg.size}")
    return struct.pack(fmt, value)


def main() -> int:
    p = argparse.ArgumentParser(
        description="inspect or capture a gfx1250 kernel's arguments"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("offsets", help="print the kernel argument table")
    o.add_argument("--asm", required=True, help="a .s emitted by llc")
    a = p.parse_args()

    if a.cmd == "offsets":
        layout = read_layout(Path(a.asm))
        print(f"kernel              {layout.name}")
        print(f"kernarg size        {layout.kernarg_size} bytes")
        print(f"static LDS          {layout.group_segment_size} bytes")
        print(
            f"caller arguments    {len(layout.args)} "
            f"({len(layout.pointers)} pointers, "
            f"{len(layout.args) - len(layout.pointers)} by value)"
        )
        print(f"compiler-added      {len(layout.hidden)} hidden")
        print()
        print(f"{'idx':>4}  {'offset':>7}  {'size':>5}  kind")
        for arg in layout.args:
            print(f"{arg.index:>4}  {arg.offset:>7}  {arg.size:>5}  {arg.kind}")
        for arg in layout.hidden:
            print(f"{'   -':>4}  {arg.offset:>7}  {arg.size:>5}  {arg.kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
