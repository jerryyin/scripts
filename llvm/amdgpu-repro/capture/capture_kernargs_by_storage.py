#!/usr/bin/env python3
"""Freeze one real launch whose pointer arguments may be views into shared allocations.

capture_kernargs.py records one buffer per pointer argument and refuses an argument that is a
view starting part-way into its storage, because the replay driver patches in the base of its
own allocation and the capture would be shifted. This recorder handles that case: it records
each STORAGE once, whole, and gives each argument a pointer INTO it, so the addresses the kernel
sees stand in the same relation to each other and to their allocation as on the device.

Giving each argument its own allocation holding only its window would be faithful only while
the kernel stays inside each window; a read past the end would read the neighbouring bytes on
the device but off the end of a fresh allocation in replay, a difference an output hash cannot
show. Recording the storage costs little (index and offset tables are small next to weights)
and is strictly more general: an argument alone on its storage is the case of one pointer at
offset zero. (The case was measured on a gfx1250 MoE launch with two index tables windowed
into one allocation: triton-investigations
investigations/a8w4-decode-performance/llvm-ticket-expert-scheduling/capture/.)

The manifest, read by ../driver/replay.cpp:

    storage <id> <bytes> <file> <sha256>
        Allocate <bytes>, upload <file>, re-upload before every execution. Patches
        no kernarg slot by itself: a storage is an allocation, not an argument.

    pointer <arg> <kernarg_offset> <storage_id> <byte_offset> <extent>
        Write storage(<storage_id>).dev + <byte_offset> into the kernarg blob at
        <kernarg_offset>. <extent> is the argument's own reachable span in bytes and
        is what the driver reads back when this argument is the output.

    output_buffer <arg>
        Names an ARGUMENT.

The driver fails on manifest keys it does not know, still reads the one-buffer-per-argument
form, and refuses a manifest that mixes the two. The code-object layout reader, the argument
table and the by-value packing are imported from capture_kernargs.py. Pointer slots are left
zero in kernarg.bin, so no device address from this machine is baked into a capture.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from capture_kernargs import Arg, KernelLayout, read_layout, _scalar_bytes


@dataclass
class _Storage:
    """One device allocation, and every argument that points into it."""

    index: int
    nbytes: int
    handle: Any  # torch UntypedStorage, kept only until the bytes are written
    members: list[int] = field(default_factory=list)


@dataclass
class PointerRecord:
    """Where one pointer argument sits inside its allocation."""

    arg_index: int
    kernarg_offset: int
    storage_index: int
    byte_offset: int
    extent: int


def tensor_extent_bytes(value: Any) -> int:
    """The bytes an indexing expression over `value` can reach, from its own base.

    Not element_size * numel: a strided tensor reaches past the end of its dense
    equivalent, and the mxfp4 weight at argument 8 is strided. The reachable span is
    the largest in-bounds linear index plus one element.
    """
    if value.numel() == 0:
        return 0
    last = sum((dim - 1) * stride for dim, stride in zip(value.shape, value.stride()))
    return (last + 1) * value.element_size()


class StorageRecorder:
    """Freeze one launch, one blob per allocation, one pointer line per argument."""

    def __init__(self, asm: str | Path, outdir: str | Path) -> None:
        self.asm_path = Path(asm)
        self.layout: KernelLayout = read_layout(self.asm_path)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def plan(self, kernel_args: Sequence[Any]) -> tuple[list[_Storage], list[PointerRecord]]:
        """Work out the allocations and pointers WITHOUT writing anything.

        Separated from record() so the caller can size a capture, and refuse it, before
        it spends the first of several gigabytes. Two attempts wrote seven gigabytes
        each to learn one fact about one argument; that is the mistake this shape
        exists to stop repeating.
        """
        import torch  # only reached at a launch site, where torch is already loaded

        layout = self.layout
        if len(kernel_args) != len(layout.args):
            raise RuntimeError(
                f"{layout.name} takes {len(layout.args)} arguments per its code-object "
                f"metadata but {len(kernel_args)} were passed. Every offset below this "
                "point would be misassigned, so this is fatal rather than a warning."
            )

        storages: list[_Storage] = []
        by_key: dict[tuple[int, int], _Storage] = {}
        pointers: list[PointerRecord] = []

        for arg, value in zip(layout.args, kernel_args):
            if not arg.is_pointer:
                continue
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"argument {arg.index} is a pointer in the code-object metadata but "
                    f"{type(value)!r} was passed; a pointer argument must be recorded as "
                    "real bytes, not as a number"
                )

            handle = value.untyped_storage()
            # Identity is the pair, not the address alone: two DISTINCT zero-length
            # allocations both report data_ptr 0, and arguments 27 and 28 -- the
            # launcher's own global-scratch and profile-scratch placeholders -- are
            # exactly that case. Pairing the address with the size keeps a real
            # allocation from ever being merged with an empty one.
            key = (int(handle.data_ptr()), int(handle.nbytes()))
            storage = by_key.get(key)
            if storage is None:
                storage = _Storage(index=len(storages), nbytes=int(handle.nbytes()),
                                   handle=handle)
                by_key[key] = storage
                storages.append(storage)
            storage.members.append(arg.index)

            byte_offset = value.storage_offset() * value.element_size()
            extent = tensor_extent_bytes(value)
            if byte_offset + extent > storage.nbytes:
                raise RuntimeError(
                    f"argument {arg.index} reaches {byte_offset + extent} bytes into a "
                    f"{storage.nbytes}-byte storage (offset {byte_offset}, extent "
                    f"{extent}). Recording it would produce a replay that reads past "
                    "its own allocation. This is a real inconsistency at the launch "
                    "site, not a limit of this format."
                )
            pointers.append(
                PointerRecord(
                    arg_index=arg.index,
                    kernarg_offset=arg.offset,
                    storage_index=storage.index,
                    byte_offset=byte_offset,
                    extent=extent,
                )
            )

        return storages, pointers

    def record(
        self,
        kernel_args: Sequence[Any],
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        lds: int,
        output_arg_index: int = 0,
    ) -> Path:
        """Record `kernel_args` -- the same tuple the kernel is launched with.

        CALL THIS BEFORE THE LAUNCH, NOT AFTER, for the reason the frozen recorder
        gives at the same place: the output buffer is recorded like any other and
        re-uploaded before every replay, so recording it afterwards replays a starting
        state the original launch never had. Nothing here can tell which side of the
        launch it was called on.
        """
        import torch

        layout = self.layout
        storages, pointers = self.plan(kernel_args)

        if output_arg_index not in {p.arg_index for p in pointers}:
            raise RuntimeError(
                f"output_arg_index {output_arg_index} is not a pointer argument; "
                f"pointer arguments are {[p.arg_index for p in pointers]}"
            )

        kernarg = bytearray(layout.kernarg_size)
        manifest: list[str] = [
            f"# captured from {self.asm_path}",
            f"# kernarg_segment_size {layout.kernarg_size}, "
            f"{len(layout.args)} caller arguments, {len(pointers)} pointers into "
            f"{len(storages)} allocations",
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
            if not arg.is_pointer:
                kernarg[arg.offset : arg.offset + arg.size] = _scalar_bytes(arg, value)
            # Pointer slots stay zero. Replay fills them in, so no device address
            # from this machine is baked into the capture.

        for storage in storages:
            fname = f"storage_{storage.index:02d}.bin"
            # A uint8 view over the WHOLE allocation, so the bytes are taken verbatim
            # regardless of any member tensor's dtype, shape or stride -- and so the
            # bytes between two windows are carried across as well.
            flat = torch.empty(0, dtype=torch.uint8, device=self._device(storage))
            flat.set_(storage.handle)
            data = bytes(flat.cpu().numpy())
            if len(data) != storage.nbytes:
                raise RuntimeError(
                    f"storage {storage.index}: read {len(data)} bytes from a "
                    f"{storage.nbytes}-byte allocation"
                )
            (self.outdir / fname).write_bytes(data)
            manifest.append(
                f"# storage {storage.index} carries argument(s) "
                + ", ".join(str(m) for m in storage.members)
            )
            manifest.append(
                f"storage {storage.index} {len(data)} {fname} "
                f"{hashlib.sha256(data).hexdigest()}"
            )
            storage.handle = None  # let the host copy go before the next one is read

        for p in pointers:
            manifest.append(
                f"pointer {p.arg_index} {p.kernarg_offset} {p.storage_index} "
                f"{p.byte_offset} {p.extent}"
            )

        (self.outdir / "kernarg.bin").write_bytes(bytes(kernarg))
        (self.outdir / "manifest.txt").write_text("\n".join(manifest) + "\n")
        return self.outdir

    @staticmethod
    def _device(storage: _Storage) -> Any:
        return storage.handle.device
