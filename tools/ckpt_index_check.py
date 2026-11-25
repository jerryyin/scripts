#!/usr/bin/env python3
"""Checkpoint completeness against model.safetensors.index.json, from stat and safetensors headers only.

ckpt_index_check.py [--split-bytes] [--json] CHECKPOINT...

For each checkpoint directory (a Hugging Face snapshot's symlinks are followed): the index's shard count and
tensor count, shards named by the index that are missing or empty, *.safetensors files the index does not
name (extra), the bytes of the shard files against the index's metadata.total_size (the shard files also
hold their JSON headers, so they exceed total_size by the header bytes). With --split-bytes it reads only
each shard's header (an 8-byte little-endian length, then that much JSON; never tensor data) and sums the
tensor data bytes by kind:
  mtp            a layer at or beyond config.json num_hidden_layers (text_config's when nested), or a tensor
                 name with an `mtp` or `nextn` component: the multi-token-prediction draft layer, whole;
  shared_expert  a name with a shared_expert(s) component;
  routed_expert  a name with an `experts` component;
  other          everything else (attention, dense MLP, embeddings, norms, routers),
and checks that their sum equals total_size. Exit status 0 when every checkpoint has its index and every
indexed shard exists and is nonempty (and, with --split-bytes, the headers name every indexed tensor), else 1.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

KINDS = ("routed_expert", "shared_expert", "mtp", "other")
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def num_hidden_layers(checkpoint: Path) -> int | None:
    path = checkpoint / "config.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    for scope in (
        config,
        config.get("text_config") or {},
        config.get("language_config") or {},
    ):
        if isinstance(scope.get("num_hidden_layers"), int):
            return scope["num_hidden_layers"]
    return None


def kind_of(name: str, layers: int | None) -> str:
    parts = name.split(".")
    match = LAYER.search(name)
    if (layers is not None and match and int(match.group(1)) >= layers) or any(
        p in ("mtp", "nextn") or p.startswith(("mtp_", "nextn_")) for p in parts
    ):
        return "mtp"
    if any(p in ("shared_expert", "shared_experts") for p in parts):
        return "shared_expert"
    if "experts" in parts:
        return "routed_expert"
    return "other"


def read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def check(checkpoint: Path, split: bool) -> dict:
    result: dict = {"checkpoint": str(checkpoint)}
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        result.update(complete=False, error=f"no {index_path.name}")
        return result
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    total_size = int(index.get("metadata", {}).get("total_size", -1))
    shards = sorted(set(weight_map.values()))
    sizes: dict[str, int] = {}
    missing, empty = [], []
    for shard in shards:
        try:
            size = os.stat(
                checkpoint / shard
            ).st_size  # follows a snapshot's symlink to its blob
        except FileNotFoundError:
            missing.append(shard)
            continue
        sizes[shard] = size
        if size == 0:
            empty.append(shard)
    extra = sorted(
        p.name for p in checkpoint.glob("*.safetensors") if p.name not in set(shards)
    )
    shard_bytes = sum(sizes.values())
    result.update(
        shards=len(shards),
        tensors=len(weight_map),
        missing=missing,
        empty=empty,
        extra=extra,
        shard_bytes=shard_bytes,
        index_total_size=total_size,
        shard_bytes_minus_total_size=shard_bytes - total_size,
    )
    complete = not missing and not empty
    if split:
        layers = num_hidden_layers(checkpoint)
        by_kind = dict.fromkeys(KINDS, 0)
        tensors = dict.fromkeys(KINDS, 0)
        seen: set[str] = set()
        for shard in shards:
            if shard in missing or shard in empty:
                continue
            for name, spec in read_header(checkpoint / shard).items():
                begin, end = spec["data_offsets"]
                kind = kind_of(name, layers)
                by_kind[kind] += end - begin
                tensors[kind] += 1
                seen.add(name)
        unheadered = sorted(set(weight_map) - seen)
        data_bytes = sum(by_kind.values())
        result.update(
            num_hidden_layers=layers,
            bytes_by_kind=by_kind,
            tensors_by_kind=tensors,
            header_data_bytes=data_bytes,
            header_data_bytes_equal_total_size=data_bytes == total_size,
            indexed_tensors_not_in_headers=len(unheadered),
        )
        complete = complete and not unheadered
    result["complete"] = complete
    return result


def render(result: dict) -> str:
    if "error" in result:
        return f"{result['checkpoint']}: INCOMPLETE ({result['error']})"
    gb = lambda n: f"{n / 1e9:,.1f} GB"  # noqa: E731 - decimal GB, as the checkpoints' sizes are quoted
    lines = [
        f"{result['checkpoint']}: {'complete' if result['complete'] else 'INCOMPLETE'}: "
        f"{result['tensors']:,} tensors, {result['shards']} shards, {len(result['missing'])} missing, "
        f"{len(result['empty'])} empty, {len(result['extra'])} extra; shard bytes {result['shard_bytes']:,} "
        f"vs index total_size {result['index_total_size']:,} (difference {result['shard_bytes_minus_total_size']:,})"
    ]
    for key in ("missing", "empty", "extra"):
        if result[key]:
            lines.append(f"  {key}: {' '.join(result[key])}")
    if "bytes_by_kind" in result:
        split = ", ".join(
            f"{kind} {gb(result['bytes_by_kind'][kind])} ({result['tensors_by_kind'][kind]:,} tensors)"
            for kind in KINDS
        )
        lines.append(
            f"  headers (num_hidden_layers {result['num_hidden_layers']}): {split}; data bytes "
            f"{result['header_data_bytes']:,} {'==' if result['header_data_bytes_equal_total_size'] else '!='} "
            f"total_size; indexed tensors not in any header: {result['indexed_tensors_not_in_headers']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--split-bytes",
        action="store_true",
        help="read each shard's header and sum bytes by kind",
    )
    parser.add_argument(
        "--json", action="store_true", help="print one JSON object per checkpoint"
    )
    args = parser.parse_args(argv)
    ok = True
    for checkpoint in args.checkpoints:
        result = check(checkpoint, args.split_bytes)
        ok = ok and result["complete"]
        print(json.dumps(result, sort_keys=True) if args.json else render(result))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
