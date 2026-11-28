#!/usr/bin/env python3
"""Check a safetensors checkpoint: completeness against its index (stat and headers), optionally identity with a Hub revision.

ckpt_index_check.py [--split-bytes] [--by-pattern] [--json] [--hub-repo REPO [--hub-revision REV] [--hub-endpoint URL]
                    [--hash-shards] [--jobs N] [--sample N] [--sample-shard NAME]... [--seed S]] CHECKPOINT...

For each checkpoint directory (a Hugging Face snapshot's symlinks are followed): the index's shard count and
tensor count, shards named by the index that are missing or empty, *.safetensors files the index does not
name (extra), the bytes of the shard files against the index's metadata.total_size (the shard files also
hold their JSON headers, so they exceed total_size by the header bytes). With --split-bytes it reads only
each shard's header (an 8-byte little-endian length, then that much JSON; never tensor data) and sums the
tensor data bytes by kind:
  mtp            a layer at or beyond config.json num_hidden_layers (text_config's or language_config's when
                 nested), or a tensor
                 name with an `mtp` or `nextn` component: the multi-token-prediction draft layer, whole;
  shared_expert  a name with a shared_expert(s) component;
  routed_expert  a name with an `experts` component;
  other          everything else (attention, dense MLP, embeddings, norms, routers),
and checks that their sum equals total_size. With --by-pattern (implies --split-bytes) it also sums bytes and
tensor counts per name pattern, the tensor name with every all-digit component replaced by `N` (so
`layers.3.ffn.experts.17.w1.weight` -> `layers.N.ffn.experts.N.w1.weight`), each with its dtypes and the shape
of its first tensor: the input a per-rank memory budget needs (which family is sharded, replicated or offloaded).

With --hub-repo it also checks the checkpoint's identity against a Hugging Face Hub repo revision without
downloading the repo: it lists the revision with the Hub API (?blobs=true) and, for every file present on
both sides (local files are walked recursively, skipping __pycache__ and a top-level .cache), compares the size and the content identity the Hub records (the LFS sha256 for LFS files, the git
blob SHA-1 of the bytes for the rest; the local side is hashed, *.safetensors only with --hash-shards, which
reads every shard). It fetches each common shard's safetensors header with HTTP Range requests and compares it
with the local header, and for each --sample-shard byte-compares --sample tensors (the smallest min(4, N),
then a --seed'ed random choice of the rest; at most the first 64 MiB of each) over HTTP Range. A file inside
a Hugging Face cache snapshot also reports the cache blob it links to, the identity the download recorded
(not a hash of the bytes). HF_TOKEN (or HUGGING_FACE_HUB_TOKEN), when set, is sent for gated repos; without it a gated or missing repo
is reported as an error, not a crash. The endpoint is HF_ENDPOINT or https://huggingface.co.

Exit status 0 when every checkpoint has its index and every indexed shard exists and is nonempty (and, with
--split-bytes, the headers name every indexed tensor; with --hub-repo, every checked fact equals the Hub's
and no file is on one side only), else 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KINDS = ("routed_expert", "shared_expert", "mtp", "other")
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
HUB_ENDPOINT = "https://huggingface.co"
SAMPLE_CAP = 64 * 1024 * 1024  # bytes of one sampled tensor compared over HTTP Range
HEADER_CAP = 256 * 1024 * 1024  # a larger safetensors header length is treated as corrupt


def pattern_of(name: str) -> str:
    """The tensor name with every all-digit component replaced by N."""
    return ".".join("N" if part.isdigit() else part for part in name.split("."))


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


def nbytes(spec: dict) -> int:
    """The tensor data bytes of one safetensors header entry."""
    begin, end = spec["data_offsets"]
    return end - begin


def local_reader(path: Path):
    """read(offset, count) -> bytes of a local file."""

    def read(offset: int, count: int) -> bytes:
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(count)

    return read


def safetensors_header(read, cap: int | None = None, where: str = "") -> tuple[int, dict]:
    """A shard's header length and header, __metadata__ included, through read(offset, count) (a local file or
    HTTP Range); with cap, a length outside 1..cap is a HubError naming `where` (so no huge Range request)."""
    (length,) = struct.unpack("<Q", read(0, 8))
    if cap is not None and not 0 < length <= cap:
        raise HubError(f"implausible safetensors header length {length} in {where}")
    return length, json.loads(read(8, length))


def check(checkpoint: Path, split: bool, by_pattern: bool = False) -> dict:
    split = split or by_pattern
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
    indexed = set(shards)
    extra = sorted(p.name for p in checkpoint.glob("*.safetensors") if p.name not in indexed)
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
        patterns: dict[str, dict] = {}
        for shard in shards:
            if shard in missing or shard in empty:
                continue
            _, header = safetensors_header(local_reader(checkpoint / shard))
            header.pop("__metadata__", None)
            for name, spec in header.items():
                size = nbytes(spec)
                kind = kind_of(name, layers)
                by_kind[kind] += size
                tensors[kind] += 1
                seen.add(name)
                if by_pattern:
                    entry = patterns.setdefault(
                        pattern_of(name),
                        {"bytes": 0, "tensors": 0, "dtypes": [], "shape": spec["shape"], "kind": kind},
                    )
                    entry["bytes"] += size
                    entry["tensors"] += 1
                    if spec["dtype"] not in entry["dtypes"]:
                        entry["dtypes"].append(spec["dtype"])
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
        if by_pattern:
            result["bytes_by_pattern"] = dict(sorted(patterns.items()))
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
    for pattern, entry in result.get("bytes_by_pattern", {}).items():
        lines.append(
            f"  {pattern}: {entry['bytes']:,} bytes ({gb(entry['bytes'])}), {entry['tensors']:,} tensors, "
            f"{'/'.join(entry['dtypes'])} {entry['shape']}, {entry['kind']}"
        )
    return "\n".join(lines)


class HubError(Exception):
    """A Hub request failed (HTTP status, network, or a server that ignored a Range request)."""


def hub_get(url: str, token: str | None, start: int | None = None, end: int | None = None) -> bytes:
    """GET url (bytes start..end inclusive when given, which must come back as 206 with exactly that length)."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as response:
            if start is None:
                return response.read()
            want = end - start + 1
            if response.status != 206:  # never read a whole multi-GB shard by accident
                raise HubError(f"HTTP {response.status} instead of 206 for a Range request to {url}")
            body = response.read(want + 1)
    except urllib.error.HTTPError as error:
        hint = " (gated or private: set HF_TOKEN)" if error.code in (401, 403) else ""
        raise HubError(f"HTTP {error.code} for {url}{hint}") from None
    except urllib.error.URLError as error:
        raise HubError(f"{error.reason} for {url}") from None
    if len(body) != want:
        raise HubError(f"{len(body)} bytes instead of {want} for range {start}-{end} of {url}")
    return body


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The git object id of a file's bytes, which the Hub lists as blobId for a non-LFS file."""
    digest = hashlib.sha1()
    digest.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def hub_reader(url: str, token: str | None):
    """read(offset, count) -> bytes of a Hub file, over HTTP Range."""
    return lambda offset, count: hub_get(url, token, offset, offset + count - 1)


def cache_blob(path: Path) -> str | None:
    """The blob a Hugging Face cache snapshot file links to (its recorded LFS sha256 or git blob SHA-1)."""
    if not path.is_symlink():
        return None
    target = Path(os.path.realpath(path))
    return target.name if target.parent.name == "blobs" else None


def local_files(checkpoint: Path) -> dict[str, Path]:
    files = {}
    for root, dirs, names in os.walk(checkpoint, followlinks=True):
        rel_root = Path(root).relative_to(checkpoint)
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and (rel_root / d).as_posix() != ".cache")
        for name in names:
            files[(rel_root / name).as_posix()] = Path(root) / name
    return files


def pick_tensors(header: dict, count: int, rng: random.Random) -> list[str]:
    """The smallest min(4, count) nonempty tensors, then a seeded random choice of the rest, count in all."""
    names = sorted(k for k in header if k != "__metadata__" and nbytes(header[k]) > 0)
    smallest = sorted(names, key=lambda k: (nbytes(header[k]), k))[: min(4, count)]
    chosen = set(smallest)
    rest = [k for k in names if k not in chosen]
    return smallest + rng.sample(rest, min(len(rest), max(0, count - len(smallest))))


def hub_identity(
    checkpoint: Path,
    repo: str,
    revision: str = "main",
    endpoint: str = HUB_ENDPOINT,
    token: str | None = None,
    hash_shards: bool = False,
    jobs: int = 6,
    sample: int = 10,
    sample_shards: tuple[str, ...] = (),
    seed: int = 0,
) -> dict:
    """Compare a local checkpoint with the Hub's repo@revision; returns the facts (or {"error": ...})."""
    endpoint = endpoint.rstrip("/")
    quote = lambda text: urllib.parse.quote(text, safe="/")  # noqa: E731
    doc: dict = {"repo": repo, "revision_requested": revision, "endpoint": endpoint}
    try:
        meta = json.loads(hub_get(f"{endpoint}/api/models/{quote(repo)}/revision/{quote(revision)}?blobs=true", token))
        resolved = meta["sha"]
        hub = {s["rfilename"]: s for s in meta["siblings"]}
        local = local_files(checkpoint)
        common = sorted(set(hub) & set(local))
        only_hub, only_local = sorted(set(hub) - set(local)), sorted(set(local) - set(hub))
        shards = [n for n in common if n.endswith(".safetensors")]
        to_hash = [n for n in common if not n.endswith(".safetensors") or hash_shards]

        def identity(name: str) -> str:
            return file_sha256(local[name]) if hub[name].get("lfs") else git_blob_sha1(local[name])

        with ThreadPoolExecutor(max(1, jobs)) as pool:
            ids = dict(zip(to_hash, pool.map(identity, to_hash)))
        files = {}
        for name in common:
            entry = hub[name]
            lfs = entry.get("lfs") or {}
            want = lfs.get("sha256") or entry.get("blobId")
            blob = cache_blob(local[name])
            local_size, hub_size = local[name].stat().st_size, entry.get("size", lfs.get("size"))
            files[name] = {
                "local_size": local_size,
                "hub_size": hub_size,
                "size_equal": local_size == hub_size,
                "id_kind": "lfs-sha256" if lfs else "git-blob-sha1",
                "hub_id": want,
                "local_id": ids.get(name),
                "id_equal": None if name not in ids else ids[name] == want,
                "cache_blob": blob,
                "cache_blob_equal": None if blob is None else blob == want,
            }
        url = lambda name: f"{endpoint}/{quote(repo)}/resolve/{resolved}/{quote(name)}"  # noqa: E731
        with ThreadPoolExecutor(max(1, jobs)) as pool:
            remote = dict(
                zip(shards, pool.map(lambda n: safetensors_header(hub_reader(url(n), token), HEADER_CAP, url(n)), shards))
            )
        headers, samples = {}, []
        rng = random.Random(seed)
        for name in shards:
            local_length, local_header = safetensors_header(local_reader(local[name]))
            headers[name] = remote[name][1] == local_header
            if name not in sample_shards:
                continue
            for tensor in pick_tensors(local_header, sample, rng):
                spec = local_header[tensor]
                start = 8 + local_length + spec["data_offsets"][0]
                count = min(nbytes(spec), SAMPLE_CAP)
                theirs = hub_reader(url(name), token)(start, count)
                ours = local_reader(local[name])(start, count)
                samples.append(
                    {
                        "shard": name,
                        "tensor": tensor,
                        "dtype": spec["dtype"],
                        "shape": spec["shape"],
                        "offset": start,
                        "bytes": count,
                        "equal": ours == theirs,
                    }
                )
    except HubError as error:
        doc["error"] = str(error)
        doc["all_checked_equal"] = False
        return doc
    summary = {
        "common": len(common),
        "only_hub": len(only_hub),
        "only_local": len(only_local),
        "size_equal": sum(v["size_equal"] for v in files.values()),
        "id_checked": sum(v["id_equal"] is not None for v in files.values()),
        "id_equal": sum(bool(v["id_equal"]) for v in files.values()),
        "cache_blob_checked": sum(v["cache_blob_equal"] is not None for v in files.values()),
        "cache_blob_equal": sum(bool(v["cache_blob_equal"]) for v in files.values()),
        "shards": len(shards),
        "shard_headers_equal": sum(headers.values()),
        "samples": len(samples),
        "samples_equal": sum(s["equal"] for s in samples),
    }
    doc.update(
        revision=resolved,
        last_modified=meta.get("lastModified"),
        only_hub=only_hub,
        only_local=only_local,
        sample_shards_not_found=sorted(set(sample_shards) - set(shards)),
        files=files,
        headers_equal=headers,
        tensor_samples=samples,
        summary=summary,
        all_checked_equal=(
            not summary["only_hub"]
            and not summary["only_local"]
            and summary["size_equal"] == summary["common"]
            and summary["id_equal"] == summary["id_checked"]
            and summary["shard_headers_equal"] == summary["shards"]
            and summary["samples_equal"] == summary["samples"]
        ),
    )
    return doc


def render_hub(doc: dict) -> str:
    where = f"{doc['repo']}@{doc.get('revision', doc['revision_requested'])}"
    if "error" in doc:
        return f"  hub {where}: ERROR {doc['error']}"
    s = doc["summary"]
    lines = [
        f"  hub {where}: {'all checked facts equal' if doc['all_checked_equal'] else 'DIFFERS'}: "
        f"{s['common']} common files ({s['only_hub']} only on the Hub, {s['only_local']} only local); sizes equal "
        f"{s['size_equal']}/{s['common']}; content ids equal {s['id_equal']}/{s['id_checked']} hashed; cache blobs "
        f"equal {s['cache_blob_equal']}/{s['cache_blob_checked']}; shard headers equal "
        f"{s['shard_headers_equal']}/{s['shards']}; sampled tensors equal {s['samples_equal']}/{s['samples']}"
    ]
    for key in ("only_hub", "only_local", "sample_shards_not_found"):
        if doc[key]:
            lines.append(f"    {key}: {' '.join(doc[key])}")
    differ = [n for n, v in doc["files"].items() if v["id_equal"] is False or not v["size_equal"]]
    if differ:
        lines.append(f"    differing files: {' '.join(differ)}")
    for sample in doc["tensor_samples"]:
        lines.append(
            f"    sample {sample['shard']} {sample['tensor']} {sample['dtype']} {sample['shape']} "
            f"{sample['bytes']:,} bytes: {'equal' if sample['equal'] else 'DIFFERS'}"
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
        "--by-pattern",
        action="store_true",
        help="also sum bytes per tensor-name pattern (digits -> N), with dtype and shape (implies --split-bytes)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print one JSON object per checkpoint"
    )
    hub = parser.add_argument_group("Hub identity (no download)")
    hub.add_argument("--hub-repo", help="compare with this Hugging Face repo, e.g. org/name")
    hub.add_argument("--hub-revision", default="main", help="branch, tag or commit (default main)")
    hub.add_argument(
        "--hub-endpoint",
        default=os.environ.get("HF_ENDPOINT") or HUB_ENDPOINT,
        help="Hub base URL (default HF_ENDPOINT or https://huggingface.co)",
    )
    hub.add_argument("--hash-shards", action="store_true", help="sha256 every local *.safetensors (reads them all)")
    hub.add_argument("--jobs", type=int, default=6, help="parallel hashes and header requests")
    hub.add_argument("--sample", type=int, default=10, help="tensors compared per --sample-shard")
    hub.add_argument("--sample-shard", action="append", default=[], help="shard file name to sample (repeatable)")
    hub.add_argument("--seed", type=int, default=0, help="seed of the random tensor choice")
    args = parser.parse_args(argv)
    if not args.hub_repo and (args.hash_shards or args.sample_shard):
        parser.error("--hash-shards and --sample-shard need --hub-repo")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    ok = True
    for checkpoint in args.checkpoints:
        result = check(checkpoint, args.split_bytes, args.by_pattern)
        ok = ok and result["complete"]
        if args.hub_repo:
            result["hub"] = hub_identity(
                checkpoint,
                args.hub_repo,
                args.hub_revision,
                args.hub_endpoint,
                token,
                args.hash_shards,
                args.jobs,
                args.sample,
                tuple(args.sample_shard),
                args.seed,
            )
            ok = ok and result["hub"]["all_checked_equal"]
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(render(result) + ("\n" + render_hub(result["hub"]) if "hub" in result else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
