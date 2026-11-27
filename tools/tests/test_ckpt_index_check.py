"""Tests for ckpt_index_check.py on a tiny synthetic checkpoint (no GPU, no model)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

TOOL = Path(__file__).resolve().parents[1] / "ckpt_index_check.py"
cic = SimpleNamespace(**runpy.run_path(str(TOOL)))

# name -> (shard, data bytes); layer 2 is the MTP layer of a two-layer model
TENSORS = {
    "model.embed_tokens.weight": ("model-00001-of-00002.safetensors", 64),
    "model.layers.0.self_attn.q_proj.weight": ("model-00001-of-00002.safetensors", 32),
    "model.layers.0.mlp.experts.0.gate_proj.weight": (
        "model-00001-of-00002.safetensors",
        100,
    ),
    "model.layers.0.mlp.experts.1.gate_proj.weight": (
        "model-00001-of-00002.safetensors",
        100,
    ),
    "model.layers.1.mlp.shared_experts.down_proj.weight": (
        "model-00002-of-00002.safetensors",
        40,
    ),
    "model.layers.1.mlp.experts.w13_weight": ("model-00002-of-00002.safetensors", 200),
    "model.layers.2.mlp.experts.0.up_proj.weight": (
        "model-00002-of-00002.safetensors",
        16,
    ),
    "model.layers.2.eh_proj.weight": ("model-00002-of-00002.safetensors", 8),
    "mtp.0.norm.weight": ("model-00002-of-00002.safetensors", 4),
}
TOTAL = sum(size for _, size in TENSORS.values())


def write_shard(path: Path, tensors: dict[str, int]) -> None:
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, size in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def make_checkpoint(root: Path, nested_config: bool = False) -> Path:
    root.mkdir(parents=True)
    layers = {"num_hidden_layers": 2}
    config = {"text_config": layers} if nested_config else layers
    (root / "config.json").write_text(json.dumps(config))
    shards: dict[str, dict[str, int]] = {}
    for name, (shard, size) in TENSORS.items():
        shards.setdefault(shard, {})[name] = size
    for shard, tensors in shards.items():
        write_shard(root / shard, tensors)
    index = {
        "metadata": {"total_size": TOTAL},
        "weight_map": {n: s for n, (s, _) in TENSORS.items()},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return root


class CkptIndexCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_complete_checkpoint_and_byte_split(self):
        ckpt = make_checkpoint(self.root / "a")
        result = cic.check(ckpt, split=True)
        self.assertTrue(result["complete"])
        self.assertEqual((result["shards"], result["tensors"]), (2, 9))
        self.assertEqual(
            (result["missing"], result["empty"], result["extra"]), ([], [], [])
        )
        self.assertEqual(result["index_total_size"], TOTAL)
        self.assertGreater(
            result["shard_bytes"], TOTAL
        )  # the shards also hold their headers
        self.assertEqual(
            result["bytes_by_kind"],
            {"routed_expert": 400, "shared_expert": 40, "mtp": 28, "other": 96},
        )
        self.assertEqual(
            result["tensors_by_kind"],
            {"routed_expert": 3, "shared_expert": 1, "mtp": 3, "other": 2},
        )
        self.assertTrue(result["header_data_bytes_equal_total_size"])
        self.assertEqual(result["indexed_tensors_not_in_headers"], 0)

    def test_nested_text_config_layers(self):
        result = cic.check(
            make_checkpoint(self.root / "n", nested_config=True), split=True
        )
        self.assertEqual(result["num_hidden_layers"], 2)
        self.assertEqual(result["bytes_by_kind"]["mtp"], 28)

    def test_stat_only_without_split_reads_no_header(self):
        ckpt = make_checkpoint(self.root / "s")
        for shard in ckpt.glob(
            "*.safetensors"
        ):  # garbage headers: only stat may touch them
            shard.write_bytes(b"\xff" * 16)
        result = cic.check(ckpt, split=False)
        self.assertTrue(result["complete"])
        self.assertNotIn("bytes_by_kind", result)
        self.assertEqual(result["shard_bytes"], 32)

    def test_missing_empty_and_extra_shards(self):
        ckpt = make_checkpoint(self.root / "m")
        (ckpt / "model-00002-of-00002.safetensors").unlink()
        (ckpt / "model-00001-of-00002.safetensors").write_bytes(b"")
        write_shard(ckpt / "stray.safetensors", {"x": 1})
        result = cic.check(ckpt, split=True)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing"], ["model-00002-of-00002.safetensors"])
        self.assertEqual(result["empty"], ["model-00001-of-00002.safetensors"])
        self.assertEqual(result["extra"], ["stray.safetensors"])
        self.assertEqual(result["indexed_tensors_not_in_headers"], 9)

    def test_symlinked_snapshot_is_followed(self):
        blobs = make_checkpoint(self.root / "blobs")
        snap = self.root / "snapshot"
        snap.mkdir()
        for path in blobs.iterdir():
            (snap / path.name).symlink_to(path)
        result = cic.check(snap, split=True)
        self.assertTrue(result["complete"])
        self.assertEqual(result["header_data_bytes"], TOTAL)

    def test_cli_exit_status_text_and_json(self):
        good = make_checkpoint(self.root / "g")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cic.main(["--split-bytes", str(good)]), 0)
        self.assertIn(
            ": complete: 9 tensors, 2 shards, 0 missing, 0 empty, 0 extra",
            out.getvalue(),
        )
        self.assertIn("routed_expert 0.0 GB (3 tensors)", out.getvalue())
        noindex = self.root / "noindex"
        noindex.mkdir()
        copied = shutil.copy2(TOOL, self.root / "elsewhere.py")  # runs copied out of the checkout
        done = subprocess.run(
            [
                sys.executable,
                str(copied),
                "--json",
                str(good),
                str(noindex),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 1)
        first, second = (json.loads(line) for line in done.stdout.splitlines())
        self.assertTrue(first["complete"])
        self.assertEqual(second["error"], "no model.safetensors.index.json")


class FakeHub:
    """A local HTTP server answering the Hub API listing and resolve URLs (with Range) from a directory."""

    def __init__(self, files: Path, repo: str = "org/model", sha: str = "a" * 40) -> None:
        self.files, self.repo, self.sha = files, repo, sha
        self.honor_range, self.status, self.requests = True, None, []
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output clean
                pass

            def do_GET(self):
                hub.requests.append((self.path, self.headers.get("Range"), self.headers.get("Authorization")))
                if hub.status:
                    self.send_response(hub.status)
                    self.end_headers()
                    return
                api = f"/api/models/{hub.repo}/revision/"
                resolve = f"/{hub.repo}/resolve/{hub.sha}/"
                if self.path.startswith(api):
                    body = json.dumps(hub.listing()).encode()
                    code = 200
                elif self.path.startswith(resolve):
                    body = (hub.files / self.path[len(resolve):]).read_bytes()
                    code = 200
                    span = self.headers.get("Range")
                    if span and hub.honor_range:
                        begin, end = (int(x) for x in span.split("=")[1].split("-"))
                        body, code = body[begin : end + 1], 206
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"

    def listing(self) -> dict:
        siblings = []
        for path in sorted(self.files.rglob("*")):
            if not path.is_file():
                continue
            data = path.read_bytes()
            entry = {"rfilename": path.relative_to(self.files).as_posix(), "size": len(data)}
            if path.suffix == ".safetensors":  # LFS: blobId is the pointer's, the content id is lfs.sha256
                entry.update(blobId="0" * 40, lfs={"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
            else:
                entry["blobId"] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            siblings.append(entry)
        return {"sha": self.sha, "lastModified": "2026-01-01T00:00:00.000Z", "siblings": siblings}

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class HubIdentityTests(unittest.TestCase):
    SHARD = "model-00002-of-00002.safetensors"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.local = make_checkpoint(self.root / "local")
        self.remote = self.root / "remote"
        shutil.copytree(self.local, self.remote)
        self.hub = FakeHub(self.remote)

    def tearDown(self) -> None:
        self.hub.close()
        self.tmp.cleanup()

    def identity(self, **kwargs) -> dict:
        kwargs.setdefault("endpoint", self.hub.endpoint)
        return cic.hub_identity(self.local, "org/model", "main", **kwargs)

    def test_identical_checkpoint(self):
        doc = self.identity(hash_shards=True, sample=3, sample_shards=(self.SHARD,))
        self.assertTrue(doc["all_checked_equal"], doc)
        self.assertEqual(doc["revision"], "a" * 40)
        s = doc["summary"]
        self.assertEqual((s["common"], s["only_hub"], s["only_local"]), (4, 0, 0))
        self.assertEqual((s["id_equal"], s["id_checked"], s["shard_headers_equal"], s["shards"]), (4, 4, 2, 2))
        self.assertEqual((s["samples_equal"], s["samples"]), (3, 3))
        self.assertEqual(doc["files"]["config.json"]["id_kind"], "git-blob-sha1")
        self.assertEqual(doc["files"][self.SHARD]["id_kind"], "lfs-sha256")
        # resolve requests use the resolved commit, and every shard request is a Range request
        shard_requests = [r for r in self.hub.requests if "/resolve/" in r[0]]
        self.assertTrue(shard_requests and all(f"/resolve/{'a' * 40}/" in r[0] and r[1] for r in shard_requests))

    def test_shards_not_hashed_without_hash_shards(self):
        doc = self.identity()
        self.assertTrue(doc["all_checked_equal"])
        self.assertIsNone(doc["files"][self.SHARD]["id_equal"])
        self.assertEqual(doc["summary"]["id_checked"], 2)  # config.json and the index only

    def test_same_headers_different_tensor_bytes(self):
        shard = self.remote / self.SHARD
        length = struct.unpack("<Q", shard.read_bytes()[:8])[0]
        data = bytearray(shard.read_bytes())
        data[8 + length :] = b"\x01" * (len(data) - 8 - length)  # every tensor's bytes change, header kept
        shard.write_bytes(bytes(data))
        doc = self.identity(hash_shards=True, sample=2, sample_shards=(self.SHARD,))
        self.assertFalse(doc["all_checked_equal"])
        s = doc["summary"]
        self.assertEqual((s["size_equal"], s["shard_headers_equal"], s["id_equal"], s["id_checked"]), (4, 2, 3, 4))
        self.assertFalse(doc["files"][self.SHARD]["id_equal"])
        # --sample 2 compares exactly the two smallest tensors (mtp.0.norm.weight 4 B, eh_proj 8 B)
        self.assertEqual([x["tensor"] for x in doc["tensor_samples"]], ["mtp.0.norm.weight", "model.layers.2.eh_proj.weight"])
        self.assertEqual([x["equal"] for x in doc["tensor_samples"]], [False, False])

    def test_header_and_non_lfs_file_differences_and_one_sided_files(self):
        write_shard(self.remote / self.SHARD, {"renamed": 4})
        (self.remote / "config.json").write_text(json.dumps({"num_hidden_layers": 3}))
        (self.remote / "README.md").write_text("hub only")
        (self.local / "local_only.txt").write_text("x")
        doc = self.identity()
        self.assertFalse(doc["headers_equal"][self.SHARD])
        self.assertFalse(doc["files"]["config.json"]["id_equal"])
        self.assertEqual((doc["only_hub"], doc["only_local"]), (["README.md"], ["local_only.txt"]))
        self.assertFalse(doc["all_checked_equal"])

    def test_sample_count_and_seeded_choice(self):
        many = {f"t{i}": 10 + i for i in range(12)}
        write_shard(self.local / "x.safetensors", many)
        shutil.copy2(self.local / "x.safetensors", self.remote / "x.safetensors")
        first = self.identity(sample=7, seed=5, sample_shards=("x.safetensors",))
        again = self.identity(sample=7, seed=5, sample_shards=("x.safetensors",))
        names = [x["tensor"] for x in first["tensor_samples"]]
        self.assertEqual(len(set(names)), 7)
        self.assertEqual(names[:4], ["t0", "t1", "t2", "t3"])
        self.assertEqual(names, [x["tensor"] for x in again["tensor_samples"]])

    def test_cache_blob_of_a_snapshot_symlink(self):
        blobs = self.root / "cache" / "blobs"
        blobs.mkdir(parents=True)
        snap = self.root / "cache" / "snapshots" / "rev"
        snap.mkdir(parents=True)
        for path in self.local.iterdir():
            data = path.read_bytes()
            name = hashlib.sha256(data).hexdigest() if path.suffix == ".safetensors" else "f" * 40
            (blobs / name).write_bytes(data)
            (snap / path.name).symlink_to(os.path.relpath(blobs / name, snap))
        doc = cic.hub_identity(snap, "org/model", endpoint=self.hub.endpoint)
        self.assertTrue(doc["files"][self.SHARD]["cache_blob_equal"])
        self.assertFalse(doc["files"]["config.json"]["cache_blob_equal"])  # wrong blob name is reported
        self.assertEqual(doc["summary"]["cache_blob_checked"], 4)

    def test_server_ignoring_range_is_an_error_not_a_full_download(self):
        self.hub.honor_range = False
        doc = self.identity()
        self.assertIn("instead of 206", doc["error"])
        self.assertFalse(doc["all_checked_equal"])

    def test_gated_repo_without_credentials_is_reported(self):
        self.hub.status = 401
        doc = self.identity()
        self.assertIn("HTTP 401", doc["error"])
        self.assertIn("HF_TOKEN", doc["error"])
        self.hub.status = None
        self.identity(token="secret")
        self.assertTrue(all(r[2] == "Bearer secret" for r in self.hub.requests[-3:]))

    def test_cli_copied_out_exit_status(self):
        copied = shutil.copy2(TOOL, self.root / "elsewhere.py")
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        argv = [sys.executable, str(copied), "--hub-repo", "org/model", "--hub-endpoint", self.hub.endpoint,
                "--sample", "1", "--sample-shard", self.SHARD]
        done = subprocess.run(argv + ["--json", str(self.local)], capture_output=True, text=True, env=env)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(json.loads(done.stdout)["hub"]["all_checked_equal"])
        (self.remote / "config.json").write_text("{}")
        done = subprocess.run(argv + [str(self.local)], capture_output=True, text=True, env=env)
        self.assertEqual(done.returncode, 1)
        self.assertIn("hub org/model@" + "a" * 40 + ": DIFFERS", done.stdout)
        self.assertIn("differing files: config.json", done.stdout)
        bad = subprocess.run([sys.executable, str(copied), "--hash-shards", str(self.local)], capture_output=True)
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
