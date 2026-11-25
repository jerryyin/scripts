"""Tests for ckpt_index_check.py on a tiny synthetic checkpoint (no GPU, no model)."""

from __future__ import annotations

import io
import json
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
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


if __name__ == "__main__":
    unittest.main()
