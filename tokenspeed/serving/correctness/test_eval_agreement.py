#!/usr/bin/env python3
"""eval_agreement.py on synthetic manage_server run directories and EvalScope review files (CPU)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_agreement as ea  # noqa: E402


def run_dir(root: Path, label: str, scores: list[list[int]]) -> Path:
    """A run directory with one EvalScope client per repetition; scores[r][q] is 1 if correct."""
    run = root / label
    clients = []
    for rep, row in enumerate(scores):
        work = run / f"gsm8k-{rep + 1}"
        review = work / "20260101" / "reviews" / "model" / "gsm8k_main.jsonl"
        review.parent.mkdir(parents=True)
        review.write_text("".join(json.dumps({"index": q, "sample_score": {"score": {"value": {"acc": s}}},
                                              "extracted_prediction": str(s)}) + "\n" for q, s in enumerate(row)))
        clients.append({"index": rep, "measure": f"gsm8k-{rep + 1}", "rc": 0, "admissible": True,
                        "command": f"evalscope eval --datasets gsm8k --work-dir {work}"})
    clients.append({"index": 9, "measure": "smoke", "rc": 0, "command": "python3 smoke.py"})
    (run / "run.json").write_text(json.dumps({"status": "complete", "clients": clients}))
    return run


def summarize(root, label, scores):
    out = root / f"{label}.json"
    assert ea.main(["summarize-run", "--run-dir", str(run_dir(root, label, scores)), "--label", label,
                    "--dataset", "gsm8k", "--out", str(out)]) == 0
    return out


def test_counts_each_repetition_and_ignores_other_clients(tmp_path):
    summary = json.loads(summarize(tmp_path, "A", [[1, 1, 0, 1], [1, 0, 0, 1]]).read_text())
    assert [(r["correct"], r["total"]) for r in summary["repetitions"]] == [(3, 4), (2, 4)]
    assert summary["dataset"] == "gsm8k"


def test_agreement_follows_the_binomial_rule(tmp_path):
    n = 400
    a = summarize(tmp_path, "A", [[1] * 360 + [0] * 40] * 3)      # 0.900
    near = summarize(tmp_path, "B", [[1] * 356 + [0] * 44] * 3)   # 0.890: |diff| 0.010 < 0.0425
    far = summarize(tmp_path, "C", [[1] * 300 + [0] * 100] * 3)   # 0.750: far outside
    out = tmp_path / "verdict.json"
    assert ea.main(["compare", str(a), str(near), str(far), "--dataset", "gsm8k", "--expected-total", str(n),
                    "--out", str(out)]) == 0
    verdict = {(p["a"], p["b"]): p["verdict"] for p in json.loads(out.read_text())["pairs"]}
    assert verdict == {("A", "B"): "agree", ("A", "C"): "disagree", ("B", "C"): "disagree"}
    assert not json.loads(out.read_text())["layouts"]["A"]["total_mismatch"]
