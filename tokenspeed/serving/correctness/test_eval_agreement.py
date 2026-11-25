#!/usr/bin/env python3
"""eval_agreement.py on synthetic manage_server run directories and EvalScope review files (CPU)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_agreement as ea  # noqa: E402


def run_dir(root: Path, label: str, scores: list[list[int]], admissible=None) -> Path:
    """A run directory with one EvalScope client per repetition; scores[r][q] is 1 if correct;
    admissible[r] is repetition r's window verdict (default true)."""
    run = root / label
    clients = []
    for rep, row in enumerate(scores):
        work = run / f"gsm8k-{rep + 1}"
        review = work / "20260101" / "reviews" / "model" / "gsm8k_main.jsonl"
        review.parent.mkdir(parents=True)
        review.write_text("".join(json.dumps({"index": q, "sample_score": {"score": {"value": {"acc": s}}},
                                              "extracted_prediction": str(s)}) + "\n" for q, s in enumerate(row)))
        ok = True if admissible is None else admissible[rep]
        clients.append({"index": rep, "measure": f"gsm8k-{rep + 1}", "rc": 0, "admissible": ok, "record_admissible": ok,
                        "command": f"evalscope eval --datasets gsm8k --work-dir {work}"})
    clients.append({"index": 9, "measure": "smoke", "rc": 0, "command": "python3 smoke.py"})
    (run / "run.json").write_text(json.dumps({"status": "complete", "clients": clients}))
    return run


def summarize(root, label, scores, admissible=None):
    out = root / f"{label}.json"
    assert ea.main(["summarize-run", "--run-dir", str(run_dir(root, label, scores, admissible)), "--label", label,
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


def test_counts_from_summary_matches_the_review_verdict_without_review_files(tmp_path):
    a = summarize(tmp_path, "A", [[1] * 360 + [0] * 40, [1] * 358 + [0] * 42, [1] * 361 + [0] * 39])
    b = summarize(tmp_path, "B", [[1] * 356 + [0] * 44] * 3)
    c = summarize(tmp_path, "C", [[1] * 300 + [0] * 100] * 3)
    reviews, counts = tmp_path / "reviews.json", tmp_path / "counts.json"
    args = [str(a), str(b), str(c), "--dataset", "gsm8k", "--expected-total", "400"]
    assert ea.main(["compare", *args, "--out", str(reviews)]) == 0
    for label in "ABC":  # only the summaries are handed over
        for d in (tmp_path / label).glob("gsm8k-*"):
            for f in sorted(d.rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
    assert ea.main(["compare", *args, "--counts-from-summary", "--out", str(counts)]) == 0
    r, c_ = json.loads(reviews.read_text()), json.loads(counts.read_text())
    assert (r["mode"], c_["mode"]) == ("per-question-reviews", "counts-from-summary")
    key = lambda rep: [(p["a"], p["b"], p["verdict"], round(p["median_diff"], 12), round(p["threshold_95"], 12))
                       for p in rep["pairs"]]
    assert key(c_) == key(r) and [p["verdict"] for p in c_["pairs"]] == ["agree", "disagree", "disagree"]
    assert c_["layouts"]["A"]["reps"] == [[1, 360, 400], [2, 358, 400], [3, 361, 400]]
    assert c_["layouts"]["A"]["within_discordance"] is None and c_["layouts"]["A"]["questions_not_stable_across_reps"] is None
    assert all(p[k] is None for p in c_["pairs"] for k in ("between_discordance", "between_discordant_median",
                                                            "between_discordant_min_max", "a_all_right_b_all_wrong",
                                                            "b_all_right_a_all_wrong"))


def test_admissible_only_drops_and_lists_inadmissible_repetitions(tmp_path):
    a = summarize(tmp_path, "A", [[1] * 360 + [0] * 40] * 3)
    # B's repetition 2 scored far lower in a window another process disturbed
    b = summarize(tmp_path, "B", [[1] * 358 + [0] * 42, [1] * 200 + [0] * 200, [1] * 356 + [0] * 44],
                  admissible=[True, False, True])
    out = tmp_path / "verdict.json"
    for mode in ([], ["--counts-from-summary"]):
        assert ea.main(["compare", str(a), str(b), "--dataset", "gsm8k", "--admissible-only", *mode,
                        "--out", str(out)]) == 0
        v = json.loads(out.read_text())
        assert v["admissible_only"] and [(x["layout"], x["rep"], x["admissible"]) for x in v["excluded"]] == [("B", 2, False)]
        assert [rep for rep, _, _ in v["layouts"]["B"]["reps"]] == [1, 3]
        assert v["layouts"]["B"]["accuracy_min"] == 356 / 400 and v["pairs"][0]["verdict"] == "agree"
    assert ea.main(["compare", str(a), str(b), "--dataset", "gsm8k", "--out", str(out)]) == 0
    v = json.loads(out.read_text())  # default: every repetition counts
    assert not v["admissible_only"] and v["excluded"] == [] and len(v["layouts"]["B"]["reps"]) == 3
