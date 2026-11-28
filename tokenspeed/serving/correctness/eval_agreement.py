#!/usr/bin/env python3
"""Do two or more serving configurations agree on an EvalScope accuracy benchmark?

  eval_agreement.py summarize-run --run-dir RUN_DIR --label LABEL --dataset NAME --out SUMMARY.json
      a `manage_server.py run --config` run directory (amd-tracking
      skills/inference_engine/benchmark/scripts/manage_server.py: RUN_DIR/run.json lists one record
      per client; a client whose command runs EvalScope on NAME with `--work-dir DIR` is one
      repetition) -> SUMMARY.json: per repetition correct/total from EvalScope's per-question review
      files, stop reasons and output tokens from its prediction files, its report score, the client's
      rc and admissibility verdicts; plus the run's status and server stop. Exit 1 when no client
      evaluated NAME.
  eval_agreement.py compare SUMMARY.json [SUMMARY.json ...] --dataset NAME [--expected-total N]
                            [--review LABEL:REP=PATH ...] [--counts-from-summary] [--admissible-only]
                            [--out VERDICT.json]
      groups repetitions by label; median and [min, max] accuracy (numerics/range_compare.py);
      binomial standard error sqrt(p(1-p)/n); per-question discordance within and between labels;
      and for each pair the verdict of the rule fixed before the data:
      agree  <=>  |median_A - median_B| <= 1.96 * sqrt(pA(1-pA)/n + pB(1-pB)/n), p = medians.
      --expected-total flags a repetition that scored a different number of questions.
      --counts-from-summary takes each repetition's correct/total from SUMMARY.json instead of
      reading its review files (when only the summaries are at hand): same verdict rule, and the
      per-question discordance fields are null ("mode": "counts-from-summary").
      --admissible-only leaves out a repetition whose measurement window was not admissible
      (summary "admissible" not true, or its record's "record_admissible" false) and lists it under
      "excluded".

EvalScope review format (1.x; checked against 1.11.1): the work dir holds
<timestamp>/reviews/<model>/<dataset>*.jsonl, one JSON object per question, with the question id in
"index" (or sample_score.sample_id / "id") and its score under "acc" or sample_score's "accuracy",
and the extracted answer under "extracted_prediction". Greedy outputs are not always repeatable, so
compare repetitions of each configuration, not single runs.
"""
import argparse, glob, itertools, json, math, os, sys
from pathlib import Path
import importlib.util

Z95 = 1.96
_RC = Path(__file__).resolve().parents[3] / "numerics" / "range_compare.py"
_spec = importlib.util.spec_from_file_location("range_compare", _RC)
range_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(range_compare)


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj and not isinstance(obj[key], (dict, list)):
            return obj[key]
        for v in obj.values():
            got = _find(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find(v, key)
            if got is not None:
                return got
    return None


def review_files(work_dir: str | Path, dataset: str) -> list[str]:
    if os.path.isfile(str(work_dir)):  # a review file given directly (compare --review)
        return [str(work_dir)]
    return sorted(glob.glob(os.path.join(str(work_dir), "**", "reviews", "**", f"*{dataset}*.jsonl"), recursive=True))


def per_question(work_dir: str | Path, dataset: str) -> dict[str, dict]:
    """question id -> {"correct": 0/1, "pred": extracted answer} from the EvalScope review file(s)."""
    out: dict[str, dict] = {}
    for f in review_files(work_dir, dataset):
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("index", _find(rec.get("sample_score", {}), "sample_id"))
            qid = rec.get("id") if qid is None else qid
            score = _find(rec, "acc")
            if score is None:  # EvalScope 1.11.1: sample_score.score.value.accuracy
                score = _find(rec.get("sample_score", {}), "accuracy")
            if qid is None or score is None:
                raise ValueError(f"{f}: review record without id or acc: {sorted(rec)}")
            out[str(qid)] = {"correct": 1 if float(score) >= 0.5 else 0,
                             "pred": _find(rec, "extracted_prediction")}
    return out


def prediction_stats(work_dir: str | Path, dataset: str) -> dict:
    """Stop reasons and output-token counts from EvalScope 1.11 prediction files
    (predictions/<model>/<dataset>*.jsonl: model_output.choices[0].stop_reason, model_output.usage.output_tokens);
    'max_tokens' is EvalScope's name for an OpenAI finish_reason 'length' ."""
    reasons: dict[str, int] = {}
    tokens = []
    for f in sorted(glob.glob(os.path.join(str(work_dir), "**", "predictions", "**", f"*{dataset}*.jsonl"), recursive=True)):
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            out = json.loads(line).get("model_output") or {}
            choices = out.get("choices") or [{}]
            r = str(choices[0].get("stop_reason"))
            reasons[r] = reasons.get(r, 0) + 1
            t = (out.get("usage") or {}).get("output_tokens")
            if t is not None:
                tokens.append(int(t))
    return {"stop_reasons": reasons, "max_tokens_stops": reasons.get("max_tokens", 0),
            "output_tokens_max": max(tokens) if tokens else None, "n_predictions": sum(reasons.values())}


def report_score(work_dir: str | Path):
    """EvalScope's own report score (reports/<model>/gsm8k.json, top-level "score") per report file."""
    out = {}
    for f in sorted(glob.glob(os.path.join(str(work_dir), "**", "reports", "**", "*.json"), recursive=True)):
        try:
            rep = json.loads(Path(f).read_text())
            score = rep.get("score")
            if score is None and rep.get("metrics"):  # EvalScope 1.11.1 report schema_version 2
                score = {m["identity"]["name"]: m["score"] for m in rep["metrics"]}
            out[f] = score
        except (ValueError, OSError) as e:
            out[f] = f"unreadable: {e}"
    return out


def _work_dir(command: str, dataset: str) -> str | None:
    """The EvalScope --work-dir of a manage_server client command that evaluates DATASET."""
    import shlex
    try:
        words = shlex.split(command or "")
    except ValueError:
        return None
    if "--work-dir" not in words or not any("evalscope" in w for w in words) or not any(dataset in w for w in words):
        return None
    i = len(words) - 1 - words[::-1].index("--work-dir")
    return words[i + 1] if i + 1 < len(words) else None


def summarize_run(args) -> int:
    run_dir = Path(args.run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    reps = []
    for client in run.get("clients", []):
        wd = _work_dir(client.get("command", ""), args.dataset)
        if wd is None:
            continue
        rec = {"rep": len(reps) + 1, "client": client.get("measure"), "index": client.get("index"),
               "work_dir": wd, "rc": client.get("rc"), "skipped": client.get("skipped"),
               "stopped": client.get("stopped"), "admissible": client.get("admissible"),
               "record_admissible": client.get("record_admissible"), "record": client.get("record"),
               "started_at": client.get("started_at"), "finished_at": client.get("finished_at"),
               "review_files": review_files(wd, args.dataset)}
        try:
            q = per_question(wd, args.dataset)
            rec.update(correct=sum(v["correct"] for v in q.values()), total=len(q))
            rec["accuracy"] = rec["correct"] / rec["total"] if rec["total"] else None
        except (ValueError, OSError) as e:
            rec.update(correct=None, total=None, accuracy=None, error=str(e))
        try:
            rec.update(prediction_stats(wd, args.dataset), evalscope_report_score=report_score(wd))
        except (ValueError, OSError) as e:
            rec["prediction_stats_error"] = str(e)
        reps.append(rec)
    stop = run.get("stop") or {}
    summary = {"schema_version": 1, "label": args.label, "dataset": args.dataset, "source": "manage_server.py run",
               "run_dir": str(run_dir), "run_status": run.get("status"), "run_error": run.get("error"),
               "started_at": run.get("started_at"), "finished_at": run.get("finished_at"),
               "stop": {"process_group": stop.get("process_group"),
                        "process_group_empty": stop.get("process_group_empty"), "reason": stop.get("reason")},
               "repetitions": reps}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"label": args.label, "run_status": summary["run_status"],
                      "reps": [(r["client"], r["correct"], r["total"], r["rc"], r["admissible"]) for r in reps]}))
    return 0 if reps else 1


def discord(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    diff = [q for q in common if a[q]["correct"] != b[q]["correct"]]
    return {"n_common": len(common), "discordant": len(diff),
            "a_only_correct": sum(a[q]["correct"] for q in diff), "b_only_correct": sum(b[q]["correct"] for q in diff),
            "answer_differs": sum(1 for q in common if a[q]["pred"] != b[q]["pred"])}


def admissible(rep: dict) -> bool:
    """A summarize-run repetition whose measurement window was admissible: the client's window verdict
    ("admissible") is true and its measurement record ("record_admissible", when recorded) is not false."""
    return rep.get("admissible") is True and rep.get("record_admissible") is not False


def compare(args) -> int:
    rc = range_compare
    from_counts = getattr(args, "counts_from_summary", False)
    admissible_only = bool(getattr(args, "admissible_only", False))
    runs: dict[str, list[dict]] = {}
    excluded = []
    override = {}
    for spec in args.review or []:  # LABEL:REP=PATH, e.g. a copied review file of another workspace's job
        key, _, path = spec.partition("=")
        override[key] = path
    for s in args.summaries:
        summ = json.loads(Path(s).read_text())
        for r in summ["repetitions"]:
            if admissible_only and not admissible(r):
                excluded.append({"layout": summ["label"], "rep": r["rep"], "client": r.get("client"),
                                 "admissible": r.get("admissible"), "record_admissible": r.get("record_admissible")})
                continue
            if from_counts:
                if not isinstance(r.get("correct"), int) or not isinstance(r.get("total"), int) or r["total"] <= 0:
                    raise SystemExit(f"{s}: {summ['label']} rep {r['rep']} has no correct/total "
                                     f"({r.get('correct')}/{r.get('total')}): {r.get('error')}")
                q, correct, total = None, r["correct"], r["total"]
            else:
                q = per_question(override.get(f"{summ['label']}:{r['rep']}", r["work_dir"]), args.dataset)
                correct, total = sum(v["correct"] for v in q.values()), len(q)
            runs.setdefault(summ["label"], []).append(
                {"layout": summ["label"], "rep": r["rep"], "q": q, "correct": correct, "total": total})
        if summ["label"] not in runs:
            raise SystemExit(f"{s}: no repetition of {summ['label']} left to compare")
    rows = [{"layout": l, "rep": r["rep"], "accuracy": r["correct"] / r["total"], "correct": r["correct"],
             "total": r["total"]} for l, rs in runs.items() for r in rs]
    summary = {row["layout"]: row for row in rc.summarize(rows, ["layout"], ["accuracy", "correct"])}
    labels = list(runs)
    report = {"dataset": args.dataset, "expected_total": args.expected_total, "rule": "agree iff |median_A-median_B| <= 1.96*sqrt(pA(1-pA)/n+pB(1-pB)/n)",
              "mode": "counts-from-summary" if from_counts else "per-question-reviews",
              "admissible_only": admissible_only, "excluded": excluded,
              "layouts": {}, "pairs": []}
    for l in labels:
        s, totals = summary[(l,)] if (l,) in summary else summary[l], {r["total"] for r in runs[l]}
        p = s["accuracy_median"]
        n = runs[l][0]["total"]
        within, unstable = None, None
        if not from_counts:
            within = [discord(a["q"], b["q"]) | {"reps": [a["rep"], b["rep"]]} for a, b in itertools.combinations(runs[l], 2)]
            always = set.intersection(*(set(r["q"]) for r in runs[l]))
            unstable = sum(1 for q in always if len({r["q"][q]["correct"] for r in runs[l]}) > 1)
        report["layouts"][l] = {
            "reps": [(r["rep"], r["correct"], r["total"]) for r in runs[l]], "totals": sorted(totals),
            "accuracy_median": p, "accuracy_min": s["accuracy_min"], "accuracy_max": s["accuracy_max"],
            "within_range": s["accuracy_max"] - s["accuracy_min"], "binomial_se": math.sqrt(p * (1 - p) / n),
            "within_discordance": within, "questions_not_stable_across_reps": unstable,
            "total_mismatch": (args.expected_total is not None
                               and any(t != args.expected_total for t in totals))}
    for a, b in itertools.combinations(labels, 2):
        la, lb = report["layouts"][a], report["layouts"][b]
        n = min(runs[a][0]["total"], runs[b][0]["total"])
        pa, pb = la["accuracy_median"], lb["accuracy_median"]
        half = Z95 * math.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
        overlap = rc.compare((pa, la["accuracy_min"], la["accuracy_max"]), (pb, lb["accuracy_min"], lb["accuracy_max"]))
        pair = {"a": a, "b": b, "median_diff": pa - pb, "threshold_95": half,
                "verdict": "agree" if abs(pa - pb) <= half else "disagree",
                "ranges_overlap": overlap["verdict"] == "overlap", "range_compare_verdict": overlap["verdict"],
                "between_discordance": None, "between_discordant_median": None, "between_discordant_min_max": None,
                "a_all_right_b_all_wrong": None, "b_all_right_a_all_wrong": None}
        if not from_counts:
            between = [discord(x["q"], y["q"]) | {"reps": [x["rep"], y["rep"]]} for x in runs[a] for y in runs[b]]
            counts = sorted(d["discordant"] for d in between)
            common = set.intersection(*(set(r["q"]) for r in runs[a] + runs[b]))
            all_ok = lambda l, q: all(r["q"][q]["correct"] for r in runs[l])
            all_bad = lambda l, q: not any(r["q"][q]["correct"] for r in runs[l])
            pair.update(
                between_discordance=between,
                between_discordant_median=counts[len(counts) // 2] if len(counts) % 2 else (counts[len(counts) // 2 - 1] + counts[len(counts) // 2]) / 2,
                between_discordant_min_max=[counts[0], counts[-1]],
                a_all_right_b_all_wrong=sorted((q for q in common if all_ok(a, q) and all_bad(b, q)), key=lambda x: (len(x), x)),
                b_all_right_a_all_wrong=sorted((q for q in common if all_ok(b, q) and all_bad(a, q)), key=lambda x: (len(x), x)))
        report["pairs"].append(pair)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(f"mode {report['mode']}" + (f"; admissible only, excluded {[(x['layout'], x['rep']) for x in excluded]}"
                                      if report["admissible_only"] else ""))
    for l, v in report["layouts"].items():
        print(f"{l}: median {v['accuracy_median']:.4f} [{v['accuracy_min']:.4f}, {v['accuracy_max']:.4f}] "
              f"range {v['within_range']:.4f} SE {v['binomial_se']:.4f} reps {v['reps']}"
              + ("" if from_counts else f" within-discordant {[d['discordant'] for d in v['within_discordance']]} "
                                        f"unstable {v['questions_not_stable_across_reps']}"))
    for p in report["pairs"]:
        print(f"{p['a']} vs {p['b']}: diff {p['median_diff']:+.4f} threshold {p['threshold_95']:.4f} -> {p['verdict']}; "
              f"ranges overlap: {p['ranges_overlap']}"
              + ("" if from_counts else
                 f"; between-discordant median {p['between_discordant_median']} "
                 f"[{p['between_discordant_min_max'][0]}, {p['between_discordant_min_max'][1]}] "
                 f"{[d['discordant'] for d in p['between_discordance']]}; "
                 f"all {p['a']} reps right & all {p['b']} reps wrong: {len(p['a_all_right_b_all_wrong'])} {p['a_all_right_b_all_wrong']}; "
                 f"all {p['b']} reps right & all {p['a']} reps wrong: {len(p['b_all_right_a_all_wrong'])} {p['b_all_right_a_all_wrong']}"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("summarize-run", help="manage_server.py run --config run directory -> compare summary")
    r.add_argument("--run-dir", required=True); r.add_argument("--label", required=True)
    r.add_argument("--dataset", required=True, help="EvalScope dataset name, e.g. gsm8k")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("summaries", nargs="+"); c.add_argument("--out")
    c.add_argument("--dataset", required=True, help="EvalScope dataset name, e.g. gsm8k")
    c.add_argument("--expected-total", type=int, help="questions each repetition should score (gsm8k test: 1319)")
    c.add_argument("--review", action="append", metavar="LABEL:REP=PATH",
                   help="read this repetition's per-question review file (or EvalScope work dir) from PATH "
                        "instead of the summary's work_dir (repeatable)")
    c.add_argument("--counts-from-summary", action="store_true",
                   help="take each repetition's correct/total from the summary instead of its review files; "
                        "per-question discordance fields are null")
    c.add_argument("--admissible-only", action="store_true",
                   help="leave out repetitions whose measurement window was not admissible; listed under excluded")
    a = ap.parse_args(argv)
    return {"summarize-run": summarize_run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
