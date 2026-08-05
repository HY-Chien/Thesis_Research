"""Build the synthesis manifest for the per-step gate-signal sweep.

For each question and each gate signal (max_ent uses the stored per-step
signal_value for exactness; min_top5m / max_renyi05 come from the step probe's
token stats), a monotone tau sweep produces at most n_steps+1 distinct
answer vectors (steps whose signal is below tau answer with the no-ctx
tentative, the rest with the retrieved-branch answer). This script enumerates
the union of those nested vectors across the three signals, dedupes them
globally, drops the recorded vector (its synthesis is already stored /
verified in conf_battery_decomp), and writes one manifest row per unique
vector that still needs a synthesis generation.

Rows are sorted by number of flips from the recorded vector so the
high-retrieve-rate region (the useful part of the sweep) is generated first.

Output: results/step_gate_manifest.jsonl  (+ per-step signal table
results/step_gate_signals.json used later by the sweep).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_confidence_battery import compute_signals  # noqa: E402

DATASETS = ["2wiki", "hotpotqa", "musique", "popqa"]
TRACE = "results/{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl"
PROBE = "results/step_gate_battery_{ds}.jsonl"
SIGNALS = ["max_ent", "min_top5m", "max_renyi05", "tl_sum"]


def vec_hash(answers: list[str]) -> str:
    return hashlib.md5(json.dumps(answers, ensure_ascii=False).encode()).hexdigest()[:16]


def main() -> None:
    manifest: list[dict] = []
    sig_table: dict[str, dict] = {}
    n_q = n_vec = 0
    for ds in DATASETS:
        probe: dict[tuple[str, int], dict] = {}
        for line in open(ROOT / PROBE.format(ds=ds), encoding="utf-8"):
            r = json.loads(line)
            probe[(str(r["id"]), r["step_id"])] = r
        for line in open(ROOT / TRACE.format(ds=ds), encoding="utf-8"):
            tr = json.loads(line)
            rid = str(tr["id"])
            steps = (tr.get("decomp") or {}).get("steps") or []
            if not steps:
                continue
            entries = []
            ok = True
            for s in steps:
                p = probe.get((rid, s["step_id"]))
                if p is None:
                    ok = False
                    break
                sig = compute_signals({"tokens": p["tokens"]}) or {}
                # retrieved-branch answer: stored final if the canonical gate
                # retrieved, else the probe-filled post answer
                ret_ans = (s.get("final_answer") if s.get("retrieved")
                           else p.get("post_pred")) or ""
                post_tok = p.get("post_tokens")
                post_ent = (max(t["ent"] for t in post_tok)
                            if post_tok else s.get("post_signal_value", -1.0))
                entries.append({
                    "step_id": s["step_id"], "sub_q": s["sub_q"],
                    "tentative": (s.get("tentative") or "").strip(),
                    "retrieved_ans": ret_ans,
                    "post_ent": post_ent,
                    "recorded_retrieved": bool(s.get("retrieved")),
                    "sig": {"max_ent": float(s.get("signal_value",
                                                   s.get("max_ent", 9.9))),
                            "min_top5m": sig.get("min_top5m", 9.9),
                            "max_renyi05": sig.get("max_renyi05", 9.9),
                            "tl_sum": sig.get("tl_sum", 9.9)},
                })
            if not ok:
                continue
            n_q += 1
            sig_table[f"{ds}:{rid}"] = {"steps": entries}
            recorded = tuple((s.get("final_answer") or "").strip() for s in steps)
            seen = {recorded}
            for name in SIGNALS:
                order = sorted(range(len(entries)),
                               key=lambda j: entries[j]["sig"][name])
                for k in range(len(entries) + 1):
                    # gate retrieves when sig >= tau, so as tau grows the
                    # lowest-signal (most confident) steps turn off first
                    off = set(order[:k])
                    answers = tuple(
                        (e["tentative"] if j in off else e["retrieved_ans"])
                        for j, e in enumerate(entries))
                    if answers in seen:
                        continue
                    seen.add(answers)
                    flips = sum(a != b for a, b in zip(answers, recorded))
                    manifest.append({
                        "ds": ds, "id": rid, "vec_hash": vec_hash(list(answers)),
                        "sub_qs": [e["sub_q"] for e in entries],
                        "answers": list(answers), "flips": flips,
                    })
                    n_vec += 1
    manifest.sort(key=lambda m: (m["flips"], m["ds"], m["id"]))
    with open(ROOT / "results/step_gate_manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(ROOT / "results/step_gate_signals.json", "w", encoding="utf-8") as f:
        json.dump(sig_table, f, ensure_ascii=False)
    print(f"questions={n_q} unique new synthesis vectors={n_vec}")


if __name__ == "__main__":
    main()
