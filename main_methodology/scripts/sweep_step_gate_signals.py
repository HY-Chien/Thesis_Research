"""Phase-3 sweep: per-step retrieval-gate signal x tau -> EM/F1 vs retrieve rate.

Assembles the overnight materials (step probe signals, manifest syntheses,
recorded-vector syntheses from conf_battery_decomp) into full cost-accuracy
curves for gate signals max_ent / min_top5m / max_renyi05, evaluated at the
main-table protocol: rescue post-fix -> verifier v2 choose_answer (no
evidence) -> typeaware replay layer -> EM/F1 (alias-aware).

First-order replay caveat: gate flips reuse the recorded downstream step
answers (no chain re-rollout); the affected-question share is reported per
config. The decomp-side verifier check stays on the max_ent scale for all
configs (fixed axis; only the GATE signal is swapped).

Sanity: signal=max_ent at tau=0.3 must reproduce the stored replay EM and
the SS4.7 retrieve rates.

Usage: python scripts/sweep_step_gate_signals.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from seek.pipeline import _rescue_value_answer  # noqa: E402
from seek.verifier import choose_answer, exact_match, token_f1  # noqa: E402
from replay_typeaware_verifier import choose_typeaware, load_jsonl  # noqa: E402
from build_step_gate_manifest import vec_hash  # noqa: E402

DATASETS = ["2wiki", "hotpotqa", "musique", "popqa"]
SIGNALS = ["max_ent", "min_top5m", "max_renyi05", "tl_sum"]
TRACE = "results/{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl"
OUT_MD = ROOT / "results" / "step_gate_signal_sweep.md"
SKIP_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def load_materials():
    sig_table = json.load(open(ROOT / "results/step_gate_signals.json",
                               encoding="utf-8"))
    synth = {}
    for name in ["step_gate_synth_shard0.jsonl", "step_gate_synth_shard1.jsonl",
                 "step_gate_synth_delta.jsonl"]:
        for line in open(ROOT / f"results/{name}", encoding="utf-8"):
            r = json.loads(line)
            synth[(r["ds"], r["id"], r["vec_hash"])] = r["pred"]
    data = {}
    for ds in DATASETS:
        rows = load_jsonl(ROOT / TRACE.format(ds=ds))
        golds = {}
        rec_synth = {}
        for line in open(ROOT / f"results/conf_battery_{ds}.jsonl",
                         encoding="utf-8"):
            r = json.loads(line)
            golds[str(r["id"])] = [r["gold"]] + (r.get("aliases") or [])
        for line in open(ROOT / f"results/conf_battery_decomp_{ds}.jsonl",
                         encoding="utf-8"):
            r = json.loads(line)
            rec_synth[str(r["id"])] = r["pred"]  # raw (pre-rescue) synthesis
        data[ds] = (rows, golds, rec_synth)
    return sig_table, synth, data


def evaluate(ds, rows, golds, rec_synth, sig_table, synth, name, tau):
    em_sum = f1_sum = 0.0
    steps_on = steps_tot = 0
    changed_q = miss = 0
    for row in rows:
        rid = str(row["id"])
        key = f"{ds}:{rid}"
        gold_list = golds[rid]
        info = sig_table.get(key)
        if not info or not info["steps"]:
            em_sum += int(row.get("replay_em", 0))
            f1_sum += token_f1(row.get("replay_pred") or "", gold_list)
            continue
        entries = info["steps"]
        recorded = tuple((e["retrieved_ans"] if e["recorded_retrieved"]
                          else e["tentative"]) for e in entries)
        answers, post_vals, on_flags = [], [], []
        for e in entries:
            on = e["sig"][name] >= tau
            answers.append(e["retrieved_ans"] if on else e["tentative"])
            post_vals.append(e["post_ent"] if on else -1.0)
            on_flags.append(on)
        steps_on += sum(on_flags)
        steps_tot += len(entries)
        answers_t = tuple(answers)
        if answers_t == recorded:
            raw = rec_synth[rid]
        else:
            changed_q += 1
            raw = synth.get((ds, rid, vec_hash(list(answers))))
            if raw is None:
                miss += 1
                raw = rec_synth[rid]
        steps_ns = [SimpleNamespace(sub_q=e["sub_q"], final_answer=a)
                    for e, a in zip(entries, answers)]
        rescued = _rescue_value_answer(row["question"], raw, steps_ns)

        decomp_rec = {
            "question": row["question"], "pred": rescued,
            "skipped_decompose": False,
            "decomposition": (row.get("decomp") or {}).get("decomposition") or [],
            "steps": [{
                "step_id": e["step_id"], "raw_sub_q": e["sub_q"],
                "sub_q": e["sub_q"], "tentative": e["tentative"],
                "final_answer": a, "retrieved": on,
                "max_ent": e["sig"]["max_ent"],
                "signal_value": e["sig"]["max_ent"],
                "post_signal_value": pv,
            } for e, a, on, pv in zip(entries, answers, on_flags, post_vals)],
        }
        ver = choose_answer(row["question"], row["direct"] or {}, decomp_rec,
                            fallback="auto")
        synth_row = {
            "question": row["question"], "choice": ver.choice,
            "reason": ver.reason, "direct_pred": row.get("direct_pred") or "",
            "decomp_pred": rescued, "direct": row.get("direct") or {},
            "decomp": decomp_rec,
        }
        choice, _ = choose_typeaware(synth_row, direct_entropy_tau=0.3,
                                     decomp_post_tau=0.8, margin=2.5)
        if choice == "direct":
            pred = row.get("direct_pred") or ""
            em = int(row.get("direct_em", 0))
        else:
            pred = rescued
            em = int(exact_match(rescued, gold_list))
        em_sum += em
        f1_sum += token_f1(pred, gold_list)
    n = len(rows)
    return {"em": 100 * em_sum / n, "f1": 100 * f1_sum / n,
            "retr": 100 * steps_on / max(steps_tot, 1),
            "changed": changed_q, "miss": miss}


def main() -> None:
    sig_table, synth, data = load_materials()
    lines = ["# Per-step gate signal sweep (first-order offline replay)\n"]

    # sanity: max_ent at canonical tau=0.3
    lines.append("## Sanity (max_ent, tau=0.3 -> must match stored replay / SS4.7)\n")
    lines.append("| ds | EM | F1 | retrieve% | stored EM | SS4.7 retr% |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    ref_retr = {"2wiki": 94.70, "hotpotqa": 92.07, "musique": 90.55, "popqa": 99.30}
    for ds in DATASETS:
        rows, golds, rec = data[ds]
        r = evaluate(ds, rows, golds, rec, sig_table, synth, "max_ent", 0.3)
        stored = 100 * np.mean([int(x.get("replay_em", 0)) for x in rows])
        lines.append(f"| {ds} | {r['em']:.1f} | {r['f1']:.2f} | {r['retr']:.2f} "
                     f"| {stored:.1f} | {ref_retr[ds]:.2f} |")

    for name in SIGNALS:
        pooled = np.array([e["sig"][name]
                           for k, v in sig_table.items() for e in v["steps"]])
        lines.append(f"\n## {name} (tau = pooled step-signal quantile)\n")
        lines.append("| skip-rate | tau | " +
                     " | ".join(f"{ds} EM/F1/retr%" for ds in DATASETS) +
                     " | macro EM | macro F1 | avg retr% | chg-q |")
        lines.append("|---:|---:|" + "---:|" * (len(DATASETS) + 4))
        for rate in SKIP_RATES:
            if rate == 0.0:
                tau = -1e9
            elif rate == 1.0:
                tau = 1e9
            else:
                tau = float(np.quantile(pooled, rate))
            ems, f1s, retrs, cells, chg = [], [], [], [], 0
            for ds in DATASETS:
                rows, golds, rec = data[ds]
                r = evaluate(ds, rows, golds, rec, sig_table, synth, name, tau)
                ems.append(r["em"]); f1s.append(r["f1"]); retrs.append(r["retr"])
                chg += r["changed"]
                cells.append(f"{r['em']:.1f}/{r['f1']:.1f}/{r['retr']:.0f}")
                if r["miss"]:
                    print(f"WARN {name} rate={rate} {ds}: {r['miss']} missing vectors")
            lines.append(f"| {rate:.2f} | {tau:.4g} | " + " | ".join(cells) +
                         f" | {np.mean(ems):.2f} | {np.mean(f1s):.2f} "
                         f"| {np.mean(retrs):.1f} | {chg} |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
