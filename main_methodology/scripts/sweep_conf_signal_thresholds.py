"""Fine threshold sweep for direct-side confidence signals (offline, no GPU).

Two integration points, three signals (max_ent baseline, min_top5m,
max_renyi05), EM and F1 (max over gold+aliases):

  A. verifier tau_d sweep — quantile-rate grid 0.02..0.60; honest selection
     via split-half (tune pooled first-500, report second-500) and LODO
     (tune on 3 datasets, report the 4th); paired bootstrap on the test half.
  B. skip-decompose gate Pareto — EM/F1 vs skip rate, plus avg retrieval
     calls per question (1 if skipped else 1 + n_steps), to see how much
     retrieval each signal can cut at a given EM.

All decisions replay the stored dual-path traces; the confidence bit is the
only free variable (decide(row, confident) precomputed for both values).

Usage: python scripts/sweep_conf_signal_thresholds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_typeaware_verifier import load_jsonl, token_f1  # noqa: E402
from replay_verifier_signal_swap import decide  # noqa: E402
from analyze_confidence_battery import compute_signals  # noqa: E402

DATASETS = ["2wiki", "hotpotqa", "musique", "popqa"]
SIGNALS = ["max_ent", "min_top5m", "max_renyi05", "tl_sum"]
TRACE = "results/{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl"
BATTERY = "results/conf_battery_{ds}.jsonl"
RATES = [round(r, 2) for r in np.arange(0.02, 0.62, 0.02)]
RNG = np.random.default_rng(13)
OUT_MD = ROOT / "results" / "conf_signal_threshold_sweep.md"


def best_f1(pred: str, golds: list[str]) -> float:
    return max(token_f1(pred, g) for g in golds) if golds else 0.0


def load() -> dict[str, dict]:
    data = {}
    for ds in DATASETS:
        rows = load_jsonl(ROOT / TRACE.format(ds=ds))
        batt = {}
        for line in open(ROOT / BATTERY.format(ds=ds), encoding="utf-8"):
            r = json.loads(line)
            s = compute_signals(r)
            if s is not None:
                batt[str(r["id"])] = (s, [r["gold"]] + (r.get("aliases") or []))
        d = {name: [] for name in SIGNALS}
        em_yes, em_no, f1_yes, f1_no = [], [], [], []
        em_dir, f1_dir, em_base, f1_base, n_steps = [], [], [], [], []
        for row in rows:
            sig, golds = batt[str(row["id"])]
            for name in SIGNALS:
                d[name].append(sig[name])
            # verifier outcome under confident / not-confident direct bit
            for conf, em_l, f1_l in ((True, em_yes, f1_yes), (False, em_no, f1_no)):
                choice, _ = decide(row, 0.0 if conf else 2.0, 1.0)
                pred = (row.get("direct_pred") if choice == "direct"
                        else row.get("decomp_pred")) or ""
                em_l.append(int(row.get("direct_em", 0)) if choice == "direct"
                            else int(row.get("decomp_em", 0)))
                f1_l.append(best_f1(pred, golds))
            em_dir.append(int(row.get("direct_em", 0)))
            f1_dir.append(best_f1(row.get("direct_pred") or "", golds))
            em_base.append(int(row.get("replay_em", 0)))
            f1_base.append(best_f1(row.get("replay_pred") or "", golds))
            n_steps.append(len((row.get("decomp") or {}).get("steps") or []))
        data[ds] = {
            "sig": {k: np.array(v) for k, v in d.items()},
            "em_yes": np.array(em_yes), "em_no": np.array(em_no),
            "f1_yes": np.array(f1_yes), "f1_no": np.array(f1_no),
            "em_dir": np.array(em_dir), "f1_dir": np.array(f1_dir),
            "em_base": np.array(em_base), "f1_base": np.array(f1_base),
            "n_steps": np.array(n_steps),
        }
    return data


def verifier_em_f1(d: dict, name: str, tau: float, idx=slice(None)):
    conf = d["sig"][name][idx] < tau
    em = np.where(conf, d["em_yes"][idx], d["em_no"][idx])
    f1 = np.where(conf, d["f1_yes"][idx], d["f1_no"][idx])
    return em, f1


def pooled_tau(data: dict, name: str, rate: float, ds_list, idx=slice(None)) -> float:
    v = np.concatenate([data[ds]["sig"][name][idx] for ds in ds_list])
    return float(np.quantile(v, rate))


def main() -> None:
    data = load()
    lines: list[str] = ["# Confidence-signal threshold sweep (offline replay)\n"]

    base_em = {ds: 100 * data[ds]["em_base"].mean() for ds in DATASETS}
    base_f1 = {ds: 100 * data[ds]["f1_base"].mean() for ds in DATASETS}
    lines.append("Base (stored replay, max_ent tau=0.3): "
                 + " ".join(f"{ds} EM={base_em[ds]:.1f}/F1={base_f1[ds]:.1f}"
                            for ds in DATASETS) + "\n")

    # ---- A. verifier tau sweep, full curves ----
    lines.append("\n## A. Verifier tau_d sweep (pooled-quantile rate grid, full 1000)\n")
    for name in SIGNALS:
        lines.append(f"\n### {name}\n")
        lines.append("| rate | tau | " + " | ".join(f"{ds} EM/F1" for ds in DATASETS)
                     + " | macro EM | macro F1 |")
        lines.append("|---:|---:|" + "---:|" * (len(DATASETS) + 2))
        for rate in RATES:
            tau = pooled_tau(data, name, rate, DATASETS)
            ems, f1s, cells = [], [], []
            for ds in DATASETS:
                em, f1 = verifier_em_f1(data[ds], name, tau)
                ems.append(100 * em.mean()); f1s.append(100 * f1.mean())
                cells.append(f"{ems[-1]:.1f}/{f1s[-1]:.1f}")
            lines.append(f"| {rate:.2f} | {tau:.4f} | " + " | ".join(cells)
                         + f" | {np.mean(ems):.2f} | {np.mean(f1s):.2f} |")

    # ---- A2. honest selection ----
    lines.append("\n## A2. Honest tau selection\n")
    lines.append("\n### Split-half (tune pooled EM on first 500/ds, report second 500/ds)\n")
    lines.append("| signal | rate* | " + " | ".join(f"{ds} EM/F1" for ds in DATASETS)
                 + " | macro EM | macro F1 |")
    lines.append("|---|---:|" + "---:|" * (len(DATASETS) + 2))
    test_em_vec: dict[str, np.ndarray] = {}
    for name in SIGNALS:
        tr, te = slice(0, 500), slice(500, None)
        best_rate, best = None, -1.0
        for rate in RATES:
            tau = pooled_tau(data, name, rate, DATASETS, tr)
            score = np.mean([verifier_em_f1(data[ds], name, tau, tr)[0].mean()
                             for ds in DATASETS])
            if score > best:
                best, best_rate = score, rate
        tau = pooled_tau(data, name, best_rate, DATASETS, tr)
        ems, f1s, cells, vecs = [], [], [], []
        for ds in DATASETS:
            em, f1 = verifier_em_f1(data[ds], name, tau, te)
            ems.append(100 * em.mean()); f1s.append(100 * f1.mean())
            cells.append(f"{ems[-1]:.1f}/{f1s[-1]:.1f}")
            vecs.append(em)
        test_em_vec[name] = np.concatenate(vecs).astype(float)
        lines.append(f"| {name} | {best_rate:.2f} | " + " | ".join(cells)
                     + f" | {np.mean(ems):.2f} | {np.mean(f1s):.2f} |")
    for name in SIGNALS[1:]:
        deltas = []
        d0, d1 = test_em_vec["max_ent"], test_em_vec[name]
        for _ in range(1000):
            idx = RNG.integers(0, len(d0), len(d0))
            deltas.append(d1[idx].mean() - d0[idx].mean())
        deltas = np.array(deltas)
        lines.append(f"\npaired bootstrap {name} vs max_ent (pooled test half): "
                     f"dEM={100 * (d1.mean() - d0.mean()):+.2f} "
                     f"CI95=[{100 * np.quantile(deltas, 0.025):+.2f},"
                     f"{100 * np.quantile(deltas, 0.975):+.2f}] "
                     f"P(d<=0)={float((deltas <= 0).mean()):.3f}")

    lines.append("\n\n### LODO (tune pooled EM on 3 datasets, report held-out)\n")
    lines.append("| signal | " + " | ".join(f"{ds} rate*/EM/F1" for ds in DATASETS)
                 + " | macro EM |")
    lines.append("|---|" + "---:|" * (len(DATASETS) + 1))
    for name in SIGNALS:
        cells, ems = [], []
        for held in DATASETS:
            rest = [ds for ds in DATASETS if ds != held]
            best_rate, best = None, -1.0
            for rate in RATES:
                tau = pooled_tau(data, name, rate, rest)
                score = np.mean([verifier_em_f1(data[ds], name, tau)[0].mean()
                                 for ds in rest])
                if score > best:
                    best, best_rate = score, rate
            tau = pooled_tau(data, name, best_rate, rest)
            em, f1 = verifier_em_f1(data[held], name, tau)
            ems.append(100 * em.mean())
            cells.append(f"{best_rate:.2f}/{ems[-1]:.1f}/{100 * f1.mean():.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {np.mean(ems):.2f} |")

    # ---- B. skip-decompose gate Pareto ----
    lines.append("\n\n## B. Skip-decompose gate (skip -> direct answer only; "
                 "retrieval calls = 1 if skipped else 1+n_steps)\n")
    for name in SIGNALS:
        lines.append(f"\n### {name}\n")
        lines.append("| skip rate | " + " | ".join(f"{ds} EM/F1" for ds in DATASETS)
                     + " | macro EM | macro F1 | avg retr | retr cut |")
        lines.append("|---:|" + "---:|" * (len(DATASETS) + 4))
        full_retr = np.mean([1 + data[ds]["n_steps"].mean() for ds in DATASETS])
        for rate in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
            ems, f1s, retrs, cells = [], [], [], []
            for ds in DATASETS:
                d = data[ds]
                v = d["sig"][name]
                tau = np.quantile(v, rate) if rate > 0 else -np.inf
                skip = v < tau
                em = np.where(skip, d["em_dir"], d["em_base"])
                f1 = np.where(skip, d["f1_dir"], d["f1_base"])
                retr = np.where(skip, 1.0, 1.0 + d["n_steps"])
                ems.append(100 * em.mean()); f1s.append(100 * f1.mean())
                retrs.append(retr.mean())
                cells.append(f"{ems[-1]:.1f}/{f1s[-1]:.1f}")
            avg_r = np.mean(retrs)
            lines.append(f"| {rate:.2f} | " + " | ".join(cells)
                         + f" | {np.mean(ems):.2f} | {np.mean(f1s):.2f} "
                         f"| {avg_r:.2f} | {100 * (1 - avg_r / full_retr):.0f}% |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
