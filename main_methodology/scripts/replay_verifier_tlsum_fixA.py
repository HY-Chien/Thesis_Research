"""Offline verifier-signal swap on the tl_sum+fixA headline trace.

The verifier consumes confidence at exactly two points:
  direct side  direct["skip_probe_ent"] < 0.3   (stage-1 _auto_fallback AND
               stage-2 typeaware +1 direct)
  decomp side  max over steps of post_signal_value > 0.8 (stage-1 unstable
               AND stage-2 -1 decomp)

The tlsum_fixA rerun stored per-step post_tl_sum, and the conf battery gives
the direct-path tl_sum, so both sides can be swapped offline. Taus are
rate-matched on the pooled 4-dataset crossing rates of the canonical signals
(no EM peeking). Configs: baseline sanity / direct-only swap / decomp-only
swap / both, plus a tau_d rate sweep and a paired bootstrap for "both".

Usage: python scripts/replay_verifier_tlsum_fixA.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from seek.verifier import choose_answer, exact_match, token_f1  # noqa: E402
from replay_typeaware_verifier import choose_typeaware, load_jsonl  # noqa: E402
from analyze_confidence_battery import compute_signals  # noqa: E402

DATASETS = ["hotpotqa", "2wiki", "musique", "popqa"]
TRACE = (
    "results/traces/{ds}/"
    "{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl"
)
NEW = "results/traces/{ds}/tlsum_fixA_{ds}_shard{sh}.jsonl"
DIRECT_TAU_ENT = 0.3
POST_TAU_ENT = 0.8
MARGIN = 2.5


def build(ds: str) -> list[dict]:
    """Join the tlsum_fixA rows with direct records + golds; keep step order."""
    trace = {str(r["id"]): r for r in load_jsonl(ROOT / TRACE.format(ds=ds))}
    golds, dsig = {}, {}
    for line in open(
        ROOT / f"results/traces/{ds}/conf_battery_{ds}.jsonl",
        encoding="utf-8",
    ):
        r = json.loads(line)
        golds[str(r["id"])] = [r["gold"]] + (r.get("aliases") or [])
        s = compute_signals(r) if r.get("tokens") else None
        if s is not None:
            dsig[str(r["id"])] = s["tl_sum"]
    rows = []
    for sh in (0, 1):
        for line in open(ROOT / NEW.format(ds=ds, sh=sh), encoding="utf-8"):
            rec = json.loads(line)
            rid = rec["id"]
            tr = trace[rid]
            g = golds[rid]
            row = {
                "ds": ds, "id": rid, "question": rec["question"],
                "stored_choice": rec["tl"].get("choice"),
                "stored_em": int(rec["tl"]["em"]),
                "stored_f1": float(rec["tl"]["f1"]),
                "unchanged": bool(rec["tl"].get("unchanged")),
                "direct": tr.get("direct") or {},
                "direct_pred": tr.get("direct_pred") or "",
                "direct_em": int(tr.get("direct_em", 0)),
                "direct_tl": dsig.get(rid),
                "decomp_pred": rec["tl"].get("decomp_pred") or "",
                "steps": rec.get("new_steps") or [],
                "decomposition": (tr.get("decomp") or {}).get("decomposition") or [],
                "golds": g,
            }
            row["direct_f1"] = token_f1(row["direct_pred"], g)
            row["decomp_em"] = int(exact_match(row["decomp_pred"], g))
            row["decomp_f1"] = token_f1(row["decomp_pred"], g)
            rows.append(row)
    return rows


def decide(row: dict, cfg: str, tau_d: float, tau_p: float) -> str:
    """Re-run choose_answer + typeaware with the configured signals."""
    if cfg in ("direct_tl", "both"):
        d_sig = row["direct_tl"] if row["direct_tl"] is not None else -1.0
        d_tau = tau_d
    else:
        d_sig = (row["direct"] or {}).get("skip_probe_ent", -1.0)
        d_tau = DIRECT_TAU_ENT
    if cfg in ("post_tl", "both"):
        p_key, p_tau = "post_tl_sum", tau_p
    else:
        p_key, p_tau = "post_max_ent", POST_TAU_ENT

    steps = [{
        "step_id": s["step_id"], "raw_sub_q": s["raw_sub_q"],
        "sub_q": s["sub_q"], "tentative": s["tentative"],
        "final_answer": s["final_answer"], "retrieved": s["retrieved"],
        "max_ent": s["max_ent"], "signal_value": s["max_ent"],
        "post_signal_value": float(s[p_key]) if s["retrieved"] else -1.0,
    } for s in row["steps"]]
    direct = dict(row["direct"] or {})
    direct["skip_probe_ent"] = d_sig if d_sig is not None else -1.0
    decomp_rec = {
        "question": row["question"], "pred": row["decomp_pred"],
        "skipped_decompose": False,
        "decomposition": row["decomposition"], "steps": steps,
    }
    ver = choose_answer(row["question"], direct, decomp_rec, fallback="auto",
                        direct_entropy_tau=d_tau, decomp_post_entropy_tau=p_tau)
    synth_row = {
        "question": row["question"], "choice": ver.choice, "reason": ver.reason,
        "direct_pred": row["direct_pred"], "decomp_pred": row["decomp_pred"],
        "direct": direct, "decomp": decomp_rec,
    }
    choice, _ = choose_typeaware(synth_row, direct_entropy_tau=d_tau,
                                 decomp_post_tau=p_tau, margin=MARGIN)
    return choice


def run(rows: list[dict], cfg: str, tau_d: float, tau_p: float) -> dict:
    em = f1 = 0.0
    n_direct = flips = rescued = lost = 0
    for row in rows:
        if row["unchanged"]:
            em += row["stored_em"]
            f1 += row["stored_f1"]
            continue
        choice = decide(row, cfg, tau_d, tau_p)
        e = row["direct_em"] if choice == "direct" else row["decomp_em"]
        em += e
        f1 += row["direct_f1"] if choice == "direct" else row["decomp_f1"]
        n_direct += choice == "direct"
        if choice != row["stored_choice"]:
            flips += 1
            rescued += int(e and not row["stored_em"])
            lost += int(row["stored_em"] and not e)
    n = len(rows)
    return {"em": 100 * em / n, "f1": 100 * f1 / n, "direct": n_direct,
            "flips": flips, "rescued": rescued, "lost": lost}


def main() -> None:
    data = {ds: build(ds) for ds in DATASETS}

    # sanity: canonical signals must reproduce the stored run
    print("== sanity: canonical signals reproduce stored tlsum_fixA run ==")
    for ds in DATASETS:
        rows = data[ds]
        res = run(rows, "canonical", DIRECT_TAU_ENT, POST_TAU_ENT)
        stored = 100 * np.mean([r["stored_em"] for r in rows])
        mism = sum(1 for r in rows if not r["unchanged"] and
                   decide(r, "canonical", DIRECT_TAU_ENT, POST_TAU_ENT)
                   != r["stored_choice"])
        print(f"{ds}: rederived EM={res['em']:.1f} (stored {stored:.1f}) "
              f"choice mismatches={mism}")

    # rate-matched taus (pooled, no EM peeking)
    d_ent = np.array([(r["direct"] or {}).get("skip_probe_ent", -1.0)
                      for ds in DATASETS for r in data[ds]
                      if (r["direct"] or {}).get("skip_probe_ent", -1.0) >= 0])
    d_tl = np.array([r["direct_tl"] for ds in DATASETS for r in data[ds]
                     if r["direct_tl"] is not None])
    rate_d = float((d_ent < DIRECT_TAU_ENT).mean())
    tau_d = float(np.quantile(d_tl, rate_d))
    p_ent, p_tl = [], []
    for ds in DATASETS:
        for r in data[ds]:
            vals_e = [s["post_max_ent"] for s in r["steps"] if s["retrieved"]]
            vals_t = [s["post_tl_sum"] for s in r["steps"] if s["retrieved"]]
            if vals_e:
                p_ent.append(max(vals_e))
                p_tl.append(max(vals_t))
    rate_p = float((np.array(p_ent) > POST_TAU_ENT).mean())
    tau_p = float(np.quantile(np.array(p_tl), 1 - rate_p))
    print(f"\nrate-matched: P(direct ent<0.3)={rate_d:.3f} -> tau_d={tau_d:.4f}; "
          f"P(max post ent>0.8)={rate_p:.3f} -> tau_p={tau_p:.4f}")

    print("\n== configs (EM/F1 per ds, macro) ==")
    header = f"{'config':>12} | " + " | ".join(f"{ds:>16}" for ds in DATASETS) \
        + " |  macro EM/F1"
    print(header)
    per_q: dict[str, dict[str, np.ndarray]] = {}
    for cfg in ["canonical", "direct_tl", "post_tl", "both"]:
        cells, ems, f1s = [], [], []
        per_q[cfg] = {}
        for ds in DATASETS:
            rows = data[ds]
            res = run(rows, cfg, tau_d, tau_p)
            cells.append(f"{res['em']:4.1f}/{res['f1']:4.1f} +{res['rescued']:2d}/-{res['lost']:2d}")
            ems.append(res["em"])
            f1s.append(res["f1"])
            qs = []
            for r in rows:
                if r["unchanged"]:
                    qs.append(r["stored_em"])
                else:
                    c = decide(r, cfg, tau_d, tau_p)
                    qs.append(r["direct_em"] if c == "direct" else r["decomp_em"])
            per_q[cfg][ds] = np.array(qs, float)
        print(f"{cfg:>12} | " + " | ".join(cells)
              + f" | {np.mean(ems):5.2f}/{np.mean(f1s):5.2f}")

    # paired bootstrap: both vs canonical (macro EM)
    rng = np.random.default_rng(11)
    diffs = []
    for _ in range(2000):
        m_a, m_b = [], []
        for ds in DATASETS:
            n = len(per_q["both"][ds])
            idx = rng.integers(0, n, n)
            m_a.append(per_q["both"][ds][idx].mean())
            m_b.append(per_q["canonical"][ds][idx].mean())
        diffs.append(np.mean(m_a) - np.mean(m_b))
    d = np.array(diffs) * 100
    mean = np.mean([per_q["both"][ds].mean() - per_q["canonical"][ds].mean()
                    for ds in DATASETS]) * 100
    p = float(min((d <= 0).mean(), (d >= 0).mean()) * 2)
    print(f"\nboth vs canonical: macro dEM {mean:+.2f} "
          f"[95% CI {np.percentile(d, 2.5):+.2f},{np.percentile(d, 97.5):+.2f}] "
          f"p={p:.3f}")

    # tau_d rate sweep for the direct side (both config)
    print("\n== tau_d rate sweep (config=both, tau_p fixed rate-matched) ==")
    for rate in [0.05, 0.10, 0.15, rate_d, 0.25, 0.30, 0.40]:
        td = float(np.quantile(d_tl, rate))
        ems = [run(data[ds], "both", td, tau_p)["em"] for ds in DATASETS]
        star = " <- rate-matched" if abs(rate - rate_d) < 1e-9 else ""
        print(f"rate={rate:4.2f} tau_d={td:8.4f} macro EM={np.mean(ems):5.2f}{star}")


if __name__ == "__main__":
    main()
