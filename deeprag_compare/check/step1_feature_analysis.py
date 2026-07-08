"""Step 1 — Type-A subtype feature analysis (no GPU).

Goal: from multi_sample_dump.jsonl (20 qids x 20 samples), compute token-level
features per sample, aggregate per qid, then compare HI_correct vs HI_wrong to
find a feature combo that separates the two groups (sc_mean cannot — see
analysis_table.md: HI_correct sc_mean=13.13 vs HI_wrong sc_mean=14.06/13.04).

Per-sample features:
  - mean_entropy_full, min_entropy_full          (over all 100 tokens)
  - mean_margin_full                              (top1_lp - top2_lp, mean over tokens)
  - mean_lp_full
  - first-entity span (pred substring match in text):
      fe_min_lp, fe_mean_lp, fe_max_entropy, fe_min_margin
  - hedge_count                                   (regex on `text`)

Per-qid aggregates (across 20 samples): mean and std of each per-sample feature.
Plus across-sample variance of first-entity min_lp (lp_variance_across_samples).

Group comparison (HI_correct vs HI_wrong, 5 vs 5):
  - mean +/- std per group
  - Cohen's d
  - AUROC (HI_correct as positive class, oriented so higher = more "correct-like")

Outputs:
  step1_output/per_qid_features.csv
  step1_output/group_comparison.md
  step1_output/group_comparison.csv
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

DEFAULT_DUMP = Path(__file__).parent / "multi_sample_dump.jsonl"
DEFAULT_OUT = Path(__file__).parent / "step1_output"

HEDGE_PATTERNS = [
    r"\bprobably\b", r"\bperhaps\b", r"\bmaybe\b", r"\blikely\b",
    r"\bpossibly\b", r"\bmight\b", r"\bcould be\b", r"\bI think\b",
    r"\bI believe\b", r"\bnot sure\b", r"\buncertain\b", r"\bunclear\b",
    r"\bguess\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


def reconstruct(tokens):
    text = ""
    offsets = []
    for t in tokens:
        offsets.append(len(text))
        text += t["tok"]
    return text, offsets


def find_first_entity_span(pred, tokens):
    pred = (pred or "").strip()
    if not pred or len(pred) < 2:
        return None
    text, offsets = reconstruct(tokens)
    idx = text.lower().find(pred.lower())
    if idx == -1:
        return None
    start_tok = None
    for i in range(len(offsets)):
        nxt = offsets[i + 1] if i + 1 < len(offsets) else len(text)
        if offsets[i] <= idx < nxt:
            start_tok = i
            break
    if start_tok is None:
        return None
    end_char = idx + len(pred)
    end_tok = start_tok + 1
    while end_tok < len(offsets) and offsets[end_tok] < end_char:
        end_tok += 1
    return start_tok, end_tok


def per_sample_features(rec):
    tokens = rec["tokens"]
    text = rec.get("text", "")
    pred = rec.get("pred", "")

    lps = [t["lp"] for t in tokens]
    ents = [t["entropy"] for t in tokens]
    margins = [t["top1_lp"] - t["top2_lp"] for t in tokens]

    feats = {
        "mean_lp_full": mean(lps),
        "mean_entropy_full": mean(ents),
        "min_entropy_full": min(ents),
        "max_entropy_full": max(ents),
        "mean_margin_full": mean(margins),
        "min_margin_full": min(margins),
        "hedge_count": len(HEDGE_RE.findall(text)),
    }

    span = find_first_entity_span(pred, tokens)
    if span is not None:
        s, e = span
        sub_lps = [tokens[i]["lp"] for i in range(s, e)]
        sub_ents = [tokens[i]["entropy"] for i in range(s, e)]
        sub_margins = [tokens[i]["top1_lp"] - tokens[i]["top2_lp"] for i in range(s, e)]
        feats.update({
            "fe_min_lp": min(sub_lps),
            "fe_mean_lp": mean(sub_lps),
            "fe_max_entropy": max(sub_ents),
            "fe_mean_entropy": mean(sub_ents),
            "fe_min_margin": min(sub_margins),
            "fe_mean_margin": mean(sub_margins),
            "has_fe": 1,
        })
    else:
        feats.update({
            "fe_min_lp": None, "fe_mean_lp": None,
            "fe_max_entropy": None, "fe_mean_entropy": None,
            "fe_min_margin": None, "fe_mean_margin": None,
            "has_fe": 0,
        })

    return feats


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def safe_std(xs):
    xs = [x for x in xs if x is not None]
    return pstdev(xs) if len(xs) > 1 else None


def aggregate_per_qid(samples):
    """samples: list of (per_sample_features, em, pred). Returns dict."""
    feat_keys = list(samples[0][0].keys())
    agg = {}
    for k in feat_keys:
        vals = [s[0][k] for s in samples]
        agg[f"{k}_mean"] = safe_mean(vals)
        agg[f"{k}_std"] = safe_std(vals)
    em_rate = mean([s[1] for s in samples])
    n_uniq = len({(s[2] or "").strip().lower() for s in samples})
    agg["em_rate"] = em_rate
    agg["n_uniq"] = n_uniq
    return agg


def cohens_d(xs, ys):
    if len(xs) < 2 or len(ys) < 2: return None
    mx, my = mean(xs), mean(ys)
    sx, sy = pstdev(xs), pstdev(ys)
    pooled = math.sqrt((sx * sx + sy * sy) / 2)
    if pooled == 0: return None
    return (mx - my) / pooled


def auroc(pos_scores, neg_scores):
    """Pos = HI_correct, Neg = HI_wrong. Higher score => more 'correct-like'."""
    if not pos_scores or not neg_scores: return None
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n: wins += 1
            elif p == n: wins += 0.5
    return wins / (n_pos * n_neg)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_file", type=str, default=str(DEFAULT_DUMP))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()
    dump_path = Path(args.dump_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    by_qid = defaultdict(list)
    qid_meta = {}
    for line in dump_path.open():
        rec = json.loads(line)
        qid = rec["qid"]
        qid_meta[qid] = {
            "stratum": rec["stratum"],
            "question": rec["question"],
            "golds": rec.get("golds", []),
        }
        feats = per_sample_features(rec)
        by_qid[qid].append((feats, rec.get("em", 0), rec.get("pred", ""),
                            rec.get("sc"), rec.get("ppl")))

    rows = []
    for qid, samples in by_qid.items():
        agg = aggregate_per_qid([(s[0], s[1], s[2]) for s in samples])
        sc_vals = [s[3] for s in samples if s[3] is not None]
        ppl_vals = [s[4] for s in samples if s[4] is not None]
        agg["sc_mean"] = safe_mean(sc_vals)
        agg["sc_std"] = safe_std(sc_vals)
        agg["ppl_mean"] = safe_mean(ppl_vals)
        rows.append({
            "qid": qid,
            "stratum": qid_meta[qid]["stratum"],
            "question_short": qid_meta[qid]["question"][:60],
            **agg,
        })

    feat_keys = [k for k in rows[0].keys()
                 if k not in ("qid", "stratum", "question_short")]
    csv_path = out_dir / "per_qid_features.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "stratum", "question_short"] + feat_keys)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {csv_path}")

    pos = [r for r in rows if r["stratum"] == "HI_correct"]
    neg = [r for r in rows if r["stratum"] == "HI_wrong"]

    HIGHER_IS_POS = {
        "mean_lp_full_mean": True, "mean_lp_full_std": False,
        "mean_entropy_full_mean": False, "mean_entropy_full_std": False,
        "min_entropy_full_mean": False, "max_entropy_full_mean": False,
        "mean_margin_full_mean": True, "min_margin_full_mean": True,
        "hedge_count_mean": False, "hedge_count_std": False,
        "fe_min_lp_mean": True, "fe_min_lp_std": False,
        "fe_mean_lp_mean": True, "fe_mean_lp_std": False,
        "fe_max_entropy_mean": False, "fe_mean_entropy_mean": False,
        "fe_min_margin_mean": True, "fe_mean_margin_mean": True,
        "em_rate": True, "n_uniq": False,
        "sc_mean": True, "sc_std": False, "ppl_mean": False,
        "has_fe_mean": True,
    }

    metrics_to_test = [k for k in feat_keys if k in HIGHER_IS_POS]

    md_lines = ["# Step 1 — HI_correct vs HI_wrong feature comparison",
                "",
                f"Source: `multi_sample_dump.jsonl` (20 qids, 20 samples each).",
                f"Pos = HI_correct (n={len(pos)}), Neg = HI_wrong (n={len(neg)}).",
                "",
                "| feature | HI_correct mean±std | HI_wrong mean±std | Cohen's d | AUROC | sep_pp |",
                "|---|---|---|---|---|---|"]

    csv_rows = []
    for k in metrics_to_test:
        pv = [r[k] for r in pos if r.get(k) is not None]
        nv = [r[k] for r in neg if r.get(k) is not None]
        if len(pv) < 2 or len(nv) < 2: continue
        higher_is_pos = HIGHER_IS_POS[k]
        d = cohens_d(pv, nv) if higher_is_pos else cohens_d(nv, pv)
        if higher_is_pos:
            au = auroc(pv, nv)
        else:
            au = auroc([-x for x in pv], [-x for x in nv])
        if au is None: continue
        sep = abs(au - 0.5) * 2
        d_str = f"{d:+.3f}" if d is not None else "  n/a"
        md_lines.append(
            f"| {k} | {mean(pv):.3f}±{pstdev(pv):.3f} | "
            f"{mean(nv):.3f}±{pstdev(nv):.3f} | "
            f"{d_str} | {au:.3f} | {sep*100:.1f} |"
        )
        csv_rows.append({"feature": k, "pos_mean": mean(pv), "pos_std": pstdev(pv),
                         "neg_mean": mean(nv), "neg_std": pstdev(nv),
                         "cohens_d": d if d is not None else "",
                         "auroc": au, "separation_pp": sep * 100})

    csv_rows.sort(key=lambda r: -r["separation_pp"])
    def _d_str(v):
        return f"{v:+.3f}" if isinstance(v, (int, float)) else "  n/a"
    md_path = out_dir / "group_comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n\n## Sorted by separation\n\n" +
                       "| feature | AUROC | sep_pp | Cohen's d |\n|---|---|---|---|\n" +
                       "\n".join(f"| {r['feature']} | {r['auroc']:.3f} | "
                                 f"{r['separation_pp']:.1f} | {_d_str(r['cohens_d'])} |"
                                 for r in csv_rows) + "\n")
    print(f"wrote {md_path}")

    cmp_csv = out_dir / "group_comparison.csv"
    with cmp_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        for r in csv_rows: w.writerow(r)
    print(f"wrote {cmp_csv}")

    print("\nTop 8 features by AUROC separation:")
    for r in csv_rows[:8]:
        print(f"  {r['feature']:35s}  AUROC={r['auroc']:.3f}  d={r['cohens_d']:+.3f}")


if __name__ == "__main__":
    main()
