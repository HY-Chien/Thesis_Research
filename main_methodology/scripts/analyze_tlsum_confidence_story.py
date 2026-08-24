"""Why tl_sum discriminates, and what is left after adopting it.

Q1 (mechanism): decompose full-distribution entropy by the chain rule
    H = H_B(m5) + m5 * H_top5 + (1-m5) * H_tail
  and show the discriminative part is exactly the escape term; then show the
  sum-vs-max gain comes from multi-token leakage accumulation, with concrete
  caught/missed example pairs.

Q2 (residuals): at the tl_sum operating point, classify the two failure tails
  on the direct-path battery (n=1000x4, alias-aware em):
    confident-wrong   em=0 and tl_sum <= median(correct)  [CW@med set]
    unconfident-right em=1 and tl_sum >= median(wrong)
  with automatic categories + example dumps, plus a selective-risk table
  (accuracy of the most-confident X% bucket) for the "use as confidence
  metric" reading.

Output: results/reproduced/tlsum_confidence_story.md
"""
from __future__ import annotations

import difflib
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["2wiki", "hotpotqa", "musique", "popqa"]
EPS = 1e-12


def auroc(u: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(u)
    ranks = np.empty(len(u), float)
    i = 0
    su = u[order]
    while i < len(u):
        j = i
        while j + 1 < len(u) and su[j + 1] == su[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].mean() - (n1 + 1) / 2) / n0


def signals(toks: list[dict]) -> dict[str, float]:
    m5 = np.clip(np.array([t["top5m"] for t in toks]), EPS, 1.0)
    p1 = np.clip(np.array([t["top1p"] for t in toks]), EPS, 1.0)
    p2 = np.clip(np.array([t["top2p"] for t in toks]), EPS, 1.0)
    ent = np.array([t["ent"] for t in toks])
    tl = -np.log(m5)
    rest = np.clip(m5 - p1 - p2, EPS, None) / 3.0
    q = np.stack([p1, p2, rest, rest, rest]) / m5
    h_top5 = -(q * np.log(np.clip(q, EPS, None))).sum(axis=0)
    tail5 = np.clip(1.0 - m5, EPS, 1.0)
    h_bin = -(m5 * np.log(m5) + tail5 * np.log(tail5))
    h_tail = np.clip((ent - h_bin - m5 * h_top5) / tail5, 0, None)
    return {
        "max_ent": float(ent.max()),
        "max_hbin": float(h_bin.max()),
        "max_htop5": float(h_top5.max()),
        "max_htail": float(h_tail.max()),
        "min_top5m": float(-m5.min()),
        "tl_sum": float(tl.sum()),
        "tl_max": float(tl.max()),
        "n_tok": len(toks),
    }


def pct_rank(v: np.ndarray) -> np.ndarray:
    """Tie-averaged percentile rank in [0, 1]."""
    order = np.argsort(v)
    ranks = np.empty(len(v), float)
    i = 0
    sv = v[order]
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks / max(len(v) - 1, 1)


DATE_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUM_RE = re.compile(r"^[\s$€£]*[\d,.]+\s*\w*$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip(" .,\"'"))


def categorize_cw(pred: str, gold: str, aliases: list[str], f1: float) -> str:
    p, g = norm(pred), norm(gold)
    golds = [g] + [norm(a) for a in aliases]
    if p in ("yes", "no") and g in ("yes", "no"):
        return "yesno_flip"
    if any(p and (p in gg or gg in p) for gg in golds if gg):
        return "boundary/alias near-miss"
    if f1 >= 0.5:
        return "boundary/alias near-miss"
    if any(difflib.SequenceMatcher(None, p, gg).ratio() >= 0.8
           for gg in golds if gg):
        return "boundary/alias near-miss"
    if DATE_RE.search(p) and DATE_RE.search(g):
        return "wrong date/number"
    if NUM_RE.match(p) and NUM_RE.match(g):
        return "wrong date/number"
    if f1 > 0:
        return "partial overlap (mixed entity)"
    return "clean wrong entity (parametric hallucination)"


def main() -> None:
    rows_all: dict[str, list[dict]] = {}
    for ds in DATASETS:
        rows = []
        for l in open(ROOT / f"results/traces/{ds}/conf_battery_{ds}.jsonl",
                      encoding="utf-8"):
            r = json.loads(l)
            if not r.get("tokens"):
                continue
            s = signals(r["tokens"])
            s.update(id=str(r["id"]), q=r["question"], pred=r.get("pred") or "",
                     gold=r["gold"], aliases=r.get("aliases") or [],
                     em=int(r["em"]), f1=float(r.get("f1") or 0.0),
                     toks=[(t["text"], -math.log(max(t["top5m"], 1e-12)))
                           for t in r["tokens"]])
            rows.append(s)
        rows_all[ds] = rows

    L = ["# tl_sum as the confidence signal: mechanism + residual failures\n",
         "Direct path battery, n=1000x4, alias-aware EM. "
         "uncertainty = tl_sum unless noted.\n"]

    # ---------- Q1a: chain-rule decomposition ----------
    L += ["## Q1a. Entropy chain-rule decomposition — the discriminative part "
          "is the escape term\n",
          "H_t = H_B(m_t) + m_t*H_top5 + (1-m_t)*H_tail. AUROC(wrong), macro:\n",
          "| component | meaning | macro AUROC |", "|---|---|---:|"]
    names = [("max_ent", "full entropy (all three terms mixed)"),
             ("max_hbin", "escape in/out top-5 only"),
             ("max_htop5", "ambiguity WITHIN the top-5 candidates"),
             ("max_htail", "composition of the tail"),
             ("min_top5m", "escape mass, weakest token"),
             ("tl_sum", "escape surprisal, summed over the answer")]
    for k, desc in names:
        aus = []
        for ds in DATASETS:
            u = np.array([r[k] for r in rows_all[ds]])
            y = np.array([1 - r["em"] for r in rows_all[ds]])
            aus.append(auroc(u, y))
        L.append(f"| {k} | {desc} | {np.mean(aus):.4f} |")

    # ---------- Q1b: sum-vs-max, multi-token accumulation ----------
    L += ["\n## Q1b. Where the sum beats the weakest link — multi-token "
          "leakage accumulation\n",
          "AUROC by answer length (pooled 4 ds):\n",
          "| len (tokens) | n | tl_sum | min_top5m | max_ent |",
          "|---|---:|---:|---:|---:|"]
    pool = [r for ds in DATASETS for r in rows_all[ds]]
    y_pool = np.array([1 - r["em"] for r in pool])
    for lo, hi, lab in [(1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"),
                        (7, 99, "7+")]:
        idx = [i for i, r in enumerate(pool) if lo <= r["n_tok"] <= hi]
        yy = y_pool[idx]
        row = [f"| {lab} | {len(idx)} |"]
        for k in ("tl_sum", "min_top5m", "max_ent"):
            u = np.array([pool[i][k] for i in idx])
            row.append(f" {auroc(u, yy):.3f} |")
        L.append("".join(row))
    frac = {}
    for lab, sel in [("wrong", 1), ("correct", 0)]:
        ratios = [r["tl_max"] / r["tl_sum"] for r in pool
                  if r["tl_sum"] > 1e-9 and (1 - r["em"]) == sel
                  and r["n_tok"] >= 2]
        frac[lab] = float(np.mean(ratios))
    L.append(f"\nshare of tl_sum carried by the single worst token (answers "
             f">=2 tokens): wrong {frac['wrong']:.2f} vs correct "
             f"{frac['correct']:.2f} — errors leak across several tokens, "
             f"which max-aggregation cannot see.\n")

    # caught-by-sum examples: high tl_sum pct, low max_ent pct, wrong
    L += ["Wrong answers CAUGHT by tl_sum but missed by max_ent "
          "(per-ds percentile ranks):\n"]
    shown = 0
    for ds in DATASETS:
        rows = rows_all[ds]
        tp = pct_rank(np.array([r["tl_sum"] for r in rows]))
        mp = pct_rank(np.array([r["max_ent"] for r in rows]))
        cand = sorted((i for i, r in enumerate(rows)
                       if r["em"] == 0 and tp[i] > 0.75 and mp[i] < 0.45),
                      key=lambda i: mp[i] - tp[i])
        for i in cand[:2]:
            r = rows[i]
            prof = " ".join(f"{t}:{v:.2f}" for t, v in r["toks"][:8])
            L.append(f"- [{ds}] q={r['q'][:70]!r} pred={r['pred'][:40]!r} "
                     f"gold={r['gold'][:40]!r} | tl_pct={tp[i]:.2f} "
                     f"ent_pct={mp[i]:.2f} | TL/token: {prof}")
            shown += 1
    # false alarms avoided: correct, high max_ent pct, low tl_sum pct
    L += ["\nCorrect answers max_ent would FALSE-ALARM but tl_sum clears "
          "(within-top-5 ambiguity, e.g. format/alias variants):\n"]
    n_fa = 0
    for ds in DATASETS:
        rows = rows_all[ds]
        tp = pct_rank(np.array([r["tl_sum"] for r in rows]))
        mp = pct_rank(np.array([r["max_ent"] for r in rows]))
        cand = sorted((i for i, r in enumerate(rows)
                       if r["em"] == 1 and mp[i] > 0.7 and tp[i] < 0.5),
                      key=lambda i: tp[i] - mp[i])
        for i in cand[:2]:
            r = rows[i]
            n_fa += 1
            L.append(f"- [{ds}] q={r['q'][:70]!r} pred={r['pred'][:40]!r} | "
                     f"tl_pct={tp[i]:.2f} ent_pct={mp[i]:.2f}")
    if n_fa == 0:
        L.append("- (none at pct gap >= 0.2 — the gain over max_ent is "
                 "asymmetric: tl_sum mostly re-ranks WRONG answers upward "
                 "in uncertainty rather than clearing correct ones)")

    # ---------- Q1c: selective risk (use as confidence metric) ----------
    L += ["\n## Q1c. Selective answering with tl_sum (pooled + per ds)\n",
          "keep the most-confident X% (lowest tl_sum), accuracy inside "
          "the kept bucket:\n",
          "| coverage | pooled acc | " + " | ".join(DATASETS) + " |",
          "|---:|---:|" + "---:|" * len(DATASETS)]
    for cov in (0.25, 0.5, 0.75, 1.0):
        cells = []
        u = np.array([r["tl_sum"] for r in pool])
        acc = np.array([r["em"] for r in pool])
        k = int(len(u) * cov)
        keep = np.argsort(u)[:k]
        cells.append(f"{acc[keep].mean()*100:.1f}")
        for ds in DATASETS:
            rows = rows_all[ds]
            uu = np.array([r["tl_sum"] for r in rows])
            aa = np.array([r["em"] for r in rows])
            kk = np.argsort(uu)[:int(len(uu) * cov)]
            cells.append(f"{aa[kk].mean()*100:.1f}")
        L.append(f"| {int(cov*100)}% | " + " | ".join(cells) + " |")

    # ---------- Q2: the two residual tails ----------
    L += ["\n## Q2. What is left: confident-wrong and unconfident-right\n"]
    cw_rows, uc_rows = [], []
    cat_counts: dict[str, dict[str, int]] = {}
    for ds in DATASETS:
        rows = rows_all[ds]
        u = np.array([r["tl_sum"] for r in rows])
        y = np.array([r["em"] for r in rows])
        med_c = np.median(u[y == 1])
        med_w = np.median(u[y == 0])
        cnt: dict[str, int] = {}
        for r in rows:
            if r["em"] == 0 and r["tl_sum"] <= med_c:
                c = categorize_cw(r["pred"], r["gold"], r["aliases"], r["f1"])
                cnt[c] = cnt.get(c, 0) + 1
                cw_rows.append((ds, r, c))
            elif r["em"] == 1 and r["tl_sum"] >= med_w:
                uc_rows.append((ds, r))
        cat_counts[ds] = cnt
    cats = sorted({c for d in cat_counts.values() for c in d},
                  key=lambda c: -sum(d.get(c, 0) for d in cat_counts.values()))
    L += ["Confident-wrong = wrong with tl_sum <= median(correct):\n",
          "| category | " + " | ".join(DATASETS) + " | total |",
          "|---|" + "---:|" * (len(DATASETS) + 1)]
    for c in cats:
        vals = [cat_counts[ds].get(c, 0) for ds in DATASETS]
        L.append(f"| {c} | " + " | ".join(map(str, vals)) +
                 f" | {sum(vals)} |")
    tot = [sum(cat_counts[ds].values()) for ds in DATASETS]
    n_wrong = [int(sum(1 - r["em"] for r in rows_all[ds])) for ds in DATASETS]
    L.append("| **total CW (share of wrong)** | " + " | ".join(
        f"{t} ({t/w*100:.0f}%)" for t, w in zip(tot, n_wrong)) +
        f" | {sum(tot)} |")

    L += ["\nConfident-wrong examples (lowest tl_sum first):\n"]
    cw_rows.sort(key=lambda t: t[1]["tl_sum"])
    seen_ds: dict[str, int] = {}
    for ds, r, c in cw_rows:
        if seen_ds.get(ds, 0) >= 3:
            continue
        seen_ds[ds] = seen_ds.get(ds, 0) + 1
        L.append(f"- [{ds}] ({c}) tl={r['tl_sum']:.4f} "
                 f"q={r['q'][:70]!r} pred={r['pred'][:40]!r} "
                 f"gold={r['gold'][:40]!r} f1={r['f1']:.2f}")

    # unconfident-right: length + first-token share
    len_uc = [r["n_tok"] for _, r in uc_rows]
    len_cc = [r["n_tok"] for ds in DATASETS for r in rows_all[ds]
              if r["em"] == 1]
    share_uc = sum(1 for _, r in uc_rows if r["n_tok"] >= 5) / max(len(uc_rows), 1)
    share_cc = sum(1 for n in len_cc if n >= 5) / max(len(len_cc), 1)
    L += [f"\nUnconfident-right = correct with tl_sum >= median(wrong): "
          f"n={len(uc_rows)} "
          f"({len(uc_rows)/max(len(len_cc),1)*100:.0f}% of all correct)."
          f" mean answer length {np.mean(len_uc):.1f} tokens vs "
          f"{np.mean(len_cc):.1f} for all correct; >=5-token answers "
          f"{share_uc*100:.0f}% vs {share_cc*100:.0f}%.\n",
          "Unconfident-right examples:\n"]
    uc_rows.sort(key=lambda t: -t[1]["tl_sum"])
    seen_ds = {}
    for ds, r in uc_rows:
        if seen_ds.get(ds, 0) >= 2:
            continue
        seen_ds[ds] = seen_ds.get(ds, 0) + 1
        L.append(f"- [{ds}] tl={r['tl_sum']:.3f} len={r['n_tok']} "
                 f"q={r['q'][:70]!r} pred={r['pred'][:40]!r}")

    out = ROOT / "results/reproduced/tlsum_confidence_story.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
