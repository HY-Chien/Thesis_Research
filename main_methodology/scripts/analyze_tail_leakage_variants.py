"""Tail-leakage metric family: principled reformulations of min_top5m.

min_top5m (SS4.13 winner) is min_t m_t where m_t = top-5 probability mass at
answer token t. This script explores whether fancier formulations of the same
"tail leakage" idea discriminate better, all computable from the stored
per-token dumps (no GPU):

  TL_t = -log m_t          per-token tail-leakage surprisal. Identities:
                           TL_t = KL(p_t^top5-renorm || p_t)  (truncated-vs-full KL)
                                = self-information of the "stay in top-5" event.
  aggregations             hard max (== min_top5m, rank-identical), sum
                           (joint stay-in-top-5 surprisal), mean, LSE_b
                           (softmax-pooled soft-max, temperature b), and
                           softmax(b*TL)-weighted mean.
  H_B(m_t)                 binary in/out-of-top-5 entropy.
  Htop5_t (approx)         entropy of the RENORMALISED top-5 distribution
                           (p1, p2 exact; p3..p5 approximated as uniform
                           remainder) -- "top-5 內部亂度".
  chain-rule combo         TL + Htop5 at the weakest token: total uncertainty
                           of "which candidate" + "escapes candidates".
  k-mass sweep             same min-mass statistic at k = 1, 2, 10, and the
                           rank-6..10 band mass / deep tail (1 - top10m).

Labels: direct-path em (alias-aware) from the bundled
results/traces/{ds}/conf_battery_{ds}.jsonl files, n=1000 x 4.
Direct-path answers never see the orig-question prompt line, so these numbers
are identical pre/post the SS4.15 prompt fix.

Output: results/reproduced/tail_leakage_variants.md
"""
from __future__ import annotations

import json
import math
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


def cw_at_med(u: np.ndarray, y: np.ndarray) -> float:
    """Share of wrong answers scored <= median uncertainty of correct ones."""
    med = np.median(u[y == 0])
    return float((u[y == 1] <= med).mean())


def variants(toks: list[dict]) -> dict[str, float]:
    m5 = np.clip(np.array([t["top5m"] for t in toks]), EPS, 1.0)
    m10 = np.clip(np.array([t["top10m"] for t in toks]), EPS, 1.0)
    p1 = np.clip(np.array([t["top1p"] for t in toks]), EPS, 1.0)
    p2 = np.clip(np.array([t["top2p"] for t in toks]), EPS, 1.0)
    ent = np.array([t["ent"] for t in toks])
    tl = -np.log(m5)

    def lse(x, b):
        return float(np.log(np.mean(np.exp(b * (x - x.max())))) / b + x.max())

    def swmean(x, b):
        w = np.exp(b * (x - x.max()))
        w /= w.sum()
        return float((w * x).sum())

    # approx renormalised top-5 entropy: p1, p2 exact, p3..5 uniform remainder
    rest = np.clip(m5 - p1 - p2, EPS, None) / 3.0
    q = np.stack([p1, p2, rest, rest, rest]) / m5
    h_top5 = -(q * np.log(np.clip(q, EPS, None))).sum(axis=0)

    tail5 = np.clip(1.0 - m5, EPS, 1.0)
    h_bin = -(m5 * np.log(m5) + tail5 * np.log(tail5))
    # estimated entropy of the tail distribution itself (chain rule residual)
    h_tail = np.clip((ent - h_bin - m5 * h_top5) / tail5, 0, None)

    band = np.clip(m10 - m5, 0, 1)          # rank 6-10 "shallow tail band"
    i_min = int(np.argmin(m5))              # weakest token

    return {
        "min_top5m(基準)": float(-m5.min()),
        "tl_sum": float(tl.sum()),
        "tl_mean": float(tl.mean()),
        "tl_lse2": lse(tl, 2.0),
        "tl_lse5": lse(tl, 5.0),
        "tl_lse10": lse(tl, 10.0),
        "tl_swmean2": swmean(tl, 2.0),
        "tl_swmean5": swmean(tl, 5.0),
        "max_hbin": float(h_bin.max()),
        "max_htop5": float(h_top5.max()),
        "htop5@weakest": float(h_top5[i_min]),
        "chain(tl+htop5)@weakest": float(tl[i_min] + h_top5[i_min]),
        "max_htail": float(h_tail.max()),
        "min_top1p": float(-p1.min()),
        "min_top2m": float(-(p1 + p2).min()),
        "min_top10m": float(-m10.min()),
        "max_band6_10": float(band.max()),
        "max_deeptail": float(-np.log(m10).max()),
        "max_ent(舊基準)": float(ent.max()),
    }


def main() -> None:
    rng = np.random.default_rng(13)
    per_ds: dict[str, dict[str, np.ndarray]] = {}
    ys: dict[str, np.ndarray] = {}
    names: list[str] = []
    for ds in DATASETS:
        rows = [json.loads(l) for l in
                open(ROOT / f"results/traces/{ds}/conf_battery_{ds}.jsonl",
                     encoding="utf-8")]
        sigs, y = [], []
        for r in rows:
            if not r.get("tokens"):
                continue
            sigs.append(variants(r["tokens"]))
            y.append(1 - int(r["em"]))
        names = list(sigs[0])
        per_ds[ds] = {k: np.array([s[k] for s in sigs]) for k in names}
        ys[ds] = np.array(y)

    lines = ["# Tail-leakage variant battery (direct path, n=1000x4, "
             "prompt-fix-invariant)\n",
             "AUROC(wrong) / CW@med;macro = 四資料集平均;P = macro AUROC "
             "paired bootstrap vs min_top5m(1000 resamples)\n",
             "| signal | macro AUROC | P vs min_top5m | " +
             " | ".join(DATASETS) + " | macro CW@med |",
             "|---|---:|---:|" + "---:|" * (len(DATASETS) + 1)]

    base = "min_top5m(基準)"
    boot_cache = {}
    for k in names:
        aus = [auroc(per_ds[ds][k], ys[ds]) for ds in DATASETS]
        cws = [cw_at_med(per_ds[ds][k], ys[ds]) for ds in DATASETS]
        boot_cache[k] = aus
        lines.append("")  # placeholder replaced below

    # paired bootstrap on macro AUROC diff vs baseline
    def macro_boot(k):
        diffs = []
        for _ in range(1000):
            vals_k, vals_b = [], []
            for ds in DATASETS:
                idx = rng.integers(0, len(ys[ds]), len(ys[ds]))
                vals_k.append(auroc(per_ds[ds][k][idx], ys[ds][idx]))
                vals_b.append(auroc(per_ds[ds][base][idx], ys[ds][idx]))
            diffs.append(np.mean(vals_k) - np.mean(vals_b))
        d = np.array(diffs)
        mean_d = np.mean([auroc(per_ds[ds][k], ys[ds]) for ds in DATASETS]) - \
            np.mean([auroc(per_ds[ds][base], ys[ds]) for ds in DATASETS])
        p = float((d <= 0).mean() if mean_d > 0 else (d >= 0).mean())
        return p

    out_rows = []
    for k in names:
        aus = [auroc(per_ds[ds][k], ys[ds]) for ds in DATASETS]
        cws = [cw_at_med(per_ds[ds][k], ys[ds]) for ds in DATASETS]
        p = 0.5 if k == base else macro_boot(k)
        out_rows.append((np.mean(aus), k, aus, cws, p))
    out_rows.sort(reverse=True)

    lines = lines[:4]
    for mau, k, aus, cws, p in out_rows:
        lines.append(f"| {k} | {mau:.4f} | {p:.3f} | " +
                     " | ".join(f"{a:.3f}" for a in aus) +
                     f" | {np.mean(cws):.3f} |")

    out = ROOT / "results" / "reproduced" / "tail_leakage_variants.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
