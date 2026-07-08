"""Compute ~30 confidence signals from conf_battery dumps and rank them.

Input : results/conf_battery_{ds}.jsonl  (from probe_confidence_battery.py)
Output: results/confidence_battery_analysis.md

Every signal is oriented so that HIGHER = MORE UNCERTAIN. The prediction
target is answer wrongness (em == 0) of the deployed direct-path answer.

Metrics per dataset and pooled:
  AUROC(wrong)        rank-based with midrank tie handling
  dAUROC vs max_ent   paired bootstrap (1000 resamples) CI95 + P(delta<=0)
  CW@med              share of wrong answers with U <= median(U | correct)
                      ("confidently wrong, no threshold can catch")
  R@10FP              recall of wrong at tau = p90 of correct U
  SelAcc@cov          selective accuracy at the coverage max_ent tau=0.4 gives
"""
from __future__ import annotations

import json
import math
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["2wiki", "hotpotqa", "musique", "popqa"]
BASELINE = "max_ent"
N_BOOT = 1000
SEED = 13

STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and",
             "or", "is", "was", "are", "were", "be", "been", "by", "with",
             "from", "as", "that", "this", "it", "its"}


def norm_text(s: str | None) -> str:
    s = (s or "").lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(a: str, b: str) -> float:
    ta, tb = norm_text(a).split(), norm_text(b).split()
    if not ta or not tb:
        return float(ta == tb)
    common = Counter(ta) & Counter(tb)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p, r = n_common / len(ta), n_common / len(tb)
    return 2 * p * r / (p + r)


def is_content(tok_text: str) -> bool:
    t = tok_text.strip()
    if not any(c.isalnum() for c in t):
        return False
    return t.lower() not in STOPWORDS


# ---------------------------------------------------------------- signals
def f1_clusters(texts: list[str]) -> list[int]:
    """Union-find clustering with token-F1 >= 0.5 linkage on normalized text."""
    n = len(texts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if norm_text(texts[i]) == norm_text(texts[j]) or token_f1(texts[i], texts[j]) >= 0.5:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
    return [find(i) for i in range(n)]


def entropy_from_masses(masses: list[float]) -> float:
    tot = sum(masses)
    if tot <= 0:
        return 0.0
    return -sum((m / tot) * math.log(m / tot) for m in masses if m > 0)


def compute_signals(r: dict) -> dict[str, float] | None:
    toks = r["tokens"]
    if not toks:
        return None
    ent = [t["ent"] for t in toks]
    lp = [t["lp"] for t in toks]
    margin = [t["top1p"] - t["top2p"] for t in toks]
    lgap = [t["lgap"] for t in toks]
    renyi2 = [t["renyi2"] for t in toks]
    renyi05 = [t["renyi05"] for t in toks]
    ent_t2 = [t["ent_t2"] for t in toks]
    top5m = [t["top5m"] for t in toks]
    cmask = [is_content(t["text"]) for t in toks]
    if not any(cmask):
        cmask = [True] * len(toks)
    c_ent = [e for e, m in zip(ent, cmask) if m]
    c_lp = [x for x, m in zip(lp, cmask) if m]
    c_margin = [x for x, m in zip(margin, cmask) if m]
    k8 = min(8, len(toks))

    sig = {
        # --- baselines (already in the thesis) ---
        "max_ent": max(ent),
        "mean_ent": sum(ent) / len(ent),
        "min_lp": -min(lp),                       # flip: higher = uncertain
        "maqa_margin": -margin[0],
        "mean_margin": -sum(margin) / len(margin),
        "targ_unc": sum(math.exp(-g / 3.0) for g in lgap[:k8]) / k8,
        # --- new token-level ---
        "mean_lp": -sum(lp) / len(lp),            # length-normalised seq logprob
        "sum_lp": -sum(lp),
        "first_ent": ent[0],
        "min_margin": -min(margin),
        "max_renyi2": max(renyi2),
        "mean_renyi2": sum(renyi2) / len(renyi2),
        "max_renyi05": max(renyi05),
        "mean_renyi05": sum(renyi05) / len(renyi05),
        "max_ent_t2": max(ent_t2),
        "mean_ent_t2": sum(ent_t2) / len(ent_t2),
        "min_lgap": -min(lgap),
        "min_top5m": -min(top5m),
        "ans_len": float(len(toks)),
        "range_ent": max(ent) - sum(ent) / len(ent),
        # --- QA-tailored content-token variants ---
        "c_max_ent": max(c_ent),
        "c_mean_ent": sum(c_ent) / len(c_ent),
        "c_min_lp": -min(c_lp),
        "c_min_margin": -min(c_margin),
    }

    # --- sample-based (10 samples, same prompt) ---
    samples = r.get("samples") or []
    pred = r.get("pred") or ""
    if samples:
        stexts = [s["text"] for s in samples]
        snorm = [norm_text(t) for t in stexts]
        pnorm = norm_text(pred)
        cnt = Counter(snorm)
        N = len(snorm)
        sig["n_uniq"] = float(len(cnt))
        sig["disc_sem_ent"] = entropy_from_masses(list(cnt.values()))
        # likelihood-weighted semantic entropy (Farquhar-style, LN weights)
        w = np.exp(np.array([s["mean_lp"] for s in samples]))
        w = w / max(w.sum(), 1e-12)
        cl_mass: dict[str, float] = defaultdict(float)
        for t, wi in zip(snorm, w):
            cl_mass[t] += float(wi)
        sig["lw_sem_ent"] = entropy_from_masses(list(cl_mass.values()))
        # F1-linkage semantic entropy (merges span variants)
        labels = f1_clusters(stexts)
        sig["f1_sem_ent"] = entropy_from_masses(list(Counter(labels).values()))
        # self-consistency support of the committed answer
        exact_sup = sum(1 for t in snorm if t == pnorm) / N
        sig["sc_support"] = -exact_sup
        sig["sc_support_f1"] = -sum(token_f1(t, pred) for t in stexts) / N
        sig["mean_max_ent"] = sum(s["max_ent"] for s in samples) / N
        sig["mean_sample_lp"] = -sum(s["mean_lp"] for s in samples) / N
        # CoCoA-style multiplicative: consistency x confidence
        sig["cocoa"] = -(exact_sup * math.exp(sum(lp) / len(lp)))

    # --- answer-level posterior over candidates ("softmax over answers") ---
    cands = r.get("cand_texts") or []
    clps = r.get("cand_mean_lps") or []
    if cands and clps:
        z = np.array(clps, dtype=float)
        post = np.exp(z - z.max())
        post = post / post.sum()
        # merge candidates that normalise to the same answer string
        cl: dict[str, float] = defaultdict(float)
        for t, p in zip(cands, post):
            cl[norm_text(t)] += float(p)
        pmass = cl.get(norm_text(pred), 0.0)
        others = [v for k, v in cl.items() if k != norm_text(pred)]
        sig["post_ent"] = entropy_from_masses(list(cl.values()))
        sig["post_top1"] = -pmass
        sig["post_margin"] = -(pmass - (max(others) if others else 0.0))

    # --- self-verification + evidence support ---
    if "p_true_ctx" in r:
        sig["p_true_ctx"] = -r["p_true_ctx"]
        sig["p_true_noctx"] = -r["p_true_noctx"]
        if samples:
            sig["ptrue_sc"] = -(r["p_true_ctx"] * (-sig["sc_support"]))
    if "noctx_mean_lp" in r and clps:
        # committed answer TF score with ctx (cands[0]) minus without ctx
        sig["delta_ctx"] = -(clps[0] - r["noctx_mean_lp"])

    return sig


# ---------------------------------------------------------------- metrics
def auroc(u: np.ndarray, y: np.ndarray) -> float:
    """AUROC of score u for label y=1 (wrong), midrank ties."""
    order = np.argsort(u, kind="mergesort")
    ranks = np.empty(len(u), dtype=float)
    su = u[order]
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def cw_at_med(u: np.ndarray, y: np.ndarray) -> float:
    """Share of wrong with uncertainty <= median uncertainty of correct."""
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return float("nan")
    med = np.median(u[y == 0])
    return float((u[y == 1] <= med).mean())


def recall_at_10fp(u: np.ndarray, y: np.ndarray) -> float:
    """Recall of wrong when flagging u > p90 of correct."""
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return float("nan")
    tau = np.quantile(u[y == 0], 0.9)
    return float((u[y == 1] > tau).mean())


def sel_acc_at_cov(u: np.ndarray, y: np.ndarray, cov: float) -> float:
    """Accuracy of the accepted set when accepting the cov fraction most confident."""
    k = max(1, int(round(cov * len(u))))
    idx = np.argsort(u, kind="mergesort")[:k]
    return float((y[idx] == 0).mean())


def paired_bootstrap(u_new: np.ndarray, u_base: np.ndarray, y: np.ndarray,
                     rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (d_lo, d_hi, p_worse) for AUROC(new) - AUROC(base)."""
    n = len(y)
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        deltas[b] = auroc(u_new[idx], y[idx]) - auroc(u_base[idx], y[idx])
    ok = deltas[~np.isnan(deltas)]
    if len(ok) == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.quantile(ok, 0.025)),
            float(np.quantile(ok, 0.975)),
            float((ok <= 0).mean()))


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def main() -> None:
    only = sys.argv[1:] or DATASETS
    data: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, np.ndarray] = {}
    skipped: dict[str, int] = {}

    for ds in only:
        path = ROOT / "results" / f"conf_battery_{ds}.jsonl"
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        sig_rows, ys = [], []
        n_skip = 0
        for r in rows:
            s = compute_signals(r)
            if s is None:
                n_skip += 1
                continue
            sig_rows.append(s)
            ys.append(1 - int(r["em"]))
        names = sorted(set().union(*[set(s) for s in sig_rows]))
        mat = {n: np.array([s.get(n, np.nan) for s in sig_rows]) for n in names}
        # z-combos (diagnostic; standardised within the eval set)
        if "sc_support" in mat:
            mat["z_ent+sc"] = zscore(mat["max_ent"]) + zscore(mat["sc_support"])
        if "p_true_ctx" in mat:
            mat["z_ent+ptrue"] = zscore(mat["max_ent"]) + zscore(mat["p_true_ctx"])
        if "sc_support" in mat and "p_true_ctx" in mat:
            mat["z_ent+sc+ptrue"] = (zscore(mat["max_ent"]) + zscore(mat["sc_support"])
                                     + zscore(mat["p_true_ctx"]))
        data[ds] = mat
        labels[ds] = np.array(ys)
        skipped[ds] = n_skip

    all_names = sorted(set().union(*[set(m) for m in data.values()]))
    rng = np.random.default_rng(SEED)

    # coverage anchor: accept rate of max_ent < 0.4 (verifier direct tau)
    cov_anchor = {ds: float((data[ds]["max_ent"] < 0.4).mean()) for ds in data}

    out = []
    out.append("# Confidence Signal Battery — direct path, deployed setting\n")
    out.append("Probe: greedy regen (100% match with stored direct_pred), wiki ES top-8, "
               "adapter mode C. Label: alias-aware EM of the direct answer. "
               "All signals oriented higher = more uncertain. "
               f"Bootstrap: paired, {N_BOOT} resamples, seed {SEED}.\n")
    out.append("Rows skipped (empty answer): "
               + ", ".join(f"{ds}={skipped[ds]}" for ds in data) + "\n")

    macro: dict[str, list[float]] = defaultdict(list)
    per_ds_auroc: dict[str, dict[str, float]] = defaultdict(dict)

    for ds in data:
        y = labels[ds]
        mat = data[ds]
        base = mat[BASELINE]
        out.append(f"\n## {ds}  (n={len(y)}, wrong={int(y.sum())}, "
                   f"direct acc={100 * (1 - y.mean()):.1f}%, "
                   f"cov anchor={100 * cov_anchor[ds]:.1f}%)\n")
        out.append("| signal | AUROC | dAUROC vs max_ent | CI95 | P(d<=0) | "
                   "CW@med | R@10FP | SelAcc@cov |")
        out.append("|---|---:|---:|---|---:|---:|---:|---:|")
        rows_out = []
        for name in all_names:
            if name not in mat or np.isnan(mat[name]).any():
                continue
            u = mat[name]
            a = auroc(u, y)
            per_ds_auroc[ds][name] = a
            macro[name].append(a)
            if name == BASELINE:
                d_lo = d_hi = p_worse = None
            else:
                d_lo, d_hi, p_worse = paired_bootstrap(u, base, y, rng)
            rows_out.append((a, name, d_lo, d_hi, p_worse,
                             cw_at_med(u, y), recall_at_10fp(u, y),
                             sel_acc_at_cov(u, y, cov_anchor[ds])))
        rows_out.sort(reverse=True)
        base_a = per_ds_auroc[ds][BASELINE]
        for a, name, d_lo, d_hi, p_worse, cw, r10, sacc in rows_out:
            star = " **" if name == BASELINE else ""
            if d_lo is None:
                delta = ci = pw = "—"
            else:
                delta = f"{a - base_a:+.3f}"
                ci = f"[{d_lo:+.3f},{d_hi:+.3f}]"
                pw = f"{p_worse:.3f}"
            out.append(f"| {name}{star} | {a:.3f} | {delta} | {ci} | {pw} | "
                       f"{cw:.3f} | {r10:.3f} | {sacc:.3f} |")

    # macro table
    out.append("\n## Macro (mean AUROC across datasets)\n")
    out.append("| signal | macro AUROC | " + " | ".join(data.keys()) + " |")
    out.append("|---|---:|" + "---:|" * len(data))
    for name in sorted(macro, key=lambda n: -np.mean(macro[n])):
        if len(macro[name]) != len(data):
            continue
        cells = " | ".join(f"{per_ds_auroc[ds].get(name, float('nan')):.3f}" for ds in data)
        out.append(f"| {name} | {np.mean(macro[name]):.3f} | {cells} |")

    # Spearman correlation of top signals with max_ent (pooled)
    out.append("\n## Pooled Spearman rank correlation with max_ent\n")
    pooled = {n: np.concatenate([data[ds][n] for ds in data])
              for n in all_names if all(n in data[ds] for ds in data)}

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra = (ra - ra.mean()) / ra.std()
        rb = (rb - rb.mean()) / rb.std()
        return float((ra * rb).mean())

    out.append("| signal | rho(max_ent) |")
    out.append("|---|---:|")
    for n in sorted(pooled, key=lambda n: -np.mean(macro[n]) if macro[n] else 0):
        if n == BASELINE:
            continue
        out.append(f"| {n} | {spearman(pooled[n], pooled[BASELINE]):.3f} |")

    dest = ROOT / "results" / "confidence_battery_analysis.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
