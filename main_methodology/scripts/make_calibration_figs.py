"""Calibration figures: does answer confidence correlate with accuracy?

Uses the DIRECT-answer probe:
  accuracy  = direct (always-skip / M1) greedy-answer EM
  confidence signals (same probe, M2b multi-sample):
      sc_agree      self-consistency (majority agreement fraction) in [0,1]
      n_uniq        # distinct sampled answers (low = confident)
      mean_max_ent  mean max per-token entropy (low = confident)
      single_maxent greedy max_ent (M2 baseline single-sample signal)

Outputs (Thesis/figures/):
  calib_reliability.{pdf,png}   reliability diagram (sc_agree vs accuracy) + ECE
  calib_nuniq.{pdf,png}         accuracy vs n_uniq bucket, per dataset
  calib_entropy_decile.{pdf,png} accuracy vs entropy decile
Prints: Pearson/Spearman corr, ECE, AUROC, majority-vote EM gain.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import re, string

FIGDIR = "/home/S113065528/Research/Thesis/figures"
RES = "/home/S113065528/Research/main_methodology/results"

DS = ["hotpotqa", "2wiki", "musique", "popqa"]
LABEL = {"hotpotqa": "HotpotQA (OOD)", "2wiki": "2WikiMultiHopQA",
         "musique": "MuSiQue", "popqa": "PopQA"}
COLOR = {"hotpotqa": "#d62728", "2wiki": "#1f77b4",
         "musique": "#2ca02c", "popqa": "#ff7f0e"}


def _norm(s):
    if s is None:
        return ""
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def load(p):
    d = {}
    for line in open(p):
        r = json.loads(line)
        d[r["id"]] = r
    return d


def auroc(score, label):
    label = np.asarray(label)
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); o = allv.argsort()
    r = np.empty(len(allv)); r[o] = np.arange(1, len(allv) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


data = {}
for ds in DS:
    S = load(f"{RES}/{ds}_1000_alwaysskip.jsonl")
    M = load(f"{RES}/{ds}_1000_msample.jsonl")
    ids = sorted(set(S) & set(M))
    d = dict(
        em=np.array([S[i]["em"] for i in ids], float),
        sc=np.array([M[i]["sc_agree"] for i in ids], float),
        n_uniq=np.array([M[i]["n_uniq"] for i in ids], float),
        mme=np.array([M[i]["mean_max_ent"] for i in ids], float),
        single=np.array([S[i].get("skip_probe_ent", np.nan) for i in ids], float),
        maj_em=np.array([1.0 if _norm(M[i]["majority"]) == _norm(M[i]["gold"]) else 0.0
                         for i in ids], float),
    )
    data[ds] = d

# pooled
P = {k: np.concatenate([data[ds][k] for ds in DS]) for k in data["hotpotqa"]}


def ece(conf, correct, nbins=10):
    bins = np.linspace(0, 1, nbins + 1)
    e = 0.0; n = len(conf)
    for b in range(nbins):
        m = (conf >= bins[b]) & (conf < bins[b + 1]) if b < nbins - 1 else \
            (conf >= bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return e


# ---------- numbers ----------
print("=== Confidence–accuracy correlation (direct answer) ===")
print(f"{'ds':10} {'Pearson(sc,em)':>14} {'Spearman':>9} {'ECE(sc)':>8} "
      f"{'AUROC(sc)':>10} {'AUROC(n_uniq)':>13} {'greedyEM':>9} {'majEM':>7}")
for ds in DS:
    d = data[ds]
    pr = pearsonr(d["sc"], d["em"])[0]
    sp = spearmanr(d["sc"], d["em"])[0]
    e = ece(d["sc"], d["em"])
    a_sc = auroc(d["sc"], d["em"].astype(int))
    a_nu = auroc(-d["n_uniq"], d["em"].astype(int))
    print(f"{ds:10} {pr:14.3f} {sp:9.3f} {e:8.3f} {a_sc:10.3f} {a_nu:13.3f} "
          f"{100*d['em'].mean():8.1f}% {100*d['maj_em'].mean():6.1f}%")
pr = pearsonr(P["sc"], P["em"])[0]; sp = spearmanr(P["sc"], P["em"])[0]
print(f"{'POOLED':10} {pr:14.3f} {sp:9.3f} {ece(P['sc'],P['em']):8.3f} "
      f"{auroc(P['sc'],P['em'].astype(int)):10.3f} {auroc(-P['n_uniq'],P['em'].astype(int)):13.3f} "
      f"{100*P['em'].mean():8.1f}% {100*P['maj_em'].mean():6.1f}%")

# ---------- Fig 1: reliability diagram ----------
fig, ax = plt.subplots(figsize=(6, 5.4))
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect calibration")
nbins = 10
bins = np.linspace(0, 1, nbins + 1)
ctr = (bins[:-1] + bins[1:]) / 2
for ds in DS:
    d = data[ds]
    acc = []; cen = []
    for b in range(nbins):
        m = (d["sc"] >= bins[b]) & (d["sc"] <= bins[b + 1]) if b == nbins - 1 else \
            (d["sc"] >= bins[b]) & (d["sc"] < bins[b + 1])
        if m.sum() >= 10:
            acc.append(d["em"][m].mean()); cen.append(ctr[b])
    ax.plot(cen, acc, "-o", color=COLOR[ds], ms=5, lw=1.8,
            label=f"{LABEL[ds]} (ECE={ece(d['sc'],d['em']):.2f})")
ax.set_xlabel("Self-consistency confidence  (fraction agreeing with majority)")
ax.set_ylabel("Empirical accuracy (EM)")
ax.set_title("Reliability diagram: confidence vs. accuracy\n(direct single-step answer)")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/calib_reliability.pdf")
fig.savefig(f"{FIGDIR}/calib_reliability.png", dpi=200)
plt.close(fig)

# ---------- Fig 2: accuracy vs n_uniq bucket ----------
buckets = [1, 2, 3, 4, 5]  # 5 = "5+"
blabels = ["1", "2", "3", "4", "5+"]
fig, ax = plt.subplots(figsize=(7, 4.6))
w = 0.2
x = np.arange(len(buckets))
for j, ds in enumerate(DS):
    d = data[ds]
    accs = []
    for b in buckets:
        m = (d["n_uniq"] == b) if b < 5 else (d["n_uniq"] >= 5)
        accs.append(100 * d["em"][m].mean() if m.sum() > 0 else np.nan)
    ax.bar(x + (j - 1.5) * w, accs, w, color=COLOR[ds], label=LABEL[ds])
ax.set_xticks(x); ax.set_xticklabels(blabels)
ax.set_xlabel("n_uniq  (number of distinct answers over 15 samples)")
ax.set_ylabel("Accuracy (EM %)")
ax.set_title("Accuracy decreases monotonically with answer dispersion")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/calib_nuniq.pdf")
fig.savefig(f"{FIGDIR}/calib_nuniq.png", dpi=200)
plt.close(fig)

# ---------- Fig 3: accuracy vs entropy decile (pooled, multiple signals) ----------
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for sig, lab, col in [("mme", "mean max-entropy (multi-sample)", "#1f77b4"),
                      ("single", "greedy max-entropy (single-sample)", "#d62728")]:
    v = P[sig]; ok = ~np.isnan(v)
    v = v[ok]; em = P["em"][ok]
    q = np.quantile(v, np.linspace(0, 1, 11))
    cen = []; acc = []
    for b in range(10):
        m = (v >= q[b]) & (v <= q[b + 1]) if b == 9 else (v >= q[b]) & (v < q[b + 1])
        if m.sum() > 0:
            cen.append((b + 0.5) / 10); acc.append(100 * em[m].mean())
    ax.plot(cen, acc, "-o", color=col, ms=5, label=lab)
ax.set_xlabel("Entropy decile  (left = low entropy / confident)")
ax.set_ylabel("Accuracy (EM %)")
ax.set_title("Pooled: accuracy falls as answer entropy rises")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/calib_entropy_decile.pdf")
fig.savefig(f"{FIGDIR}/calib_entropy_decile.png", dpi=200)
plt.close(fig)

print("\nSaved figures to", FIGDIR)
for f in ["calib_reliability", "calib_nuniq", "calib_entropy_decile"]:
    print("  -", f + ".pdf / .png")
