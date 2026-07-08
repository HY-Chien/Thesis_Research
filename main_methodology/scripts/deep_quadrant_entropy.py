#!/usr/bin/env python3
"""Deeper both_correct vs both_wrong direct-entropy analysis (advisor p45 follow-up).

Adds to quadrant_confidence_analysis.md:
  A. distribution percentiles per quadrant (overlap check)
  B. direct_ent quintile -> direct-path accuracy calibration (monotonicity)
  C. threshold operating points (flag wrong if ent > tau)
  D. zero/near-zero entropy plateau in the deployed single-pass traces
"""
import json, math
from pathlib import Path

RESULTS = Path("/home/S113065528/Research/main_methodology/results")
FILES = {
    "2Wiki": "2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "HotpotQA": "hotpotqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "MuSiQue": "musique_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "PopQA": "popqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
}

def pct(vals, q):
    vals = sorted(vals)
    if not vals: return float("nan")
    idx = q * (len(vals) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)

def auroc(neg, pos):
    both = sorted([(v, 0) for v in neg] + [(v, 1) for v in pos])
    # rank-based AUROC with tie handling
    ranks, i = {}, 0
    vals = [v for v, _ in both]
    n = len(vals)
    r = [0.0] * n
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    pos_rank_sum = sum(rk for rk, (_, lab) in zip(r, both) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (pos_rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)

all_bc, all_bw, all_pairs = [], [], []
print("## A. direct_ent percentiles per quadrant")
print("| Dataset | Quadrant | n | p10 | p25 | p50 | p75 | p90 | %ent=0 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
per_ds = {}
for ds, fn in FILES.items():
    rows = [json.loads(l) for l in (RESULTS / fn).read_text().splitlines() if l.strip()]
    data = []
    for row in rows:
        ent = (row.get("direct") or {}).get("skip_probe_ent")
        try:
            ent = float(ent)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ent): continue
        d, c = int(row.get("direct_em") or 0), int(row.get("decomp_em") or 0)
        quad = "both_correct" if d and c else "direct_only" if d else "decomp_only" if c else "both_wrong"
        data.append((ent, d, quad))
    per_ds[ds] = data
    for quad in ["both_correct", "direct_only", "decomp_only", "both_wrong"]:
        e = [x[0] for x in data if x[2] == quad]
        z = sum(1 for v in e if v < 1e-6) / len(e) * 100
        print(f"| {ds} | {quad} | {len(e)} | " + " | ".join(f"{pct(e,q):.3f}" for q in (.1,.25,.5,.75,.9)) + f" | {z:.1f} |")
    all_bc += [x[0] for x in data if x[2] == "both_correct"]
    all_bw += [x[0] for x in data if x[2] == "both_wrong"]

print()
print("## B. direct_ent quintile -> direct-path accuracy (all 1000 q, per dataset)")
print("| Dataset | Q1 (lowest ent) | Q2 | Q3 | Q4 | Q5 (highest) | Spearman-ish monotone? |")
print("|---|---:|---:|---:|---:|---:|---|")
for ds, data in per_ds.items():
    srt = sorted(data, key=lambda x: x[0])
    n = len(srt)
    accs = []
    for i in range(5):
        chunk = srt[i * n // 5:(i + 1) * n // 5]
        accs.append(100 * sum(d for _, d, _ in chunk) / len(chunk))
    mono = all(accs[i] >= accs[i + 1] for i in range(4))
    print(f"| {ds} | " + " | ".join(f"{a:.1f}" for a in accs) + f" | {'yes' if mono else 'no'} |")

print()
print("## C. operating points: flag WRONG if direct_ent > tau (both_correct vs both_wrong only)")
print("| Dataset | tau = bc p75 | wrong flagged (recall%) | correct falsely flagged% | tau = bc p90 | recall% | false flag% |")
print("|---|---:|---:|---:|---:|---:|---:|")
for ds, data in per_ds.items():
    bc = [x[0] for x in data if x[2] == "both_correct"]
    bw = [x[0] for x in data if x[2] == "both_wrong"]
    for label_q in [(.75,), (.9,)]:
        pass
    t75, t90 = pct(bc, .75), pct(bc, .9)
    r75 = 100 * sum(1 for v in bw if v > t75) / len(bw)
    f75 = 100 * sum(1 for v in bc if v > t75) / len(bc)
    r90 = 100 * sum(1 for v in bw if v > t90) / len(bw)
    f90 = 100 * sum(1 for v in bc if v > t90) / len(bc)
    print(f"| {ds} | {t75:.3f} | {r75:.1f} | {f75:.1f} | {t90:.3f} | {r90:.1f} | {f90:.1f} |")

print()
print("## D. zero-entropy plateau in deployed single-pass direct traces")
print("| Dataset | n ent<1e-6 | direct acc in plateau% | share of all direct-wrong% |")
print("|---|---:|---:|---:|")
for ds, data in per_ds.items():
    plateau = [x for x in data if x[0] < 1e-6]
    wrong_all = sum(1 for _, d, _ in data if not d)
    wrong_pl = sum(1 for _, d, _ in plateau if not d)
    acc = 100 * sum(d for _, d, _ in plateau) / len(plateau) if plateau else float("nan")
    print(f"| {ds} | {len(plateau)} | {acc:.1f} | {100*wrong_pl/wrong_all:.1f} |")

print()
print(f"pooled both_correct vs both_wrong AUROC = {auroc(all_bc, all_bw):.3f}")
print(f"pooled bc median={pct(all_bc,.5):.3f}  bw median={pct(all_bw,.5):.3f}")
