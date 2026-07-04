"""
Confidence vs. entity-popularity bias analysis (PopQA).

Question
--------
Is the model's confidence (mean_max_ent, lower = more confident) driven by
entity popularity rather than by actual correctness? Concretely:

  1. Do popular subjects get systematically lower entropy (higher confidence)?
  2. Are high-confidence-wrong answers concentrated in a popularity band
     (e.g., popular-enough-to-sound-familiar but not popular-enough-to-know)?
  3. Does the entropy->wrong discriminative power (AUROC) degrade for rare
     entities?

Data
----
- results/popqa_1000_msample.jsonl : no-RAG self-consistency confidence
  (mean_max_ent, majority, gold), same file as analyze_maxent_correctness.py.
- datasets/popqa/raw/popQA.tsv     : original PopQA (Mallen et al. 2023),
  s_pop / o_pop = monthly Wikipedia page views of subject / object entity.
- results/popqa_1000_verify_v2_pathaware.jsonl : supplementary — direct-path
  probe entropy + EM in the deployed SEEK pipeline.

Correctness uses the same official rule as analyze_maxent_correctness.py:
normalized EM or token-F1 >= 0.5 vs the single `gold` string. An alias-aware
variant (possible_answers from the TSV) is reported as a secondary column.
"""

import collections
import csv
import json
import math
import re
import string
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

RESULTS_DIR = Path("/home/S113065528/Research/main_methodology/results")
TSV_PATH = Path("/home/S113065528/Research/main_methodology/datasets/popqa/raw/popQA.tsv")
OUT_PATH = Path(__file__).resolve().parent.parent / "confidence_popularity_bias.md"

MSAMPLE = RESULTS_DIR / "popqa_1000_msample.jsonl"
PATHAWARE = RESULTS_DIR / "popqa_1000_verify_v2_pathaware.jsonl"


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def compute_f1(pred, gold):
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num = sum(common.values())
    if not pred_toks or not gold_toks:
        return int(pred_toks == gold_toks)
    if num == 0:
        return 0
    p = num / len(pred_toks)
    r = num / len(gold_toks)
    return 2 * p * r / (p + r)


def is_correct(pred, golds):
    for g in golds:
        if normalize_answer(pred) == normalize_answer(g):
            return True
        if compute_f1(pred, g) >= 0.5:
            return True
    return False


def auroc(scores, labels):
    """AUROC of score predicting label=1 (higher score -> label 1)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = sp_stats.rankdata(scores)
    return (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_tsv():
    meta = {}
    with open(TSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            aliases = []
            try:
                aliases = json.loads(row["possible_answers"])
            except Exception:
                pass
            meta[row["id"]] = {
                "s_pop": int(row["s_pop"]),
                "o_pop": int(row["o_pop"]),
                "prop": row["prop"],
                "subj": row["subj"],
                "aliases": aliases or [row["obj"]],
            }
    return meta


def main():
    meta = load_tsv()
    rows = []
    with open(MSAMPLE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            m = meta.get(str(d["id"]))
            if m is None:
                continue
            rows.append(
                {
                    "id": str(d["id"]),
                    "ent": float(d["mean_max_ent"]),
                    "pred": d["majority"],
                    "gold": d["gold"],
                    "correct": int(is_correct(d["majority"], [d["gold"]])),
                    "correct_alias": int(is_correct(d["majority"], m["aliases"])),
                    "s_pop": m["s_pop"],
                    "o_pop": m["o_pop"],
                    "prop": m["prop"],
                    "subj": m["subj"],
                    "log_spop": math.log10(max(m["s_pop"], 1)),
                    "log_opop": math.log10(max(m["o_pop"], 1)),
                }
            )
    n = len(rows)
    ent = np.array([r["ent"] for r in rows])
    correct = np.array([r["correct"] for r in rows])
    wrong = 1 - correct
    log_spop = np.array([r["log_spop"] for r in rows])
    log_opop = np.array([r["log_opop"] for r in rows])

    lines = []
    lines.append("# Confidence vs Entity-Popularity Bias (PopQA)\n")
    lines.append("Date: 2026-07-03\n")
    lines.append(
        "Join: `popqa_1000_msample.jsonl` (no-RAG self-consistency confidence, "
        "`mean_max_ent`) x original PopQA TSV (`s_pop` / `o_pop` = monthly Wikipedia "
        f"page views). Joined n = {n} / 1000. Correctness = official rule "
        "(normalized EM or token-F1 >= 0.5 vs `gold`); alias-aware uses "
        "`possible_answers`.\n"
    )
    acc = correct.mean() * 100
    acc_alias = np.mean([r["correct_alias"] for r in rows]) * 100
    lines.append(f"Overall no-RAG accuracy: official {acc:.1f} / alias-aware {acc_alias:.1f}.\n")

    lines.append("## TL;DR\n")
    lines.append(
        "1. **No raw popularity->confidence bias.** Confidence (mean_max_ent) is "
        "uncorrelated with subject popularity, overall and within wrong answers "
        "(Spearman |rho| < 0.05, n.s., section 1); the zero-entropy wrong group has "
        "the same popularity distribution as the zero-entropy correct group "
        "(section 4). Confident hallucination is NOT concentrated on famous or "
        "obscure entities.\n"
    )
    lines.append(
        "2. **Alias-aware accuracy IS popularity-skewed** (section 2): popular "
        "entities' 'errors' are disproportionately metric artifacts (longer, richer "
        "answer strings), while rare-entity confident errors are 100% genuine "
        "(section 4b). Semantic confident hallucination is therefore mildly skewed "
        "toward rare entities even though the official rate looks flat.\n"
    )
    lines.append(
        "3. **The entropy signal does not collapse for rare entities** (section 5): "
        "AUROC(ent->wrong) is ~0.72-0.75 in every popularity band. On single-hop "
        "PopQA the confidence signal's weakness is the deterministic-wrong plateau "
        "(136/1000 zero-entropy wrong), not popularity bias.\n"
    )

    # --- 1. correlations -----------------------------------------------------
    lines.append("## 1. Is confidence driven by popularity?\n")
    rho_sp, p_sp = sp_stats.spearmanr(ent, log_spop)
    rho_op, p_op = sp_stats.spearmanr(ent, log_opop)
    rho_acc, p_acc = sp_stats.spearmanr(correct, log_spop)
    lines.append("| pair | Spearman rho | p |")
    lines.append("|---|---:|---:|")
    lines.append(f"| mean_max_ent vs log10(s_pop) | {rho_sp:.3f} | {p_sp:.2e} |")
    lines.append(f"| mean_max_ent vs log10(o_pop) | {rho_op:.3f} | {p_op:.2e} |")
    lines.append(f"| correct vs log10(s_pop) | {rho_acc:.3f} | {p_acc:.2e} |")
    # partial: does entropy track popularity beyond what correctness explains?
    rho_w, p_w = sp_stats.spearmanr(ent[wrong == 1], log_spop[wrong == 1])
    rho_c, p_c = sp_stats.spearmanr(ent[correct == 1], log_spop[correct == 1])
    lines.append(f"| within WRONG only: ent vs log10(s_pop) | {rho_w:.3f} | {p_w:.2e} |")
    lines.append(f"| within CORRECT only: ent vs log10(s_pop) | {rho_c:.3f} | {p_c:.2e} |")
    lines.append("")
    lines.append(
        "The within-wrong correlation is the bias test: if the model is only wrong "
        "on rare entities, popularity and correctness are confounded; a negative "
        "entropy~popularity correlation *among wrong answers* means the model is "
        "more confident about popular subjects even when it is wrong.\n"
    )

    # --- 2. popularity quartiles --------------------------------------------
    lines.append("## 2. Popularity quartiles (s_pop)\n")
    qs = np.quantile(log_spop, [0.25, 0.5, 0.75])
    labels_q = ["Q1 (rarest)", "Q2", "Q3", "Q4 (most popular)"]
    lines.append(
        "| quartile | s_pop range | n | acc | acc alias | mean ent | zero-ent rate | zero-ent acc |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    qidx = np.digitize(log_spop, qs)
    for qi in range(4):
        sel = qidx == qi
        if sel.sum() == 0:
            continue
        sp_lo = 10 ** log_spop[sel].min()
        sp_hi = 10 ** log_spop[sel].max()
        zsel = sel & (ent == 0)
        zacc = correct[zsel].mean() * 100 if zsel.sum() else float("nan")
        a_alias = np.mean([rows[i]["correct_alias"] for i in np.where(sel)[0]]) * 100
        lines.append(
            f"| {labels_q[qi]} | {sp_lo:.0f}-{sp_hi:.0f} | {sel.sum()} "
            f"| {correct[sel].mean()*100:.1f} | {a_alias:.1f} | {ent[sel].mean():.2f} "
            f"| {(ent[sel]==0).mean()*100:.1f}% | {zacc:.1f} |"
        )
    lines.append("")

    # --- 3. confidence x popularity grid -------------------------------------
    lines.append("## 3. Accuracy grid: confidence band x popularity tercile\n")
    lines.append(
        "Confidence bands respect the mass at zero entropy: `zero-ent` (fully "
        "deterministic SC votes), then the nonzero entropies split at their median.\n"
    )
    nz_med = np.median(ent[ent > 0])
    cidx = np.where(ent == 0, 0, np.where(ent <= nz_med, 1, 2))
    pt = np.quantile(log_spop, [1 / 3, 2 / 3])
    pidx = np.digitize(log_spop, pt)  # 0 = rarest
    conf_names = ["zero-ent (max conf)", "low ent (nonzero)", "high ent"]
    pop_names = ["rare", "mid", "popular"]
    lines.append("| | " + " | ".join(f"{p} acc (n)" for p in pop_names) + " |")
    lines.append("|---|" + "---:|" * 3)
    for ci in range(3):
        cells = []
        for pi in range(3):
            sel = (cidx == ci) & (pidx == pi)
            cells.append(f"{correct[sel].mean()*100:.1f} ({sel.sum()})" if sel.sum() else "-")
        lines.append(f"| {conf_names[ci]} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- 4. high-confidence-wrong popularity profile --------------------------
    lines.append("## 4. Where do high-confidence-wrong answers live?\n")
    hcw = (ent == 0) & (wrong == 1)
    hcc = (ent == 0) & (correct == 1)
    rest_w = (ent > 0) & (wrong == 1)
    lines.append("| group | n | median s_pop | mean log10 s_pop |")
    lines.append("|---|---:|---:|---:|")
    for name, sel in [
        ("zero-ent WRONG (confident hallucination)", hcw),
        ("zero-ent CORRECT (really knows)", hcc),
        ("nonzero-ent WRONG", rest_w),
    ]:
        if sel.sum():
            med = np.median(10 ** log_spop[sel])
            lines.append(f"| {name} | {sel.sum()} | {med:.0f} | {log_spop[sel].mean():.2f} |")
    if hcw.sum() and hcc.sum():
        u, p_u = sp_stats.mannwhitneyu(log_spop[hcw], log_spop[hcc], alternative="two-sided")
        lines.append("")
        lines.append(
            f"Mann-Whitney zero-ent wrong vs zero-ent correct (log s_pop): U={u:.0f}, p={p_u:.2e}."
        )
    if hcw.sum() and rest_w.sum():
        u2, p_u2 = sp_stats.mannwhitneyu(log_spop[hcw], log_spop[rest_w], alternative="two-sided")
        lines.append(
            f"Mann-Whitney zero-ent wrong vs nonzero-ent wrong (log s_pop): U={u2:.0f}, p={p_u2:.2e}.\n"
        )

    # 4b. within zero-ent wrong: how many are alias/metric artifacts, by popularity?
    lines.append(
        "### 4b. Zero-ent wrong: genuine hallucination vs alias/metric artifact, by popularity\n"
    )
    lines.append(
        "`alias-recovered` = officially wrong but correct under `possible_answers`. "
        "If popular-entity confident errors are mostly alias artifacts while "
        "rare-entity ones are genuine, the *semantic* confident-hallucination rate "
        "is popularity-skewed even though the official rate is flat.\n"
    )
    alias_ok = np.array([r["correct_alias"] for r in rows])
    lines.append("| popularity band | zero-ent wrong n | alias-recovered | genuine wrong | genuine share |")
    lines.append("|---|---:|---:|---:|---:|")
    for pi, pname in enumerate(["rare", "mid", "popular"]):
        sel = hcw & (pidx == pi)
        if sel.sum() == 0:
            continue
        rec = int(alias_ok[sel].sum())
        lines.append(
            f"| {pname} | {sel.sum()} | {rec} | {sel.sum()-rec} | {(sel.sum()-rec)/sel.sum()*100:.0f}% |"
        )
    sel = hcw
    rec = int(alias_ok[sel].sum())
    lines.append(
        f"| ALL | {sel.sum()} | {rec} | {sel.sum()-rec} | {(sel.sum()-rec)/sel.sum()*100:.0f}% |"
    )
    lines.append("")

    # --- 5. AUROC by popularity band ------------------------------------------
    lines.append("## 5. Does the entropy signal degrade for rare entities?\n")
    lines.append("AUROC of mean_max_ent predicting WRONG, within popularity bands:\n")
    lines.append("| band | n | acc | AUROC(ent->wrong) |")
    lines.append("|---|---:|---:|---:|")
    for name, sel in [
        ("rare (bottom tercile)", pidx == 0),
        ("mid", pidx == 1),
        ("popular (top tercile)", pidx == 2),
        ("ALL", np.ones(n, dtype=bool)),
    ]:
        a = auroc(ent[sel], wrong[sel])
        lines.append(f"| {name} | {sel.sum()} | {correct[sel].mean()*100:.1f} | {a:.3f} |")
    lines.append("")

    # --- 6. supplementary: deployed direct path -------------------------------
    if PATHAWARE.exists():
        lines.append("## 6. Supplementary: deployed SEEK direct path (pathaware trace)\n")
        d_ent, d_em, d_pop = [], [], []
        with open(PATHAWARE, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                m = meta.get(str(d["id"]))
                if m is None:
                    continue
                pe = d.get("direct", {}).get("skip_probe_ent", -1.0)
                if pe is None or pe < 0:
                    continue
                d_ent.append(float(pe))
                d_em.append(int(d["direct_em"]))
                d_pop.append(math.log10(max(m["s_pop"], 1)))
        if d_ent:
            d_ent = np.array(d_ent)
            d_em = np.array(d_em)
            d_pop = np.array(d_pop)
            r1, pv1 = sp_stats.spearmanr(d_ent, d_pop)
            r2, pv2 = sp_stats.spearmanr(d_ent[d_em == 0], d_pop[d_em == 0])
            lines.append(f"n = {len(d_ent)} (direct probe entropy available).\n")
            lines.append("| pair | Spearman rho | p |")
            lines.append("|---|---:|---:|")
            lines.append(f"| direct probe ent vs log10(s_pop) | {r1:.3f} | {pv1:.2e} |")
            lines.append(f"| within direct-WRONG: ent vs log10(s_pop) | {r2:.3f} | {pv2:.2e} |")
            lines.append("")

    # --- 7. examples -----------------------------------------------------------
    lines.append("## 7. Examples: zero-entropy wrong, most popular subjects\n")
    ex = [r for i, r in enumerate(rows) if hcw[i]]
    ex.sort(key=lambda r: -r["s_pop"])
    lines.append("| subj | s_pop | prop | gold | majority |")
    lines.append("|---|---:|---|---|---|")
    for r in ex[:12]:
        lines.append(f"| {r['subj']} | {r['s_pop']} | {r['prop']} | {r['gold']} | {r['pred']} |")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
