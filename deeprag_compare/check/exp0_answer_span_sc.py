"""
Exp-0: Compare confidence-metric variants over different spans of the CoT generation.

Hypothesis (motivated by professor's "QA-specific metric" feedback + structural
mismatch between SC's design (whole-generation = answer) and QA's design (final
short entity = answer)):

    Aggregate-CoT SC dilutes the actual answer-confidence signal across ~95%
    filler tokens. Span-restricted confidence (answer-only, first-entity-only,
    or CoT-only) should be more discriminative for predicting EM.

For each of 400 sample records, compute:
  - mean log-prob of chosen token  (>= higher = more confident)
  - mean entropy                    (lower = more confident)
  - mean (top1_lp - top2_lp) margin (higher = more confident)
  - min log-prob, max entropy        (worst-token diagnostics)

over four spans:
  - full       : the whole generation (current SC behavior)
  - answer     : tokens after "So the answer is " until end of that sentence
  - cot        : everything before "So the answer is "
  - first_ent  : first occurrence of the predicted answer string in the CoT

Output:
  exp0_output/exp0_summary.md
  exp0_output/auroc_table.json
  exp0_output/per_record.jsonl
  exp0_output/figures/{boxplot,auroc}.png
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/S113065528/Research/deeprag_compare")
DUMP_FILE = ROOT / "check/multi_sample_dump.jsonl"
OUT_DIR = ROOT / "check/exp0_output"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Span detection
# ---------------------------------------------------------------------------

ANSWER_PHRASE_RE = re.compile(r"(?i)so the answer is\s+|the answer is\s+")


def reconstruct(tokens):
    """Concat token strings; return text and per-token start-char offsets."""
    text = ""
    offsets = []
    for t in tokens:
        offsets.append(len(text))
        text += t["tok"]
    return text, offsets


def char_to_token_idx(offsets, char_pos, n_tokens):
    """First token whose start >= char_pos (i.e., first token at or after this char)."""
    for i, off in enumerate(offsets):
        if off >= char_pos:
            return i
    return n_tokens  # past end


def find_answer_span(tokens):
    """Tokens for the answer in 'So the answer is X.' — exclude the trailing period."""
    if not tokens:
        return None
    text, offsets = reconstruct(tokens)
    m = ANSWER_PHRASE_RE.search(text)
    if not m:
        return None
    start = char_to_token_idx(offsets, m.end(), len(tokens))
    if start >= len(tokens):
        return None
    end = start
    for i in range(start, len(tokens)):
        tok = tokens[i]["tok"]
        if "<|eot_id|>" in tok or "\n" in tok:
            break
        if tok.strip() == "." or (tok.strip().endswith(".") and i > start):
            # include up to (but not including) the final period token
            end = i
            break
        end = i + 1
    if end <= start:
        return None
    return (start, end)


def find_first_entity_span(record):
    """First occurrence of the predicted answer entity in the generation."""
    pred = (record.get("pred") or "").strip()
    if not pred or len(pred) < 2:
        return None
    tokens = record["tokens"]
    text, offsets = reconstruct(tokens)
    text_lc = text.lower()
    pred_lc = pred.lower()
    idx = text_lc.find(pred_lc)
    if idx == -1:
        return None
    start_tok = None
    for i in range(len(offsets)):
        next_off = offsets[i + 1] if i + 1 < len(offsets) else len(text)
        if offsets[i] <= idx < next_off:
            start_tok = i
            break
    if start_tok is None:
        return None
    end_char = idx + len(pred_lc)
    end_tok = start_tok + 1
    while end_tok < len(offsets) and offsets[end_tok] < end_char:
        end_tok += 1
    return (start_tok, end_tok)


# ---------------------------------------------------------------------------
# Span-level confidence metrics
# ---------------------------------------------------------------------------

def span_metrics(tokens, span):
    if span is None:
        return None
    s, e = span
    if e <= s or s < 0 or e > len(tokens):
        return None
    sub = tokens[s:e]
    lps = [t["lp"] for t in sub]
    ents = [t["entropy"] for t in sub]
    margins = [t["top1_lp"] - t["top2_lp"] for t in sub]
    return {
        "len": e - s,
        "start_pos": s,
        "mean_lp": float(mean(lps)),
        "min_lp": float(min(lps)),
        "mean_ent": float(mean(ents)),
        "max_ent": float(max(ents)),
        "mean_margin": float(mean(margins)),
        "min_margin": float(min(margins)),
    }


def process(record):
    tokens = record["tokens"]
    n = len(tokens)
    full = span_metrics(tokens, (0, n))
    ans = find_answer_span(tokens)
    cot_span = (0, ans[0]) if ans else None
    fe = find_first_entity_span(record)

    # Trim full to drop the next-question continuation noise:
    # find first "\n\nQuestion" or pre-eot block end
    text, offsets = reconstruct(tokens)
    cut_match = re.search(r"\n\n(?:Question|Q:)", text)
    cut_tok_idx = char_to_token_idx(offsets, cut_match.start(), n) if cut_match else n
    full_clean = span_metrics(tokens, (0, cut_tok_idx)) if cut_tok_idx > 0 else full

    return {
        "qid": record["qid"],
        "stratum": record["stratum"],
        "sample_idx": record["sample_idx"],
        "em": record["em"],
        "f1": record["f1"],
        "pred": record["pred"],
        "sc_orig": record["sc"],
        "n_tokens_total": n,
        "spans": {
            "full": full,
            "full_clean": full_clean,
            "answer": span_metrics(tokens, ans),
            "cot": span_metrics(tokens, cot_span),
            "first_entity": span_metrics(tokens, fe),
        },
        "has_answer_span": ans is not None,
        "has_first_entity": fe is not None,
        "answer_span_len": (ans[1] - ans[0]) if ans else None,
    }


# ---------------------------------------------------------------------------
# Manual AUROC (no sklearn)
# ---------------------------------------------------------------------------

def auroc(y_true, scores):
    """Mann-Whitney U / rank-sum AUROC.  y_true ∈ {0,1}; higher score = positive."""
    pairs = list(zip(scores, y_true))
    n = len(pairs)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # ranks are 1-indexed
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRIC_DEFS = [
    # (name, span_key, metric_key, higher_means_more_confident)
    ("sc_orig",                None,           None,           True),
    ("full_lp",                "full",         "mean_lp",      True),
    ("full_clean_lp",          "full_clean",   "mean_lp",      True),
    ("full_neg_ent",           "full",         "mean_ent",     False),
    ("full_clean_neg_ent",     "full_clean",   "mean_ent",     False),
    ("full_margin",            "full",         "mean_margin",  True),
    ("answer_lp",              "answer",       "mean_lp",      True),
    ("answer_min_lp",          "answer",       "min_lp",       True),
    ("answer_neg_ent",         "answer",       "mean_ent",     False),
    ("answer_max_neg_ent",     "answer",       "max_ent",      False),
    ("answer_margin",          "answer",       "mean_margin",  True),
    ("answer_min_margin",      "answer",       "min_margin",   True),
    ("cot_lp",                 "cot",          "mean_lp",      True),
    ("cot_neg_ent",            "cot",          "mean_ent",     False),
    ("first_ent_lp",           "first_entity", "mean_lp",      True),
    ("first_ent_min_lp",       "first_entity", "min_lp",       True),
    ("first_ent_neg_ent",      "first_entity", "mean_ent",     False),
]


def get_score(p, defn):
    name, span_key, metric_key, hi_conf = defn
    if name == "sc_orig":
        return p["sc_orig"]
    s = p["spans"].get(span_key)
    if s is None:
        return None
    v = s.get(metric_key)
    if v is None:
        return None
    return v if hi_conf else -v


def main():
    print("Loading dump...")
    records = [json.loads(l) for l in open(DUMP_FILE)]
    print(f"  loaded {len(records)} records")

    print("Computing span metrics...")
    processed = [process(r) for r in records]
    n_with_ans = sum(1 for p in processed if p["has_answer_span"])
    n_with_fe = sum(1 for p in processed if p["has_first_entity"])
    print(f"  detected answer span: {n_with_ans}/{len(processed)}")
    print(f"  detected first-entity span: {n_with_fe}/{len(processed)}")

    # Save per-record processed view
    per_rec_path = OUT_DIR / "per_record.jsonl"
    with open(per_rec_path, "w") as f:
        for p in processed:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  per-record dump → {per_rec_path}")

    # ============================================================
    # AUROC: which metric best predicts EM (sample-level)?
    # ============================================================
    print("\n=== Sample-level AUROC for predicting EM ===")
    auroc_rows = []
    for defn in METRIC_DEFS:
        ys, ss = [], []
        for p in processed:
            sc = get_score(p, defn)
            if sc is None:
                continue
            ys.append(p["em"])
            ss.append(sc)
        a = auroc(ys, ss) if len(ys) >= 10 else float("nan")
        auroc_rows.append({"metric": defn[0], "auroc": a, "n": len(ys),
                           "n_pos": sum(ys), "n_neg": len(ys) - sum(ys)})
    auroc_rows.sort(key=lambda r: -r["auroc"] if not math.isnan(r["auroc"]) else 0)
    print(f"{'metric':<25s} {'AUROC':>7s} {'n':>5s} {'n_pos':>6s}")
    for r in auroc_rows:
        print(f"{r['metric']:<25s} {r['auroc']:>7.4f} {r['n']:>5d} {r['n_pos']:>6d}")
    with open(OUT_DIR / "auroc_table.json", "w") as f:
        json.dump(auroc_rows, f, indent=2)

    # ============================================================
    # Per-stratum mean ± std for top metrics
    # ============================================================
    print("\n=== Per-stratum stats (mean ± std) ===")
    strata = ["HI_correct", "HI_wrong", "LO_correct", "LO_wrong"]
    interesting = ["sc_orig", "full_lp", "answer_lp", "answer_min_lp",
                   "answer_neg_ent", "first_ent_lp", "cot_lp"]
    defn_by_name = {d[0]: d for d in METRIC_DEFS}

    stratum_stats = defaultdict(dict)
    for m in interesting:
        defn = defn_by_name[m]
        for st in strata:
            vals = [get_score(p, defn) for p in processed if p["stratum"] == st]
            vals = [v for v in vals if v is not None]
            if not vals:
                stratum_stats[m][st] = None
                continue
            stratum_stats[m][st] = {
                "mean": float(mean(vals)), "std": float(stdev(vals)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }

    print(f"{'metric':<25s} | " + " | ".join(f"{s:^15s}" for s in strata))
    for m in interesting:
        cells = []
        for st in strata:
            s = stratum_stats[m][st]
            cells.append(f"{s['mean']:>6.2f} ± {s['std']:.2f}" if s else "n/a")
        print(f"{m:<25s} | " + " | ".join(f"{c:^15s}" for c in cells))

    # ============================================================
    # Boxplot per stratum × span
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_metrics = [
        ("sc_orig",         "Original SC (full CoT)"),
        ("full_lp",         "Full mean log-prob"),
        ("answer_lp",       "Answer-span mean log-prob"),
        ("answer_min_lp",   "Answer-span MIN log-prob"),
        ("first_ent_lp",    "First-entity mean log-prob"),
        ("cot_lp",          "CoT-only mean log-prob"),
    ]
    colors = {"HI_correct": "#2c7", "HI_wrong": "#c33",
              "LO_correct": "#27c", "LO_wrong": "#a3a"}
    for ax, (m, title) in zip(axes.flat, plot_metrics):
        defn = defn_by_name[m]
        data = []
        labels = []
        for st in strata:
            vals = [get_score(p, defn) for p in processed if p["stratum"] == st]
            vals = [v for v in vals if v is not None]
            data.append(vals)
            labels.append(st)
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        for patch, st in zip(bp["boxes"], strata):
            patch.set_facecolor(colors[st]); patch.set_alpha(0.5)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Confidence metric distribution by stratum (lower = less confident)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplot_by_stratum.png", dpi=110)
    plt.close(fig)
    print(f"  → {FIG_DIR/'boxplot_by_stratum.png'}")

    # ============================================================
    # AUROC bar chart
    # ============================================================
    fig, ax = plt.subplots(figsize=(11, 6))
    rows = [r for r in auroc_rows if not math.isnan(r["auroc"])]
    rows.sort(key=lambda r: r["auroc"])
    names = [r["metric"] for r in rows]
    aurocs = [r["auroc"] for r in rows]
    bar_colors = []
    for n in names:
        if n.startswith("answer"): bar_colors.append("#c33")
        elif n.startswith("first_ent"): bar_colors.append("#e80")
        elif n.startswith("cot"): bar_colors.append("#27c")
        elif n.startswith("full"): bar_colors.append("#888")
        else: bar_colors.append("#2c7")
    ax.barh(names, aurocs, color=bar_colors)
    ax.axvline(0.5, ls="--", c="k", alpha=0.5)
    for i, v in enumerate(aurocs):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("AUROC predicting EM (higher = better)")
    ax.set_title("Discriminative power per confidence variant (sample-level, N=400)")
    ax.set_xlim(0.4, max(aurocs) + 0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "auroc_bars.png", dpi=110)
    plt.close(fig)
    print(f"  → {FIG_DIR/'auroc_bars.png'}")

    # ============================================================
    # Question-level: do the 4 strata separate better with answer-span SC?
    # ============================================================
    print("\n=== Per-question (mean over 20 samples) ===")
    by_qid = defaultdict(list)
    for p in processed:
        by_qid[p["qid"]].append(p)

    qid_summary = []
    for qid, ps in by_qid.items():
        st = ps[0]["stratum"]
        em_rate = mean(p["em"] for p in ps)
        full_lp_mean = mean(p["spans"]["full"]["mean_lp"] for p in ps)
        ans_lps = [p["spans"]["answer"]["mean_lp"] for p in ps if p["spans"]["answer"]]
        ans_lp_mean = mean(ans_lps) if ans_lps else None
        ans_lp_std = stdev(ans_lps) if len(ans_lps) > 1 else 0.0
        cot_lps = [p["spans"]["cot"]["mean_lp"] for p in ps if p["spans"]["cot"]]
        cot_lp_mean = mean(cot_lps) if cot_lps else None
        gap = (ans_lp_mean - cot_lp_mean) if (ans_lp_mean is not None and cot_lp_mean is not None) else None
        qid_summary.append({
            "qid": qid, "stratum": st, "em_rate": em_rate,
            "full_lp": full_lp_mean,
            "ans_lp": ans_lp_mean, "ans_lp_std": ans_lp_std,
            "cot_lp": cot_lp_mean,
            "ans_minus_cot": gap,
            "question": ps[0]["pred"][:0] + "",  # filled below
        })
    # attach question text
    qmap = {r["qid"]: r["question"] for r in records}
    for q in qid_summary:
        q["question"] = qmap.get(q["qid"], "")[:60]

    qid_summary.sort(key=lambda x: (x["stratum"], -x["em_rate"]))
    fmt = lambda v, prec=3, w=8: ("n/a".rjust(w) if v is None else f"{v:>{w}.{prec}f}")
    print(f"{'stratum':<12s} {'qid':<10s} {'em%':>5s} {'full_lp':>8s} "
          f"{'ans_lp':>8s} {'cot_lp':>8s} {'Δ':>7s}  q")
    for q in qid_summary:
        print(f"{q['stratum']:<12s} {q['qid'][:10]:<10s} "
              f"{q['em_rate']*100:>5.1f} {fmt(q['full_lp'])} "
              f"{fmt(q['ans_lp'])} {fmt(q['cot_lp'])} "
              f"{fmt(q['ans_minus_cot'], 3, 7)}  {q['question']}")

    # ============================================================
    # Markdown report
    # ============================================================
    md = []
    md.append("# Exp-0: Answer-Span vs Full-CoT Confidence Metrics")
    md.append("")
    md.append("Hypothesis: aggregate-CoT SC dilutes the actual answer-confidence "
              "signal across ~95% filler tokens, so span-restricted confidence "
              "should be more discriminative for predicting EM.")
    md.append("")
    md.append(f"- N samples: {len(processed)} (20 questions × 20 samples)")
    md.append(f"- Detected `So the answer is X` span: {n_with_ans}/{len(processed)} "
              f"({100*n_with_ans/len(processed):.1f}%)")
    md.append(f"- Detected first-entity span (pred occurs in CoT): "
              f"{n_with_fe}/{len(processed)} ({100*n_with_fe/len(processed):.1f}%)")
    md.append("")

    md.append("## Sample-level AUROC for predicting EM")
    md.append("")
    md.append("| Metric | AUROC | n | n_correct |")
    md.append("|---|---|---|---|")
    for r in auroc_rows:
        a = "n/a" if math.isnan(r["auroc"]) else f"{r['auroc']:.4f}"
        md.append(f"| `{r['metric']}` | {a} | {r['n']} | {r['n_pos']} |")
    md.append("")
    md.append("![AUROC bars](figures/auroc_bars.png)")
    md.append("")

    md.append("## Per-stratum stats (mean ± std)")
    md.append("")
    md.append("| Metric | HI_correct | HI_wrong | LO_correct | LO_wrong |")
    md.append("|---|---|---|---|---|")
    for m in interesting:
        cells = []
        for st in strata:
            s = stratum_stats[m][st]
            cells.append(f"{s['mean']:.2f} ± {s['std']:.2f}" if s else "n/a")
        md.append(f"| `{m}` | " + " | ".join(cells) + " |")
    md.append("")
    md.append("![Boxplots](figures/boxplot_by_stratum.png)")
    md.append("")

    md.append("## Per-question summary (mean over 20 samples)")
    md.append("")
    md.append("| stratum | qid | em% | full_lp | ans_lp | cot_lp | Δ(ans−cot) | question |")
    md.append("|---|---|---|---|---|---|---|---|")
    fmt = lambda v, p=3, sign=False: ("n/a" if v is None else (f"{v:+.{p}f}" if sign else f"{v:.{p}f}"))
    for q in qid_summary:
        md.append(f"| {q['stratum']} | {q['qid'][:10]} | {q['em_rate']*100:.1f} "
                  f"| {fmt(q['full_lp'])} | {fmt(q['ans_lp'])} | {fmt(q['cot_lp'])} "
                  f"| {fmt(q['ans_minus_cot'], sign=True)} | {q['question']} |")
    md.append("")

    # Read-out
    best = max([r for r in auroc_rows if not math.isnan(r["auroc"])],
               key=lambda r: r["auroc"])
    sc_orig_auroc = next(r["auroc"] for r in auroc_rows if r["metric"] == "sc_orig")
    md.append("## Take-away")
    md.append("")
    md.append(f"- Original `sc_orig` AUROC = **{sc_orig_auroc:.4f}**")
    md.append(f"- Best variant: `{best['metric']}` AUROC = **{best['auroc']:.4f}** "
              f"(Δ = {best['auroc'] - sc_orig_auroc:+.4f})")
    md.append("")
    md.append("Interpretation guide:")
    md.append("- AUROC ≈ 0.5 → metric is **random** for distinguishing correct vs wrong")
    md.append("- If `answer_*` AUROC > `full_*` AUROC → CoT-span dilutes the signal "
              "(supports the hypothesis)")
    md.append("- If both are similar and weak → SC is fundamentally limited on QA "
              "regardless of span (need new metric, not new span)")

    out_md = OUT_DIR / "exp0_summary.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    print(f"\nReport → {out_md}")
    print("Done.")


if __name__ == "__main__":
    main()
