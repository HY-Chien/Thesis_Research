#!/usr/bin/env python3
"""Per-quadrant confidence analysis for the direct-vs-decompose 2x2 (slide p45).

Buckets each question into both_correct / direct_only / decomp_only / both_wrong
using the saved direct_em / decomp_em from the current final typeaware-replay
traces, then compares the confidence signals of the two paths inside each
quadrant:

  - direct path:  direct.skip_probe_ent  (skip-decompose probe entropy)
  - decomp path:  max over steps of max_ent (plus mean_ent / min_lp variants)

Questions answered:
  1. Are both-wrong questions detectably more uncertain than both-correct ones?
  2. How often is the model *confidently wrong* (both-wrong entropy at or below
     the both-correct median)?
  3. In the disagreement quadrants, does picking the more-confident path find
     the correct one, and how does the current verifier compare?
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
THESIS_FIG = ROOT.parent / "Thesis" / "figures" / "fig_quadrant_confidence.png"
REPORT = RESULTS / "quadrant_confidence_analysis.md"

FILES = {
    "2Wiki": "2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "HotpotQA": "hotpotqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "MuSiQue": "musique_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "PopQA": "popqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
}

QUADRANTS = ["both_correct", "direct_only", "decomp_only", "both_wrong"]
DIRECT_COLOR = "#244a8f"
DECOMP_COLOR = "#0b8f6a"


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def quadrant(row: dict[str, Any]) -> str:
    d = int(row.get("direct_em") or 0)
    c = int(row.get("decomp_em") or 0)
    if d and c:
        return "both_correct"
    if d:
        return "direct_only"
    if c:
        return "decomp_only"
    return "both_wrong"


def direct_ent(row: dict[str, Any]) -> float | None:
    return safe_float((row.get("direct") or {}).get("skip_probe_ent"))


def decomp_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    return (row.get("decomp") or {}).get("steps") or []


def step_agg(row: dict[str, Any], key: str, agg: str) -> float | None:
    vals = [v for step in decomp_steps(row) if (v := safe_float(step.get(key))) is not None]
    if not vals:
        return None
    if agg == "max":
        return max(vals)
    if agg == "min":
        return min(vals)
    return mean(vals)


def neg_min_lp(row: dict[str, Any]) -> float | None:
    v = step_agg(row, "min_lp", "min")
    return None if v is None else -v


# Question-level uncertainty signals: higher value = more uncertain.
SIGNALS = {
    "direct_ent": direct_ent,
    "decomp_max_ent": lambda row: step_agg(row, "max_ent", "max"),
    "decomp_mean_ent": lambda row: step_agg(row, "mean_ent", "mean"),
    "decomp_neg_min_lp": neg_min_lp,
}


def signal_values(rows: list[dict[str, Any]], name: str) -> list[float]:
    fn = SIGNALS[name]
    return [v for row in rows if (v := fn(row)) is not None]


def router_choice(row: dict[str, Any]) -> str:
    return row.get("replay_choice") or row.get("choice") or "decomp"


def auroc(scores: list[float], labels: list[int]) -> float | None:
    pairs = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    n_pos = sum(y == 1 for _, y in pairs)
    n_neg = sum(y == 0 for _, y in pairs)
    if n_pos == 0 or n_neg == 0:
        return None

    pairs.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(y == 1 for _, y in pairs[i:j])
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def cohens_d(pos: list[float], neg: list[float]) -> float | None:
    if len(pos) < 2 or len(neg) < 2:
        return None
    var_pos = pstdev(pos) ** 2
    var_neg = pstdev(neg) ** 2
    pooled = math.sqrt(
        ((len(pos) - 1) * var_pos + (len(neg) - 1) * var_neg) / (len(pos) + len(neg) - 2)
    )
    if pooled == 0:
        return None
    return (mean(pos) - mean(neg)) / pooled


def fmt(val: float | None, spec: str = ".3f") -> str:
    return "NA" if val is None else format(val, spec)


def bucket_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {q: [] for q in QUADRANTS}
    for row in rows:
        buckets[quadrant(row)].append(row)
    return buckets


def render_quadrant_table(data: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    lines = [
        "| Dataset | Quadrant | n | direct_ent mean | direct_ent median | decomp_max_ent mean | decomp_max_ent median | verifier picks direct % |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, buckets in data.items():
        for quad in QUADRANTS:
            rows = buckets[quad]
            if not rows:
                continue
            d_ent = signal_values(rows, "direct_ent")
            c_ent = signal_values(rows, "decomp_max_ent")
            direct_rate = 100 * sum(router_choice(r) == "direct" for r in rows) / len(rows)
            lines.append(
                f"| {ds} | {quad} | {len(rows)} | {fmt(mean(d_ent) if d_ent else None)} | "
                f"{fmt(median(d_ent) if d_ent else None)} | {fmt(mean(c_ent) if c_ent else None)} | "
                f"{fmt(median(c_ent) if c_ent else None)} | {direct_rate:.1f} |"
            )
    return "\n".join(lines)


def render_separation_table(data: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    lines = [
        "| Dataset | Signal | mean both_correct | mean both_wrong | AUROC (wrong=pos) | Cohen's d |",
        "|---|---|---:|---:|---:|---:|",
    ]
    pooled: dict[str, tuple[list[float], list[float]]] = {s: ([], []) for s in SIGNALS}
    for ds, buckets in data.items():
        for sig in SIGNALS:
            correct = signal_values(buckets["both_correct"], sig)
            wrong = signal_values(buckets["both_wrong"], sig)
            pooled[sig][0].extend(correct)
            pooled[sig][1].extend(wrong)
            scores = correct + wrong
            labels = [0] * len(correct) + [1] * len(wrong)
            lines.append(
                f"| {ds} | {sig} | {fmt(mean(correct) if correct else None)} | "
                f"{fmt(mean(wrong) if wrong else None)} | {fmt(auroc(scores, labels))} | "
                f"{fmt(cohens_d(wrong, correct), '+.2f')} |"
            )
    for sig, (correct, wrong) in pooled.items():
        scores = correct + wrong
        labels = [0] * len(correct) + [1] * len(wrong)
        lines.append(
            f"| ALL | {sig} | {fmt(mean(correct) if correct else None)} | "
            f"{fmt(mean(wrong) if wrong else None)} | {fmt(auroc(scores, labels))} | "
            f"{fmt(cohens_d(wrong, correct), '+.2f')} |"
        )
    return "\n".join(lines)


def confidently_wrong_rate(buckets: dict[str, list[dict[str, Any]]], sig: str) -> tuple[float | None, int]:
    correct = signal_values(buckets["both_correct"], sig)
    if not correct:
        return None, 0
    threshold = median(correct)
    fn = SIGNALS[sig]
    wrong_vals = [v for row in buckets["both_wrong"] if (v := fn(row)) is not None]
    if not wrong_vals:
        return None, 0
    hits = sum(v <= threshold for v in wrong_vals)
    return 100 * hits / len(wrong_vals), len(wrong_vals)


def render_confidently_wrong_table(data: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    lines = [
        "| Dataset | both_wrong n | direct_ent <= both_correct median % | decomp_max_ent <= both_correct median % |",
        "|---|---:|---:|---:|",
    ]
    for ds, buckets in data.items():
        d_rate, n = confidently_wrong_rate(buckets, "direct_ent")
        c_rate, _ = confidently_wrong_rate(buckets, "decomp_max_ent")
        lines.append(f"| {ds} | {n} | {fmt(d_rate, '.1f')} | {fmt(c_rate, '.1f')} |")
    return "\n".join(lines)


def zscore_params(rows: list[dict[str, Any]], sig: str) -> tuple[float, float]:
    vals = signal_values(rows, sig)
    mu = mean(vals) if vals else 0.0
    sd = pstdev(vals) if len(vals) > 1 else 1.0
    return mu, sd if sd > 0 else 1.0


def render_disagreement_table(
    data: dict[str, dict[str, list[dict[str, Any]]]],
    all_rows: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        "| Dataset | Quadrant | n | verifier picks correct path % | raw lower-entropy path correct % | z-scored lower-entropy path correct % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ds, buckets in data.items():
        d_mu, d_sd = zscore_params(all_rows[ds], "direct_ent")
        c_mu, c_sd = zscore_params(all_rows[ds], "decomp_max_ent")
        for quad, right in (("direct_only", "direct"), ("decomp_only", "decomp")):
            rows = buckets[quad]
            if not rows:
                continue
            verifier_hits = sum(router_choice(r) == right for r in rows)
            raw_hits = raw_n = z_hits = z_n = 0
            for row in rows:
                d = direct_ent(row)
                c = SIGNALS["decomp_max_ent"](row)
                if d is None or c is None:
                    continue
                raw_n += 1
                raw_pick = "direct" if d < c else "decomp"
                raw_hits += raw_pick == right
                z_n += 1
                z_pick = "direct" if (d - d_mu) / d_sd < (c - c_mu) / c_sd else "decomp"
                z_hits += z_pick == right
            lines.append(
                f"| {ds} | {quad} | {len(rows)} | {100 * verifier_hits / len(rows):.1f} | "
                f"{fmt(100 * raw_hits / raw_n if raw_n else None, '.1f')} | "
                f"{fmt(100 * z_hits / z_n if z_n else None, '.1f')} |"
            )
    return "\n".join(lines)


def make_fig(data: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=170, sharey=False)
    quad_labels = ["Both\ncorrect", "Direct\nonly", "Decomp\nonly", "Both\nwrong"]
    for ax, (ds, buckets) in zip(axes.flat, data.items()):
        d_series = [signal_values(buckets[q], "direct_ent") for q in QUADRANTS]
        c_series = [signal_values(buckets[q], "decomp_max_ent") for q in QUADRANTS]
        pos_d = [i - 0.19 for i in range(len(QUADRANTS))]
        pos_c = [i + 0.19 for i in range(len(QUADRANTS))]
        box_kw = dict(widths=0.32, showfliers=False, patch_artist=True, medianprops=dict(color="#222222"))
        bp_d = ax.boxplot(d_series, positions=pos_d, **box_kw)
        bp_c = ax.boxplot(c_series, positions=pos_c, **box_kw)
        for patch in bp_d["boxes"]:
            patch.set_facecolor(DIRECT_COLOR)
            patch.set_alpha(0.75)
        for patch in bp_c["boxes"]:
            patch.set_facecolor(DECOMP_COLOR)
            patch.set_alpha(0.75)
        for i, q in enumerate(QUADRANTS):
            ax.text(i, ax.get_ylim()[0], f"n={len(buckets[q])}", ha="center", va="bottom", fontsize=7.5, color="#666666")
        ax.set_title(ds, fontsize=12)
        ax.set_xticks(range(len(QUADRANTS)))
        ax.set_xticklabels(quad_labels, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0][0].set_ylabel("Entropy (uncertainty)")
    axes[1][0].set_ylabel("Entropy (uncertainty)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DIRECT_COLOR, alpha=0.75, label="Direct path (skip-probe entropy)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=DECOMP_COLOR, alpha=0.75, label="Decompose path (max step entropy)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=2, frameon=False, fontsize=10)
    fig.suptitle("Confidence by Direct-vs-Decompose Correctness Quadrant", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    THESIS_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS_FIG, bbox_inches="tight")


def write_report(
    data: dict[str, dict[str, list[dict[str, Any]]]],
    all_rows: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Per-Quadrant Confidence Analysis (Direct-vs-Decompose 2x2, slide p45)",
        "",
        "Setting: current final candidate traces with path-aware retrieval, original-question",
        "evidence union, and type-aware rule-based verifier replay (same files as",
        "`direct_decompose_2x2_current_final.md`). Question-level uncertainty signals:",
        "",
        "```text",
        "direct_ent        = direct.skip_probe_ent          (skip-decompose probe entropy)",
        "decomp_max_ent    = max over steps of max_ent      (canonical gate signal)",
        "decomp_mean_ent   = mean over steps of mean_ent",
        "decomp_neg_min_lp = -min over steps of min_lp",
        "higher value = more uncertain, for every signal",
        "```",
        "",
        "## 1. Confidence by quadrant",
        "",
        render_quadrant_table(data),
        "",
        "## 2. Both-correct vs both-wrong separation",
        "",
        "AUROC of each uncertainty signal at ranking both_wrong above both_correct",
        "(disagreement quadrants excluded).",
        "",
        render_separation_table(data),
        "",
        "## 3. Confidently-wrong rate inside both_wrong",
        "",
        "Fraction of both_wrong questions whose uncertainty is at or below the *median*",
        "uncertainty of both_correct questions - i.e. wrong answers that look exactly like",
        "confident correct ones and cannot be flagged by any entropy threshold.",
        "",
        render_confidently_wrong_table(data),
        "",
        "## 4. Disagreement quadrants: signal-driven or guessing?",
        "",
        "Inside direct_only / decomp_only, compare (a) the current verifier's pick and",
        "(b) a naive rule that picks the path with lower entropy (raw, and z-scored per",
        "dataset because the two entropies live on different scales).",
        "",
        render_disagreement_table(data, all_rows),
        "",
        "## Interpretation",
        "",
        "(filled in after first run)",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_rows = {ds: load_rows(RESULTS / fn) for ds, fn in FILES.items()}
    data = {ds: bucket_rows(rows) for ds, rows in all_rows.items()}
    for ds, buckets in data.items():
        counts = {q: len(buckets[q]) for q in QUADRANTS}
        print(ds, counts, flush=True)
    write_report(data, all_rows)
    make_fig(data)
    print(f"Wrote {REPORT}")
    print(f"Wrote {THESIS_FIG}")


if __name__ == "__main__":
    main()
