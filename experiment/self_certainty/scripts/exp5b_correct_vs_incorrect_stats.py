"""
Exp5b: summary statistics of SC / PPL broken down by correctness.

Also slices by answer length (n_tokens) bins to test the hypothesis
"SC wins on long answers, PPL wins on short factoid".
"""

import json
import statistics
from pathlib import Path

ROOT = Path("/home/S113065528/Adaptive-RAG/experiment/self_certainty")
RESULTS_IN = ROOT / "results" / "exp4_nq_results.json"
OUT_MD = ROOT / "results" / "exp5b_correct_vs_incorrect_stats.md"


def stats(xs):
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    }


def auc(correct_scores, incorrect_scores, higher_is_confident=True):
    """Probability that a random correct sample has a higher (or lower) score than
    a random incorrect one. Mann-Whitney U / N1*N2."""
    if not correct_scores or not incorrect_scores:
        return float("nan")
    n_c, n_i = len(correct_scores), len(incorrect_scores)
    wins = 0.0
    for c in correct_scores:
        for i in incorrect_scores:
            if higher_is_confident:
                wins += (c > i) + 0.5 * (c == i)
            else:
                wins += (c < i) + 0.5 * (c == i)
    return wins / (n_c * n_i)


def fmt(s):
    return f"n={s['n']:3d}  mean={s['mean']:7.3f}  median={s['median']:7.3f}  std={s['std']:6.3f}"


def line(title, correct, incorrect, key, higher_is_confident):
    c_vals = [r[key] for r in correct]
    i_vals = [r[key] for r in incorrect]
    a = auc(c_vals, i_vals, higher_is_confident=higher_is_confident)
    return (
        f"  {title:5s}  correct: {fmt(stats(c_vals))}\n"
        f"         incorrect: {fmt(stats(i_vals))}\n"
        f"         gap(c-i)={stats(c_vals)['mean'] - stats(i_vals)['mean']:+.3f}   AUC={a:.4f}"
    )


def section(name, items):
    correct = [r for r in items if r["is_correct"]]
    incorrect = [r for r in items if not r["is_correct"]]
    acc = 100 * len(correct) / len(items) if items else 0.0
    out = [f"\n### {name}   (n={len(items)}, acc={acc:.1f}%)"]
    out.append(line("SC", correct, incorrect, "sc", higher_is_confident=True))
    out.append(line("PPL", correct, incorrect, "ppl", higher_is_confident=False))
    return "\n".join(out)


def main():
    with open(RESULTS_IN) as f:
        results = json.load(f)

    lines = [
        "# Exp5b — SC / PPL by Correctness\n",
        "Higher SC = more confident. Lower PPL = more confident.",
        "AUC = P(confident score ranks correct answer above incorrect one). 0.5 = chance, 1.0 = perfect.",
    ]

    lines.append("\n## Overall")
    lines.append(section("ALL", results))

    lines.append("\n## Sliced by answer length (n_tokens)")
    bins = [("short 1-4", lambda r: r["n_tokens"] <= 4),
            ("med   5-15", lambda r: 5 <= r["n_tokens"] <= 15),
            ("long  16+",  lambda r: r["n_tokens"] >= 16)]
    for name, pred in bins:
        sub = [r for r in results if pred(r)]
        lines.append(section(name, sub))

    text = "\n".join(lines) + "\n"
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(text)
    print(text)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
