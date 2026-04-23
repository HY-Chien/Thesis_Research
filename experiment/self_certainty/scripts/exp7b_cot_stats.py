"""
Exp7b: SC/PPL summary statistics on CoT results (exp7).

Same structure as exp5b but operates on exp7_cot_<dataset>_results.json.
Also slices by answer length (n_tokens) to see whether the SC advantage
grows with output length (predicted by our hypothesis).
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path("/home/S113065528/Adaptive-RAG/experiment/self_certainty")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="2wiki")
args = parser.parse_args()

RESULTS_IN = ROOT / "results" / f"exp7_cot_{args.dataset}_results.json"
OUT_MD = ROOT / "results" / f"exp7b_cot_stats_{args.dataset}.md"


def stats(xs):
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {"n": len(xs), "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "std": statistics.stdev(xs) if len(xs) > 1 else 0.0}


def auc(correct, incorrect, higher_is_confident):
    if not correct or not incorrect:
        return float("nan")
    wins = 0.0
    for c in correct:
        for i in incorrect:
            if higher_is_confident:
                wins += (c > i) + 0.5 * (c == i)
            else:
                wins += (c < i) + 0.5 * (c == i)
    return wins / (len(correct) * len(incorrect))


def fmt(s):
    return f"n={s['n']:3d}  mean={s['mean']:7.3f}  median={s['median']:7.3f}  std={s['std']:6.3f}"


def line(title, correct, incorrect, key, higher_is_confident):
    c = [r[key] for r in correct]; i = [r[key] for r in incorrect]
    a = auc(c, i, higher_is_confident)
    gap = stats(c)["mean"] - stats(i)["mean"] if c and i else float("nan")
    return (f"  {title:5s}  correct: {fmt(stats(c))}\n"
            f"         incorrect: {fmt(stats(i))}\n"
            f"         gap(c-i)={gap:+.3f}   AUC={a:.4f}")


def section(name, items):
    c = [r for r in items if r["is_correct"]]
    i = [r for r in items if not r["is_correct"]]
    acc = 100 * len(c) / len(items) if items else 0.0
    mean_tok = statistics.mean([r["n_tokens"] for r in items]) if items else 0
    out = [f"\n### {name}   (n={len(items)}, acc={acc:.1f}%, mean_n_tok={mean_tok:.0f})"]
    out.append(line("SC", c, i, "sc", True))
    out.append(line("PPL", c, i, "ppl", False))
    return "\n".join(out)


def main():
    with open(RESULTS_IN) as f:
        results = json.load(f)

    lines = [
        f"# Exp7b — CoT SC / PPL by Correctness  ({args.dataset.upper()})\n",
        "Hypothesis check: under CoT, does SC beat PPL, and does the gap grow with length?",
    ]
    lines.append("\n## Overall")
    lines.append(section("ALL", results))

    lines.append("\n## Sliced by output length (CoT token count)")
    bins = [
        ("short  <50",    lambda r: r["n_tokens"] < 50),
        ("med   50-150",  lambda r: 50 <= r["n_tokens"] <= 150),
        ("long  151-300", lambda r: 151 <= r["n_tokens"] <= 300),
        ("xlong 301+",    lambda r: r["n_tokens"] > 300),
    ]
    for name, pred in bins:
        sub = [r for r in results if pred(r)]
        lines.append(section(name, sub))

    text = "\n".join(lines) + "\n"
    with open(OUT_MD, "w") as f:
        f.write(text)
    print(text)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
