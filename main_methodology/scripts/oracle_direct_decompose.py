"""Oracle upper bound for choosing direct vs. decomposed SEEK answers.

The oracle is deliberately gold-label based: for each question, it chooses the
direct answer if direct is correct, otherwise the decomposed answer if that is
correct. This is not a deployable verifier; it measures how much headroom a
future verifier/router could possibly have when restricted to the two existing
answers.

Run from Research/main_methodology:
    python scripts/oracle_direct_decompose.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PAIRS = {
    "hotpotqa": ("results/hotpotqa_1000_wiki_single_k8.jsonl",
                 "results/hotpotqa_1000_wiki_k8.jsonl"),
    "2wiki": ("results/2wiki_1000_single_k8.jsonl",
              "results/2wiki_1000_wiki_k8.jsonl"),
    "musique": ("results/musique_1000_single_k8.jsonl",
                "results/musique_1000_wiki_k8.jsonl"),
}

ROUTERS = {
    "hotpotqa": [
        ("tau=0.5", "results/hotpotqa_1000_router05.jsonl"),
        ("tau=0.8", "results/hotpotqa_1000_router08.jsonl"),
        ("tau=1.2", "results/hotpotqa_1000_router12.jsonl"),
    ],
    "2wiki": [
        ("tau=0.5", "results/2wiki_1000_router05.jsonl"),
        ("tau=0.8", "results/2wiki_1000_router08.jsonl"),
        ("tau=1.2", "results/2wiki_1000_router12.jsonl"),
    ],
    "musique": [
        ("tau=0.5", "results/musique_1000_router05.jsonl"),
        ("tau=0.8", "results/musique_1000_router08.jsonl"),
        ("tau=1.2", "results/musique_1000_router12.jsonl"),
    ],
}


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


def mean_em(path: Path) -> float:
    rows = load_jsonl(path)
    return pct(sum(int(r.get("em", 0)) for r in rows.values()), len(rows))


def best_router(root: Path, dataset: str) -> tuple[str, float] | None:
    vals = []
    for label, rel in ROUTERS.get(dataset, []):
        path = root / rel
        if path.exists():
            vals.append((label, mean_em(path)))
    if not vals:
        return None
    return max(vals, key=lambda item: item[1])


def analyze_pair(root: Path, dataset: str, direct_rel: str, decomp_rel: str) -> dict:
    direct = load_jsonl(root / direct_rel)
    decomp = load_jsonl(root / decomp_rel)
    ids = sorted(set(direct) & set(decomp))
    missing_direct = sorted(set(decomp) - set(direct))
    missing_decomp = sorted(set(direct) - set(decomp))

    both_correct = direct_only = decomp_only = both_wrong = 0
    examples = {"direct_only": [], "decomp_only": [], "both_wrong": []}
    for qid in ids:
        d = int(direct[qid].get("em", 0))
        c = int(decomp[qid].get("em", 0))
        if d and c:
            both_correct += 1
        elif d:
            direct_only += 1
            bucket = examples["direct_only"]
            if len(bucket) < 3:
                bucket.append((qid, direct[qid].get("question", ""),
                               direct[qid].get("pred", ""), decomp[qid].get("pred", ""),
                               direct[qid].get("gold", "")))
        elif c:
            decomp_only += 1
            bucket = examples["decomp_only"]
            if len(bucket) < 3:
                bucket.append((qid, direct[qid].get("question", ""),
                               direct[qid].get("pred", ""), decomp[qid].get("pred", ""),
                               direct[qid].get("gold", "")))
        else:
            both_wrong += 1
            bucket = examples["both_wrong"]
            if len(bucket) < 3:
                bucket.append((qid, direct[qid].get("question", ""),
                               direct[qid].get("pred", ""), decomp[qid].get("pred", ""),
                               direct[qid].get("gold", "")))

    n = len(ids)
    direct_em = pct(both_correct + direct_only, n)
    decomp_em = pct(both_correct + decomp_only, n)
    oracle_em = pct(both_correct + direct_only + decomp_only, n)
    best_pure = max(direct_em, decomp_em)
    router = best_router(root, dataset)

    return {
        "dataset": dataset,
        "n": n,
        "missing_direct": len(missing_direct),
        "missing_decomp": len(missing_decomp),
        "both_correct": both_correct,
        "direct_only": direct_only,
        "decomp_only": decomp_only,
        "both_wrong": both_wrong,
        "direct_em": direct_em,
        "decomp_em": decomp_em,
        "oracle_em": oracle_em,
        "best_pure": best_pure,
        "oracle_gain": oracle_em - best_pure,
        "best_router": router,
        "examples": examples,
        "direct_file": direct_rel,
        "decomp_file": decomp_rel,
    }


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# Oracle Direct vs Decompose Upper Bound",
        "",
        "Setting: honest wiki-21M, top_k=8, 12K decomposer, Mode C. "
        "The oracle chooses the correct answer between the forced direct path "
        "and the forced decomposition path using gold EM labels.",
        "",
        "> This is an upper bound for a future verifier/router, not a deployable "
        "method. PopQA is excluded here because the existing forced-direct file "
        "is from a different setting (non-wiki corpus, top_k=5).",
        "",
        "| Dataset | Direct EM | Decomp EM | Best router EM | Oracle EM | Gain vs best pure | Gap vs best router | directOnly | decompOnly | bothWrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        router_label = "-"
        router_em = None
        if r["best_router"]:
            router_label, router_em = r["best_router"]
            router_cell = f"{router_em:.1f} ({router_label})"
            router_gap = r["oracle_em"] - router_em
            router_gap_cell = f"{router_gap:+.1f}"
        else:
            router_cell = "-"
            router_gap_cell = "-"
        lines.append(
            f"| {r['dataset']} | {r['direct_em']:.1f} | {r['decomp_em']:.1f} | "
            f"{router_cell} | **{r['oracle_em']:.1f}** | "
            f"{r['oracle_gain']:+.1f} | {router_gap_cell} | "
            f"{r['direct_only']} | {r['decomp_only']} | {r['both_wrong']} |"
        )

    lines.extend([
        "",
        "## 2x2 Counts",
        "",
    ])
    for r in results:
        lines.extend([
            f"### {r['dataset']} (N={r['n']})",
            "",
            "| | Decompose correct | Decompose wrong |",
            "|---|---:|---:|",
            f"| Direct correct | {r['both_correct']} | {r['direct_only']} |",
            f"| Direct wrong | {r['decomp_only']} | {r['both_wrong']} |",
            "",
        ])
        if r["missing_direct"] or r["missing_decomp"]:
            lines.append(
                f"ID mismatch: missing_direct={r['missing_direct']}, "
                f"missing_decomp={r['missing_decomp']}"
            )
            lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- `directOnly` is the maximum number of cases a verifier can rescue from decomposition failures by choosing direct.",
        "- `decompOnly` is the maximum number of cases it can rescue from direct failures by choosing decomposition.",
        "- `bothWrong` is unreachable for this two-path verifier; improving those cases needs better retrieval, decomposition, synthesis, or answer generation.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path,
                    default=Path("results/wiki21M_oracle_direct_decompose.md"))
    args = ap.parse_args()

    results = [
        analyze_pair(args.root, dataset, direct_rel, decomp_rel)
        for dataset, (direct_rel, decomp_rel) in DEFAULT_PAIRS.items()
    ]
    md = render_markdown(results)
    out = args.root / args.out
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
