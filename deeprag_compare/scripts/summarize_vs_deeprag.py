"""
Build a comparison table matching DeepRAG Table 2 columns, plus PopQA.

Reads the per-dataset summaries from `results/` and prints two tables:
  Table 1 — HotpotQA / 2Wiki / MuSiQue  (compare vs DeepRAG paper)
  Table 2 — PopQA                        (no paper baseline; ours only)
"""

import argparse
import json
import os

RES_DIR = "/home/S113065528/Research/deeprag_compare/results"
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
DATASETS_ALL = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]


def find_in_results(filename):
    """Look for filename at RES_DIR root, then in any first-level subdir.
    Returns the matched path, or RES_DIR/filename so callers' os.path.exists fails cleanly."""
    import glob
    direct = os.path.join(RES_DIR, filename)
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(RES_DIR, "*", filename))
    return matches[0] if matches else direct

# DeepRAG paper Table 2 numbers on Llama-3-8B-Instruct (zero-shot baselines)
DEEPRAG_REF = {
    "CoT (paper)":     {"hotpotqa": (27.2, 37.8), "2wikimultihopqa": (28.2, 34.9), "musique": (13.7, 23.0)},
    "FLARE (paper)":   {"hotpotqa": (23.4, 32.9), "2wikimultihopqa": (30.5, 37.5), "musique": (17.6, 23.6)},
    "DRAGIN (paper)":  {"hotpotqa": (27.6, 38.1), "2wikimultihopqa": (29.1, 35.7), "musique": (11.8, 20.0)},
    "UAR (paper)":     {"hotpotqa": (29.7, 40.7), "2wikimultihopqa": (34.8, 42.4), "musique": (19.1, 28.4)},
    "DeepRAG (paper)": {"hotpotqa": (40.7, 51.5), "2wikimultihopqa": (48.1, 53.3), "musique": (22.5, 30.4)},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", default="meta-llama-3-8b-instruct")
    args = parser.parse_args()

    ours = {}
    for ds in DATASETS_ALL:
        path = find_in_results(f"{ds}_{args.model_tag}_summary.json")
        if os.path.exists(path):
            ours[ds] = json.load(open(path))
        else:
            ours[ds] = None

    # ---- Table 1: HotpotQA / 2Wiki / MuSiQue vs DeepRAG paper ----
    rows = []

    def ours_row(method_key, label, ds_list):
        vals = []
        for ds in ds_list:
            s = ours[ds]
            if s is None or method_key not in s.get("methods", {}):
                vals.append(("-", "-"))
            else:
                m = s["methods"][method_key]
                vals.append((f"{m['em']*100:.1f}", f"{m['f1']*100:.1f}"))
        rows.append((label, vals))

    for mk, lb in [
        ("no_rag",          "Ours CoT (no-RAG)"),
        ("single_rag",      "Ours Single-RAG"),
        ("dynamic_rag",     "Ours Dynamic SC-RAG"),
        ("sc_gated_rate20", "Ours SC-gated@20"),
        ("sc_gated_rate50", "Ours SC-gated@50"),
        ("sc_gated_rate80", "Ours SC-gated@80"),
    ]:
        ours_row(mk, lb, DATASETS)

    rows.append(("", [("", "") for _ in DATASETS]))

    for label, v in DEEPRAG_REF.items():
        vals = [(f"{v[ds][0]:.1f}", f"{v[ds][1]:.1f}") for ds in DATASETS]
        rows.append((label, vals))

    print("=== Table 1: Multi-hop QA vs DeepRAG Table 2 ===")
    head = f"{'Method':<25s} | " + " | ".join(f"{ds[:12]:>12s}" for ds in DATASETS)
    print(head)
    print(f"{'':<25s} | " + " | ".join(f"{'  EM   F1':>12s}" for _ in DATASETS))
    print("-" * len(head))
    for label, vals in rows:
        cells = " | ".join(f"{em:>5s} {f1:>5s}" for em, f1 in vals)
        print(f"{label:<25s} | {cells}")

    # ---- Table 2: PopQA (no paper baseline) ----
    print("\n=== Table 2: PopQA (single-hop factoid, no DeepRAG baseline) ===")
    popqa_methods = [
        ("no_rag",          "CoT (no-RAG)"),
        ("single_rag",      "Single-RAG"),
        ("dynamic_rag",     "Dynamic SC-RAG"),
        ("sc_gated_rate20", "SC-gated@20"),
        ("sc_gated_rate50", "SC-gated@50"),
        ("sc_gated_rate80", "SC-gated@80"),
    ]
    s = ours.get("popqa")
    print(f"{'Method':<22s}   EM     F1")
    print("-" * 38)
    for mk, lb in popqa_methods:
        if s is None or mk not in s.get("methods", {}):
            print(f"{lb:<22s}    -      -")
        else:
            m = s["methods"][mk]
            print(f"{lb:<22s} {m['em']*100:5.1f}  {m['f1']*100:5.1f}")

    # ---- Save markdown ----
    md_path = os.path.join(RES_DIR, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("## Table 1: Multi-hop QA vs DeepRAG\n\n")
        f.write("| Method | " + " | ".join(f"{ds} EM | {ds} F1" for ds in DATASETS) + " |\n")
        f.write("|---|" + "|".join(["---"] * (len(DATASETS) * 2)) + "|\n")
        for label, vals in rows:
            if not label:
                continue
            cells = " | ".join(f"{em} | {f1}" for em, f1 in vals)
            f.write(f"| {label} | {cells} |\n")
        f.write("\n## Table 2: PopQA\n\n")
        f.write("| Method | EM | F1 |\n|---|---|---|\n")
        for mk, lb in popqa_methods:
            if s is None or mk not in s.get("methods", {}):
                f.write(f"| {lb} | - | - |\n")
            else:
                m = s["methods"][mk]
                f.write(f"| {lb} | {m['em']*100:.1f} | {m['f1']*100:.1f} |\n")
    print(f"\nMarkdown table -> {md_path}")


if __name__ == "__main__":
    main()
