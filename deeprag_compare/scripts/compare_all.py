"""
Aggregate results across ALL methods (baselines + dynamic variants + DeepRAG-SC + paper)
into a single comparison table.

Reads from /home/S113065528/Research/deeprag_compare/results/:
  - baseline:        <ds>_<model_tag>_summary.json
  - dyn_continue:    <ds>_<model_tag>_dyn_continue_summary.json
  - dyn_restart:     <ds>_<model_tag>_dyn_restart_summary.json
  - dyn_planA:       <ds>_<model_tag>_dyn_planA_summary.json
  - DeepRAG-SC:      <ds>_deeprag-8b_sc_th<thr>_summary.json   (one row per threshold)
  - DeepRAG-learned: <ds>_deeprag-8b_learned_summary.json

Writes:
  results/comparison_all.md
  prints both tables to stdout
"""
import argparse
import glob
import json
import os
import re

RES_DIR = "/home/S113065528/Research/deeprag_compare/results"
DS_MULTI = ["hotpotqa", "2wikimultihopqa", "musique"]
DS_ALL = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]


def find_in_results(filename):
    """Look for filename at RES_DIR root, then in any first-level subdir.
    Returns the matched path, or RES_DIR/filename so callers' os.path.exists fails cleanly."""
    direct = os.path.join(RES_DIR, filename)
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(RES_DIR, "*", filename))
    return matches[0] if matches else direct

DEEPRAG_REF = {
    "CoT (paper)":     {"hotpotqa": (27.2, 37.8), "2wikimultihopqa": (28.2, 34.9), "musique": (13.7, 23.0)},
    "FLARE (paper)":   {"hotpotqa": (23.4, 32.9), "2wikimultihopqa": (30.5, 37.5), "musique": (17.6, 23.6)},
    "DRAGIN (paper)":  {"hotpotqa": (27.6, 38.1), "2wikimultihopqa": (29.1, 35.7), "musique": (11.8, 20.0)},
    "UAR (paper)":     {"hotpotqa": (29.7, 40.7), "2wikimultihopqa": (34.8, 42.4), "musique": (19.1, 28.4)},
    "DeepRAG (paper)": {"hotpotqa": (40.7, 51.5), "2wikimultihopqa": (48.1, 53.3), "musique": (22.5, 30.4)},
}


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def cell_em_f1(method_summary, key):
    """Return (em_str, f1_str) or ('-', '-') if missing."""
    if method_summary is None:
        return ("-", "-")
    m = method_summary.get("methods", {}).get(key)
    if not m:
        return ("-", "-")
    return (f"{m['em']*100:.1f}", f"{m['f1']*100:.1f}")


def cell_deeprag_sc(deeprag_sc, jsonl_path=None):
    """DeepRAG-SC summary stores em + avg_retrievals but no f1.
    Fall back to aggregating f1 from the per-record JSONL when missing."""
    if deeprag_sc is None:
        return ("-", "-")
    em = deeprag_sc.get("em", 0) * 100
    f1 = deeprag_sc.get("f1")
    if f1 is None and jsonl_path and os.path.exists(jsonl_path):
        f1s = []
        with open(jsonl_path) as fh:
            for line in fh:
                r = json.loads(line)
                if "f1" in r:
                    f1s.append(r["f1"])
        if f1s:
            f1 = sum(f1s) / len(f1s)
    f1_str = f"{f1*100:.1f}" if f1 else "-"
    return (f"{em:.1f}", f1_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", default="meta-llama-3-8b-instruct")
    ap.add_argument("--out", default=os.path.join(RES_DIR, "comparison_all.md"))
    args = ap.parse_args()

    # Load every variant per dataset
    summaries = {}
    drag_thresholds = set()  # union of threshold values seen across datasets
    for ds in DS_ALL:
        # Discover DeepRAG-SC threshold files: <ds>_deeprag-8b_sc_th<thr>_summary.json
        drag_sc = {}
        for path in glob.glob(f"{RES_DIR}/**/{ds}_deeprag-8b_sc_th*_summary.json", recursive=True):
            m = re.search(r"_sc_th([0-9.]+)_summary\.json$", path)
            if m:
                thr = m.group(1)
                drag_sc[thr] = load_json(path)
                drag_thresholds.add(thr)
        summaries[ds] = {
            "base":   load_json(find_in_results(f"{ds}_{args.model_tag}_summary.json")),
            "cont":   load_json(find_in_results(f"{ds}_{args.model_tag}_dyn_continue_summary.json")),
            "rest":   load_json(find_in_results(f"{ds}_{args.model_tag}_dyn_restart_summary.json")),
            "planA":  load_json(find_in_results(f"{ds}_{args.model_tag}_dyn_planA_summary.json")),
            "drag_learned": load_json(find_in_results(f"{ds}_deeprag-8b_learned_summary.json")),
            "drag_sc": drag_sc,  # dict[threshold_str] -> summary
        }
    # Sort thresholds numerically for stable row order
    drag_thresholds_sorted = sorted(drag_thresholds, key=float)

    # Method spec for our methods (label, source_key, method_key)
    our_methods = [
        ("Ours CoT (no-RAG)",                     "base",  "no_rag"),
        ("Ours Single-RAG",                       "base",  "single_rag"),
        ("Ours SC-gated@20",                      "base",  "sc_gated_rate20"),
        ("Ours SC-gated@50",                      "base",  "sc_gated_rate50"),
        ("Ours SC-gated@80",                      "base",  "sc_gated_rate80"),
        ("Ours Dynamic-RAG (continue, basic)",    "cont",  "dynamic_rag"),
        ("Ours Dynamic-RAG (restart, basic)",     "rest",  "dynamic_rag"),
        ("Ours Dynamic-RAG (regen+masked, planA)","planA", "dynamic_rag"),
    ]

    rows = []
    for label, src_key, method_key in our_methods:
        cells = []
        for ds in DS_MULTI:
            src = summaries[ds][src_key]
            cells.append(cell_em_f1(src, method_key))
        rows.append((label, cells))

    # DeepRAG-8b rows: one per threshold + learned gate
    if any(summaries[ds]["drag_learned"] for ds in DS_MULTI):
        cells = []
        for ds in DS_MULTI:
            jp = find_in_results(f"{ds}_deeprag-8b_learned.jsonl")
            cells.append(cell_deeprag_sc(summaries[ds]["drag_learned"], jp))
        rows.append(("DeepRAG-8b + learned gate", cells))
    for thr in drag_thresholds_sorted:
        cells = []
        for ds in DS_MULTI:
            jp = find_in_results(f"{ds}_deeprag-8b_sc_th{thr}.jsonl")
            cells.append(cell_deeprag_sc(summaries[ds]["drag_sc"].get(thr), jp))
        rows.append((f"DeepRAG-8b + SC gate (th={thr})", cells))

    # PopQA row collection (separate)
    popqa_rows = []
    for label, src_key, method_key in our_methods:
        src = summaries["popqa"][src_key]
        popqa_rows.append((label, cell_em_f1(src, method_key)))

    # ---- Print Table 1 ----
    print("=== Table 1: Multi-hop QA (HotpotQA / 2Wiki / MuSiQue) ===\n")
    head = f"{'Method':<42s} | " + " | ".join(f"{ds[:14]:>14s}" for ds in DS_MULTI)
    print(head)
    print(f"{'':<42s} | " + " | ".join(f"{' EM    F1':>14s}" for _ in DS_MULTI))
    print("-" * len(head))
    for label, cells in rows:
        c = " | ".join(f"{em:>6s} {f1:>6s}" for em, f1 in cells)
        print(f"{label:<42s} | {c}")
    print()
    for label, ref in DEEPRAG_REF.items():
        cells = [(f"{ref[ds][0]:.1f}", f"{ref[ds][1]:.1f}") for ds in DS_MULTI]
        c = " | ".join(f"{em:>6s} {f1:>6s}" for em, f1 in cells)
        print(f"{label:<42s} | {c}")

    # ---- Print Table 2 (PopQA) ----
    print(f"\n=== Table 2: PopQA (single-hop, no paper baseline) ===\n")
    print(f"{'Method':<42s}    EM      F1")
    print("-" * 60)
    for label, (em, f1) in popqa_rows:
        print(f"{label:<42s}  {em:>5s}   {f1:>5s}")

    # ---- Save markdown ----
    with open(args.out, "w") as f:
        f.write("# Full Comparison — All Methods\n\n")
        f.write("Model: Llama-3-8B-Instruct (base) + DeepRAG-8b (decomp).  N=1000 dev.\n\n")

        f.write("## Table 1: Multi-hop QA vs DeepRAG paper\n\n")
        f.write("| Method | " + " | ".join(f"{ds} EM | {ds} F1" for ds in DS_MULTI) + " |\n")
        f.write("|---|" + "|".join(["---"] * (len(DS_MULTI) * 2)) + "|\n")
        for label, cells in rows:
            cs = " | ".join(f"{em} | {f1}" for em, f1 in cells)
            f.write(f"| {label} | {cs} |\n")
        f.write("|  |  |  |  |  |  |  |\n")
        for label, ref in DEEPRAG_REF.items():
            cells = [(f"{ref[ds][0]:.1f}", f"{ref[ds][1]:.1f}") for ds in DS_MULTI]
            cs = " | ".join(f"{em} | {f1}" for em, f1 in cells)
            f.write(f"| {label} | {cs} |\n")

        f.write("\n## Table 2: PopQA (single-hop factoid)\n\n")
        f.write("| Method | EM | F1 |\n|---|---|---|\n")
        for label, (em, f1) in popqa_rows:
            f.write(f"| {label} | {em} | {f1} |\n")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
