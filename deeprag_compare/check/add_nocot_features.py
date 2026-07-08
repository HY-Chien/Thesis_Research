"""Augment exp1_output/{ds}_per_q.jsonl with no-CoT first-ent confidence features.

For each of the 1000 questions per dataset:
  - regenerate a no-CoT direct answer with greedy decode (T=0, deterministic)
  - score EM/F1 → nocot_em / nocot_f1
  - find first-entity span on the direct answer
  - compute span aggregations (min_lp, min_sc, mean_lp, mean_sc, neg_max_ent, ...)

Existing rag_em is preserved (merged from per_q.jsonl by qid). The output is
ready for production_sweep.py to apply gates using no-CoT signal (the strongest
metric per exp0b: no-CoT first-ent min_lp AUROC ≈ 0.866).

Usage:
    conda run -n prag python check/add_nocot_features.py            # all 4 datasets, N=1000
    conda run -n prag python check/add_nocot_features.py --n 50     # smoke test
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch  # noqa: F401  (imported via calibrate_thresholds)
from tqdm import tqdm

ROOT = Path("/home/S113065528/Research/deeprag_compare")
sys.path.insert(0, str(ROOT / "check"))
sys.path.insert(0, str(ROOT / "src"))

from calibrate_thresholds import (  # noqa: E402
    DATASET_CLASSES, WikiMultiHopQA, ALIAS_FILE,
    build_fewshot_prompt, score, build_no_cot_prefix, no_cot_test_input,
    greedy_generate_with_scoring, clean_pred, find_first_entity_span,
    span_aggs, load_model, N_SHOTS, MAX_NEW_DIRECT,
)

INPUT_DIR = ROOT / "check/exp1_output"
DATA_DIR = ROOT / "data/top1000"
OUT_DIR = ROOT / "check/production_sweep_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]


def load_per_q(ds):
    """Existing per-q records keyed by qid (has rag_em, nr_em, first_ent_min_lp from CoT)."""
    fp = INPUT_DIR / f"{ds}_per_q.jsonl"
    out = {}
    for ln in fp.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        out[r["qid"]] = r
    return out


def load_top1000(ds):
    """Top-1000 raw rows (qid, question, answer, ...)."""
    fp = DATA_DIR / f"{ds}.jsonl"
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]


def process_dataset(ds, model, tok, n_max):
    ds_cls = DATASET_CLASSES[ds]
    if ds == "2wikimultihopqa":
        WikiMultiHopQA.load_aliases(str(ALIAS_FILE))

    per_q_existing = load_per_q(ds)
    raw_rows = load_top1000(ds)
    # only process rows that exist in per_q.jsonl (so we have nr_em / rag_em for them)
    rows = [r for r in raw_rows if r["qid"] in per_q_existing][:n_max]
    print(f"\n=== {ds}: {len(rows)} questions to score (limit={n_max}) ===")

    direct_prefix = build_no_cot_prefix(ds_cls, N_SHOTS)
    out_path = OUT_DIR / f"{ds}_per_q_v2.jsonl"
    n_em = 0
    n_with_span = 0

    with out_path.open("w") as fout:
        for q in tqdm(rows, desc=f"  {ds}"):
            qid = q["qid"]
            existing = per_q_existing[qid]
            prompt = direct_prefix + no_cot_test_input(q["question"])
            tokens, text = greedy_generate_with_scoring(model, tok, prompt, MAX_NEW_DIRECT)
            pred = clean_pred(text)
            golds = ds_cls.golds(q)
            sc_out = score(pred, golds)
            em = sc_out["em"]; f1 = sc_out["f1"]
            n_em += em
            ent_span = find_first_entity_span(pred, tokens) if pred else None
            agg_ent = span_aggs(tokens, ent_span)
            if agg_ent: n_with_span += 1
            agg_full = span_aggs(tokens, (0, len(tokens))) if tokens else None

            rec = {
                "qid": qid,
                "question": q["question"][:120],
                "golds": golds,
                # existing CoT path (preserved for comparison)
                "nr_em": existing["nr_em"],
                "rag_em": existing["rag_em"],
                "cot_first_ent_min_lp": existing.get("first_ent_min_lp"),
                "cot_first_ent_mean_lp": existing.get("first_ent_mean_lp"),
                # new no-CoT features
                "nocot_pred": pred,
                "nocot_em": em,
                "nocot_f1": f1,
                "nocot_n_tok": len(tokens),
                "nocot_first_ent": agg_ent,   # dict or None
                "nocot_full":      agg_full,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"  → {out_path}  nocot_em={n_em}/{len(rows)} ({n_em/len(rows):.1%})  "
          f"with_first_ent_span={n_with_span}/{len(rows)}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    print("loading model...")
    tok, model = load_model()

    for ds in args.datasets:
        process_dataset(ds, model, tok, args.n)

    print("\nALL DONE")


if __name__ == "__main__":
    main()
