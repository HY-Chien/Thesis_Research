"""Step 2 — Perturbation experiment for confident-hallucination subtype probing.

Goal: distinguish A1 (parametric override — model ignores input) vs A2
(in-context filling — model uses input but builds a plausible wrong answer)
vs HI_correct, by measuring how confidence/predictions shift under input
perturbations.

Design:
  10 questions = 5 HI_correct + 5 HI_wrong (from analysis_table.md).
  4 conditions per question:
    1. original
    2. mask     — replace key entity with [X]
    3. replace  — swap key entity with same-type unrelated entity (changes the
                  ground-truth answer, but here we measure model's *confidence shift*)
    4. distractor — prepend an irrelevant fact to the question
  N_SAMPLES per condition under T=0.7 (matches multi_sample run).

Hypothesis:
  - HI_correct: confidence drops sharply when key entity masked/replaced
                (model relied on input)
  - A1 hallucination (e.g. 5a75383a55, n_uniq=1): confidence stays high
                regardless — model committed internally before reading input
  - A2 hallucination (e.g. 5a7c2f1455, n_uniq=14): confidence already low,
                may shift unpredictably

Outputs:
  step2_output/perturbation_dump.jsonl    one record per (qid, condition, sample_idx)
  step2_output/sensitivity_summary.csv    per (qid, condition) aggregates
  step2_output/sensitivity_table.md       per-qid sensitivity (delta vs original)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path("/home/S113065528/Research/deeprag_compare")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_multi_sample import (  # noqa: E402  reuse infra
    load_model, sample_once, MAX_NEW_TOKENS,
)
from dataset_eval import HotpotQA, build_fewshot_prompt, extract_answer, score  # noqa: E402

OUT_DIR = Path(__file__).parent / "step2_output"
OUT_DIR.mkdir(exist_ok=True)
DUMP_FILE = OUT_DIR / "perturbation_dump.jsonl"
SUMMARY_CSV = OUT_DIR / "sensitivity_summary.csv"
TABLE_MD = OUT_DIR / "sensitivity_table.md"

DISTRACTOR_PREFIX = (
    "Note: The capital of Burkina Faso is Ouagadougou, "
    "and the Pacific Ocean is the largest ocean on Earth. "
)

# Perturbation config: each entry chooses ONE key entity per question.
# - mask_target:  exact substring (case-sensitive) to mask out / replace
# - replacement:  same-type plausible swap that should change the ground-truth answer
# - rationale:    why this entity is the "key" anchor
PERTURB = [
    # ---------- HI_correct ----------
    {
        "qid": "5ae74fc45542997b22f6a6b4", "stratum": "HI_correct",
        "question": "What is the full name of the viral respiratory disease in which Yi Guan's research helped to prevent the outbreak of?",
        "gold": "severe acute respiratory syndrome",
        "mask_target": "Yi Guan",
        "replacement": "Marcus Reynolds",
        "rationale": "Yi Guan is the proper-noun anchor; without it the question has no referent.",
    },
    {
        "qid": "5ab51dae5542991779162d82", "stratum": "HI_correct",
        "question": "Are both Elko Regional Airport and Gerald R. Ford International Airport located in Michigan?",
        "gold": "no",
        "mask_target": "Gerald R. Ford International Airport",
        "replacement": "Detroit Metropolitan Airport",
        "rationale": "Replacing with another Michigan airport would flip the answer to 'yes' — measures whether model uses input.",
    },
    {
        "qid": "5ae1316555429901ffe4adda", "stratum": "HI_correct",
        "question": "Were Goo Goo Dolls and Echosmith formed in the same city?",
        "gold": "no",
        "mask_target": "Echosmith",
        "replacement": "Maroon 5",
        "rationale": "Echosmith is the more obscure half; masking forces model to admit ignorance if it was using input.",
    },
    {
        "qid": "5a738225554299623ed4abc3", "stratum": "HI_correct",
        "question": "Which musical fantasy film is older, Bedknobs and Broomsticks or The Muppet Christmas Carol?",
        "gold": "Bedknobs and Broomsticks",
        "mask_target": "Bedknobs and Broomsticks",
        "replacement": "Mary Poppins",
        "rationale": "Replacing with Mary Poppins (1964) keeps the comparison sensible but flips which film is older.",
    },
    {
        "qid": "5ab67bd355429954757d32f2", "stratum": "HI_correct",
        "question": "Curt Menefee is the host of the Fox network's NFL show with co-host Michael Anthony Strahan who spent his entire 15-year career with what team?",
        "gold": "New York Giants",
        "mask_target": "Michael Anthony Strahan",
        "replacement": "Tom Brady",
        "rationale": "Strahan is the answer-bearing entity (15-year career → Giants). Tom Brady is a confound (Patriots).",
    },
    # ---------- HI_wrong ----------
    {
        "qid": "5a75383a5542996c70cfaeb3", "stratum": "HI_wrong",
        "question": "Who was born first, Ulf Merbold or Mirosław Hermaszewski?",
        "gold": "Ulf Dietrich Merbold",
        "mask_target": "Mirosław Hermaszewski",
        "replacement": "Albert Einstein",
        "rationale": "A1 candidate (n_uniq=1, locked-in answer). Replacing one comparand with Einstein (1879) should drastically shift any model that uses the input.",
    },
    {
        "qid": "5a7c2f145542996dd594b8b4", "stratum": "HI_wrong",
        "question": "The actress who made her film debut in Friday Night Lights also starred in a 2014 French-American crime-thriller directed by whi?",
        "gold": "McG",
        "mask_target": "Friday Night Lights",
        "replacement": "Saturday Night Live",
        "rationale": "A2 candidate (n_uniq=14). Friday Night Lights anchors the actress; replacing should change which actress comes to mind.",
    },
    {
        "qid": "5ade69e455429975fa854ec5", "stratum": "HI_wrong",
        "question": "What director worked with Vikram Bhatt on a film starring actors Rajneesh Duggal and Adah Sharma?",
        "gold": "Tinu Suresh Desai",
        "mask_target": "Vikram Bhatt",
        "replacement": "Karan Johar",
        "rationale": "Vikram Bhatt is the producer anchor; swap to a different Bollywood producer should change collaborator.",
    },
    {
        "qid": "5a88d745554299206df2b378", "stratum": "HI_wrong",
        "question": "What occupation was shared by David Yates and Pietro Germi?",
        "gold": "director",
        "mask_target": "Pietro Germi",
        "replacement": "Stephen King",
        "rationale": "Pietro Germi is the obscure half. Stephen King is a writer (not director) — shared occupation flips.",
    },
    {
        "qid": "5abd01205542992ac4f3819b", "stratum": "HI_wrong",
        "question": "Mummulgum is a stop on the Bruxner Highway before reaching the town on the banks of which river?",
        "gold": "Richmond",
        "mask_target": "Mummulgum",
        "replacement": "Brisbane",
        "rationale": "Mummulgum is the obscure anchor; Brisbane is on a different river, so output should change.",
    },
]


def perturb_question(q: str, cfg: dict, mode: str) -> str:
    if mode == "original":
        return q
    if mode == "mask":
        if cfg["mask_target"] not in q:
            raise ValueError(f"mask_target '{cfg['mask_target']}' not in question")
        return q.replace(cfg["mask_target"], "[X]")
    if mode == "replace":
        if cfg["mask_target"] not in q:
            raise ValueError(f"mask_target '{cfg['mask_target']}' not in question")
        return q.replace(cfg["mask_target"], cfg["replacement"])
    if mode == "distractor":
        return DISTRACTOR_PREFIX + q
    raise ValueError(f"unknown mode {mode}")


def show_dry_run():
    print("=" * 80)
    print("PERTURBATION CONFIG — DRY RUN")
    print("=" * 80)
    for cfg in PERTURB:
        print(f"\n[{cfg['stratum']}] {cfg['qid'][:10]}  gold={cfg['gold']!r}")
        print(f"  rationale: {cfg['rationale']}")
        for mode in ("original", "mask", "replace", "distractor"):
            q = perturb_question(cfg["question"], cfg, mode)
            print(f"  {mode:11s}: {q}")
    print("\n" + "=" * 80)
    print(f"Total: {len(PERTURB)} questions x 4 conditions x N_SAMPLES generations")


def per_sample_features(out: dict, pred: str):
    tokens = out["tokens"]
    if not tokens:
        return None
    lps = [t["lp"] for t in tokens]
    ents = [t["entropy"] for t in tokens]
    margins = [t["top1_lp"] - t["top2_lp"] for t in tokens]
    feats = {
        "sc": out["sc"],
        "ppl": out["ppl"],
        "mean_lp_full": mean(lps),
        "mean_entropy_full": mean(ents),
        "min_margin_full": min(margins),
    }
    text = out["text"]
    pred_norm = (pred or "").strip().lower()
    if pred_norm and len(pred_norm) >= 2:
        text_lo = text.lower()
        idx = text_lo.find(pred_norm)
        if idx != -1:
            offsets, cur = [], 0
            for t in tokens:
                offsets.append(cur); cur += len(t["tok"])
            start = next((i for i in range(len(offsets))
                          if offsets[i] <= idx < (offsets[i+1] if i+1 < len(offsets) else cur)), None)
            if start is not None:
                end_char = idx + len(pred_norm)
                end = start + 1
                while end < len(offsets) and offsets[end] < end_char:
                    end += 1
                if end > start:
                    sub_lp = [tokens[i]["lp"] for i in range(start, end)]
                    sub_ent = [tokens[i]["entropy"] for i in range(start, end)]
                    sub_mg = [tokens[i]["top1_lp"] - tokens[i]["top2_lp"] for i in range(start, end)]
                    feats.update({
                        "fe_min_lp": min(sub_lp),
                        "fe_mean_entropy": mean(sub_ent),
                        "fe_min_margin": min(sub_mg),
                    })
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print perturbed questions for review, no inference")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed_base", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first K questions (smoke test)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    show_dry_run()
    if args.dry_run:
        return

    print("\nLoading model ...")
    tok, model = load_model()
    print("Model loaded.")
    fewshot = build_fewshot_prompt(HotpotQA, n_shots=8)

    done = set()
    if args.resume and DUMP_FILE.exists():
        for line in DUMP_FILE.open():
            r = json.loads(line)
            done.add((r["qid"], r["mode"], r["sample_idx"]))
        print(f"Resume: {len(done)} records already done")

    mode_flag = "a" if (args.resume and DUMP_FILE.exists()) else "w"
    fout = DUMP_FILE.open(mode_flag)

    cfgs = PERTURB[: args.limit] if args.limit else PERTURB
    total_gens = len(cfgs) * 4 * args.n_samples
    completed = 0
    t0 = time.time()

    for cfg in cfgs:
        for mode in ("original", "mask", "replace", "distractor"):
            try:
                q_perturbed = perturb_question(cfg["question"], cfg, mode)
            except ValueError as e:
                print(f"  SKIP {cfg['qid'][:10]} {mode}: {e}")
                continue
            prompt = fewshot + HotpotQA.demo_input(q_perturbed) + " "
            for s_idx in range(args.n_samples):
                if (cfg["qid"], mode, s_idx) in done:
                    completed += 1
                    continue
                seed = args.seed_base + s_idx
                out = sample_once(model, tok, prompt,
                                  args.max_new_tokens, args.temperature, seed)
                pred = extract_answer(out["text"])
                sc_dict = score(pred, [cfg["gold"]])
                em = int(sc_dict.get("em", 0))
                f1 = float(sc_dict.get("f1", 0.0))
                feats = per_sample_features(out, pred)
                rec = {
                    "qid": cfg["qid"], "stratum": cfg["stratum"], "mode": mode,
                    "sample_idx": s_idx, "seed": seed,
                    "perturbed_question": q_perturbed,
                    "pred": pred, "em": em, "f1": f1,
                    "text": out["text"], "n_tokens": out["n_tokens"],
                    **(feats or {}),
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                completed += 1
                if completed % 10 == 0:
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-9)
                    remain = (total_gens - completed) / max(rate, 1e-9)
                    print(f"  [{completed}/{total_gens}] "
                          f"{cfg['qid'][:10]} {mode} s{s_idx} "
                          f"pred={pred!r:30s} em={em} "
                          f"({rate:.2f} gen/s, ETA {remain/60:.1f} min)")

    fout.close()
    print(f"\nDone. Wrote {DUMP_FILE}")


if __name__ == "__main__":
    main()
