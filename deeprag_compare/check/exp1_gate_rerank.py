"""
Exp-1: Replace sc_orig gate with first_ent_min_lp gate on existing 1000Q results.

For each dataset, score the saved nr_text via a model forward pass (no generation
needed) to get per-token log-probs, then compute first_ent_min_lp. Simulate gating
at all thresholds and compare accuracy/retrieve-rate curves vs. sc_orig gating.

Usage:
    conda run -n prag python check/exp1_gate_rerank.py [--datasets hotpotqa ...]

Output:
    check/exp1_output/{dataset}_per_q.jsonl    per-question metric dump
    check/exp1_output/exp1_summary.md          comparison table
    check/exp1_output/figures/gate_curve_*.png accuracy-vs-retrieverate curves
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/S113065528/Research/deeprag_compare")
sys.path.insert(0, str(ROOT / "src"))
from dataset_eval import DATASET_CLASSES, build_fewshot_prompt  # noqa: E402

OUTPUT_DIR = ROOT / "check/exp1_output"
RESULTS_DIR = ROOT / "results"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_CACHE_CANDIDATES = [
    "/home/S113065528/Adaptive-RAG/hf_cache",
    "/workspace/S113065528/hf_cache/hub",
    "/dataspace/P76125041/.cache/huggingface/hub",
    "/dataspace/Q56124068/.cache/huggingface/hub",
]
N_SHOTS = 8
RESULT_SUFFIX = "meta-llama-3-8b-instruct"

ANSWER_PHRASE_RE = re.compile(r"[Ss]o the answer is\s+", re.I)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_cache_dir():
    model_dir = "models--" + MODEL_ID.replace("/", "--")
    for c in HF_CACHE_CANDIDATES:
        if os.path.isdir(os.path.join(c, model_dir)):
            return c
    raise FileNotFoundError(f"Cannot find {model_dir} in any of: {HF_CACHE_CANDIDATES}")


def load_model():
    cache_dir = _find_cache_dir()
    print(f"Using HF cache: {cache_dir}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=cache_dir, device_map="cuda",
        torch_dtype=torch.bfloat16, local_files_only=True,
    )
    m.eval()
    return tok, m


# ---------------------------------------------------------------------------
# Token span detection (mirrors exp0_answer_span_sc.py)
# ---------------------------------------------------------------------------

def reconstruct(tokens):
    text = ""
    offsets = []
    for t in tokens:
        offsets.append(len(text))
        text += t["tok"]
    return text, offsets


def char_to_token_idx(offsets, char_pos, n_tokens):
    for i, off in enumerate(offsets):
        if off >= char_pos:
            return i
    return n_tokens


def find_answer_span(tokens):
    """Tokens for the answer in 'So the answer is X.' — excludes the trailing period."""
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
        tok_str = tokens[i]["tok"]
        if "<|eot_id|>" in tok_str or "\n" in tok_str:
            break
        if tok_str.strip() == "." or (tok_str.strip().endswith(".") and i > start):
            end = i
            break
        end = i + 1
    if end <= start:
        return None
    return (start, end)


def find_first_entity_span(pred, tokens):
    """First occurrence of pred entity in the token sequence."""
    pred = (pred or "").strip()
    if not pred or len(pred) < 2:
        return None
    text, offsets = reconstruct(tokens)
    idx = text.lower().find(pred.lower())
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
    end_char = idx + len(pred)
    end_tok = start_tok + 1
    while end_tok < len(offsets) and offsets[end_tok] < end_char:
        end_tok += 1
    return (start_tok, end_tok)


def span_metrics(tokens, span):
    if span is None:
        return None
    s, e = span
    if e <= s or s < 0 or e > len(tokens):
        return None
    sub = tokens[s:e]
    lps = [t["lp"] for t in sub]
    return {
        "len": e - s,
        "start_pos": s,
        "mean_lp": float(mean(lps)),
        "min_lp": float(min(lps)),
    }


# ---------------------------------------------------------------------------
# Scoring: forward pass on existing generated text
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_text(model, tok_obj, prompt: str, gen_text: str):
    """Run forward pass on prompt+gen_text; return per-token dicts for gen_text tokens."""
    # Tokenize prompt alone to get P
    p_ids = tok_obj(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    P = p_ids.shape[1]

    # Tokenize prompt + gen_text jointly (avoids boundary tokenization artifacts)
    # gen_text starts with a space because decode().strip() removed it — re-add one
    full_ids = tok_obj(
        prompt + " " + gen_text.lstrip(),
        add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(model.device)

    A = full_ids.shape[1] - P
    if A <= 0:
        return []

    # Single forward pass
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits[0, P - 1: P + A - 1, :].float()  # (A, V)

    log_probs = F.log_softmax(logits, dim=-1)  # (A, V)
    ans_ids = full_ids[0, P:]                   # (A,)

    token_lp = log_probs.gather(1, ans_ids.unsqueeze(1)).squeeze(1)  # (A,)
    top2_lp, _ = log_probs.topk(2, dim=-1)                           # (A, 2)

    tokens = []
    for i in range(A):
        tid = ans_ids[i].item()
        tokens.append({
            "pos": i,
            "id": tid,
            "tok": tok_obj.decode([tid]),
            "lp": token_lp[i].item(),
            "top1_lp": top2_lp[i, 0].item(),
            "top2_lp": top2_lp[i, 1].item(),
        })
    return tokens


# ---------------------------------------------------------------------------
# Gating simulation
# ---------------------------------------------------------------------------

def auroc(y_true, scores):
    """Mann-Whitney U AUROC: fraction of (correct, wrong) pairs where score_correct > score_wrong."""
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    count = sum(1 for p in pos for n in neg if p > n)
    count += sum(0.5 for p in pos for n in neg if p == n)
    return count / (len(pos) * len(neg))


def simulate_gate(records, metric_key, retrieve_when_below):
    """
    Sweep thresholds over metric_key values.
    retrieve_when_below=True: use_rag = (metric < threshold)
    retrieve_when_below=False: use_rag = (metric >= threshold)  [inverted for sc_orig]

    Returns list of (retrieve_rate, em_acc) sorted by retrieve_rate.
    """
    vals = [(r[metric_key], r["nr_em"], r["rag_em"])
            for r in records if r.get(metric_key) is not None]
    thresholds = sorted(set(v[0] for v in vals))

    results = []
    for th in thresholds:
        em_total = 0
        n_retrieve = 0
        for metric_val, nr_em, rag_em in vals:
            if retrieve_when_below:
                use_rag = metric_val < th
            else:
                use_rag = metric_val >= th
            if use_rag:
                em_total += rag_em
                n_retrieve += 1
            else:
                em_total += nr_em
        n = len(vals)
        results.append({
            "threshold": th,
            "retrieve_rate": n_retrieve / n,
            "em": em_total / n,
        })
    # Add always-NR and always-RAG endpoints
    n = len(vals)
    results.insert(0, {
        "threshold": None,
        "retrieve_rate": 0.0,
        "em": mean(v[1] for v in vals),
    })
    results.append({
        "threshold": None,
        "retrieve_rate": 1.0,
        "em": mean(v[2] for v in vals),
    })
    results.sort(key=lambda x: x["retrieve_rate"])
    return results


def best_gate_em(sim_results):
    """Max EM over all threshold settings."""
    return max(r["em"] for r in sim_results)


def em_at_rate(sim_results, target_rate, tol=0.05):
    """EM at the threshold closest to target_rate."""
    best = min(sim_results, key=lambda r: abs(r["retrieve_rate"] - target_rate))
    if abs(best["retrieve_rate"] - target_rate) > tol + 0.05:
        return None
    return best["em"]


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(ds_name, model, tok_obj):
    ds_cls = DATASET_CLASSES[ds_name]
    fewshot_prefix = build_fewshot_prompt(ds_cls, N_SHOTS)

    results_file = RESULTS_DIR / f"{ds_name}_{RESULT_SUFFIX}.jsonl"
    if not results_file.exists():
        print(f"  [SKIP] {results_file} not found")
        return None

    records_raw = [json.loads(l) for l in results_file.read_text().splitlines() if l.strip()]
    print(f"  Loaded {len(records_raw)} questions from {results_file.name}")

    out_file = OUTPUT_DIR / f"{ds_name}_per_q.jsonl"
    per_q = []

    with open(out_file, "w") as fout:
        for rec in tqdm(records_raw, desc=f"  scoring {ds_name}"):
            question = rec["question"]
            nr_text = rec.get("nr_text", "")
            nr_pred = rec.get("nr_pred", "")
            nr_sc = rec.get("nr_sc")
            nr_em = rec.get("nr_em", 0)
            rag_em = rec.get("rag_em", 0)

            prompt = fewshot_prefix + ds_cls.test_input(question)
            tokens = score_text(model, tok_obj, prompt, nr_text) if nr_text else []

            # first-entity span
            ent_span = find_first_entity_span(nr_pred, tokens)
            ent_m = span_metrics(tokens, ent_span)

            # answer-phrase span ("So the answer is X.")
            ans_span = find_answer_span(tokens)
            ans_m = span_metrics(tokens, ans_span)

            pq = {
                "qid": rec.get("qid", ""),
                "question": question[:80],
                "nr_pred": nr_pred,
                "nr_sc": nr_sc,
                "nr_em": nr_em,
                "rag_em": rag_em,
                "n_tok": len(tokens),
                "first_ent_min_lp": ent_m["min_lp"] if ent_m else None,
                "first_ent_mean_lp": ent_m["mean_lp"] if ent_m else None,
                "answer_min_lp": ans_m["min_lp"] if ans_m else None,
                "answer_mean_lp": ans_m["mean_lp"] if ans_m else None,
            }
            per_q.append(pq)
            fout.write(json.dumps(pq) + "\n")

    return per_q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["hotpotqa", "2wikimultihopqa", "musique", "popqa"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)

    print("Loading model...")
    tok_obj, model = load_model()

    all_stats = {}

    for ds_name in args.datasets:
        print(f"\n=== {ds_name} ===")
        per_q = process_dataset(ds_name, model, tok_obj)
        if per_q is None:
            continue

        # --- AUROCs ---
        valid_ent = [r for r in per_q if r["first_ent_min_lp"] is not None]
        valid_ans = [r for r in per_q if r["answer_min_lp"] is not None]

        auroc_sc   = auroc([r["nr_em"] for r in per_q], [r["nr_sc"] for r in per_q])
        auroc_ent  = auroc([r["nr_em"] for r in valid_ent],
                           [r["first_ent_min_lp"] for r in valid_ent])
        auroc_ans  = auroc([r["nr_em"] for r in valid_ans],
                           [r["answer_min_lp"] for r in valid_ans]) if valid_ans else float("nan")

        # --- Gating simulation ---
        # sc_orig: higher SC → confident → NR; lower SC → retrieve
        # threshold sweep: retrieve_when_below=True (retrieve if sc < th)
        sc_sim = simulate_gate(per_q, "nr_sc", retrieve_when_below=True)

        # first_ent_min_lp: lower (more negative) → less confident → retrieve
        # retrieve_when_below=True: retrieve if first_ent_min_lp < th
        ent_sim = simulate_gate(valid_ent, "first_ent_min_lp", retrieve_when_below=True)

        # answer_min_lp
        ans_sim = simulate_gate(valid_ans, "answer_min_lp", retrieve_when_below=True) if valid_ans else []

        nr_em_baseline  = mean(r["nr_em"] for r in per_q)
        rag_em_baseline = mean(r["rag_em"] for r in per_q)

        stats = {
            "n": len(per_q),
            "n_ent_found": len(valid_ent),
            "nr_em": nr_em_baseline,
            "rag_em": rag_em_baseline,
            "auroc_sc": auroc_sc,
            "auroc_ent": auroc_ent,
            "auroc_ans": auroc_ans,
            "best_sc_gate":  best_gate_em(sc_sim),
            "best_ent_gate": best_gate_em(ent_sim),
            "best_ans_gate": best_gate_em(ans_sim) if ans_sim else float("nan"),
        }

        # EM at matched retrieve rates
        for rate in [0.2, 0.4, 0.6, 0.8]:
            sc_val  = em_at_rate(sc_sim, rate)
            ent_val = em_at_rate(ent_sim, rate)
            stats[f"sc_em@{int(rate*100)}"]  = sc_val
            stats[f"ent_em@{int(rate*100)}"] = ent_val

        all_stats[ds_name] = stats

        # --- Plot gate curve ---
        fig, ax = plt.subplots(figsize=(7, 5))
        sc_rr  = [r["retrieve_rate"] for r in sc_sim]
        sc_em  = [r["em"] for r in sc_sim]
        ent_rr = [r["retrieve_rate"] for r in ent_sim]
        ent_em = [r["em"] for r in ent_sim]

        ax.plot(sc_rr, sc_em, "b-o", markersize=3, label=f"sc_orig gate (AUROC={auroc_sc:.3f})")
        ax.plot(ent_rr, ent_em, "r-o", markersize=3,
                label=f"first_ent_min_lp gate (AUROC={auroc_ent:.3f})")
        ax.axhline(nr_em_baseline,  color="gray", linestyle="--", linewidth=1, label=f"NR baseline ({nr_em_baseline:.3f})")
        ax.axhline(rag_em_baseline, color="green", linestyle="--", linewidth=1, label=f"RAG baseline ({rag_em_baseline:.3f})")
        ax.set_xlabel("Retrieve Rate (fraction of questions that trigger RAG)")
        ax.set_ylabel("EM Accuracy")
        ax.set_title(f"{ds_name}: Gate accuracy vs retrieve rate")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"figures/gate_curve_{ds_name}.png", dpi=150)
        plt.close(fig)
        print(f"  NR={nr_em_baseline:.3f}  RAG={rag_em_baseline:.3f}  "
              f"best_sc_gate={stats['best_sc_gate']:.3f}  best_ent_gate={stats['best_ent_gate']:.3f}")
        print(f"  AUROC: sc={auroc_sc:.4f}  first_ent={auroc_ent:.4f}  answer={auroc_ans:.4f}")

    # --- Summary markdown ---
    lines = ["# Exp-1: first_ent_min_lp vs sc_orig as Retrieval Gate\n"]
    lines.append("## Per-dataset results\n")
    lines.append("| Dataset | N | NR EM | RAG EM | AUROC sc | AUROC ent | Best sc gate | Best ent gate | Δ |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for ds, s in all_stats.items():
        delta = s["best_ent_gate"] - s["best_sc_gate"]
        lines.append(
            f"| {ds} | {s['n']} "
            f"| {s['nr_em']:.3f} | {s['rag_em']:.3f} "
            f"| {s['auroc_sc']:.4f} | {s['auroc_ent']:.4f} "
            f"| {s['best_sc_gate']:.3f} | {s['best_ent_gate']:.3f} "
            f"| {delta:+.3f} |"
        )

    lines.append("\n## EM at matched retrieve rates\n")
    lines.append("| Dataset | sc@20% | ent@20% | sc@40% | ent@40% | sc@60% | ent@60% | sc@80% | ent@80% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for ds, s in all_stats.items():
        def fmt(v): return f"{v:.3f}" if v is not None else "n/a"
        lines.append(
            f"| {ds} "
            f"| {fmt(s.get('sc_em@20'))} | {fmt(s.get('ent_em@20'))} "
            f"| {fmt(s.get('sc_em@40'))} | {fmt(s.get('ent_em@40'))} "
            f"| {fmt(s.get('sc_em@60'))} | {fmt(s.get('ent_em@60'))} "
            f"| {fmt(s.get('sc_em@80'))} | {fmt(s.get('ent_em@80'))} |"
        )

    lines.append("\n## Notes")
    lines.append("- `first_ent_min_lp`: min log-prob of the token span where the predicted answer first appears in the CoT")
    lines.append("- `sc_orig`: aggregate Self-Certainty over full CoT (current gate)")
    lines.append("- Gate simulation: at each threshold, questions with metric below threshold trigger RAG; rest use NR")
    lines.append("- `best_*_gate`: maximum EM over all threshold settings (oracle-selected threshold)")

    summary_md = "\n".join(lines)
    (OUTPUT_DIR / "exp1_summary.md").write_text(summary_md)
    print("\n" + summary_md)
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
