"""
Calibration: per-dataset distribution of no-CoT first-ent min_lp / min_sc.

Goal: pick threshold sweep centers for the 1000Q production gate run.
For each of {HotpotQA, 2WikiMultiHopQA, MuSiQue, PopQA}:
  - sample N_CAL questions from top-1000
  - generate no-CoT direct answer (greedy, T=0)
  - compute first-ent min_lp / min_sc on the answer span
  - score EM, split by correct / wrong, dump quantiles

Usage:
    conda run -n prag python check/calibrate_thresholds.py
    conda run -n prag python check/calibrate_thresholds.py --datasets hotpotqa popqa --n-cal 50
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/S113065528/Research/deeprag_compare")
sys.path.insert(0, str(ROOT / "src"))
from dataset_eval import (  # noqa: E402
    DATASET_CLASSES, WikiMultiHopQA, build_fewshot_prompt, score,
)

OUTPUT_DIR = ROOT / "check/calibration_output"
DATA_DIR = ROOT / "data/top1000"
ALIAS_FILE = DATA_DIR / "2wikimultihopqa_id_aliases.jsonl"

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_CACHE_CANDIDATES = [
    "/home/S113065528/Adaptive-RAG/hf_cache",
    "/workspace/S113065528/hf_cache/hub",
    "/dataspace/P76125041/.cache/huggingface/hub",
    "/dataspace/Q56124068/.cache/huggingface/hub",
]
N_SHOTS = 8
MAX_NEW_DIRECT = 20
SEED = 42


# ---------------------------------------------------------------------------
# Model loading (mirrors exp0b)
# ---------------------------------------------------------------------------

def find_cache_dir():
    model_dir = "models--" + MODEL_ID.replace("/", "--")
    for c in HF_CACHE_CANDIDATES:
        if os.path.isdir(os.path.join(c, model_dir)):
            return c
    raise FileNotFoundError(f"No cache for {model_dir}")


def load_model():
    cache = find_cache_dir()
    print(f"HF cache: {cache}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=cache, device_map="cuda",
        torch_dtype=torch.bfloat16, local_files_only=True,
    )
    m.eval()
    return tok, m


# ---------------------------------------------------------------------------
# Prompt + span helpers (no-CoT, mirrors exp0b)
# ---------------------------------------------------------------------------

def build_no_cot_prefix(ds_cls, n_shots=N_SHOTS):
    lines = []
    for ex in ds_cls.examplars[:n_shots]:
        lines.append(ds_cls.demo_input(ex["question"]))
        lines.append(f" {ex['answer']}.")
        lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def no_cot_test_input(q):
    return f"Answer the following question with a short answer (one phrase, no explanation).\nQuestion: {q}\nAnswer:"


def reconstruct(tokens):
    text = ""
    offsets = []
    for t in tokens:
        offsets.append(len(text))
        text += t["tok"]
    return text, offsets


def find_first_entity_span(pred, tokens):
    pred = (pred or "").strip()
    if not pred or len(pred) < 2:
        return None
    text, offsets = reconstruct(tokens)
    idx = text.lower().find(pred.lower())
    if idx == -1:
        return None
    start_tok = None
    for i in range(len(offsets)):
        nxt = offsets[i + 1] if i + 1 < len(offsets) else len(text)
        if offsets[i] <= idx < nxt:
            start_tok = i
            break
    if start_tok is None:
        return None
    end_char = idx + len(pred)
    end_tok = start_tok + 1
    while end_tok < len(offsets) and offsets[end_tok] < end_char:
        end_tok += 1
    return (start_tok, end_tok)


def span_aggs(tokens, span):
    if span is None:
        return None
    s, e = span
    if e <= s or s < 0 or e > len(tokens):
        return None
    sub = tokens[s:e]
    lps = [t["lp"] for t in sub]
    scs = [t["sc"] for t in sub]
    ents = [t["ent"] for t in sub]
    return {
        "min_lp": float(min(lps)),
        "mean_lp": float(mean(lps)),
        "min_sc": float(min(scs)),
        "mean_sc": float(mean(scs)),
        "max_sc": float(max(scs)),
        "min_ent": float(min(ents)),
        "max_ent": float(max(ents)),
        "neg_max_ent": -float(max(ents)),
    }


# ---------------------------------------------------------------------------
# Greedy generation + scoring (T=0, no sampling — calibration only needs
# the distribution of one deterministic sample per question)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate_with_scoring(model, tok_obj, prompt, max_new):
    p_ids = tok_obj(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    P = p_ids.shape[1]
    out = model.generate(
        p_ids,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tok_obj.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )
    seq = out.sequences[0]
    A = seq.shape[0] - P
    if A <= 0:
        return [], ""

    eos_id = tok_obj.eos_token_id
    pad_id = tok_obj.pad_token_id
    end = A
    while end > 0 and seq[P + end - 1].item() in (eos_id, pad_id):
        end -= 1
    if end <= 0:
        return [], ""
    full_ids = seq[: P + end].unsqueeze(0)
    A = end

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits[0, P - 1: P + A - 1, :].float()
    log_probs = F.log_softmax(logits, dim=-1)
    V = log_probs.shape[-1]
    ans_ids = full_ids[0, P:]
    token_lp = log_probs.gather(1, ans_ids.unsqueeze(1)).squeeze(1)
    sc_per_pos = -math.log(V) - log_probs.mean(dim=-1)
    probs = log_probs.exp()
    ent_per_pos = -(probs * log_probs).sum(dim=-1)
    tokens = []
    for i in range(A):
        tid = ans_ids[i].item()
        tokens.append({
            "pos": i, "id": tid, "tok": tok_obj.decode([tid]),
            "lp": token_lp[i].item(),
            "sc": sc_per_pos[i].item(),
            "ent": ent_per_pos[i].item(),
        })
    text = tok_obj.decode(ans_ids, skip_special_tokens=True).strip()
    return tokens, text


def clean_pred(text):
    pred = text.split("\n")[0].strip()
    for sep in [". ", "."]:
        if sep in pred:
            pred = pred.split(sep)[0].strip()
            break
    return pred


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def quantiles(vals):
    if not vals:
        return None
    vs = sorted(vals)
    n = len(vs)
    p = lambda q: vs[max(0, min(n - 1, int(q * n)))]
    return {
        "n": n, "min": vs[0], "max": vs[-1], "mean": mean(vs),
        "p10": p(0.10), "p25": p(0.25), "p50": p(0.50),
        "p75": p(0.75), "p90": p(0.90),
    }


def auroc(y_true, scores):
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    count = sum(1 for p in pos for n in neg if p > n)
    count += sum(0.5 for p in pos for n in neg if p == n)
    return count / (len(pos) * len(neg))


def threshold_sweep(c_vals, w_vals, qs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """Returns rows: (threshold, trigger_rate, TPR, FPR, balanced)."""
    if not c_vals or not w_vals:
        return []
    all_vals = sorted(c_vals + w_vals)
    rows = []
    for q in qs:
        th = all_vals[int(q * len(all_vals))]
        tw = sum(1 for x in w_vals if x < th)
        tc = sum(1 for x in c_vals if x < th)
        tpr = tw / len(w_vals)
        fpr = tc / len(c_vals)
        rows.append((th, q, tpr, fpr, tpr - fpr))
    return rows


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def calibrate_one(model, tok, ds_tag, n_cal, model_loaded=True):
    ds_cls = DATASET_CLASSES[ds_tag]
    if ds_tag == "2wikimultihopqa":
        WikiMultiHopQA.load_aliases(str(ALIAS_FILE))

    data_path = DATA_DIR / f"{ds_tag}.jsonl"
    with open(data_path) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]
    rng = random.Random(SEED)
    sample = rng.sample(all_rows, min(n_cal, len(all_rows)))
    print(f"\n=== {ds_tag}: calibrating on {len(sample)} questions ===")

    direct_prefix = build_no_cot_prefix(ds_cls, N_SHOTS)
    records = []
    for q in tqdm(sample):
        prompt = direct_prefix + no_cot_test_input(q["question"])
        tokens, text = greedy_generate_with_scoring(model, tok, prompt, MAX_NEW_DIRECT)
        pred = clean_pred(text)
        golds = ds_cls.golds(q)
        sc_out = score(pred, golds)
        em = sc_out["em"]

        ent_span = find_first_entity_span(pred, tokens) if pred else None
        agg_ent = span_aggs(tokens, ent_span)
        agg_full = span_aggs(tokens, (0, len(tokens))) if tokens else None
        records.append({
            "qid": q["qid"],
            "question": q["question"],
            "golds": golds,
            "pred": pred,
            "em": em,
            "f1": sc_out["f1"],
            "n_tok": len(tokens),
            "first_ent_aggs": agg_ent,
            "full_aggs": agg_full,
        })
    return records


def summarize(ds_tag, records):
    """Per-dataset distribution + threshold sweep."""
    n_total = len(records)
    n_em = sum(r["em"] for r in records)
    n_with_span = sum(1 for r in records if r["first_ent_aggs"])
    print(f"  total={n_total}  em={n_em}/{n_total} ({n_em/n_total:.1%})  with_first_ent_span={n_with_span}")

    out = {"dataset": ds_tag, "n_total": n_total, "n_em": n_em,
           "n_with_span": n_with_span, "metrics": {}}

    for key in ("min_lp", "min_sc", "mean_lp", "mean_sc", "neg_max_ent"):
        c_vals, w_vals = [], []
        for r in records:
            a = r["first_ent_aggs"]
            if not a:
                continue
            v = a[key]
            if v is None or math.isnan(v) or math.isinf(v):
                continue
            (c_vals if r["em"] == 1 else w_vals).append(v)
        out["metrics"][key] = {
            "auroc": auroc([1] * len(c_vals) + [0] * len(w_vals), c_vals + w_vals),
            "correct": quantiles(c_vals),
            "wrong": quantiles(w_vals),
            "sweep": threshold_sweep(c_vals, w_vals),
        }
    return out


def render_markdown(all_results):
    L = ["# Threshold calibration: no-CoT first-ent SC / lp",
         "",
         f"- Per-dataset N_CAL = {all_results[0]['n_total']} (greedy decode, T=0)",
         f"- Model: {MODEL_ID}",
         "- Metric: confidence on first-entity span of direct (no-CoT) answer",
         "- gate trigger rule: `score < threshold` ⇒ retrieve",
         ""]

    L.append("## EM and span-coverage per dataset\n")
    L.append("| Dataset | N | EM | EM% | with_first_ent_span |")
    L.append("|---|---|---|---|---|")
    for r in all_results:
        L.append(f"| {r['dataset']} | {r['n_total']} | {r['n_em']} | "
                 f"{r['n_em']/max(r['n_total'],1):.1%} | {r['n_with_span']} |")

    L.append("\n## AUROC per dataset (first-ent span)\n")
    L.append("| Dataset | min_lp | min_sc | mean_lp | mean_sc | neg_max_ent |")
    L.append("|---|---|---|---|---|---|")
    for r in all_results:
        m = r["metrics"]
        cells = [f"{m[k]['auroc']:.3f}" if not math.isnan(m[k]['auroc']) else "—"
                 for k in ("min_lp", "min_sc", "mean_lp", "mean_sc", "neg_max_ent")]
        L.append(f"| {r['dataset']} | " + " | ".join(cells) + " |")

    for k in ("min_lp", "min_sc"):
        L.append(f"\n## Distribution of `{k}` (correct vs wrong)\n")
        L.append("| Dataset | side | n | p10 | p25 | p50 | p75 | p90 | mean |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in all_results:
            for side in ("correct", "wrong"):
                q = r["metrics"][k][side]
                if not q:
                    continue
                L.append(f"| {r['dataset']} | {side} | {q['n']} | "
                         f"{q['p10']:.3f} | {q['p25']:.3f} | {q['p50']:.3f} | "
                         f"{q['p75']:.3f} | {q['p90']:.3f} | {q['mean']:.3f} |")

    for k in ("min_lp", "min_sc"):
        L.append(f"\n## Threshold sweep on `{k}` (gate fires when score < threshold)\n")
        for r in all_results:
            L.append(f"### {r['dataset']}\n")
            L.append("| threshold | trigger% | TPR (wrong caught) | FPR (correct disturbed) | TPR−FPR |")
            L.append("|---|---|---|---|---|")
            for (th, q, tpr, fpr, bal) in r["metrics"][k]["sweep"]:
                L.append(f"| {th:.3f} | {q*100:.0f}% | {tpr:.3f} | {fpr:.3f} | {bal:+.3f} |")
            L.append("")

    L.append("\n## Recommended threshold sweep for production 1000Q runs\n")
    L.append("Per dataset, take p25/p40/p60 of the **wrong** distribution as natural sweep centers (where TPR ≈ 0.25/0.40/0.60), then add ±0.5/±1.0 on each side.")
    L.append("")
    L.append("| Dataset | metric | suggested 6-pt sweep |")
    L.append("|---|---|---|")
    for r in all_results:
        for k in ("min_lp", "min_sc"):
            w = r["metrics"][k]["wrong"]
            if not w:
                continue
            sweep = sorted({round(x, 2) for x in [
                w["p10"], w["p25"], w["p50"], w["p75"], w["p90"], w["mean"]
            ]})
            L.append(f"| {r['dataset']} | {k} | " +
                     ", ".join(f"{x:.2f}" for x in sweep) + " |")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["hotpotqa", "2wikimultihopqa", "musique", "popqa"])
    p.add_argument("--n-cal", type=int, default=100)
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading model...")
    tok, model = load_model()

    all_results = []
    for ds_tag in args.datasets:
        if ds_tag not in DATASET_CLASSES:
            print(f"  skip unknown dataset: {ds_tag}")
            continue
        records = calibrate_one(model, tok, ds_tag, args.n_cal)
        # dump raw per-question records
        (OUTPUT_DIR / f"{ds_tag}_records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
        result = summarize(ds_tag, records)
        all_results.append(result)

    summary_md = render_markdown(all_results)
    (OUTPUT_DIR / "calibration_summary.md").write_text(summary_md)
    (OUTPUT_DIR / "calibration_summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {OUTPUT_DIR}/")
    print("\n" + summary_md)


if __name__ == "__main__":
    main()
