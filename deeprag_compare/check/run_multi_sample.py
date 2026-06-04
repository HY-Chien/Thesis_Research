"""
Multi-sample diagnostic for SC stability and confident-hallucination analysis.

For each of the 20 stratified-sampled HotpotQA questions, run No-RAG CoT
N times under sampling (temperature=0.7) and dump:
  - sample-level: SC, PPL, predicted answer, EM/F1
  - token-level: chosen token, chosen log-prob, top-5 alternatives, entropy

This matches run_experiment.py's prompt format (8-shot DRAGIN-style CoT),
just swapping greedy decoding for sampling.

Output:
  check/multi_sample_dump.jsonl   (one record per (qid, sample_idx))
  check/multi_sample_summary.json (per-qid aggregate: sc mean/std/min/max, em rate)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/S113065528/Research/deeprag_compare")
sys.path.insert(0, str(ROOT / "src"))
from dataset_eval import HotpotQA, build_fewshot_prompt, extract_answer, score  # noqa: E402

SAMPLED_FILE = ROOT / "check/sampled_20q.json"
DUMP_FILE = ROOT / "check/multi_sample_dump.jsonl"
SUMMARY_FILE = ROOT / "check/multi_sample_summary.json"

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# Try multiple cache locations: user's cache first, then peer caches
HF_CACHE_CANDIDATES = [
    "/home/S113065528/Adaptive-RAG/hf_cache",
    "/workspace/S113065528/hf_cache/hub",
    "/dataspace/P76125041/.cache/huggingface/hub",
    "/dataspace/Q56124068/.cache/huggingface/hub",
]
N_SAMPLES = 20
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_NEW_TOKENS = 100
TOP_K_DUMP = 5  # how many top alternatives to record per token


def _find_cache_dir():
    model_dir = "models--" + MODEL_ID.replace("/", "--")
    for c in HF_CACHE_CANDIDATES:
        if os.path.isdir(os.path.join(c, model_dir)):
            return c
    raise FileNotFoundError(
        f"Cannot find {model_dir} in any of: {HF_CACHE_CANDIDATES}"
    )


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


@torch.no_grad()
def sample_once(model, tok, prompt: str, max_new: int, temperature: float, seed: int):
    """Sample once and compute SC + per-token diagnostics."""
    torch.manual_seed(seed)

    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    out_ids = model.generate(
        prompt_ids, max_new_tokens=max_new, do_sample=True,
        temperature=temperature, top_p=TOP_P,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    ans_ids = out_ids[0][prompt_len:]
    text = tok.decode(ans_ids, skip_special_tokens=True).strip()
    n_tok = int(ans_ids.shape[0])
    if n_tok == 0:
        return {"text": "", "sc": 0.0, "ppl": float("inf"), "n_tokens": 0, "tokens": []}

    full_ids = out_ids[:, : prompt_len + n_tok]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits

    ans_logits = logits[:, prompt_len - 1 : prompt_len + n_tok - 1, :].squeeze(0).float()
    log_probs = F.log_softmax(ans_logits, dim=-1)  # [n_tok, V]

    ans_token_ids = ans_ids.to(model.device)
    token_lp = log_probs.gather(1, ans_token_ids.unsqueeze(1)).squeeze(1)  # [n_tok]
    avg_nll = -token_lp.mean().item()
    ppl = math.exp(min(avg_nll, 20))

    V = float(log_probs.size(-1))
    lp_sum = log_probs.sum(dim=-1)  # [n_tok]
    sc = (-lp_sum.mean().item() / V) - math.log(V)

    # Per-token diagnostics: top-k + entropy
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [n_tok], nats
    topk_lp, topk_ids = log_probs.topk(TOP_K_DUMP, dim=-1)  # [n_tok, K]

    tokens = []
    for i in range(n_tok):
        chosen_id = int(ans_token_ids[i].item())
        chosen_str = tok.decode([chosen_id], skip_special_tokens=False)
        topk = []
        for k in range(TOP_K_DUMP):
            tid = int(topk_ids[i, k].item())
            topk.append({
                "id": tid,
                "tok": tok.decode([tid], skip_special_tokens=False),
                "lp": float(topk_lp[i, k].item()),
            })
        tokens.append({
            "pos": i,
            "id": chosen_id,
            "tok": chosen_str,
            "lp": float(token_lp[i].item()),
            "entropy": float(entropy[i].item()),
            "top1_lp": float(topk_lp[i, 0].item()),
            "top2_lp": float(topk_lp[i, 1].item()),
            "topk": topk,
        })

    return {"text": text, "sc": sc, "ppl": ppl, "n_tokens": n_tok, "tokens": tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first K of 20 questions (for smoke test)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (qid, sample_idx) pairs already in dump file")
    parser.add_argument("--sampled_file", type=str, default=str(SAMPLED_FILE),
                        help="Stratified question file (default: sampled_20q.json)")
    parser.add_argument("--dump_file", type=str, default=str(DUMP_FILE),
                        help="Output JSONL dump path")
    parser.add_argument("--summary_file", type=str, default=str(SUMMARY_FILE),
                        help="Output summary JSON path")
    args = parser.parse_args()

    sampled_path = Path(args.sampled_file)
    dump_path = Path(args.dump_file)
    summary_path = Path(args.summary_file)

    with open(sampled_path) as f:
        sampled = json.load(f)
    samples = sampled["samples"]
    if args.limit:
        samples = samples[: args.limit]

    print(f"Loading model {MODEL_ID} ...")
    tok, model = load_model()
    print("Model loaded.")

    fewshot = build_fewshot_prompt(HotpotQA, n_shots=8)

    done = set()
    if args.resume and dump_path.exists():
        for line in open(dump_path):
            try:
                r = json.loads(line)
                done.add((r["qid"], r["sample_idx"]))
            except Exception:
                pass
        print(f"Resume: skipping {len(done)} existing records")

    mode = "a" if args.resume and dump_path.exists() else "w"
    fout = open(dump_path, mode)

    for q_idx, s in enumerate(samples):
        qid = s["qid"]
        question = s["question"]
        golds = s["golds"]
        prompt = fewshot + HotpotQA.demo_input(question) + " "
        t0 = time.time()
        for sidx in range(args.n_samples):
            if (qid, sidx) in done:
                continue
            seed = 1000 * (q_idx + 1) + sidx
            out = sample_once(model, tok, prompt, args.max_new_tokens, args.temperature, seed)
            pred = extract_answer(out["text"])
            sc_dict = score(pred, golds)
            rec = {
                "qid": qid,
                "stratum": s["stratum"],
                "sample_idx": sidx,
                "seed": seed,
                "question": question,
                "golds": golds,
                "pred": pred,
                "em": sc_dict["em"],
                "f1": sc_dict["f1"],
                "sc": out["sc"],
                "ppl": out["ppl"],
                "n_tokens": out["n_tokens"],
                "text": out["text"],
                "tokens": out["tokens"],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        dt = time.time() - t0
        print(f"[{q_idx + 1:2d}/{len(samples)}] {s['stratum']:11s} qid={qid[:10]} "
              f"({dt:.1f}s for {args.n_samples} samples)")

    fout.close()

    # Compute per-qid summary
    by_qid = {}
    for line in open(dump_path):
        r = json.loads(line)
        by_qid.setdefault(r["qid"], []).append(r)

    summary = []
    for s in samples:
        recs = by_qid.get(s["qid"], [])
        if not recs:
            continue
        scs = [r["sc"] for r in recs]
        ems = [r["em"] for r in recs]
        ppls = [r["ppl"] for r in recs]
        preds = [r["pred"] for r in recs]
        from statistics import mean, stdev
        summary.append({
            "qid": s["qid"],
            "stratum": s["stratum"],
            "question": s["question"],
            "golds": s["golds"],
            "greedy_nr_sc": s["nr_sc"],
            "greedy_nr_pred": s["nr_pred"],
            "greedy_nr_em": s["nr_em"],
            "n_samples": len(recs),
            "sc_mean": mean(scs),
            "sc_std": stdev(scs) if len(scs) > 1 else 0.0,
            "sc_min": min(scs),
            "sc_max": max(scs),
            "ppl_mean": mean(ppls),
            "em_rate": mean(ems),
            "unique_preds": sorted(set(preds)),
            "n_unique_preds": len(set(preds)),
        })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDump:    {dump_path}")
    print(f"Summary: {summary_path}")
    print(f"\nQuick view:")
    for r in summary:
        print(f"  [{r['stratum']:11s}] sc_mean={r['sc_mean']:5.2f} ±{r['sc_std']:.2f}  "
              f"em_rate={r['em_rate']:.2f}  unique_preds={r['n_unique_preds']:2d}  "
              f"| {r['question'][:50]}")


if __name__ == "__main__":
    main()
