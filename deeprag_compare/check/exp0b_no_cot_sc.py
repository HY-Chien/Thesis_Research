"""
Exp-0b: no-CoT comparison + span-restricted SC variants.

Goals:
  1. Augment existing CoT samples with span-restricted SC (was missing in exp0)
  2. Generate fresh no-CoT direct-answer samples for the same 20 questions
  3. Compare AUROCs across (lp, sc, entropy) × (CoT, no-CoT) × (span agg)

Why both:
  - Span-SC vs span-lp: addresses whether the SC FORMULA itself is good when
    restricted to answer span (not just whether log-prob works there)
  - CoT vs no-CoT: addresses whether the answer-span signal is a CoT artifact
    (in-context self-reinforcement boosts repeated tokens' lp)

Usage:
    conda run -n prag python check/exp0b_no_cot_sc.py
"""

import json
import math
import os
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/S113065528/Research/deeprag_compare")
sys.path.insert(0, str(ROOT / "src"))
from dataset_eval import HotpotQA, build_fewshot_prompt, extract_answer, score  # noqa: E402

OUTPUT_DIR = ROOT / "check/exp0b_output"
EXISTING_COT_DUMP = ROOT / "check/multi_sample_dump.jsonl"
SAMPLED_FILE = ROOT / "check/sampled_20q.json"

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_CACHE_CANDIDATES = [
    "/home/S113065528/Adaptive-RAG/hf_cache",
    "/workspace/S113065528/hf_cache/hub",
    "/dataspace/P76125041/.cache/huggingface/hub",
    "/dataspace/Q56124068/.cache/huggingface/hub",
]
N_SHOTS = 8
N_SAMPLES = 20
TEMPERATURE = 0.7
MAX_NEW_DIRECT = 20  # short answer, no CoT


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
# Span detection (re-implemented from exp0)
# ---------------------------------------------------------------------------

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
    """Return all span aggregations (min/mean for lp, sc, entropy)."""
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
        "min_sc": float(min(scs)),     # min over span — sensitive to dips
        "mean_sc": float(mean(scs)),
        "max_sc": float(max(scs)),     # max (high SC = confident)
        "min_ent": float(min(ents)),
        "max_ent": float(max(ents)),
        "neg_max_ent": -float(max(ents)),  # so higher = more confident
    }


# ---------------------------------------------------------------------------
# Forward-pass scoring (full vocab distribution → SC + entropy + lp)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_text_full(model, tok_obj, prompt, gen_text):
    p_ids = tok_obj(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    P = p_ids.shape[1]
    full_ids = tok_obj(
        prompt + " " + gen_text.lstrip(),
        add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(model.device)
    A = full_ids.shape[1] - P
    if A <= 0:
        return []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits[0, P - 1: P + A - 1, :].float()
    log_probs = F.log_softmax(logits, dim=-1)  # (A, V)
    V = log_probs.shape[-1]
    ans_ids = full_ids[0, P:]
    token_lp = log_probs.gather(1, ans_ids.unsqueeze(1)).squeeze(1)
    sc_per_pos = -math.log(V) - log_probs.mean(dim=-1)            # SC formula
    probs = log_probs.exp()
    ent_per_pos = -(probs * log_probs).sum(dim=-1)                # standard entropy
    out = []
    for i in range(A):
        tid = ans_ids[i].item()
        out.append({
            "pos": i, "id": tid, "tok": tok_obj.decode([tid]),
            "lp": token_lp[i].item(),
            "sc": sc_per_pos[i].item(),
            "ent": ent_per_pos[i].item(),
        })
    return out


# ---------------------------------------------------------------------------
# Generation (no-CoT direct QA)
# ---------------------------------------------------------------------------

def build_no_cot_prefix(ds_cls, n_shots=N_SHOTS):
    """Same examplars but answers only — no CoT."""
    lines = []
    for ex in ds_cls.examplars[:n_shots]:
        lines.append(ds_cls.demo_input(ex["question"]))
        lines.append(f" {ex['answer']}.")
        lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def no_cot_test_input(ds_cls, q):
    return f"Answer the following question with a short answer (one phrase, no explanation).\nQuestion: {q}\nAnswer:"


@torch.no_grad()
def generate_with_scoring(model, tok_obj, prompt, max_new, temperature, seed):
    """Generate with sampling, then forward-pass with the same IDs to get
    clean (un-temperature-scaled) log-probs for SC/entropy."""
    torch.manual_seed(seed)
    p_ids = tok_obj(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    P = p_ids.shape[1]
    out = model.generate(
        p_ids,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        pad_token_id=tok_obj.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )
    seq = out.sequences[0]                  # (P + A,)
    A = seq.shape[0] - P
    if A <= 0:
        return [], ""

    # Trim trailing EOS / pad tokens
    eos_id = tok_obj.eos_token_id
    pad_id = tok_obj.pad_token_id
    end = A
    while end > 0 and seq[P + end - 1].item() in (eos_id, pad_id):
        end -= 1
    if end <= 0:
        return [], ""
    full_ids = seq[: P + end].unsqueeze(0)
    A = end

    # Clean forward pass — use the EXACT IDs that were sampled
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


def auroc(y_true, scores):
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    count = sum(1 for p in pos for n in neg if p > n)
    count += sum(0.5 for p in pos for n in neg if p == n)
    return count / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load 20 sampled questions
    sampled = json.loads(SAMPLED_FILE.read_text())
    questions = sampled["samples"]  # list of {qid, stratum, question, golds, ...}
    print(f"Loaded {len(questions)} sampled questions")

    # 2) Load existing CoT multi-sample dump
    cot_records = [json.loads(l) for l in EXISTING_COT_DUMP.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(cot_records)} existing CoT samples")

    print("Loading model...")
    tok, model = load_model()

    cot_prefix = build_fewshot_prompt(HotpotQA, N_SHOTS)
    direct_prefix = build_no_cot_prefix(HotpotQA, N_SHOTS)

    # ----------------------------------------------------------------- Phase 1
    print("\n--- Phase 1: re-score existing CoT samples with span-SC ---")
    cot_aug = []
    for rec in tqdm(cot_records):
        prompt = cot_prefix + HotpotQA.test_input(rec["question"])
        tokens = score_text_full(model, tok, prompt, rec["text"])
        ent_span = find_first_entity_span(rec["pred"], tokens)
        agg = span_aggs(tokens, ent_span)
        cot_aug.append({
            "qid": rec["qid"],
            "stratum": rec["stratum"],
            "sample_idx": rec["sample_idx"],
            "pred": rec["pred"],
            "em": rec["em"],
            "sc_orig": rec["sc"],
            "n_tok": len(tokens),
            "first_ent_aggs": agg,
        })
    (OUTPUT_DIR / "cot_aug.jsonl").write_text(
        "\n".join(json.dumps(r) for r in cot_aug) + "\n"
    )

    # ----------------------------------------------------------------- Phase 2
    print("\n--- Phase 2: generate no-CoT samples ---")
    direct_records = []
    for q in tqdm(questions):
        prompt = direct_prefix + no_cot_test_input(HotpotQA, q["question"])
        for s_idx in range(N_SAMPLES):
            seed = 1000 * hash(q["qid"]) % (2**31) + s_idx
            tokens, text = generate_with_scoring(
                model, tok, prompt, MAX_NEW_DIRECT, TEMPERATURE, seed
            )
            # the "answer" is the entire generation up to first newline / period
            # use whole gen as span (no CoT to dilute)
            # but extract pred via simple cleanup: split on \n or '.'
            pred = text.split("\n")[0].strip()
            for sep in [". ", "."]:
                if sep in pred:
                    pred = pred.split(sep)[0].strip()
                    break
            sc_out = score(pred, q["golds"])
            em, f1 = sc_out["em"], sc_out["f1"]

            # span = whole generation (no CoT)
            span_full = (0, len(tokens)) if tokens else None
            agg_full = span_aggs(tokens, span_full)
            # also try first-entity span if pred is in text
            ent_span = find_first_entity_span(pred, tokens) if pred else None
            agg_ent = span_aggs(tokens, ent_span)

            direct_records.append({
                "qid": q["qid"],
                "stratum": q["stratum"],
                "sample_idx": s_idx,
                "text": text,
                "pred": pred,
                "em": em,
                "f1": f1,
                "n_tok": len(tokens),
                "full_aggs": agg_full,
                "first_ent_aggs": agg_ent,
            })
    (OUTPUT_DIR / "direct_dump.jsonl").write_text(
        "\n".join(json.dumps(r) for r in direct_records) + "\n"
    )

    # ----------------------------------------------------------------- Phase 3
    print("\n--- Phase 3: AUROC comparison ---")
    AGG_KEYS = ["min_lp", "mean_lp", "min_sc", "mean_sc", "max_sc",
                "min_ent", "max_ent", "neg_max_ent"]

    def aurocs_for(records, agg_field):
        out = {}
        for k in AGG_KEYS:
            ys, ss = [], []
            for r in records:
                a = r.get(agg_field)
                if a is None:
                    continue
                v = a[k]
                if v is None or math.isnan(v) or math.isinf(v):
                    continue
                ys.append(r["em"])
                ss.append(v)
            out[k] = (auroc(ys, ss), len(ys))
        return out

    cot_aurocs = aurocs_for(cot_aug, "first_ent_aggs")
    direct_full_aurocs = aurocs_for(direct_records, "full_aggs")
    direct_ent_aurocs = aurocs_for(direct_records, "first_ent_aggs")

    # also sc_orig baseline for CoT
    sc_orig_auroc = auroc([r["em"] for r in cot_aug],
                          [r["sc_orig"] for r in cot_aug])

    # ----- Markdown summary -----
    L = ["# Exp-0b: no-CoT comparison + span-SC variants\n"]
    L.append(f"- Questions: {len(questions)} (HotpotQA stratified)")
    L.append(f"- Samples per question: {N_SAMPLES} (T={TEMPERATURE})")
    L.append(f"- CoT samples re-scored: {len(cot_aug)}")
    L.append(f"- No-CoT samples generated: {len(direct_records)}")
    L.append("")
    L.append("**Reference**: full-CoT `sc_orig` AUROC = "
             f"{sc_orig_auroc:.4f} (baseline from exp0)\n")

    L.append("## AUROC: CoT (span = first-entity in CoT)\n")
    L.append("| Metric | AUROC | n |")
    L.append("|---|---|---|")
    for k in AGG_KEYS:
        a, n = cot_aurocs[k]
        L.append(f"| `{k}` | {a:.4f} | {n} |")

    L.append("\n## AUROC: no-CoT (span = entire short answer)\n")
    L.append("| Metric | AUROC | n |")
    L.append("|---|---|---|")
    for k in AGG_KEYS:
        a, n = direct_full_aurocs[k]
        L.append(f"| `{k}` | {a:.4f} | {n} |")

    L.append("\n## AUROC: no-CoT (span = first-entity within direct answer)\n")
    L.append("| Metric | AUROC | n |")
    L.append("|---|---|---|")
    for k in AGG_KEYS:
        a, n = direct_ent_aurocs[k]
        L.append(f"| `{k}` | {a:.4f} | {n} |")

    L.append("\n## Side-by-side (key metrics)\n")
    L.append("| Metric | CoT first-ent | no-CoT full | no-CoT first-ent |")
    L.append("|---|---|---|---|")
    for k in ["min_lp", "min_sc", "mean_sc", "max_sc", "neg_max_ent"]:
        c = cot_aurocs[k][0]
        df = direct_full_aurocs[k][0]
        de = direct_ent_aurocs[k][0]
        L.append(f"| `{k}` | {c:.4f} | {df:.4f} | {de:.4f} |")

    L.append("\n## Interpretation guide\n")
    L.append("- **span-SC vs span-lp** (rows in CoT first-ent table): if `min_sc`/`max_sc` ≈ `min_lp`, then SC formula is fine — exp0's failure was just full-CoT averaging. If span-SC < span-lp, log-prob is the better primitive.")
    L.append("- **CoT vs no-CoT** (across columns): if no-CoT AUROC ≈ CoT AUROC, signal is independent of CoT structure. If no-CoT << CoT, CoT amplifies signal (likely via in-context repetition self-reinforcement on the answer phrase).")
    L.append("- `neg_max_ent` is `-max(entropy)` over span, so higher = more confident — directly comparable to other metrics.")

    summary = "\n".join(L)
    (OUTPUT_DIR / "exp0b_summary.md").write_text(summary)
    print("\n" + summary)
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
