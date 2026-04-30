"""
run_deeprag_sc.py

Use DeepRAG-8b checkpoint with SC-based retrieval gating instead of DeepRAG's
learned "Let" gate. Compares:
  --gate learned : original DeepRAG decision (model generates "Let" → retrieval)
  --gate sc      : SC threshold on intermediate answer confidence
  --gate hybrid  : retrieve if model says "Let" OR SC < threshold

Usage:
  python run_deeprag_sc.py --dataset hotpotqa --gate sc --sc_threshold 12.0
  python run_deeprag_sc.py --dataset hotpotqa --gate learned
  python run_deeprag_sc.py --dataset hotpotqa --gate hybrid --sc_threshold 12.0
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from dataset_eval import DATASET_CLASSES, score
from retriever_bm25 import BM25Retriever

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True,
                    choices=["hotpotqa", "2wikimultihopqa", "musique", "popqa"])
parser.add_argument("--gate", default="sc", choices=["learned", "sc", "hybrid"],
                    help="learned=DeepRAG's Let-token gate; sc=SC threshold; hybrid=either")
parser.add_argument("--sc_threshold", type=float, default=18.0,
                    help="SC below this → retrieve (only used with gate=sc or hybrid)")
parser.add_argument("--model_id", default="xinyan233333/DeepRAG-8b")
parser.add_argument("--model_tag", default="deeprag-8b")
parser.add_argument("--hf_cache", default="/home/S113065528/Adaptive-RAG/hf_cache")
parser.add_argument("--data_jsonl",
                    help="Input JSONL file (defaults to results/{dataset}_{model_tag}.jsonl)")
parser.add_argument("--output_dir",
                    default="/home/S113065528/Research/deeprag_compare/results")
parser.add_argument("--log_dir",
                    default="/home/S113065528/Research/deeprag_compare/logs")
parser.add_argument("--n_samples", type=int, default=None,
                    help="Limit to first N samples (for quick debug)")
parser.add_argument("--max_depth", type=int, default=6,
                    help="Max BFS sub-question depth")
parser.add_argument("--max_new_tokens_followup", type=int, default=80)
parser.add_argument("--max_new_tokens_answer", type=int, default=100)
parser.add_argument("--topk", type=int, default=3)
parser.add_argument("--es_index", default="wiki")
parser.add_argument("--device", default="cuda")
parser.add_argument("--prompt_style", default="auto",
                    choices=["auto", "deeprag", "vanilla"],
                    help="Final-answer prompt format. "
                         "deeprag = '<answer long>...</answer long><answer short>X</answer short>' "
                         "(only DeepRAG-trained models can produce this); "
                         "vanilla = 'the answer is X' (works for any LLM); "
                         "auto = deeprag if model_id contains 'deeprag', else vanilla.")
parser.add_argument("--use_fewshot", default="auto",
                    choices=["auto", "yes", "no"],
                    help="Include DeepRAG's 4-shot decomposition examples in the system "
                         "prompt (from official ICL-prompt/question_decompose.txt). "
                         "DeepRAG-8b is already SFT-tuned on this format, doesn't need shots; "
                         "vanilla / Qwen / non-DeepRAG models benefit from the shots. "
                         "auto = no if model_id contains 'deeprag', yes otherwise.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(args.log_dir, exist_ok=True)
LOG_FILE = os.path.join(
    args.log_dir,
    f"{args.dataset}_{args.model_tag}_{args.gate}_sc{args.sc_threshold}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

if args.gate == "learned":
    _GATE_TAG = "learned"
else:
    # Include threshold in filename so sweeps don't overwrite each other
    _GATE_TAG = f"{args.gate}_th{args.sc_threshold:g}"

OUT_FILE = os.path.join(
    args.output_dir,
    f"{args.dataset}_{args.model_tag}_{_GATE_TAG}.jsonl"
)

# ---------------------------------------------------------------------------
# DeepRAG ICL instruction (from gxy-gxy/DeepRAG/ICL-prompt/instruction.txt)
# ---------------------------------------------------------------------------

DEEPRAG_INSTRUCTION = (
    "Instruction: You are a helpful Retrieve-Augmented Generation (RAG) model. "
    "Your task is to answer questions by logically decomposing them into clear "
    "sub-questions and iteratively addressing each one. "
    'Use "Follow up:" to introduce each sub-question and "Intermediate answer:" '
    "to provide answers. For each sub-question, decide whether you can provide a "
    "direct answer or if additional information is required. If additional "
    'information is needed, state, "Let\'s search the question in Wikipedia." '
    "and then use the retrieved information to respond comprehensively. "
    "If a direct answer is possible, provide it immediately without searching."
)

# 4-shot question decomposition examples — copied verbatim from DeepRAG official repo:
# https://github.com/gxy-gxy/DeepRAG/blob/main/ICL-prompt/question_decompose.txt
# This is the prompt the DeepRAG authors used to ask Qwen-2.5-72B to decompose
# multi-hop questions during binary tree search data synthesis.
DEEPRAG_DECOMPOSE_FEWSHOT = """\
Decomposition guidelines:
1. If the given problem is already simple enough or there is no way to break it down, then no breakdown is needed.
2. Follow the output format and thought process in the Examples, and do not provide any additional explanatory notes.

Examples:
##########################
Question: Who was born first out of Walter Lambertus and Ingeborg Meyer-Rey?

Follow up: What is the birth year of Walter Lambertus?
Intermediate answer: 14 August 1955
Follow up: What is the birth year of Ingeborg Meyer-Rey?
Intermediate answer: 14 December 1920
So the final answer is: <answer long>Walter Lambertus born in 14 August 1955, Ingeborg Meyer-Rey born in 14 December 1920, Ingeborg Meyer-Rey was born first. </answer long><answer short>Ingeborg Meyer-Rey</answer short>

##########################
Question: When was the last time the death penalty was used in pa?

Follow up: When was the last time the death penalty was used in Pennsylvania?
Intermediate answer: July 6, 1999
So the final answer is: <answer long>The last time the death penalty was used in pa was on July 6, 1999.</answer long><answer short>July 6, 1999</answer short>

##########################
Question: Who won the war between ethiopia and italy?

Follow up: Who won the First Italo-Ethiopian War?
Intermediate answer: Ethiopia
Follow up: Who won the Second Italo-Ethiopian War?
Intermediate answer: Italy
So the final answer is: <answer long>The first war between Italy and Ethiopia took place from 1895 to 1896. This war was won by the Ethiopian army after the Battle of Adwa. The second war between the two nations was fought from October 1935 to February 1937 and was won by the Italians in the Battle of Maychew. </answer long><answer short>Ethiopia for the first war, Italy for the second war</answer short>

##########################
Question: What is the date of death of the director of film Spitfire (1934 Film)?

Follow up: Who is the director of film Spitfire (1934 Film)?
Intermediate answer: John Cromwell
Follow up: What is the date of death of John Cromwell?
Intermediate answer: September 26, 1979
So the final answer is: <answer long>The director of film Spitfire (1934 Film) is John Cromwell, and his date of death is September 26, 1979.</answer long><answer short>September 26, 1979</answer short>

##########################"""

# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

class StopStringCriteria(StoppingCriteria):
    """Stops generation when any of the stop_strings appears in the output."""

    def __init__(self, stop_strings, tokenizer, input_length):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.stopped_at = None

    def __call__(self, input_ids, scores, **kwargs):
        new_ids = input_ids[0][self.input_length:]
        if len(new_ids) == 0:
            return False
        # Decode only the tail to avoid O(n^2) cost
        tail = self.tokenizer.decode(new_ids[-30:], skip_special_tokens=False)
        for s in self.stop_strings:
            if s in tail:
                self.stopped_at = s
                return True
        return False


# ---------------------------------------------------------------------------
# Core generation with SC computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sc(model, tokenizer, prompt_text, stop_strings, max_new_tokens):
    """
    Generate text from prompt_text, stopping at stop_strings.
    Returns (clean_text, sc_value, stopped_at_string).
    sc_value is the mean per-token Self-Certainty over generated tokens.
    """
    V = float(model.config.vocab_size)
    log_V = math.log(V)

    enc = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )
    input_ids = enc.input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    input_length = input_ids.shape[1]

    stopper = StopStringCriteria(stop_strings, tokenizer, input_length)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        max_length=None,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        stopping_criteria=StoppingCriteriaList([stopper]),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs.sequences[0][input_length:]
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Remove the matched stop string from the end of the text
    stopped_at = stopper.stopped_at
    clean_text = full_text
    if stopped_at:
        # Find last occurrence (in case stop string appears earlier too)
        idx = full_text.rfind(stopped_at)
        if idx >= 0:
            clean_text = full_text[:idx].strip()

    # Compute SC: mean over tokens of (-sum(log_softmax) / V) - log(V)
    sc_values = []
    for step_logits in outputs.scores:
        lp = F.log_softmax(step_logits[0].float(), dim=-1)
        sc_values.append((-lp.sum() / V - log_V).item())

    sc = float(np.mean(sc_values)) if sc_values else 0.0
    return clean_text, sc, stopped_at


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, user_content, assistant_prefix):
    """
    Build a Llama-3 chat-template prompt that ends mid-assistant-turn
    so the model can continue generating from assistant_prefix.
    """
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_prefix},
        ],
        add_generation_prompt=False,
        tokenize=False,
    )
    # Remove trailing end-of-turn so generation continues
    for suffix in ["<|eot_id|>", "<|im_end|>"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def format_history(history):
    """Format reasoning history into DeepRAG's Follow up/Intermediate answer format."""
    parts = []
    for step in history:
        parts.append(f"Follow up: {step['follow_up'].strip()}")
        if "docs" in step:
            parts.append("Let's search the question in Wikipedia.")
            parts.append(f"Context:\n{step['docs']}")
        parts.append(f"Intermediate answer: {step['answer'].strip()}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# DeepRAG BFS reasoning
# ---------------------------------------------------------------------------

def _build_answer_prompt(tokenizer, user_content, history, followup_text, docs_str=None):
    """Build the prompt for generating an intermediate answer."""
    hist_text = format_history(history)
    if docs_str:
        if hist_text:
            prefix = (hist_text + f"\nFollow up: {followup_text}\n"
                      "Let's search the question in Wikipedia.\n"
                      f"Context:\n{docs_str}\nIntermediate answer:")
        else:
            prefix = (f"Follow up: {followup_text}\n"
                      "Let's search the question in Wikipedia.\n"
                      f"Context:\n{docs_str}\nIntermediate answer:")
    else:
        if hist_text:
            prefix = hist_text + f"\nFollow up: {followup_text}\nIntermediate answer:"
        else:
            prefix = f"Follow up: {followup_text}\nIntermediate answer:"
    return build_prompt(tokenizer, user_content, prefix)


def run_deeprag_reasoning(
    model, tokenizer, retriever, question, gate, sc_threshold, max_depth
):
    """
    Run DeepRAG-style BFS reasoning with either learned gate or SC gate.

    Returns (history, final_answer_text, n_retrievals).
    history: list of {follow_up, answer, docs (opt), sc (opt), retrieved_with}
    """
    # Resolve few-shot setting
    use_fewshot = args.use_fewshot
    if use_fewshot == "auto":
        use_fewshot = "no" if "deeprag" in args.model_id.lower() else "yes"

    if use_fewshot == "yes":
        terseness_rules = (
            "\n\nStrict format rules (must follow exactly):\n"
            "- Each \"Follow up:\" must be a concise question only — no meta-commentary, no \"Note:\", no \"However\", no explanation.\n"
            "- Each \"Intermediate answer:\" must be the minimal factual answer (an entity, date, or yes/no), no explanation.\n"
            "- The <answer short> tag must contain ONLY the minimal answer entity (e.g., \"yes\", \"Shirley Temple\", \"Greenwich Village\"). Do NOT write a sentence inside <answer short>."
        )
        user_content = (
            DEEPRAG_INSTRUCTION
            + "\n\n" + DEEPRAG_DECOMPOSE_FEWSHOT
            + terseness_rules
            + "\n\nNow answer the following question following the same format:\n\nQuestion: " + question
        )
    else:
        user_content = DEEPRAG_INSTRUCTION + "\nQuestion: " + question
    history = []
    n_retrievals = 0
    seen_followups = set()

    for depth in range(max_depth):
        hist_text = format_history(history)

        # ---- Generate next step ----
        # Force "Follow up:" prefix at EVERY step so the model doesn't drift
        # into meta-commentary like "Note: I will continue...". Termination
        # comes from: max_depth, empty followup, or loop detection.
        if not history:
            asst_prefix = "Follow up:"
        else:
            asst_prefix = hist_text + "\nFollow up:"

        # "Let's search" signals model's retrieval intent; stop on it for all gates.
        # learned/hybrid: use it as the decision signal.
        # sc: ignore the signal but stop here to keep followup text clean.
        # Extra stops "\nNote", "\nHowever" suppress Qwen-Instruct meta-commentary.
        followup_stop = ["Intermediate answer:", "So the final",
                         "Let's search", "<|eot_id|>", "<|im_end|>",
                         "\nNote", "\nHowever", "\n#####"]

        raw_next, _, stopped_at = generate_sc(
            model, tokenizer,
            build_prompt(tokenizer, user_content, asst_prefix),
            stop_strings=followup_stop,
            max_new_tokens=args.max_new_tokens_followup,
        )

        # ---- Termination checks ----
        if stopped_at == "So the final" or "So the final" in raw_next:
            break

        # Parse follow-up question
        followup_text = raw_next.strip()
        if history and followup_text.startswith("Follow up:"):
            followup_text = followup_text[len("Follow up:"):].strip()
        if not followup_text:
            break

        # Simple loop detection
        if followup_text in seen_followups:
            break
        seen_followups.add(followup_text)

        # ---- Gate decision ----
        # "Let's search" in stopped_at means model signals retrieval
        wants_retrieval_learned = (stopped_at == "Let's search")

        parametric_text = None
        sc_val = None

        if gate == "learned":
            use_retrieval = wants_retrieval_learned

        elif gate == "sc":
            # Always generate intermediate answer parametrically, then check SC
            answer_prompt = _build_answer_prompt(tokenizer, user_content, history, followup_text)
            parametric_text, sc_val, _ = generate_sc(
                model, tokenizer, answer_prompt,
                stop_strings=["Follow up:", "So the final", "<|eot_id|>", "<|im_end|>"],
                max_new_tokens=args.max_new_tokens_answer,
            )
            parametric_text = parametric_text.strip()
            use_retrieval = (sc_val < sc_threshold)

        elif gate == "hybrid":
            if wants_retrieval_learned:
                use_retrieval = True
            else:
                answer_prompt = _build_answer_prompt(tokenizer, user_content, history, followup_text)
                parametric_text, sc_val, _ = generate_sc(
                    model, tokenizer, answer_prompt,
                    stop_strings=["Follow up:", "So the final", "<|eot_id|>", "<|im_end|>"],
                    max_new_tokens=args.max_new_tokens_answer,
                )
                parametric_text = parametric_text.strip()
                use_retrieval = (sc_val < sc_threshold)

        # ---- Execute chosen path ----
        if use_retrieval:
            docs_raw = retriever.search(followup_text, topk=args.topk)
            docs_str = BM25Retriever.format_context(docs_raw)
            rag_prompt = _build_answer_prompt(tokenizer, user_content, history, followup_text, docs_str)
            answer_text, _, _ = generate_sc(
                model, tokenizer, rag_prompt,
                stop_strings=["Follow up:", "So the final", "<|eot_id|>", "<|im_end|>"],
                max_new_tokens=args.max_new_tokens_answer,
            )
            answer_text = answer_text.strip()
            n_retrievals += 1
            step = {
                "follow_up": followup_text,
                "answer": answer_text,
                "docs": docs_str,
                "retrieved_with": "retrieval",
            }
            if sc_val is not None:
                step["sc"] = sc_val
        else:
            # Parametric path
            if parametric_text is None:
                answer_prompt = _build_answer_prompt(tokenizer, user_content, history, followup_text)
                parametric_text, _, _ = generate_sc(
                    model, tokenizer, answer_prompt,
                    stop_strings=["Follow up:", "So the final", "<|eot_id|>", "<|im_end|>"],
                    max_new_tokens=args.max_new_tokens_answer,
                )
                parametric_text = parametric_text.strip()
            answer_text = parametric_text
            step = {
                "follow_up": followup_text,
                "answer": answer_text,
                "retrieved_with": "parametric",
            }
            if sc_val is not None:
                step["sc"] = sc_val

        history.append(step)

    # ---- Generate final answer ----
    # Resolve prompt style: if model is DeepRAG-trained or we provided 4-shot examples
    # that teach the <answer long>/<answer short> format, prime with deeprag style.
    # Otherwise (vanilla model, no shots), prime with plain-text style.
    prompt_style = args.prompt_style
    if prompt_style == "auto":
        if "deeprag" in args.model_id.lower() or use_fewshot == "yes":
            prompt_style = "deeprag"
        else:
            prompt_style = "vanilla"

    hist_text = format_history(history)
    if prompt_style == "deeprag":
        # Prime with "<answer long>" so DeepRAG-trained model emits its native format:
        # <answer long>...explanation...</answer long><answer short>X</answer short>
        if not history:
            asst_final = "So the final answer is: <answer long>"
        else:
            asst_final = hist_text + "\nSo the final answer is: <answer long>"
    else:
        # Vanilla format: works for any instruction-tuned model.
        # The extract_final_answer regex picks up "the answer is X".
        if not history:
            asst_final = "So the final answer is: the answer is "
        else:
            asst_final = hist_text + "\nSo the final answer is: the answer is "

    final_prompt = build_prompt(tokenizer, user_content, asst_final)
    final_text, _, _ = generate_sc(
        model, tokenizer, final_prompt,
        stop_strings=["<|eot_id|>", "<|im_end|>", "Follow up:", "Question:"],
        max_new_tokens=200,
    )
    final_text = final_text.strip()

    return history, final_text, n_retrievals


# ---------------------------------------------------------------------------
# Answer extraction (handles both DeepRAG and DRAGIN formats)
# ---------------------------------------------------------------------------

def extract_final_answer(final_text, history):
    """
    Extract the short answer from the final answer generation.
    Priority: <answer short> > <answer long> first sentence > "the answer is" > raw text.
    """
    text = final_text

    # Highest priority: DeepRAG <answer short> tag
    if "<answer short>" in text:
        text = text.split("<answer short>")[1].split("</answer short>")[0].strip()
        for tok in ["<|eot_id|>", "<|im_end|>", "</s>"]:
            text = text.replace(tok, "")
        text = text.strip()
        # Verbose-answer cleanup (Qwen-Instruct tends to write full sentences here):
        # If output starts with yes/no, take just that token (matches gold format).
        low = text.lower().lstrip(".,! ")
        if low.startswith("yes"):
            return "yes"
        if low.startswith("no"):
            return "no"
        # Strip leading "the answer is" / "it is" prefixes if present
        for prefix in ["the answer is ", "it is ", "the final answer is "]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break
        # Take first sentence only (cut at period/comma followed by clause)
        for sep in [". ", "; ", " — "]:
            if sep in text:
                text = text.split(sep)[0].strip()
                break
        return text.strip().rstrip(".,!?")

    # <answer long> tag — take first sentence
    if "<answer long>" in text:
        text = text.split("<answer long>")[1]
        if "</answer long>" in text:
            text = text.split("</answer long>")[0]
        text = text.strip().split(".")[0].strip()

    # Standard DRAGIN/FLARE "the answer is" format
    if "the answer is" in text.lower():
        idx = text.lower().find("the answer is") + len("the answer is")
        text = text[idx:].strip().split("\n")[0].split(".")[0].strip()

    # Strip special tokens
    for tok in ["<|eot_id|>", "<|im_end|>", "</s>"]:
        text = text.replace(tok, "")

    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Load questions from existing JSONL ----
    if args.data_jsonl:
        in_file = args.data_jsonl
    else:
        in_file = os.path.join(
            args.output_dir,
            f"{args.dataset}_meta-llama-3-8b-instruct.jsonl"
        )

    logging.info(f"Loading questions from {in_file}")
    records = [json.loads(l) for l in open(in_file) if l.strip()]
    if args.n_samples:
        records = records[: args.n_samples]
    logging.info(f"  {len(records)} questions")

    ds_cls = DATASET_CLASSES[args.dataset]

    # ---- Load model ----
    logging.info(f"Loading model {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, cache_dir=args.hf_cache
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.hf_cache,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    logging.info("Model loaded.")

    # ---- Load retriever ----
    retriever = BM25Retriever(index=args.es_index)
    _ = retriever.search("test", topk=1)
    logging.info("Retriever OK.")

    logging.info(
        f"Gate={args.gate}  SC_threshold={args.sc_threshold}  "
        f"Dataset={args.dataset}"
    )

    # ---- Run inference ----
    results = []
    n_correct = n_total = 0
    n_retrieved_total = 0
    t0 = time.time()

    for rec in tqdm(records, desc=f"DeepRAG-SC [{args.dataset}]"):
        question = rec["question"]
        golds = rec["golds"] if isinstance(rec["golds"], list) else [rec["golds"]]

        try:
            history, final_text, n_retrievals = run_deeprag_reasoning(
                model, tokenizer, retriever,
                question=question,
                gate=args.gate,
                sc_threshold=args.sc_threshold,
                max_depth=args.max_depth,
            )
        except Exception as e:
            logging.warning(f"Error on '{question[:60]}': {e}")
            history, final_text, n_retrievals = [], "", 0

        pred = extract_final_answer(final_text, history)
        sc_result = score(pred, golds)

        n_total += 1
        n_correct += sc_result["em"]
        n_retrieved_total += n_retrievals

        result = {
            "question": question,
            "golds": golds,
            "pred": pred,
            "em": sc_result["em"],
            "f1": sc_result["f1"],
            "n_retrievals": n_retrievals,
            "final_text": final_text,
            "history": history,
        }
        results.append(result)

    elapsed = time.time() - t0
    em = n_correct / n_total if n_total else 0
    avg_retr = n_retrieved_total / n_total if n_total else 0

    logging.info(f"\n=== Results ===")
    logging.info(f"Gate={args.gate}  SC_threshold={args.sc_threshold}")
    logging.info(f"EM={em*100:.2f}%  N={n_total}  AvgRetrievals={avg_retr:.2f}")
    logging.info(f"Elapsed={elapsed/60:.1f} min")

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "dataset": args.dataset,
        "model": args.model_id,
        "gate": args.gate,
        "sc_threshold": args.sc_threshold,
        "n_questions": n_total,
        "em": em,
        "avg_retrievals": avg_retr,
        "elapsed_min": elapsed / 60,
    }
    summary_file = OUT_FILE.replace(".jsonl", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Saved → {OUT_FILE}")
    logging.info(f"Summary → {summary_file}")


if __name__ == "__main__":
    main()
