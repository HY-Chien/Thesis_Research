"""
fig_token_sc_demo.py
=====================
Generate a multi-example token-level SC trajectory figure for thesis presentation.

Key insight (corrected):
    SC = -1/V * Σ log P(v|context) - log(V)
      → SC = 0  when distribution is UNIFORM  (most uncertain)
      → SC high when distribution is PEAKED   (most certain)

    So: HIGH SC = certain,  LOW SC = uncertain.

    Function words (the, of, a) are PREDICTABLE → model is certain → HIGH SC.
    Content words (named entities, numbers)     → model is uncertain → LOW SC.

    The problem with token-level SC triggering:
      SC fluctuates WILDLY between adjacent tokens (can differ by >10 units).
      A content word might be surrounded by certain function words, making the
      sentence-level picture misleading if we just look at individual tokens.
      Sentence-level mean SC smooths this volatility into a stable signal.

Output:
    figures/fig_token_sc_demo.png   — grid of 6 examples
    figures/fig_token_sc_single.png — single example with zoomed panel + CoT text
"""

import os
import sys
sys.path.insert(0, "/home/S113065528/Adaptive-RAG/retriever_server")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import math
import re
import textwrap

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_ID  = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE  = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN  = os.environ.get("HF_TOKEN", None)
EXP7_FILE = "/home/S113065528/Adaptive-RAG/experiment/self_certainty/results/exp7_cot_nq_results.json"
OUT_DIR   = "/home/S113065528/Adaptive-RAG/experiment/self_certainty/figures"
OUT_GRID  = os.path.join(OUT_DIR, "fig_token_sc_demo.png")
OUT_SINGLE = os.path.join(OUT_DIR, "fig_token_sc_single.png")
os.makedirs(OUT_DIR, exist_ok=True)

FUNCTION_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "and", "or", "but", "is", "was", "are", "were",
    "be", "been", "being", "that", "this", "it", "its", "as", "he",
    "she", "they", "we", "i", "his", "her", "their", "our", "my",
    "which", "who", "what", "when", "where", "how", "had", "has",
    "have", "will", "would", "could", "should", "may", "might",
    "not", "no", "if", "so", "do", "did", "does", "then", "than",
    "also", "up", "out", "about", "into", "through", "during",
    "before", "after", "between", "each", "more", "most", "other",
}

def is_function_word(tok_str):
    clean = tok_str.strip().lstrip("▁Ġ ").lower().strip(".,;:!?\"'()")
    return clean in FUNCTION_WORDS or len(clean) <= 1


# ── pick examples ───────────────────────────────────────────────────────────
def pick_examples(exp7_path, n=6, token_range=(70, 140)):
    with open(exp7_path, encoding="utf-8") as f:
        data = json.load(f)
    chosen = []
    for item in data:
        if not item.get("is_correct"):
            continue
        cot = item.get("cot", "")
        nt = item.get("n_tokens", len(cot.split()))
        if token_range[0] <= nt <= token_range[1]:
            if "Final Answer" in cot or "final answer" in cot.lower():
                chosen.append(item)
        if len(chosen) >= n:
            break
    # fallback: fill with any correct items
    if len(chosen) < n:
        for item in data:
            if item.get("is_correct") and item not in chosen:
                chosen.append(item)
            if len(chosen) >= n:
                break
    return chosen[:n]


# ── SC forward pass ────────────────────────────────────────────────────────
@torch.no_grad()
def compute_token_sc(model, tokenizer, prompt_text, response_text):
    device = model.device
    V      = model.config.vocab_size
    log_V  = math.log(V)

    prompt_ids   = tokenizer(prompt_text,   return_tensors="pt",
                             add_special_tokens=False).input_ids.to(device)
    response_ids = tokenizer(response_text, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(device)

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits[0].float()   # (L, V)

    prompt_len = prompt_ids.shape[1]
    scs, tokens = [], []

    for t in range(response_ids.shape[1]):
        pos = prompt_len - 1 + t
        if pos >= logits.shape[0]:
            break
        log_prob = F.log_softmax(logits[pos], dim=-1)
        sc = (-log_prob.sum() / V - log_V).item()
        scs.append(sc)
        tokens.append(tokenizer.decode([response_ids[0, t].item()]))

    return tokens, scs


def build_prompt(tokenizer, question):
    msgs = [
        {"role": "system", "content":
         "Answer the given question. Always conduct and show your "
         "step-by-step reasoning first, then give the final answer. "
         "End with a line: 'Final Answer: <your answer>'."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_sentence_boundaries(tokens):
    boundaries = []
    running = ""
    for i, tok in enumerate(tokens):
        running += tok
        if re.search(r'[.!?]\s*$|\n', running):
            boundaries.append(i)
            running = ""
    return boundaries


def get_sentence_means(scs, boundaries, n_tokens):
    means = []
    prev = 0
    for bnd in boundaries + [n_tokens - 1]:
        seg = scs[prev:bnd + 1]
        if seg:
            means.append((prev, bnd, float(np.mean(seg))))
        prev = bnd + 1
    return means


# ── Figure 1: 6-example grid ───────────────────────────────────────────────
def plot_grid(examples_data, out_path, sc_threshold=12.0):
    """
    examples_data: list of dict with keys: question, answer, tokens, scs
    """
    n = len(examples_data)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.5 * nrows))
    fig.suptitle(
        "Token-level SC Trajectories across Multiple Examples\n"
        "HIGH SC = model is certain  |  LOW SC = model is uncertain\n"
        "Blue line = token SC  |  Orange bars = sentence mean SC  |  Red dashed = threshold",
        fontsize=12, fontweight="bold", y=1.01
    )
    axes = axes.flatten()

    for idx, ex in enumerate(examples_data):
        ax = axes[idx]
        tokens = ex["tokens"][:100]
        scs    = ex["scs"][:100]
        xs     = list(range(len(tokens)))

        boundaries  = get_sentence_boundaries(tokens)
        sent_means  = get_sentence_means(scs, boundaries, len(tokens))
        func_mask   = [is_function_word(t) for t in tokens]

        # Threshold zone
        ax.axhspan(0, sc_threshold, alpha=0.07, color="red", zorder=0)
        ax.axhline(sc_threshold, color="red", linewidth=1.0,
                   linestyle="--", zorder=2, alpha=0.8)

        # Sentence mean bars
        for s, e, m in sent_means:
            ax.barh(y=m, width=(e - s + 1), left=s,
                    height=max(scs) * 0.03 + 0.3,
                    color="darkorange", alpha=0.35, zorder=1)
            ax.text((s + e) / 2, m + 0.4, f"{m:.1f}",
                    ha="center", fontsize=6.5, color="darkorange", zorder=3)

        # Sentence boundary lines
        for bnd in boundaries:
            ax.axvline(bnd + 0.5, color="gray", linewidth=0.6,
                       linestyle=":", alpha=0.5)

        # Token dots
        for i, (x, sc, is_func) in enumerate(zip(xs, scs, func_mask)):
            # HIGH SC = function word (predictable) → grey
            # LOW SC  = content word (uncertain)    → blue
            color = "#AAAAAA" if is_func else "#3A72C0"
            ax.scatter(x, sc, color=color, s=15, zorder=4, linewidths=0)

        # Line
        ax.plot(xs, scs, color="#BBBBBB", linewidth=0.7, zorder=3)

        # Annotate large drops (volatility illustration)
        for i in range(1, len(scs)):
            drop = scs[i - 1] - scs[i]
            if drop > 8:
                ax.annotate("", xy=(i, scs[i]),
                            xytext=(i - 1, scs[i - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="red",
                                            lw=1.2),
                            zorder=5)

        # Title
        q_short = ex["question"][:60] + ("…" if len(ex["question"]) > 60 else "")
        ax.set_title(f"Q: {q_short}\nA: {ex['answer']}", fontsize=7.5, loc="left")
        ax.set_xlabel("Token index", fontsize=8)
        ax.set_ylabel("SC", fontsize=8)
        ax.set_ylim(max(0, min(scs) - 1), max(scs) + 2)
        ax.tick_params(labelsize=7)

    # Hide empty subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2: single example — full + zoomed + CoT text ───────────────────
def plot_single(ex, out_path, sc_threshold=12.0, max_display=110):
    tokens = ex["tokens"][:max_display]
    scs    = ex["scs"][:max_display]
    xs     = list(range(len(tokens)))
    func_mask  = [is_function_word(t) for t in tokens]
    boundaries = get_sentence_boundaries(tokens)
    sent_means = get_sentence_means(scs, boundaries, len(tokens))

    cot_display = ex["cot"].replace("\n", " ").strip()
    cot_wrapped = "\n".join(textwrap.wrap(cot_display, width=155))

    fig = plt.figure(figsize=(19, 13))
    gs  = fig.add_gridspec(3, 1, hspace=0.6, height_ratios=[0.5, 1.4, 1.0])
    ax_text = fig.add_subplot(gs[0])
    ax1     = fig.add_subplot(gs[1])
    ax2     = fig.add_subplot(gs[2])

    fig.suptitle(
        "Token-level SC Trajectory — Why Sentence-level Aggregation is Needed",
        fontsize=13, fontweight="bold", y=0.99
    )

    # ── Text box ──
    ax_text.axis("off")
    ax_text.text(0.0, 1.0,
        f"Q: {ex['question']}\n\nModel output:\n{cot_wrapped}",
        transform=ax_text.transAxes,
        fontsize=8.5, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F7F7",
                  edgecolor="#CCCCCC", linewidth=1)
    )

    def draw(ax, xs, scs, tokens, func_mask, boundaries, sent_means,
             show_labels, zoom_window=None):
        if zoom_window:
            a, b = zoom_window
            xs        = list(range(b - a))
            scs       = scs[a:b]
            tokens    = tokens[a:b]
            func_mask = func_mask[a:b]
            bounds_l  = [s - a for s in boundaries if a <= s < b]
            means_l   = [(s - a, e - a, m) for s, e, m in sent_means if a <= s < b]
        else:
            bounds_l = boundaries
            means_l  = sent_means

        ax.axhspan(0, sc_threshold, alpha=0.07, color="red", zorder=0)
        ax.axhline(sc_threshold, color="red", linewidth=1.2, linestyle="--",
                   label=f"SC threshold = {sc_threshold:.0f}", zorder=2)

        for s, e, m in means_l:
            ax.barh(y=m, width=(e - s + 1), left=s, height=0.5,
                    color="darkorange", alpha=0.30, zorder=1)
            ax.text((s + e) / 2, m + 0.35, f"μ={m:.1f}",
                    ha="center", fontsize=7, color="darkorange", zorder=3)

        for bnd in bounds_l:
            ax.axvline(bnd + 0.5, color="gray", linewidth=0.7,
                       linestyle=":", alpha=0.6)

        for i, (x, sc, is_func) in enumerate(zip(xs, scs, func_mask)):
            color = "#AAAAAA" if is_func else "#3A72C0"
            ax.scatter(x, sc, color=color, s=28, zorder=4, linewidths=0)

        ax.plot(xs, scs, color="#CCCCCC", linewidth=0.9, zorder=3)

        # Highlight large drops with red arrows
        for i in range(1, len(scs)):
            if scs[i - 1] - scs[i] > 8:
                ax.annotate("", xy=(i, scs[i]),
                            xytext=(i - 1, scs[i - 1]),
                            arrowprops=dict(arrowstyle="-|>", color="red", lw=1.3),
                            zorder=5)

        if show_labels:
            mean_sc = float(np.mean(scs))
            for i, (x, sc, tok, is_func) in enumerate(zip(xs, scs, tokens, func_mask)):
                display = tok.strip().replace("▁", "").replace("Ġ", "")
                if not display:
                    continue
                color  = "#888888" if is_func else "#1A3A6B"
                offset = 7 if sc >= mean_sc else -14
                ax.annotate(
                    display, (x, sc),
                    textcoords="offset points", xytext=(0, offset),
                    ha="center", fontsize=7.5, color=color,
                    rotation=50 if len(display) > 4 else 0,
                    fontweight="bold" if not is_func else "normal",
                )

        ax.set_xlim(-1, len(xs))
        ax.set_ylim(max(0, min(scs) - 1), max(scs) + 2.5)
        ax.set_ylabel("Self-Certainty (SC)", fontsize=10)

    # ── Panel 1: full trace ──
    draw(ax1, xs, scs, tokens, func_mask, boundaries, sent_means,
         show_labels=False)
    ax1.set_xlabel("Token index", fontsize=10)
    ax1.set_title(
        "Full CoT — token-level SC  (no per-token labels; see zoomed panel below)\n"
        "Grey dots = function words (predictable → HIGH SC)  |  "
        "Blue dots = content words (uncertain → LOW SC)\n"
        "Orange bars = sentence-mean SC  |  Red arrows = large SC drops (>8 units)",
        fontsize=9, loc="left"
    )
    ax1.legend(handles=[
        mpatches.Patch(color="#3A72C0", label="Content word (uncertain, lower SC)"),
        mpatches.Patch(color="#AAAAAA", label="Function word (predictable, higher SC)"),
        mpatches.Patch(color="darkorange", alpha=0.5, label="Sentence mean SC"),
    ], loc="upper right", fontsize=8)

    # ── Panel 2: zoomed ──
    zoom_start = max(0, len(tokens) // 4)
    zoom_end   = min(len(tokens), zoom_start + 40)
    draw(ax2, xs, scs, tokens, func_mask, boundaries, sent_means,
         show_labels=True, zoom_window=(zoom_start, zoom_end))
    ax2.set_xlabel("Token index (zoomed)", fontsize=10)
    ax2.set_title(
        f"Zoomed (tokens {zoom_start}–{zoom_end}) — individual token labels\n"
        "Adjacent tokens can differ by >10 SC units → token-level threshold is unstable.\n"
        "Sentence-level mean SC (orange) is far more stable and semantically meaningful.",
        fontsize=9, loc="left"
    )
    ax2.legend(handles=[
        mpatches.Patch(color="#3A72C0", label="Content word"),
        mpatches.Patch(color="#AAAAAA", label="Function word"),
        plt.Line2D([0], [0], color="red", linestyle="--",
                   label=f"Threshold={sc_threshold:.0f}"),
    ], loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_idx", type=int, default=1,
                        help="Which example to use for single figure (0-based, default: 1)")
    parser.add_argument("--n_examples", type=int, default=8,
                        help="How many examples to load (default: 8)")
    args = parser.parse_args()

    print("Loading examples from exp7...")
    items = pick_examples(EXP7_FILE, n=args.n_examples, token_range=(70, 140))
    print(f"  Selected {len(items)} examples")

    idx = min(args.example_idx, len(items) - 1)
    print(f"  Will draw single figure for example #{idx}: {items[idx]['question'][:70]}")

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, token=HF_TOKEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, device_map="cuda",
        torch_dtype=torch.bfloat16, token=HF_TOKEN, local_files_only=True)
    model.eval()
    print("Model loaded.\n")

    # Only compute SC for the chosen example (faster)
    item   = items[idx]
    q      = item["question"]
    cot    = item["cot"]
    ans    = item.get("final_answer", "")
    prompt = build_prompt(tokenizer, q)

    print(f"Computing SC for: {q[:70]}")
    tokens, scs = compute_token_sc(model, tokenizer, prompt, cot)
    print(f"  tokens={len(tokens)}  SC range=[{min(scs):.1f}, {max(scs):.1f}]  "
          f"mean={np.mean(scs):.1f}")

    ex = {"question": q, "answer": ans, "cot": cot, "tokens": tokens, "scs": scs}

    print("Drawing single-example figure...")
    plot_single(ex, OUT_SINGLE, sc_threshold=12.0)

    print("Done.")


if __name__ == "__main__":
    main()
