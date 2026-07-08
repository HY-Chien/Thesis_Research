#!/usr/bin/env python3
"""Build the SEER (full dual + direct-context-union) result table across tau.

For tau in {0.3, 0.5, 1.0} reports, per dataset:
  - decompose-path retrieval rate (cost)
  - base Verifier EM/F1 (run_seek_dual)
  - type-aware replay EM/F1 (the canonical SEER headline post-processing)
tau=0.3 reuses the existing *_directctx runs; 0.5/1.0 use *_directctx_tau* runs.

Usage: python scripts/seer_tau_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seek.verifier import token_f1  # noqa: E402  (alias-aware, multi-gold)
from replay_typeaware_verifier import load_jsonl, choose_typeaware  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT.parent / "results"
DATASETS = ["hotpotqa", "2wiki", "musique", "popqa"]
TAUS = [0.3, 0.5, 0.8, 1.0]
DEVS = {
    "hotpotqa": "datasets/hotpotqa/raw/dev_first1000.jsonl",
    "2wiki": "datasets/2wiki/raw/dev_first1000.jsonl",
    "musique": "datasets/musique/raw/dev_first1000.jsonl",
    "popqa": "datasets/popqa/raw/dev.jsonl",
}
# replay defaults (validated to reproduce the SEER headline)
REPLAY_KW = dict(direct_entropy_tau=0.3, decomp_post_tau=0.8, margin=2.5)


def input_path(ds: str, tau: float) -> Path:
    if abs(tau - 0.3) < 1e-9:
        return RES / f"{ds}_1000_verify_v2_pathaware_directctx.jsonl"
    return RES / f"{ds}_directctx_tau{str(tau).replace('.', 'p')}.jsonl"


def load_golds(ds: str) -> dict[str, list[str]]:
    """id -> [answer, *answer_aliases] from the dev set (alias-aware F1/EM)."""
    golds: dict[str, list[str]] = {}
    for line in Path(ROOT.parent / DEVS[ds]).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        golds[str(r.get("id"))] = [r["answer"]] + (r.get("answer_aliases") or [])
    return golds


def stats(ds: str, path: Path):
    rows = load_jsonl(path)
    n = len(rows)
    retr = tot = 0
    for r in rows:
        for s in r.get("decomp", {}).get("steps", []):
            tot += 1
            retr += bool(s.get("retrieved"))
    rr = 100 * retr / tot if tot else 0.0

    # type-aware replay -> final per-row choice/pred (replay EM already alias-aware)
    choices = [choose_typeaware(r, **REPLAY_KW) for r in rows]
    golds = load_golds(ds)
    em_sum = f1_sum = 0.0
    for r, (choice, _) in zip(rows, choices):
        pred = (r.get("direct_pred") if choice == "direct" else r.get("decomp_pred")) or ""
        g = golds.get(str(r.get("id"))) or [r.get("gold") or ""]
        em_sum += int(r.get("direct_em", 0)) if choice == "direct" else int(r.get("decomp_em", 0))
        f1_sum += token_f1(pred, g)              # alias-aware, max over golds
    return {"n": n, "rr": rr, "rep_em": 100 * em_sum / n, "rep_f1": 100 * f1_sum / n}


def main() -> None:
    data = {}
    for ds in DATASETS:
        for tau in TAUS:
            p = input_path(ds, tau)
            if p.exists():
                data[(ds, tau)] = stats(ds, p)

    lines = ["# SEER (dual + original-question evidence union) — main result across retrieval-gate tau",
             "",
             "honest wiki-21M, 1000Q, mode C, max_ent, path-aware, no-evidence, "
             "direct_tau=0.4, decomp_tau=0.8, + type-aware verifier replay. "
             "EM/F1 = post-replay headline (the SEER (Ours) numbers). "
             "retr% = decompose-path retrieval rate (cost). tau=0.3 is the current final/canonical setting; higher tau values are cost-saving alternatives.",
             ""]
    # per-tau headline rows + averages
    for tau in TAUS:
        present = [ds for ds in DATASETS if (ds, tau) in data]
        if not present:
            continue
        lines.append(f"## tau = {tau}" + (" (final/canonical)" if tau == 0.3 else ""))
        lines.append("| dataset | retr% | EM | F1 |")
        lines.append("|---------|------:|---:|---:|")
        ems, f1s = [], []
        for ds in DATASETS:
            d = data.get((ds, tau))
            if not d:
                continue
            lines.append(f"| {ds} | {d['rr']:.2f} | {d['rep_em']:.2f} | {d['rep_f1']:.2f} |")
            ems.append(d["rep_em"]); f1s.append(d["rep_f1"])
        avg_rr = sum(data[(ds, tau)]["rr"] for ds in present) / len(present)
        lines.append(f"| **Avg** | **{avg_rr:.2f}** | **{sum(ems)/len(ems):.2f}** | **{sum(f1s)/len(f1s):.2f}** |")
        lines.append("")

    # compact comparison: EM/F1 by (dataset x tau)
    lines.append("## EM / F1 comparison (post-replay, alias-aware)")
    lines.append("| dataset | " + " | ".join(f"τ={t}" for t in TAUS) + " |")
    lines.append("|---------|" + "------|" * len(TAUS))
    for ds in DATASETS:
        cells = [f"{data[(ds, t)]['rep_em']:.2f} / {data[(ds, t)]['rep_f1']:.2f}"
                 if (ds, t) in data else "—" for t in TAUS]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    avg_cells = []
    for t in TAUS:
        es = [data[(ds, t)]["rep_em"] for ds in DATASETS if (ds, t) in data]
        fs = [data[(ds, t)]["rep_f1"] for ds in DATASETS if (ds, t) in data]
        avg_cells.append(f"**{sum(es)/len(es):.2f} / {sum(fs)/len(fs):.2f}**" if es else "—")
    lines.append("| **Avg** | " + " | ".join(avg_cells) + " |")
    lines.append("")

    # retrieval-rate (cost) by (dataset x tau)
    lines.append("## Retrieval rate per dataset (decompose-path, cost)")
    lines.append("| dataset | " + " | ".join(f"τ={t}" for t in TAUS) + " |")
    lines.append("|---------|" + "------|" * len(TAUS))
    for ds in DATASETS:
        cells = [f"{data[(ds, t)]['rr']:.2f}%" if (ds, t) in data else "—" for t in TAUS]
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    avg_cells = []
    for t in TAUS:
        rs = [data[(ds, t)]["rr"] for ds in DATASETS if (ds, t) in data]
        avg_cells.append(f"**{sum(rs)/len(rs):.2f}%**" if rs else "—")
    lines.append("| **Avg** | " + " | ".join(avg_cells) + " |")
    lines.append("")

    out = RES / "seer_tau_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
