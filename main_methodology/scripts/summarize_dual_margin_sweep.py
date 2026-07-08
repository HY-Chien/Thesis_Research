#!/usr/bin/env python3
"""Aggregate the dual-path margin/TARG sweep (scripts/run_dual_margin_sweep.sh) into one table.

Reads whatever `{ds}_dual_{signal}_tau*.log` / `.jsonl` pairs already exist under
scripts/ and results/ for datasets in {hotpotqa, 2wiki, musique, popqa} and signals in
{targ_unc, mean_margin, max_margin, maqa_margin, min_lp}. Configs whose log lacks a
"Verifier EM" line (still running / never started) are skipped, mirroring the
exclusion rule used by the original results/dual_margin_sweep_completed_summary.md.

Retrieval rate is computed the same way as scripts/seer_tau_table.py: count
decompose-path steps with `retrieved: true` in the *.jsonl trace, divided by the
total decompose-path step count for that dataset.

Usage: python scripts/summarize_dual_margin_sweep.py
Writes: results/dual_margin_sweep_completed_summary.md, .tsv
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"

DATASETS = ["hotpotqa", "2wiki", "musique", "popqa"]
SIGNALS = ["targ_unc", "mean_margin", "max_margin", "maqa_margin", "min_lp"]

VERIFIER_EM_RE = re.compile(r"Verifier EM:\s*([\d.]+)%")
VERIFIER_F1_RE = re.compile(r"Verifier F1:\s*([\d.]+)%")
CHOICES_RE = re.compile(r"choices:\s*decomp=(\d+), direct=(\d+)")


def tag_tau(tau: str) -> str:
    return tau.replace(".", "p")


def find_configs(ds: str, signal: str) -> list[str]:
    """Return the tau tags (e.g. '0p30') that have a log file for this ds/signal."""
    pattern = f"{ds}_dual_{signal}_tau*.log"
    tags = []
    for log_path in sorted(SCRIPTS.glob(pattern)):
        m = re.search(rf"{re.escape(ds)}_dual_{re.escape(signal)}_tau(.+)\.log$", log_path.name)
        if m:
            tags.append(m.group(1))
    return tags


def retrieval_rate(jsonl_path: Path) -> tuple[float, int]:
    """(retrieval_rate_pct, total_decompose_steps) over the decompose-path trace."""
    total = retrieved = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for step in row.get("decomp", {}).get("steps", []):
                total += 1
                retrieved += bool(step.get("retrieved"))
    rate = 100 * retrieved / total if total else 0.0
    return rate, total


def load_config(ds: str, signal: str, tag: str) -> dict | None:
    log_path = SCRIPTS / f"{ds}_dual_{signal}_tau{tag}.log"
    jsonl_path = RESULTS / f"{ds}_dual_{signal}_tau{tag}.jsonl"
    text = log_path.read_text(encoding="utf-8", errors="replace")

    em_m = VERIFIER_EM_RE.search(text)
    f1_m = VERIFIER_F1_RE.search(text)
    if not em_m or not f1_m:
        return None  # incomplete run, skip (matches original summary's exclusion rule)

    choices_m = CHOICES_RE.search(text)
    choice_decomp, choice_direct = (int(choices_m.group(1)), int(choices_m.group(2))) if choices_m else (0, 0)
    n = choice_decomp + choice_direct

    if not jsonl_path.exists():
        return None
    rate, steps = retrieval_rate(jsonl_path)

    tau = float(tag.replace("p", "."))
    return {
        "dataset": ds,
        "signal": signal,
        "tau": tau,
        "n": n,
        "em": float(em_m.group(1)),
        "f1": float(f1_m.group(1)),
        "retrieval_rate": rate,
        "steps": steps,
        "choice_direct": 100 * choice_direct / n if n else 0.0,
        "choice_decomp": 100 * choice_decomp / n if n else 0.0,
    }


def collect() -> list[dict]:
    rows = []
    for ds in DATASETS:
        for signal in SIGNALS:
            for tag in find_configs(ds, signal):
                cfg = load_config(ds, signal, tag)
                if cfg is not None:
                    rows.append(cfg)
    rows.sort(key=lambda r: (r["dataset"], r["signal"], r["tau"]))
    return rows


def write_tsv(rows: list[dict], path: Path) -> None:
    cols = ["dataset", "signal", "tau", "n", "em", "f1", "retrieval_rate", "steps",
            "choice_direct", "choice_decomp"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join([
            r["dataset"], r["signal"], f"{r['tau']:.3f}", str(r["n"]),
            f"{r['em']:.2f}", f"{r['f1']:.2f}", f"{r['retrieval_rate']:.2f}",
            str(r["steps"]), f"{r['choice_direct']:.2f}", f"{r['choice_decomp']:.2f}",
        ]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["# Dual Margin Sweep Completed Summary", "",
              "Only configs whose per-run log contains `Verifier EM` are included. "
              "Currently running/partial JSONL files are excluded.", ""]

    # macro over datasets that have completed runs for a given (signal, tau)
    by_key: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["signal"], r["tau"]), []).append(r)

    lines.append("## Macro over completed datasets")
    lines.append("")
    lines.append("| signal | tau | datasets | EM | F1 | retrieval rate |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for signal in SIGNALS:
        taus = sorted({tau for (sig, tau) in by_key if sig == signal})
        for tau in taus:
            group = by_key[(signal, tau)]
            ds_list = ",".join(r["dataset"] for r in group)
            em = sum(r["em"] for r in group) / len(group)
            f1 = sum(r["f1"] for r in group) / len(group)
            rr = sum(r["retrieval_rate"] for r in group) / len(group)
            lines.append(f"| {signal} | {tau:.3f} | {ds_list} | {em:.1f} | {f1:.1f} | {rr:.1f} |")
    lines.append("")

    lines.append("## Best per dataset and signal")
    lines.append("")
    lines.append("| dataset | signal | tau | EM | F1 | retrieval rate | n |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for ds in DATASETS:
        for signal in SIGNALS:
            group = [r for r in rows if r["dataset"] == ds and r["signal"] == signal]
            if not group:
                continue
            best = max(group, key=lambda r: r["em"])
            lines.append(f"| {best['dataset']} | {best['signal']} | {best['tau']:.3f} | "
                          f"{best['em']:.1f} | {best['f1']:.1f} | {best['retrieval_rate']:.1f} | {best['n']} |")
    lines.append("")

    # excluded/partial: any log present without a usable row
    excluded = []
    for ds in DATASETS:
        for signal in SIGNALS:
            for tag in find_configs(ds, signal):
                jsonl_name = f"{ds}_dual_{signal}_tau{tag}.jsonl"
                if load_config(ds, signal, tag) is None and (SCRIPTS / f"{ds}_dual_{signal}_tau{tag}.log").exists():
                    excluded.append(jsonl_name)
    if excluded:
        lines.append("## Excluded partial files")
        lines.append("")
        for name in excluded:
            lines.append(f"- `{name}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = collect()
    if not rows:
        print("No completed configs found.")
        return
    write_tsv(rows, RESULTS / "dual_margin_sweep_completed_summary.tsv")
    write_markdown(rows, RESULTS / "dual_margin_sweep_completed_summary.md")
    print(f"{len(rows)} completed configs across "
          f"{sorted(set(r['dataset'] for r in rows))} / {sorted(set(r['signal'] for r in rows))}")
    print(f"wrote {RESULTS / 'dual_margin_sweep_completed_summary.tsv'}")
    print(f"wrote {RESULTS / 'dual_margin_sweep_completed_summary.md'}")


if __name__ == "__main__":
    main()
