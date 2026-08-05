"""Summarise the tl_sum-gated prompt-fix-A rerun (rerun_tlsum_fixA.py).

Reads results/tlsum_fixA_{ds}_shard{0,1}.jsonl and reports per dataset:
  EM / F1 of the new run, canonical replay EM as stored in the same rows,
  prompt-fix-A (max_ent gate) EM/F1 from prompt_fix_{ds}_shard*.jsonl,
  step retrieve-rate under the tl_sum gate vs what max_ent tau=0.3 would do,
  and verifier direct/decomp choice rates.

Output: results/tlsum_fixA_summary.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["hotpotqa", "2wiki", "musique", "popqa"]
MAXENT_TAU = 0.3


def load(pattern: str, ds: str) -> list[dict]:
    rows = []
    for sh in (0, 1):
        p = ROOT / f"results/{pattern.format(ds=ds)}_shard{sh}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    return rows


def main() -> None:
    lines = ["# tl_sum-gated prompt-fix-A full rerun (n=1000x4)\n",
             "gate: tl_sum >= 0.008461 (old-sweep skip=0.10 quantile); "
             "steps + tentative probes carry the original question (fix A); "
             "rescue -> verifier v2 -> typeaware -> alias-aware EM/F1\n",
             "| ds | n | canonical EM | fixA(max_ent) EM/F1 | "
             "**tl_sum+fixA EM/F1** | step retr% (tl_sum) | retr% (max_ent would) "
             "| direct-choice% |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    mac = {"em": [], "f1": [], "fa_em": [], "fa_f1": [], "old": []}
    for ds in DATASETS:
        rows = load("tlsum_fixA_{ds}", ds)
        if not rows:
            lines.append(f"| {ds} | 0 | - | - | - | - | - | - |")
            continue
        n = len(rows)
        em = sum(r["tl"]["em"] for r in rows) / n * 100
        f1 = sum(r["tl"]["f1"] for r in rows) / n * 100
        old = sum(r["replay_em"] for r in rows) / n * 100
        n_steps = n_ret = n_ret_me = 0
        n_direct = n_choice = 0
        for r in rows:
            for s in r.get("new_steps") or []:
                n_steps += 1
                n_ret += int(bool(s["retrieved"]))
                n_ret_me += int(s["max_ent"] >= MAXENT_TAU)
            if not r["tl"].get("unchanged"):
                n_choice += 1
                n_direct += int(r["tl"].get("choice") == "direct")
        fa = load("prompt_fix_{ds}", ds)
        fa_em = sum(r["A"]["em"] for r in fa) / max(len(fa), 1) * 100
        fa_f1 = sum(r["A"]["f1"] for r in fa) / max(len(fa), 1) * 100
        lines.append(
            f"| {ds} | {n} | {old:.1f} | {fa_em:.1f}/{fa_f1:.1f} | "
            f"**{em:.1f}/{f1:.1f}** | {n_ret/max(n_steps,1)*100:.1f} | "
            f"{n_ret_me/max(n_steps,1)*100:.1f} | "
            f"{n_direct/max(n_choice,1)*100:.1f} |")
        mac["em"].append(em)
        mac["f1"].append(f1)
        mac["fa_em"].append(fa_em)
        mac["fa_f1"].append(fa_f1)
        mac["old"].append(old)
    if mac["em"]:
        k = len(mac["em"])
        lines.append(
            f"| **macro** | | {sum(mac['old'])/k:.2f} | "
            f"{sum(mac['fa_em'])/k:.2f}/{sum(mac['fa_f1'])/k:.2f} | "
            f"**{sum(mac['em'])/k:.2f}/{sum(mac['f1'])/k:.2f}** | | | |")
    out = ROOT / "results/tlsum_fixA_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
