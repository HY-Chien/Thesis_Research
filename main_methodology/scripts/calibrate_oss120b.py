"""Generate a calibration report for the OSS-120B decompositions vs MuSiQue gold.

Run:
    conda activate adaptive_rag
    python /home/S113065528/Research/main_methodology/scripts/calibrate_oss120b.py

Writes:
    datasets/musique/oss-120b/calibration_report.md
    datasets/musique/oss-120b/calibration_stats.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/S113065528/Research/main_methodology/datasets/musique")
OSS_PATH = ROOT / "oss-120b" / "selected_2048_with_oss120b_decomposition.jsonl"
OUT_MD = ROOT / "oss-120b" / "calibration_report.md"
OUT_JSON = ROOT / "oss-120b" / "calibration_stats.json"


def hop_class(qid: str) -> str:
    return qid.split("__")[0]


def expected_hops(qid: str) -> int:
    return int(hop_class(qid)[0])


def load_rows(p: Path) -> list[dict]:
    rows = []
    with p.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def check_depends_on(gen: list[dict]) -> tuple[bool, str]:
    """Return (valid, reason). depends_on must reference earlier step_ids only."""
    seen: set[int] = set()
    for step in gen:
        sid = step.get("step_id")
        if sid is None:
            return False, "missing step_id"
        deps = step.get("depends_on") or []
        for d in deps:
            if isinstance(d, str) and d.startswith("#"):
                try:
                    d = int(d[1:])
                except ValueError:
                    return False, f"unparseable dep {d!r}"
            if not isinstance(d, int):
                return False, f"non-int dep {d!r}"
            if d not in seen:
                return False, f"step {sid} depends on {d} not yet defined"
        seen.add(sid)
    return True, ""


def main() -> None:
    rows = load_rows(OSS_PATH)
    n = len(rows)

    # Overall
    n_err = sum(1 for r in rows if r.get("decomposition_error"))
    n_match = 0
    delta_hist: Counter[int] = Counter()   # generated - gold
    per_hop_total: Counter[str] = Counter()
    per_hop_match: Counter[str] = Counter()
    per_hop_delta_hist: dict[str, Counter[int]] = defaultdict(Counter)
    dep_invalid: list[tuple[str, str]] = []

    # Sample buckets
    sample_match: dict[str, dict] = {}
    sample_under: dict[str, dict] = {}   # generated < gold
    sample_over: dict[str, dict] = {}    # generated > gold

    for r in rows:
        qid = r["id"]
        hc = hop_class(qid)
        per_hop_total[hc] += 1

        gold = r.get("question_decomposition") or []
        gen = r.get("generated_decomposition") or []
        g_n, p_n = len(gold), len(gen)
        delta = p_n - g_n
        delta_hist[delta] += 1
        per_hop_delta_hist[hc][delta] += 1

        if delta == 0:
            n_match += 1
            per_hop_match[hc] += 1
            sample_match.setdefault(hc, r)
        elif delta < 0:
            sample_under.setdefault(hc, r)
        else:
            sample_over.setdefault(hc, r)

        ok, reason = check_depends_on(gen)
        if not ok:
            dep_invalid.append((qid, reason))

    stats = {
        "n_rows": n,
        "n_decomposition_error": n_err,
        "n_count_match": n_match,
        "count_match_rate": round(n_match / n, 4),
        "n_depends_on_invalid": len(dep_invalid),
        "delta_histogram": dict(sorted(delta_hist.items())),
        "per_hop": {
            hc: {
                "total": per_hop_total[hc],
                "match": per_hop_match[hc],
                "match_rate": round(per_hop_match[hc] / per_hop_total[hc], 4),
                "expected_hops": expected_hops(hc + "__"),
                "delta_hist": dict(sorted(per_hop_delta_hist[hc].items())),
            }
            for hc in sorted(per_hop_total)
        },
    }

    OUT_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    # ---------- Markdown report ----------
    def fmt_sample(label: str, ex: dict) -> str:
        lines = [f"**{label}** — `{ex['id']}`", ""]
        lines.append(f"- Question: {ex['question']}")
        lines.append("- Gold decomposition:")
        for s in ex["question_decomposition"]:
            lines.append(f"  - `{s['question']}` → `{s['answer']}`")
        lines.append("- Generated decomposition:")
        for s in ex["generated_decomposition"]:
            deps = s.get("depends_on") or []
            lines.append(f"  - `{s['id']}` (deps={deps}): {s['sub_question']}")
        return "\n".join(lines)

    md = []
    md.append("# OSS-120B Decomposition — Calibration Report")
    md.append("")
    md.append(f"Source: `{OSS_PATH.relative_to(ROOT.parent.parent)}`")
    md.append("")
    md.append("## 1. Overall")
    md.append("")
    md.append(f"- Rows: **{n}**")
    md.append(f"- `decomposition_error` non-null: **{n_err}** ({n_err/n*100:.1f}%)")
    md.append(f"- Sub-question count matches gold: **{n_match} / {n}** ({n_match/n*100:.1f}%)")
    md.append(f"- `depends_on` graph invalid (forward / unparseable refs): **{len(dep_invalid)}**")
    md.append("")
    md.append("### Delta histogram (generated − gold)")
    md.append("")
    md.append("| Δ | Count | % |")
    md.append("|---|---:|---:|")
    for d, c in sorted(delta_hist.items()):
        md.append(f"| {d:+d} | {c} | {c/n*100:.1f}% |")
    md.append("")
    md.append("## 2. Per-hop breakdown")
    md.append("")
    md.append("| Hop class | Expected sub-Qs | Total | Match | Match rate | Δ histogram |")
    md.append("|---|---:|---:|---:|---:|---|")
    for hc in sorted(per_hop_total):
        deltas = per_hop_delta_hist[hc]
        delta_str = ", ".join(f"{d:+d}:{c}" for d, c in sorted(deltas.items()))
        md.append(
            f"| `{hc}` | {expected_hops(hc + '__')} | {per_hop_total[hc]} | "
            f"{per_hop_match[hc]} | {per_hop_match[hc]/per_hop_total[hc]*100:.1f}% | {delta_str} |"
        )
    md.append("")
    md.append("## 3. Samples")
    md.append("")
    md.append("### 3.1 Count-matched")
    md.append("")
    for hc in sorted(sample_match):
        md.append(fmt_sample(f"[{hc}] match", sample_match[hc]))
        md.append("")
    md.append("### 3.2 Generated < Gold (OSS-120B under-decomposed)")
    md.append("")
    for hc in sorted(sample_under):
        md.append(fmt_sample(f"[{hc}] under", sample_under[hc]))
        md.append("")
    md.append("### 3.3 Generated > Gold (OSS-120B over-decomposed)")
    md.append("")
    for hc in sorted(sample_over):
        md.append(fmt_sample(f"[{hc}] over", sample_over[hc]))
        md.append("")
    md.append("## 4. Interpretation notes")
    md.append("")
    md.append(
        "- **Hop number ≠ sub-Q count required.** A 4-hop MuSiQue question may decompose into 4 atomic sub-Qs, "
        "but OSS-120B sometimes splits a single hop into multiple steps (over) or fuses two hops (under). "
        "Count agreement alone is necessary but not sufficient — the thesis Stage 3.1 step (2) "
        "self-verify filter (oracle paragraph + check final answer) is what actually gates SFT inclusion."
    )
    md.append(
        "- **Hop classes with low match rate** (typically 3-hop / 4-hop) are the ones to watch when "
        "deciding whether to retrain the decomposer or to widen self-verify tolerance."
    )
    md.append(
        "- `depends_on` integrity should be 100% if the JSON parsing was sound; any non-zero count here flags "
        "rows that would need manual repair before SFT."
    )

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote: {OUT_MD}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
