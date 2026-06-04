"""Compute gated EM + F1 for the two configs cited in EXPERIMENTS_SUMMARY section 5:
  - min_lp @ -0.80
  - neg_max_ent @ -0.80
Joins per_q_v2.jsonl (no-CoT signal) with results/{ds}_meta-llama-3-8b-instruct.jsonl (F1).
"""
import json
from pathlib import Path

ROOT = Path("/home/S113065528/Research/deeprag_compare")
SWEEP_IN = ROOT / "check/production_sweep_output"
RES_IN = ROOT / "results"
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]

CONFIGS = [
    ("min_lp", -0.80, "min_lp@-0.80"),
    ("neg_max_ent", -0.80, "neg_max_ent@-0.80"),
]


def load_v2(ds):
    fp = SWEEP_IN / f"{ds}_per_q_v2.jsonl"
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]


def load_f1(ds):
    fp = RES_IN / f"{ds}_meta-llama-3-8b-instruct.jsonl"
    out = {}
    for l in fp.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        out[r["qid"]] = (r.get("nr_f1", 0.0), r.get("rag_f1", 0.0))
    return out


def evaluate(records, f1_map, metric_key, threshold):
    n = len(records)
    em_t = 0.0
    f1_t = 0.0
    f1_nr_t = 0.0
    f1_rag_t = 0.0
    em_nr_t = 0
    em_rag_t = 0
    n_trig = 0
    matched = 0
    for r in records:
        nr_em = r["nr_em"]
        rag_em = r["rag_em"]
        nr_f1, rag_f1 = f1_map.get(r["qid"], (None, None))
        if nr_f1 is None:
            continue
        matched += 1
        em_nr_t += nr_em
        em_rag_t += rag_em
        f1_nr_t += nr_f1
        f1_rag_t += rag_f1
        ag = r.get("nocot_first_ent")
        m = ag.get(metric_key) if ag else None
        if m is not None and m < threshold:
            n_trig += 1
            em_t += rag_em
            f1_t += rag_f1
        else:
            em_t += nr_em
            f1_t += nr_f1
    return {
        "n": matched,
        "trig": n_trig / matched,
        "em": em_t / matched,
        "f1": f1_t / matched,
        "nr_em": em_nr_t / matched,
        "nr_f1": f1_nr_t / matched,
        "rag_em": em_rag_t / matched,
        "rag_f1": f1_rag_t / matched,
    }


def main():
    rows = []
    rows.append("# Gated EM + F1 (with F1 joined from results/)\n")
    rows.append("| Dataset | always_NR EM/F1 | always_RAG EM/F1 | "
                + " | ".join(f"{tag} EM/F1 (trig%)" for _, _, tag in CONFIGS) + " |")
    rows.append("|" + "---|" * (3 + len(CONFIGS)))
    for ds in DATASETS:
        recs = load_v2(ds)
        f1m = load_f1(ds)
        cells = []
        for metric, thr, tag in CONFIGS:
            r = evaluate(recs, f1m, metric, thr)
            cells.append(f"{r['em']*100:.1f} / {r['f1']*100:.1f} ({r['trig']*100:.0f}%)")
        # Use nr/rag from one of the runs (same across configs)
        r0 = evaluate(recs, f1m, CONFIGS[0][0], CONFIGS[0][1])
        rows.append(f"| {ds} | {r0['nr_em']*100:.1f} / {r0['nr_f1']*100:.1f} | "
                    f"{r0['rag_em']*100:.1f} / {r0['rag_f1']*100:.1f} | "
                    + " | ".join(cells) + " |")
    out = "\n".join(rows)
    print(out)
    (SWEEP_IN / "sweep_nocot_with_f1.md").write_text(out + "\n")


if __name__ == "__main__":
    main()
