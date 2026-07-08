"""Step 3 — Role-overlap detector for B3 (role confusion) hallucination.

Hypothesis: When the model treats a question's *constraint entity* as the
*answer entity* (e.g. asked "What director worked with Vikram Bhatt..." and
answers "Vikram Bhatt"), the prediction substring overlaps with the question.
This is a QA-native failure mode invisible to SC and to all token-level
features we tried in Step 1.

Naive overlap is ambiguous though: comparison questions ("Which is older,
X or Y?") legitimately yield pred ∈ question. So we split by question schema:

  comparison_q   — contains "or" + (older|first|more|less|...) — pred ∈ Q is expected
  fact_finding_q — open-ended — pred ∈ Q ⇒ likely B3 role confusion

Inputs:
  multi_sample_dump.jsonl       (Step 1: 20Q × 20 samples, no perturbation)
  step2_output/perturbation_dump.jsonl  (Step 2: 10Q × 4 modes × 10 samples)

Outputs:
  step3_output/role_overlap_step1.csv
  step3_output/role_overlap_step2.csv
  step3_output/role_overlap_summary.md
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent
DUMP1 = ROOT / "multi_sample_dump.jsonl"
DUMP2 = ROOT / "step2_output/perturbation_dump.jsonl"
OUT_DIR = ROOT / "step3_output"
OUT_DIR.mkdir(exist_ok=True)

# Stopwords that are commonly capitalized in questions but aren't entities.
STOP_CAPS = {
    "What", "Who", "Where", "When", "Which", "Why", "How",
    "The", "A", "An", "Are", "Is", "Was", "Were", "Do", "Does",
    "Did", "In", "Of", "At", "On", "And", "Or", "But", "By",
    "Both", "Either", "Neither", "Yes", "No", "Note", "Also",
}

ENT_RE = re.compile(
    r"\b([A-Z][\w\.\-]*(?:\s+(?:[A-Z][\w\.\-]*|of|the|de|von|van|la|in|on|and))*?\s+[A-Z][\w\.\-]*"
    r"|[A-Z][\w\.\-]*)\b"
)

COMPARISON_PAT = re.compile(
    r"\b(or|older|younger|first|more|less|larger|smaller|earlier|later|"
    r"better|worse|both|either|neither|which.*(?:older|younger|first|"
    r"older|or|more)|same)\b",
    re.IGNORECASE,
)

YES_NO_PAT = re.compile(r"^\s*(are|is|was|were|do|does|did|has|have|can|could|will|would)\b",
                        re.IGNORECASE)


def extract_entity_spans(question: str):
    """Return list of unique capitalized multi/single-word spans, minus stopword caps."""
    spans = set()
    for m in ENT_RE.finditer(question):
        s = m.group(0).strip()
        # Drop if it's only a stopword cap.
        words = s.split()
        if all(w in STOP_CAPS for w in words):
            continue
        # Drop leading stopwords ("What is the X" -> "X")
        while words and words[0] in STOP_CAPS:
            words = words[1:]
        if not words: continue
        cleaned = " ".join(words)
        if len(cleaned) < 2: continue
        spans.add(cleaned)
    return spans


def question_schema(question: str) -> str:
    q = question.strip()
    if YES_NO_PAT.match(q) and "?" in q:
        # Yes/no — pred is "yes"/"no", overlap mostly irrelevant
        return "yes_no"
    if COMPARISON_PAT.search(q):
        return "comparison"
    return "fact_finding"


def pred_overlap(pred: str, ent_spans: set) -> tuple[bool, str]:
    """Return (overlapping, matched_span_or_empty)."""
    p = (pred or "").strip().lower()
    if not p: return False, ""
    for span in ent_spans:
        sp = span.lower()
        # Either pred contains span or span contains pred (handles
        # "Vikram Bhatt" pred matching "Vikram Bhatt" span etc.)
        if sp in p or p in sp:
            # Filter out trivial overlaps like "the" (pred="the", span="the X")
            if len(sp) >= 3 and len(p) >= 3:
                return True, span
    return False, ""


def analyze_step1():
    by_qid = defaultdict(list)
    qid_meta = {}
    for line in DUMP1.open():
        r = json.loads(line)
        qid = r["qid"]
        by_qid[qid].append(r)
        if qid not in qid_meta:
            qid_meta[qid] = {
                "stratum": r["stratum"],
                "question": r["question"],
                "ent_spans": extract_entity_spans(r["question"]),
                "schema": question_schema(r["question"]),
            }
    rows = []
    for qid, recs in by_qid.items():
        meta = qid_meta[qid]
        spans = meta["ent_spans"]
        overlaps = [pred_overlap(r.get("pred", ""), spans)[0] for r in recs]
        ems = [r.get("em", 0) for r in recs]
        sample_matches = [pred_overlap(r.get("pred", ""), spans)[1] for r in recs]
        modal_match = Counter(m for m in sample_matches if m).most_common(1)
        rows.append({
            "qid": qid, "stratum": meta["stratum"], "schema": meta["schema"],
            "n_samples": len(recs),
            "overlap_rate": mean(overlaps) if overlaps else 0.0,
            "em_rate": mean(ems) if ems else 0.0,
            "n_entities": len(spans),
            "modal_overlap": modal_match[0][0] if modal_match else "",
            "modal_overlap_count": modal_match[0][1] if modal_match else 0,
            "entities": " | ".join(sorted(spans)[:6]),
        })
    return rows, qid_meta


def analyze_step2():
    by_qm = defaultdict(list)
    qid_meta = {}
    for line in DUMP2.open():
        r = json.loads(line)
        key = (r["qid"], r["mode"])
        by_qm[key].append(r)
        qid = r["qid"]
        if qid not in qid_meta:
            qid_meta[qid] = {
                "stratum": r["stratum"],
                "question": r["perturbed_question"]
                if r["mode"] == "original" else None,
            }
    # Use original-condition question for entity extraction (we want to detect
    # whether pred overlaps with the *original* entities)
    for line in DUMP2.open():
        r = json.loads(line)
        if r["mode"] == "original":
            qid_meta[r["qid"]]["question"] = r["perturbed_question"]
            qid_meta[r["qid"]]["ent_spans"] = extract_entity_spans(r["perturbed_question"])
            qid_meta[r["qid"]]["schema"] = question_schema(r["perturbed_question"])

    rows = []
    for (qid, mode), recs in by_qm.items():
        meta = qid_meta[qid]
        spans = meta["ent_spans"]
        overlaps = [pred_overlap(r.get("pred", ""), spans)[0] for r in recs]
        ems = [r.get("em", 0) for r in recs]
        rows.append({
            "qid": qid, "stratum": meta["stratum"], "schema": meta["schema"],
            "mode": mode, "n_samples": len(recs),
            "overlap_rate": mean(overlaps),
            "em_rate": mean(ems),
        })
    return rows, qid_meta


def main():
    s1_rows, s1_meta = analyze_step1()
    s2_rows, s2_meta = analyze_step2()

    s1_csv = OUT_DIR / "role_overlap_step1.csv"
    with s1_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s1_rows[0].keys()))
        w.writeheader()
        for r in s1_rows: w.writerow(r)
    print(f"wrote {s1_csv}")

    s2_csv = OUT_DIR / "role_overlap_step2.csv"
    with s2_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s2_rows[0].keys()))
        w.writeheader()
        for r in s2_rows: w.writerow(r)
    print(f"wrote {s2_csv}")

    md = ["# Step 3 — Role-overlap detector",
          "",
          "Detects whether the predicted answer is a substring of an entity "
          "appearing in the question. Combined with question schema "
          "(comparison / yes-no / fact-finding) this isolates **B3 role "
          "confusion**: in fact-finding questions, pred ∈ Q is a strong "
          "signal that the model misparsed the constraint entity as the "
          "answer entity.",
          "",
          "## Step 1 (multi_sample_dump, 20Q × 20 samples, no perturbation)",
          "",
          "| qid | stratum | schema | overlap_rate | em_rate | modal_overlap | entities |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(s1_rows, key=lambda x: -x["overlap_rate"]):
        md.append(
            f"| {r['qid'][:10]} | {r['stratum']} | {r['schema']} | "
            f"{r['overlap_rate']:.2f} | {r['em_rate']:.2f} | "
            f"`{r['modal_overlap']}`×{r['modal_overlap_count']} | {r['entities']} |"
        )

    md += ["",
           "## Step 1 grouped by (stratum × schema)",
           "",
           "| stratum | schema | n_qids | mean_overlap | mean_em |",
           "|---|---|---|---|---|"]
    grp = defaultdict(list)
    for r in s1_rows:
        grp[(r["stratum"], r["schema"])].append(r)
    for (st, sc), rs in sorted(grp.items()):
        md.append(f"| {st} | {sc} | {len(rs)} | "
                  f"{mean(r['overlap_rate'] for r in rs):.2f} | "
                  f"{mean(r['em_rate'] for r in rs):.2f} |")

    md += ["",
           "## Step 2 (perturbation, 10Q × 4 modes × 10 samples)",
           "",
           "Compare overlap_rate across conditions. **B3 signature**: high "
           "overlap in `original`, drops sharply under `mask` (the entity is "
           "no longer in question) → confirms model was role-confused.",
           "",
           "| qid | stratum | schema | original | mask | replace | distractor |",
           "|---|---|---|---|---|---|---|"]

    by_qid_mode = defaultdict(dict)
    for r in s2_rows:
        by_qid_mode[(r["qid"], r["stratum"], r["schema"])][r["mode"]] = r["overlap_rate"]

    for (qid, st, sch), modes in sorted(by_qid_mode.items()):
        md.append(
            f"| {qid[:10]} | {st} | {sch} | "
            f"{modes.get('original', 0):.2f} | {modes.get('mask', 0):.2f} | "
            f"{modes.get('replace', 0):.2f} | {modes.get('distractor', 0):.2f} |"
        )

    md += ["",
           "## Interpretation cheat-sheet",
           "",
           "- **fact_finding + high overlap** → B3 (role confusion). "
           "Expected: mask drops overlap, replace re-attaches to new entity, "
           "distractor unchanged.",
           "- **comparison + high overlap** → expected (pred is one of the candidates).",
           "- **yes_no schema** → overlap not informative.",
           ""]

    summary_md = OUT_DIR / "role_overlap_summary.md"
    summary_md.write_text("\n".join(md) + "\n")
    print(f"wrote {summary_md}")

    print("\n=== Step 1 sorted by overlap_rate ===")
    for r in sorted(s1_rows, key=lambda x: -x["overlap_rate"])[:10]:
        print(f"  {r['qid'][:10]} {r['stratum']:11s} {r['schema']:13s} "
              f"overlap={r['overlap_rate']:.2f} em={r['em_rate']:.2f} "
              f"-> {r['modal_overlap'][:30]!r}")


if __name__ == "__main__":
    main()
