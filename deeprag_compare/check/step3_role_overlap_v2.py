"""Step 3 v2 — Role-overlap detector with spaCy dependency parsing.

v1 (step3_role_overlap.py) flagged any "pred ∈ Q entities" in fact_finding
schema as B3. False positive: 5abdf12255 ("Album of Rock singer Robert C
Seger... what was the singer's stage name?") — Seger appears in question
because the question asks ABOUT him, not because he's a constraint.

v2 fixes this by classifying each entity's syntactic role:
  - CONSTRAINT  : entity is `pobj` of a constraint preposition
                  ({with, by, starring, featuring, alongside, opposite, against,
                  produced, directed, written}-by, or "co-host/co-star X")
  - SUBJECT     : entity is `nsubj` / `nsubjpass` of the main clause
  - DESCRIPTOR  : entity is `pobj` of a non-constraint preposition that
                  attaches to a NOUN (modifies a noun phrase, not a verb)
  - OTHER       : everything else

B3 = pred matches a CONSTRAINT entity AND schema=fact_finding.

Also adds:
  - ner_entities: spaCy NER (more accurate than regex caps) + falls back to
                  regex caps for cases spaCy misses (e.g. "Echosmith")
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import spacy

ROOT = Path(__file__).parent
DEFAULT_DUMP1 = ROOT / "multi_sample_dump.jsonl"
DUMP2 = ROOT / "step2_output/perturbation_dump.jsonl"
DEFAULT_OUT_DIR = ROOT / "step3_output"

NLP = spacy.load("en_core_web_sm")

CONSTRAINT_PREPS = {"with", "by", "starring", "featuring", "alongside",
                    "opposite", "against"}
# "co-host X" / "co-star X" — X is appositive of a preceding compound noun.
# We also handle these via prep-chain inspection.
COMPOUND_CONSTRAINT_HEADS = {"co-host", "co-star", "co-author", "co-writer"}

COMPARISON_PAT = re.compile(
    r"\b(or|older|younger|first|more|less|larger|smaller|earlier|later|"
    r"better|worse|both|either|neither|same)\b",
    re.IGNORECASE,
)
YES_NO_PAT = re.compile(r"^\s*(are|is|was|were|do|does|did|has|have|can|could|will|would)\b",
                        re.IGNORECASE)

STOP_CAPS = {"What", "Who", "Where", "When", "Which", "Why", "How",
             "The", "A", "An", "Are", "Is", "Was", "Were", "Do", "Does",
             "Did", "In", "Of", "At", "On", "And", "Or", "But", "By",
             "Both", "Either", "Neither", "Yes", "No", "Note", "Also"}
ENT_RE = re.compile(
    r"\b([A-Z][\w\.\-]*(?:\s+(?:[A-Z][\w\.\-]*|of|the|de|von|van|la|in|on|and))*?\s+[A-Z][\w\.\-]*"
    r"|[A-Z][\w\.\-]*)\b"
)


def regex_entity_spans(question: str):
    spans = set()
    for m in ENT_RE.finditer(question):
        s = m.group(0).strip()
        words = s.split()
        if all(w in STOP_CAPS for w in words): continue
        while words and words[0] in STOP_CAPS:
            words = words[1:]
        if not words: continue
        cleaned = " ".join(words)
        if len(cleaned) >= 2: spans.add(cleaned)
    return spans


def question_schema(question: str) -> str:
    q = question.strip()
    if YES_NO_PAT.match(q) and "?" in q:
        return "yes_no"
    if COMPARISON_PAT.search(q):
        return "comparison"
    return "fact_finding"


def classify_entity_role(ent) -> str:
    """Classify a spaCy span's syntactic role w.r.t. the question structure."""
    root = ent.root
    dep = root.dep_

    if dep in ("nsubj", "nsubjpass"):
        return "SUBJECT"

    if dep == "pobj":
        prep = root.head           # the preposition token
        prep_head = prep.head      # what the prep attaches to
        prep_text = prep.text.lower()

        if prep_text in CONSTRAINT_PREPS:
            # "X with [E]" / "X by [E]" / "starring [E]" — verbal/predicate constraint
            return "CONSTRAINT"

        # "directed by E" / "produced by E" — by attaches to a verb (acl/relcl)
        if prep_text == "by" and prep_head.pos_ == "VERB":
            return "CONSTRAINT"

        # "of E" attaching to a NOUN is descriptive (e.g. "Album of Seger")
        if prep_head.pos_ in ("NOUN", "PROPN", "PRON"):
            return "DESCRIPTOR"

        # Fallback for prep on verbs: still constraint-ish
        if prep_head.pos_ in ("VERB", "AUX"):
            return "CONSTRAINT"

    # appositive after a compound constraint head: "co-host [E]"
    if dep == "appos":
        head = root.head
        if head.text.lower() in COMPOUND_CONSTRAINT_HEADS:
            return "CONSTRAINT"

    return "OTHER"


def extract_constraint_entities(question: str):
    """Return list of (entity_text, role) using spaCy + regex fallback."""
    doc = NLP(question)
    out = []
    seen = set()
    for ent in doc.ents:
        text = ent.text.strip()
        if text in seen or len(text) < 2: continue
        seen.add(text)
        role = classify_entity_role(ent)
        out.append((text, role))

    regex_spans = regex_entity_spans(question)
    for span in regex_spans:
        if span in seen: continue
        # Try to look up its dep role by finding the span in the parse
        # Best-effort: find the rightmost token of the span and treat as ent.root surrogate.
        for token in doc:
            if token.text == span.split()[-1]:
                # Build a fake "entity" for role classification
                class FakeEnt:
                    def __init__(self, root): self.root = root
                role = classify_entity_role(FakeEnt(token))
                out.append((span, role))
                seen.add(span)
                break
        else:
            out.append((span, "OTHER"))
            seen.add(span)
    return out


def pred_matches(pred: str, ent_text: str) -> bool:
    p = (pred or "").strip().lower()
    e = ent_text.lower()
    if not p or len(p) < 3 or len(e) < 3: return False
    return p in e or e in p


def detect_b3(pred: str, entities, schema: str):
    """Return (is_b3, matched_role, matched_entity)."""
    if schema != "fact_finding":
        return False, "", ""
    for ent_text, role in entities:
        if pred_matches(pred, ent_text):
            if role == "CONSTRAINT":
                return True, role, ent_text
            else:
                # Matched but not B3 (subject / descriptor / other)
                return False, role, ent_text
    return False, "", ""


def analyze_step1(dump_path):
    by_qid = defaultdict(list)
    qid_meta = {}
    for line in dump_path.open():
        r = json.loads(line)
        qid = r["qid"]
        by_qid[qid].append(r)
        if qid not in qid_meta:
            qid_meta[qid] = {
                "stratum": r["stratum"],
                "question": r["question"],
                "schema": question_schema(r["question"]),
                "entities": extract_constraint_entities(r["question"]),
            }
    rows = []
    for qid, recs in by_qid.items():
        meta = qid_meta[qid]
        b3_flags, roles_matched = [], []
        for r in recs:
            is_b3, role, ent = detect_b3(r.get("pred", ""), meta["entities"], meta["schema"])
            b3_flags.append(int(is_b3))
            if role: roles_matched.append((role, ent))
        ems = [r.get("em", 0) for r in recs]
        roles_counter = Counter(r for r, _ in roles_matched)
        rows.append({
            "qid": qid, "stratum": meta["stratum"], "schema": meta["schema"],
            "n_samples": len(recs),
            "v2_b3_rate": mean(b3_flags) if b3_flags else 0.0,
            "em_rate": mean(ems) if ems else 0.0,
            "matched_roles": "; ".join(f"{r}×{c}" for r, c in roles_counter.most_common()),
            "n_constraint_ents": sum(1 for _, role in meta["entities"] if role == "CONSTRAINT"),
            "ents": " | ".join(f"{e}({r})" for e, r in meta["entities"][:8]),
        })
    return rows, qid_meta


def analyze_step2():
    by_qm = defaultdict(list)
    qid_to_origQ = {}
    for line in DUMP2.open():
        r = json.loads(line)
        by_qm[(r["qid"], r["mode"])].append(r)
        if r["mode"] == "original":
            qid_to_origQ[r["qid"]] = r["perturbed_question"]
    qid_meta = {}
    for qid, q in qid_to_origQ.items():
        qid_meta[qid] = {
            "schema": question_schema(q),
            "entities": extract_constraint_entities(q),
            "stratum": next(iter(by_qm[(qid, "original")]))["stratum"],
        }
    rows = []
    for (qid, mode), recs in by_qm.items():
        meta = qid_meta[qid]
        b3_flags = []
        for r in recs:
            is_b3, role, ent = detect_b3(r.get("pred", ""), meta["entities"], meta["schema"])
            b3_flags.append(int(is_b3))
        rows.append({
            "qid": qid, "stratum": meta["stratum"], "schema": meta["schema"],
            "mode": mode, "n_samples": len(recs),
            "v2_b3_rate": mean(b3_flags),
            "em_rate": mean(r.get("em", 0) for r in recs),
        })
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_file", type=str, default=str(DEFAULT_DUMP1))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--skip_step2", action="store_true",
                    help="Skip step2 perturbation analysis (e.g. when dump_file is 100Q with no perturbation)")
    args = ap.parse_args()
    dump_path = Path(args.dump_file)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)

    s1_rows, s1_meta = analyze_step1(dump_path)
    s2_rows = [] if args.skip_step2 else analyze_step2()

    s1_csv = OUT_DIR / "role_overlap_v2_step1.csv"
    with s1_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s1_rows[0].keys()))
        w.writeheader()
        for r in s1_rows: w.writerow(r)
    print(f"wrote {s1_csv}")

    if s2_rows:
        s2_csv = OUT_DIR / "role_overlap_v2_step2.csv"
        with s2_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(s2_rows[0].keys()))
            w.writeheader()
            for r in s2_rows: w.writerow(r)
        print(f"wrote {s2_csv}")

    # --- v1 vs v2 comparison ---
    v1_step1 = {}
    v1_csv_path = OUT_DIR / "role_overlap_step1.csv"
    if v1_csv_path.exists():
        with v1_csv_path.open() as f:
            for row in csv.DictReader(f):
                v1_step1[row["qid"]] = float(row["overlap_rate"])

    md = ["# Step 3 v2 — Role-overlap detector with spaCy dependency parse",
          "",
          "v2 fixes v1's false-positive on 5abdf12255 by classifying entity "
          "syntactic role (CONSTRAINT / SUBJECT / DESCRIPTOR / OTHER) before "
          "flagging B3.",
          "",
          "## Step 1 (multi_sample_dump, 20Q × 20 samples) — v1 vs v2",
          "",
          "| qid | stratum | schema | em_rate | v1_overlap | **v2_b3_rate** | matched_roles | constraint_ents |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(s1_rows, key=lambda x: -x["v2_b3_rate"]):
        v1 = v1_step1.get(r["qid"], 0.0)
        md.append(
            f"| {r['qid'][:10]} | {r['stratum']} | {r['schema']} | "
            f"{r['em_rate']:.2f} | {v1:.2f} | **{r['v2_b3_rate']:.2f}** | "
            f"{r['matched_roles'] or '—'} | {r['n_constraint_ents']} |"
        )

    md += ["", "## Step 1 grouped (stratum × schema): v2_b3_rate", "",
           "| stratum | schema | n_qids | mean_v1 | **mean_v2** | mean_em |",
           "|---|---|---|---|---|---|"]
    grp = defaultdict(list)
    for r in s1_rows:
        grp[(r["stratum"], r["schema"])].append(r)
    for (st, sc), rs in sorted(grp.items()):
        v1_mean = mean(v1_step1.get(r["qid"], 0.0) for r in rs)
        md.append(f"| {st} | {sc} | {len(rs)} | {v1_mean:.2f} | "
                  f"**{mean(r['v2_b3_rate'] for r in rs):.2f}** | "
                  f"{mean(r['em_rate'] for r in rs):.2f} |")

    md += ["", "## Step 2 (perturbation, 10Q × 4 modes × 10 samples)", "",
           "**B3 mechanistic test**: high in `original`, drops under `mask`.",
           "",
           "| qid | stratum | schema | original | mask | replace | distractor |",
           "|---|---|---|---|---|---|---|"]
    by_qid_mode = defaultdict(dict)
    for r in s2_rows:
        by_qid_mode[(r["qid"], r["stratum"], r["schema"])][r["mode"]] = r["v2_b3_rate"]
    for (qid, st, sch), modes in sorted(by_qid_mode.items()):
        md.append(
            f"| {qid[:10]} | {st} | {sch} | "
            f"{modes.get('original', 0):.2f} | {modes.get('mask', 0):.2f} | "
            f"{modes.get('replace', 0):.2f} | {modes.get('distractor', 0):.2f} |"
        )

    md += ["", "## Per-qid entity role breakdown (Step 1)", ""]
    for r in sorted(s1_rows, key=lambda x: -x["v2_b3_rate"])[:6]:
        md.append(f"### {r['qid'][:10]} ({r['stratum']}, {r['schema']})")
        md.append(f"- v1_overlap={v1_step1.get(r['qid'], 0):.2f}  "
                  f"v2_b3_rate={r['v2_b3_rate']:.2f}  em_rate={r['em_rate']:.2f}")
        md.append(f"- entities: {r['ents']}")
        md.append("")

    summary = OUT_DIR / "role_overlap_v2_summary.md"
    summary.write_text("\n".join(md) + "\n")
    print(f"wrote {summary}")

    print("\n=== v1 vs v2 sorted by v2_b3_rate ===")
    for r in sorted(s1_rows, key=lambda x: -x["v2_b3_rate"]):
        v1 = v1_step1.get(r["qid"], 0.0)
        change = ""
        if v1 >= 0.5 and r["v2_b3_rate"] < 0.5: change = " [FIXED]"
        if v1 < 0.5 and r["v2_b3_rate"] >= 0.5: change = " [NEW FLAG]"
        print(f"  {r['qid'][:10]} {r['stratum']:11s} {r['schema']:13s} "
              f"v1={v1:.2f} v2={r['v2_b3_rate']:.2f}  em={r['em_rate']:.2f}"
              f"  roles=[{r['matched_roles']}]{change}")


if __name__ == "__main__":
    main()
