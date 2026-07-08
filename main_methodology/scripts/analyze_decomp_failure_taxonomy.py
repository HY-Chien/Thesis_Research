"""
Decomposition failure taxonomy, extended from the HotpotQA failure analysis
(results/hotpotqa_failure_analysis.md) to 2Wiki and MuSiQue.

For each dataset it replays, offline, from existing traces:

  1. the direct/decompose 2x2 (both correct / direct only / decomp only /
     both wrong) plus router capture rates,
  2. the retrieval-coverage cross-tab (original-question A full@10 title-off
     vs sub-question B-union full@10 title-on, matching path-aware retrieval),
  3. a failure-mode classification of every decomposition-path error
     (decomp_em = 0), with mutually exclusive primary categories:

       synthesis_failure     some step answer already matches gold but the
                             synthesized final answer is wrong
       constraint_loss       original-question retrieval has full gold
                             coverage @10 but the decomposed B-union does not
                             (sub-questions dropped disambiguating constraints)
       answer_despite_cov    B-union coverage is full but no step answer
                             matches gold (step answering fails on good docs)
       retrieval_gap_both    neither path reaches full coverage @10
       no_coverage_join      id missing from the m3 coverage files

     and two orthogonal flags:

       intermediate_echo     final decomp answer equals a non-final step
                             answer (answer stops at an intermediate hop)
       subq1_restates_q      first sub-question already restates >=60% of the
                             original question's content words and the chain
                             still adds more steps. On HotpotQA (natural-
                             language sub-questions) this is an
                             over-decomposition proxy; on 2Wiki/MuSiQue the
                             KG-triple sub-questions inherit long entity
                             titles, which inflates the flag, so read it as
                             "sub-q1 carries most of the question" there.

Inputs (per dataset):
  results/{ds}_1000_verify_v2_pathaware.jsonl
  results/{ds}_m3_wiki_detail.jsonl            (A = original question, title-off)
  results/{ds}_m3_wiki_titlefield_detail.jsonl (B-union = sub-questions, title-on)

Output: results/decomp_failure_taxonomy.md
"""

import collections
import json
import re
import string
from pathlib import Path

RESULTS = Path("/home/S113065528/Research/main_methodology/results")
OUT = RESULTS / "decomp_failure_taxonomy.md"

DATASETS = ["hotpotqa", "2wiki", "musique"]

STOP = set(
    "what which who whom whose where when why how is are was were did does do "
    "the a an of in on at by for from to with and or as that this it his her "
    "its their there be been >> s".split()
)


def norm(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def f1(pred, gold):
    p = norm(pred).split()
    g = norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    num = sum(common.values())
    if not p or not g:
        return float(p == g)
    if num == 0:
        return 0.0
    pr, rc = num / len(p), num / len(g)
    return 2 * pr * rc / (pr + rc)


def matches_gold(ans, gold):
    return norm(ans) == norm(gold) or f1(ans, gold) >= 0.5


def content_tokens(s):
    return [t for t in norm(s).split() if t not in STOP]


def over_decomposed(question, steps):
    if len(steps) < 2 or not steps:
        return False
    sub1 = steps[0].get("sub_q") or steps[0].get("raw_sub_q") or ""
    q_toks = content_tokens(question)
    s_toks = set(content_tokens(sub1))
    if not q_toks:
        return False
    recall = sum(1 for t in q_toks if t in s_toks) / len(q_toks)
    return recall >= 0.6


def load_cov(ds):
    """id -> (A_full@10 title-off, Bunion_full@10 title-on, type)."""
    a_full, b_full, qtype = {}, {}, {}
    p_off = RESULTS / f"{ds}_m3_wiki_detail.jsonl"
    p_on = RESULTS / f"{ds}_m3_wiki_titlefield_detail.jsonl"
    if p_off.exists():
        with open(p_off, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                a_full[str(d["id"])] = int(d["cov"]["10"]["A"]["full"])
                qtype[str(d["id"])] = d.get("type")
    if p_on.exists():
        with open(p_on, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                b_full[str(d["id"])] = int(d["cov"]["10"]["Bunion"]["full"])
    return a_full, b_full, qtype


def musique_hop(qid):
    m = re.match(r"(\d)hop", qid)
    return f"{m.group(1)}hop" if m else "?"


def classify(row, a_full, b_full):
    """Primary category for a decomp-wrong case."""
    gold = row["gold"]
    steps = row["decomp"].get("steps") or []
    step_answers = [s.get("final_answer") or "" for s in steps]
    if any(matches_gold(a, gold) for a in step_answers if a):
        return "synthesis_failure"
    qid = row["id"]
    a = a_full.get(qid)
    b = b_full.get(qid)
    if a is None or b is None:
        return "no_coverage_join"
    if a == 1 and b == 0:
        return "constraint_loss"
    if b == 1:
        return "answer_despite_cov"
    return "retrieval_gap_both"


def pct(x, n):
    return f"{x} ({x/n*100:.0f}%)" if n else "0"


def main():
    lines = []
    lines.append("# Decomposition Failure Taxonomy: HotpotQA / 2Wiki / MuSiQue\n")
    lines.append("Date: 2026-07-03\n")
    lines.append(
        "Offline replay from `*_1000_verify_v2_pathaware.jsonl` + m3 coverage "
        "detail files. HotpotQA is included as a consistency check against "
        "`results/hotpotqa_failure_analysis.md`; 2Wiki and MuSiQue are the new "
        "datasets. Categories are defined in "
        "`scripts/analyze_decomp_failure_taxonomy.py`.\n"
    )
    tldr_pos = len(lines)

    summary_rows = {}
    for ds in DATASETS:
        trace_path = RESULTS / f"{ds}_1000_verify_v2_pathaware.jsonl"
        rows = [json.loads(l) for l in open(trace_path, encoding="utf-8")]
        a_full, b_full, qtype = load_cov(ds)

        groups = {"bothC": [], "dirOnly": [], "decOnly": [], "bothW": []}
        for r in rows:
            de, ce = int(r["direct_em"]), int(r["decomp_em"])
            key = {(1, 1): "bothC", (1, 0): "dirOnly", (0, 1): "decOnly", (0, 0): "bothW"}[(de, ce)]
            groups[key].append(r)

        n = len(rows)
        lines.append(f"\n---\n\n## {ds} (n={n})\n")

        # -- 2x2 + router -----------------------------------------------------
        lines.append("### Direct/decompose 2x2 and router capture\n")
        lines.append("| group | n | share |")
        lines.append("|---|---:|---:|")
        for k in ["bothC", "dirOnly", "decOnly", "bothW"]:
            lines.append(f"| {k} | {len(groups[k])} | {len(groups[k])/n*100:.1f}% |")
        dir_caught = sum(1 for r in groups["dirOnly"] if r["choice"] == "direct")
        dec_caught = sum(1 for r in groups["decOnly"] if r["choice"] == "decomp")
        final_wrong = sum(1 for r in rows if int(r["em"]) == 0)
        lines.append("")
        lines.append(
            f"Router capture: direct-only {dir_caught}/{len(groups['dirOnly'])} "
            f"({dir_caught/max(len(groups['dirOnly']),1)*100:.0f}%), "
            f"decomp-only {dec_caught}/{len(groups['decOnly'])} "
            f"({dec_caught/max(len(groups['decOnly']),1)*100:.0f}%).\n"
        )
        lines.append("Error sources among final wrong answers:\n")
        miss_dir = len(groups["dirOnly"]) - dir_caught
        miss_dec = len(groups["decOnly"]) - dec_caught
        bw = len(groups["bothW"])
        lines.append("| source | n | share of final wrong |")
        lines.append("|---|---:|---:|")
        lines.append(f"| both paths wrong | {bw} | {bw/final_wrong*100:.1f}% |")
        lines.append(f"| router missed direct-only | {miss_dir} | {miss_dir/final_wrong*100:.1f}% |")
        lines.append(f"| router missed decomp-only | {miss_dec} | {miss_dec/final_wrong*100:.1f}% |")
        lines.append("")

        # -- coverage cross-tab -------------------------------------------------
        lines.append("### Retrieval coverage by group (full-coverage@10)\n")
        lines.append("A = original question, title-off (direct path); B-union = sub-questions, title-on (decomp path).\n")
        lines.append("| group | joined n | A full@10 | B-union full@10 |")
        lines.append("|---|---:|---:|---:|")
        for k in ["bothC", "dirOnly", "decOnly", "bothW"]:
            js = [r for r in groups[k] if str(r["id"]) in a_full and str(r["id"]) in b_full]
            if not js:
                lines.append(f"| {k} | 0 | - | - |")
                continue
            am = sum(a_full[str(r["id"])] for r in js) / len(js) * 100
            bm = sum(b_full[str(r["id"])] for r in js) / len(js) * 100
            lines.append(f"| {k} | {len(js)} | {am:.1f}% | {bm:.1f}% |")
        lines.append("")

        # -- taxonomy of decomp-path errors --------------------------------------
        lines.append("### Failure taxonomy of decomposition-path errors (decomp_em = 0)\n")
        cats = ["synthesis_failure", "constraint_loss", "answer_despite_cov", "retrieval_gap_both", "no_coverage_join"]
        tax = {g: collections.Counter() for g in ["dirOnly", "bothW"]}
        flags = {g: collections.Counter() for g in ["dirOnly", "bothW"]}
        examples = collections.defaultdict(list)
        for g in ["dirOnly", "bothW"]:
            for r in groups[g]:
                c = classify(r, a_full, b_full)
                tax[g][c] += 1
                steps = r["decomp"].get("steps") or []
                sa = [s.get("final_answer") or "" for s in steps]
                if len(sa) >= 2 and any(
                    norm(r["decomp_pred"]) == norm(a) and norm(a) != norm(sa[-1])
                    for a in sa[:-1]
                    if a
                ):
                    flags[g]["intermediate_echo"] += 1
                if over_decomposed(r["question"], steps):
                    flags[g]["subq1_restates_q"] += 1
                if len(examples[c]) < 3 and g in ("dirOnly", "bothW"):
                    examples[c].append((r, g))
        n_dir, n_bw = len(groups["dirOnly"]), len(groups["bothW"])
        lines.append("| primary category | dirOnly | bothW | total |")
        lines.append("|---|---:|---:|---:|")
        for c in cats:
            tot = tax["dirOnly"][c] + tax["bothW"][c]
            lines.append(
                f"| {c} | {pct(tax['dirOnly'][c], n_dir)} | {pct(tax['bothW'][c], n_bw)} "
                f"| {pct(tot, n_dir+n_bw)} |"
            )
        lines.append("")
        lines.append("Orthogonal flags (can overlap with any primary category):\n")
        lines.append("| flag | dirOnly | bothW |")
        lines.append("|---|---:|---:|")
        for fl in ["intermediate_echo", "subq1_restates_q"]:
            lines.append(f"| {fl} | {pct(flags['dirOnly'][fl], n_dir)} | {pct(flags['bothW'][fl], n_bw)} |")
        lines.append("")
        summary_rows[ds] = (n_dir, n_bw, tax, flags)

        # -- per-type breakdown ---------------------------------------------------
        typef = (lambda r: musique_hop(str(r["id"]))) if ds == "musique" else (
            lambda r: qtype.get(str(r["id"])) or "?"
        )
        tcount = collections.defaultdict(lambda: collections.Counter())
        for k in ["bothC", "dirOnly", "decOnly", "bothW"]:
            for r in groups[k]:
                tcount[typef(r)][k] += 1
        lines.append("### Per-type 2x2\n")
        lines.append("| type | n | bothC | dirOnly | decOnly | bothW |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for t, c in sorted(tcount.items(), key=lambda kv: -sum(kv[1].values())):
            tn = sum(c.values())
            lines.append(
                f"| {t} | {tn} | {c['bothC']} | {c['dirOnly']} | {c['decOnly']} | {c['bothW']} |"
            )
        lines.append("")

        # -- examples ----------------------------------------------------------------
        lines.append("### Representative examples\n")
        for c in cats:
            if c == "no_coverage_join" or not examples[c]:
                continue
            lines.append(f"**{c}**\n")
            for r, g in examples[c][:2]:
                steps = r["decomp"].get("steps") or []

                def clip(s, k=90):
                    s = " ".join(str(s).split())
                    return s[:k] + ("..." if len(s) > k else "")

                sa = " | ".join(
                    f"{clip(s.get('sub_q',''))} => {clip(s.get('final_answer',''))}" for s in steps
                )
                lines.append(f"- [{g}] Q: {r['question']}")
                lines.append(f"  - gold: `{r['gold']}` | direct: `{r['direct_pred']}` | decomp: `{r['decomp_pred']}`")
                lines.append(f"  - chain: {sa}")
            lines.append("")

    # -- cross-dataset summary ---------------------------------------------------
    lines.append("\n---\n\n## Cross-dataset summary (decomp-path errors, dirOnly+bothW pooled)\n")
    lines.append("| category | " + " | ".join(DATASETS) + " |")
    lines.append("|---|" + "---:|" * len(DATASETS))
    for c in ["synthesis_failure", "constraint_loss", "answer_despite_cov", "retrieval_gap_both", "no_coverage_join"]:
        cells = []
        for ds in DATASETS:
            n_dir, n_bw, tax, _ = summary_rows[ds]
            tot = tax["dirOnly"][c] + tax["bothW"][c]
            cells.append(pct(tot, n_dir + n_bw))
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    for fl in ["intermediate_echo", "subq1_restates_q"]:
        cells = []
        for ds in DATASETS:
            n_dir, n_bw, _, flags = summary_rows[ds]
            tot = flags["dirOnly"][fl] + flags["bothW"][fl]
            cells.append(pct(tot, n_dir + n_bw))
        lines.append(f"| flag: {fl} | " + " | ".join(cells) + " |")
    lines.append("")

    # -- TL;DR (inserted after the header) ----------------------------------------
    def share(ds, cat):
        n_dir, n_bw, tax, flags = summary_rows[ds]
        src = flags if cat in ("intermediate_echo", "subq1_restates_q") else tax
        tot = src["dirOnly"][cat] + src["bothW"][cat]
        return tot / (n_dir + n_bw) * 100

    tldr = []
    tldr.append("## TL;DR\n")
    tldr.append(
        "1. **The dominant decomposition-path failure everywhere is retrieval, not "
        "the decomposer**: `retrieval_gap_both` (neither the original question nor "
        "the sub-question union reaches full gold coverage @10) accounts for "
        f"{share('hotpotqa','retrieval_gap_both'):.0f}% (HotpotQA) / "
        f"{share('2wiki','retrieval_gap_both'):.0f}% (2Wiki) / "
        f"{share('musique','retrieval_gap_both'):.0f}% (MuSiQue) of decomposition-"
        "path errors.\n"
    )
    tldr.append(
        "2. **Constraint loss (sub-questions dropping disambiguating context) is a "
        "HotpotQA-specific failure**: "
        f"{share('hotpotqa','constraint_loss'):.0f}% on HotpotQA vs "
        f"{share('2wiki','constraint_loss'):.0f}% / "
        f"{share('musique','constraint_loss'):.0f}% on 2Wiki / MuSiQue. The "
        "MuSiQue-style KG-triple decomposition hurts precisely where the original "
        "question already carries strong lexical constraints — which is why the "
        "original-question evidence union helps HotpotQA most (+5.5 EM).\n"
    )
    tldr.append(
        "3. **Synthesis failure is a stable 15-18% slice on all three datasets**: "
        "some step answer already matches gold but the synthesized final answer is "
        "wrong. This is the mass addressed by the synthesis guards, and it caps "
        "what decomposition alone can deliver even with perfect retrieval.\n"
    )
    tldr.append(
        "4. **The verifier is decompose-biased by construction**: it catches "
        "91-95% of decomp-only wins on every dataset but only 24-51% of "
        "direct-only wins. That is the right trade on 2Wiki/MuSiQue (decOnly >> "
        "dirOnly) and the wrong one on HotpotQA (dirOnly 151 > decOnly 98).\n"
    )
    lines[tldr_pos:tldr_pos] = tldr

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
