"""Offline application of the synthesis value->entity rescue (see
seek/pipeline.py::_rescue_value_answer) to already-generated result JSONLs.

Comparison questions must be answered with an entity, but the LoRA-off
synthesis step sometimes returns the winning VALUE (e.g. '1932') or selects the
wrong entity despite having correct date sub-answers. The deterministic rescue
maps values back to entities and fixes conservative date comparisons — no GPU
needed, results files are updated in place with `pred_orig` / `em_orig` kept on
every changed record (fully reversible).

Usage:  python scripts/rescue_value_answers.py [glob ...]
        default glob: results/*.jsonl
"""
import glob
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_CMP_Q_RE = re.compile(
    r"\b(first|earlier|later|older|younger|longer|earliest|latest|more recently|newer)\b", re.I)
_VALUE_ANS_RE = re.compile(
    r"^\s*\d{3,4}\s*$"
    r"|^\s*\d{1,2}\s+\w+\s+\d{4}\s*$"
    r"|^\s*(january|february|march|april|may|june|july|august"
    r"|september|october|november|december)\b.*\d{4}", re.I)
_YEAR_RE = re.compile(r"\b(\d{3,4})\b")
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH_FIRST_RE = re.compile(
    rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})?(?:st|nd|rd|th)?[,]?\s+(\d{{3,4}})\b",
    re.I,
)
_DAY_FIRST_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?[,]?\s+(\d{{3,4}})\b",
    re.I,
)
_BAD_CANDIDATE_RE = re.compile(
    r"^\s*(who|what|which|when|where|how|why|is|are|was|were|did|does|do)\b",
    re.I,
)


def _norm_span(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _parse_date_key(text: str | None) -> tuple[int, int, int] | None:
    if not text:
        return None
    m = _MONTH_FIRST_RE.search(text)
    if m:
        return int(m.group(3)), _MONTHS[m.group(1).lower().rstrip(".")], int(m.group(2) or 1)
    m = _DAY_FIRST_RE.search(text)
    if m:
        return int(m.group(3)), _MONTHS[m.group(2).lower().rstrip(".")], int(m.group(1))
    m = _YEAR_RE.search(text)
    if m:
        return int(m.group(1)), 1, 1
    return None


def _date_compare_direction(question: str) -> str | None:
    q = (question or "").lower()
    if re.match(r"\s*(is|are|was|were|do|does|did|has|have|had)\b", q):
        return None
    if re.search(r"\b(younger|later|latest|newer|more recently|born later)\b", q):
        return "max"
    if re.search(r"\b(older|earlier|earliest|first|born first|came first)\b", q):
        return "min"
    return None


def _candidate_from_step(step: dict) -> str:
    sub_q = step.get("sub_q") or step.get("raw_sub_q") or ""
    if ">>" in sub_q:
        cand = sub_q.split(">>", 1)[0].strip()
    else:
        cand = sub_q.strip().rstrip("?")
        cand = re.sub(r"^\s*(?:when|what year)\s+(?:did|was|were|is)\s+",
                      "", cand, flags=re.I)
        cand = re.sub(
            r"\s+\b(?:take place|happen|occur|start|begin|release|"
            r"released|published|founded|established|born|opened|open)\b.*$",
            "", cand, flags=re.I,
        ).strip()
    cand = cand.strip(" .,:;\"'")
    if not cand or "#" in cand or _BAD_CANDIDATE_RE.search(cand):
        return ""
    return cand


def _comparison_candidate_for_step(step: dict, question: str) -> str:
    cand = _candidate_from_step(step)
    cand_norm = _norm_span(cand)
    if cand_norm and cand_norm in _norm_span(question):
        return cand
    return ""


def _rescue_date_comparison(question: str, answer: str, steps: list[dict]) -> str:
    direction = _date_compare_direction(question)
    if direction is None:
        return answer
    comparable = []
    for step in steps:
        cand = _comparison_candidate_for_step(step, question)
        date_key = _parse_date_key(step.get("final_answer"))
        if cand and date_key is not None:
            comparable.append((date_key, cand))
    if len(comparable) < 2:
        return answer
    comparable.sort(key=lambda x: x[0])
    winner = comparable[0][1] if direction == "min" else comparable[-1][1]
    if not winner or _norm_span(winner) == _norm_span(answer):
        return answer
    return winner


def rescue(question: str, answer: str, steps: list[dict]) -> str:
    rescued = answer
    if _CMP_Q_RE.search(question) and _VALUE_ANS_RE.match(answer or ""):
        hits = [s for s in steps if answer.lower() in (s.get("final_answer") or "").lower()]
        if len(hits) == 1 and ">>" in hits[0].get("sub_q", ""):
            rescued = hits[0]["sub_q"].split(">>")[0].strip()
    return _rescue_date_comparison(question, rescued, steps)


# --- scoring, identical to scripts/run_seek.py ---
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, golds: list[str]) -> bool:
    p = _norm(pred)
    return any(p == _norm(g) for g in golds)


def f1(pred: str, golds: list[str]) -> float:
    p = _norm(pred).split()
    best = 0.0
    for g in golds:
        gt = _norm(g).split()
        if not p or not gt:
            best = max(best, float(p == gt))
            continue
        common = Counter(p) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(p), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def load_aliases(ds: str) -> dict[str, list[str]]:
    path = ROOT / "datasets" / ds / "raw" / "dev.jsonl"
    out = {}
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            out[str(r["id"])] = [r["answer"]] + (r.get("answer_aliases") or [])
    return out


def dataset_of(fname: str) -> str | None:
    for ds in ("2wiki", "hotpotqa", "musique", "popqa"):
        if Path(fname).name.startswith(ds):
            return ds
    return None


def main() -> None:
    patterns = sys.argv[1:] or [str(ROOT / "results" / "*.jsonl")]
    files = sorted(f for pat in patterns for f in glob.glob(pat))
    alias_cache: dict[str, dict] = {}
    rows = []
    for fp in files:
        ds = dataset_of(fp)
        if ds is None:
            continue
        if ds not in alias_cache:
            alias_cache[ds] = load_aliases(ds)
        aliases = alias_cache[ds]
        recs = [json.loads(l) for l in open(fp)]
        if not recs or "pred" not in recs[0] or "steps" not in recs[0]:
            continue
        changed = 0
        for r in recs:
            new_pred = rescue(r["question"], r["pred"], r.get("steps") or [])
            if new_pred == r["pred"]:
                continue
            golds = aliases.get(str(r["id"])) or [r["gold"]]
            r["pred_orig"], r["em_orig"] = r["pred"], r["em"]
            r["pred"], r["em"] = new_pred, int(em(new_pred, golds))
            r["rescued"] = True
            changed += 1
        n = len(recs)
        em_now = 100 * sum(r["em"] for r in recs) / n
        em_was = 100 * sum(r["em_orig"] if r.get("rescued") else r["em"] for r in recs) / n
        f1_now = f1_was = 0.0
        for r in recs:
            golds = aliases.get(str(r["id"])) or [r["gold"]]
            f1_now += f1(r["pred"], golds)
            f1_was += f1(r["pred_orig"] if r.get("rescued") else r["pred"], golds)
        f1_now, f1_was = 100 * f1_now / n, 100 * f1_was / n
        if changed:
            with open(fp, "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows.append((Path(fp).name, n, changed, em_was, em_now, f1_was, f1_now))

    print(f"{'file':52s} {'n':>5s} {'chg':>4s} {'EM_old':>7s} {'EM_new':>7s} {'F1_old':>7s} {'F1_new':>7s}")
    for name, n, chg, em_was, em_now, f1_was, f1_now in rows:
        flag = "  <-- " if chg else ""
        print(f"{name:52s} {n:5d} {chg:4d} {em_was:7.1f} {em_now:7.1f} {f1_was:7.1f} {f1_now:7.1f}{flag}")


if __name__ == "__main__":
    main()
