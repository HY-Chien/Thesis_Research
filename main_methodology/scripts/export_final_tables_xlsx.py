"""Export the final SEEK result tables to a single .xlsx workbook.

Sources (all read-only):
  results/tlsum_fixA_step_gate_final.json  step-gate policies: EM/F1, retrieval
                                           rate, direct-selection counts
  results/verifier_tlsum_fixA_swap.md      final adopted verifier config
  Thesis/contents/chapter04.tex            main comparison table

Usage:
    conda run -n thesis python scripts/export_final_tables_xlsx.py
"""
from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/tlsum_fixA_step_gate_final.json"
OUT = ROOT / "results/SEEK_final_tables.xlsx"

DATASETS = ["hotpotqa", "2wiki", "musique", "popqa"]
PRETTY = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki",
          "musique": "MuSiQue", "popqa": "PopQA"}
# json key -> (display label, deployed?)
POLICIES = [
    ("never", "Never retrieve (tau=inf)"),
    ("tau_0.1165", "tau=0.116500"),
    ("tau_0.07467", "tau=0.074670"),
    ("tau_0.04417", "tau=0.044170"),
    ("tau_0.02216", "tau=0.022160"),
    ("deployed", "tau=0.008461 (DEPLOYED)"),
    ("tau_0.002305", "tau=0.002305 (accuracy-first)"),
    ("always", "Always retrieve (tau=0)"),
]

HDR_FILL = PatternFill("solid", fgColor="1F4E79")
HDR_FONT = Font(color="FFFFFF", bold=True, size=10)
HI_FILL = PatternFill("solid", fgColor="FFF2CC")     # deployed
ACC_FILL = PatternFill("solid", fgColor="DDEBF7")    # accuracy-first
TITLE_FONT = Font(bold=True, size=12)
NOTE_FONT = Font(italic=True, size=9, color="666666")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def style_header(ws, row: int, ncols: int) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border = BORDER


def autosize(ws, widths: dict[int, int]) -> None:
    for col, w in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = w


def write_rows(ws, rows, start: int, ncols: int, highlight=None) -> int:
    r = start
    for row in rows:
        for c, val in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c, value=val)
            cell.border = BORDER
            if c > 1 and isinstance(val, (int, float)):
                cell.alignment = Alignment(horizontal="center")
                if isinstance(val, float):
                    cell.number_format = "0.00"
        tag = highlight(row) if highlight else None
        if tag:
            fill = HI_FILL if tag == "deployed" else ACC_FILL
            for c in range(1, ncols + 1):
                ws.cell(row=r, column=c).fill = fill
            ws.cell(row=r, column=1).font = Font(bold=True, size=10)
        r += 1
    return r


def sheet_main(wb) -> None:
    """Thesis main comparison table (chapter04.tex tab:main-results)."""
    ws = wb.create_sheet("1_Main_Results")
    ws["A1"] = "Main Results - full-corpus Wikipedia (21M passages), first 1000 dev questions"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("Baseline numbers are the Llama-3-8B results reported by DeepRAG. "
                "Averages recomputed over these four datasets. Each cell: EM / F1.")
    ws["A2"].font = NOTE_FONT

    hdr = ["Type", "Method",
           "HotpotQA EM", "HotpotQA F1", "2Wiki EM", "2Wiki F1",
           "MuSiQue EM", "MuSiQue F1", "PopQA EM", "PopQA F1",
           "Avg EM", "Avg F1"]
    ws.append([])
    ws.append(hdr)
    style_header(ws, 4, len(hdr))

    rows = [
        ["Reasoning", "CoT", 27.20, 37.75, 28.20, 34.85, 13.70, 22.97, 21.20, 25.33, 22.58, 30.23],
        ["Reasoning", "CoT-Retrieve", 34.90, 46.85, 35.80, 43.41, 19.10, 28.18, 32.80, 45.87, 30.65, 41.08],
        ["Adaptive", "FLARE", 23.80, 32.88, 30.30, 37.45, 14.50, 23.57, 28.80, 40.61, 24.35, 33.63],
        ["Adaptive", "DRAGIN", 27.60, 38.05, 29.10, 35.68, 11.80, 19.97, 22.60, 28.53, 22.78, 30.56],
        ["Learned adaptive", "DeepRAG", 40.70, 51.54, 48.10, 53.25, 22.50, 30.40, 42.50, 47.80, 38.45, 45.75],
        ["Ours", "SEEK final", 37.50, 48.72, 48.70, 55.63, 26.60, 34.74, 53.70, 55.53, 41.62, 48.66],
    ]
    end = write_rows(ws, rows, 5, len(hdr),
                     highlight=lambda r: "deployed" if r[1] == "SEEK final" else None)

    ws.cell(row=end + 1, column=1,
            value="SEEK vs DeepRAG: +3.17 macro EM, +2.91 macro F1. "
                  "Wins on 2Wiki, MuSiQue, PopQA; HotpotQA -3.20 EM.").font = NOTE_FONT
    autosize(ws, {1: 17, 2: 15, **{i: 11 for i in range(3, 13)}})
    ws.freeze_panes = "C5"


def sheet_threshold(wb, data) -> None:
    """Step-gate threshold sensitivity: per-dataset EM/F1 + retrieval rate."""
    ws = wb.create_sheet("2_StepGate_Thresholds")
    ws["A1"] = "Step-gate threshold sensitivity (tl_sum), per dataset"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("Fixed-chain replay: only the deployed row is a live end-to-end run. "
                "Other rows hold later resolved sub-questions fixed and resynthesise. "
                "Verifier held constant at direct tau=0.0005, post tau=0.0158.")
    ws["A2"].font = NOTE_FONT

    hdr = ["Policy"]
    for ds in DATASETS:
        hdr += [f"{PRETTY[ds]} EM", f"{PRETTY[ds]} F1", f"{PRETTY[ds]} retr%"]
    hdr += ["Macro EM", "Macro F1", "Pooled step retr%"]
    ws.append([])
    ws.append(hdr)
    style_header(ws, 4, len(hdr))

    rows = []
    for key, label in POLICIES:
        row = [label]
        ems, f1s, ret, tot = [], [], 0, 0
        for ds in DATASETS:
            r = data["results"][ds][key]
            row += [r["em"], r["f1"], r["retrieval_rate"]]
            ems.append(r["em"])
            f1s.append(r["f1"])
            ret += r["retrieved_steps"]
            tot += r["total_steps"]
        row += [sum(ems) / len(ems), sum(f1s) / len(f1s), ret / tot * 100]
        rows.append(row)

    def hl(r):
        if "DEPLOYED" in r[0]:
            return "deployed"
        if "accuracy-first" in r[0]:
            return "acc"
        return None

    end = write_rows(ws, rows, 5, len(hdr), highlight=hl)
    ws.cell(row=end + 1, column=1,
            value="Deployed tau was frozen from the development sweep, where it beat "
                  "tau=0.002305 on both macro EM/F1 while retrieving less "
                  "(40.60/47.69 @ 90.8% vs 40.58/47.62 @ 95.5%). Prompt fix A later "
                  "shifted the score distribution.").font = NOTE_FONT
    autosize(ws, {1: 30, **{i: 11 for i in range(2, len(hdr) + 1)}})
    ws.freeze_panes = "B5"


def sheet_routing(wb, data) -> None:
    """Verifier direct-vs-decompose selection counts."""
    ws = wb.create_sheet("3_Routing_Direct_vs_Decomp")
    ws["A1"] = "Final verifier routing: direct vs decomposition candidate"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("Every question runs BOTH paths; these counts are which candidate the "
                "verifier adopted as the final answer. n=1000 per dataset.")
    ws["A2"].font = NOTE_FONT

    hdr = ["Policy"]
    for ds in DATASETS:
        hdr += [f"{PRETTY[ds]} direct", f"{PRETTY[ds]} direct%",
                f"{PRETTY[ds]} decomp%"]
    hdr += ["Pooled direct", "Pooled direct%", "Pooled decomp%"]
    ws.append([])
    ws.append(hdr)
    style_header(ws, 4, len(hdr))

    rows = []
    for key, label in POLICIES:
        row, tot_d, tot_n = [label], 0, 0
        for ds in DATASETS:
            r = data["results"][ds][key]
            n, d = r["n"], r["direct_selected"]
            row += [d, d / n * 100, (n - d) / n * 100]
            tot_d += d
            tot_n += n
        row += [tot_d, tot_d / tot_n * 100, (tot_n - tot_d) / tot_n * 100]
        rows.append(row)

    def hl(r):
        if "DEPLOYED" in r[0]:
            return "deployed"
        if "accuracy-first" in r[0]:
            return "acc"
        return None

    end = write_rows(ws, rows, 5, len(hdr), highlight=hl)
    ws.cell(row=end + 1, column=1,
            value="Routing is nearly invariant to the step-gate threshold (pooled direct "
                  "8.1%-8.2% across all policies): the step gate controls retrieval INSIDE "
                  "the decomposition path, while the verifier selects between paths.").font = NOTE_FONT
    autosize(ws, {1: 30, **{i: 12 for i in range(2, len(hdr) + 1)}})
    ws.freeze_panes = "B5"


def sheet_candidate(wb, data) -> None:
    """Final-system vs decomposition-candidate-only scores."""
    ws = wb.create_sheet("4_Final_vs_DecompCandidate")
    ws["A1"] = "Final system vs decomposition candidate alone"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("'Final' applies the verifier; 'Decomp candidate' scores the "
                "decomposition path answer for every question with no routing.")
    ws["A2"].font = NOTE_FONT

    hdr = ["Policy"]
    for ds in DATASETS:
        hdr += [f"{PRETTY[ds]} final EM", f"{PRETTY[ds]} decomp EM"]
    hdr += ["Macro final EM", "Macro decomp EM", "Routing gain (EM)"]
    ws.append([])
    ws.append(hdr)
    style_header(ws, 4, len(hdr))

    rows = []
    for key, label in POLICIES:
        row, fin, dec = [label], [], []
        for ds in DATASETS:
            r = data["results"][ds][key]
            row += [r["em"], r["decomp_em"]]
            fin.append(r["em"])
            dec.append(r["decomp_em"])
        mf, md = sum(fin) / len(fin), sum(dec) / len(dec)
        row += [mf, md, mf - md]
        rows.append(row)

    def hl(r):
        if "DEPLOYED" in r[0]:
            return "deployed"
        if "accuracy-first" in r[0]:
            return "acc"
        return None

    write_rows(ws, rows, 5, len(hdr), highlight=hl)
    autosize(ws, {1: 30, **{i: 13 for i in range(2, len(hdr) + 1)}})
    ws.freeze_panes = "B5"


def sheet_protocol(wb, data) -> None:
    ws = wb.create_sheet("0_Protocol")
    ws["A1"] = "SEEK final protocol"
    ws["A1"].font = TITLE_FONT

    items = [
        ("Base model", "Llama-3-8B-Instruct"),
        ("Decomposer", "12K three-source LoRA (MuSiQue + 2Wiki + PopQA), r=16 alpha=32"),
        ("Corpus", "psgs_w100 Wikipedia, ~21M passages, shared Elasticsearch index"),
        ("Retrieval", "BM25 top-k=8 for ALL FOUR datasets"),
        ("Title policy", "path-aware: direct=text only, sub-questions=title+text"),
        ("Evidence", "original-question evidence union on the decomposition path"),
        ("Step gate signal", data["protocol"]["step_gate_signal"]),
        ("Step gate rule", data["protocol"]["step_gate_rule"]),
        ("Step gate tau (deployed)", data["protocol"]["deployed_step_tau"]),
        ("Verifier direct signal", data["protocol"]["direct_verifier_signal"]),
        ("Verifier direct tau", data["protocol"]["direct_verifier_tau"]),
        ("Verifier post signal", data["protocol"]["post_verifier_signal"]),
        ("Verifier post tau", data["protocol"]["post_verifier_tau"]),
        ("Entropy operational?", str(data["protocol"]["entropy_operational"])),
        ("Replay type", data["protocol"]["replay"]),
        ("Eval", "alias-aware normalised EM / token F1, first 1000 dev questions"),
    ]
    ws.append([])
    ws.append(["Field", "Value"])
    style_header(ws, 3, 2)
    r = 4
    for k, v in items:
        ws.cell(row=r, column=1, value=k).border = BORDER
        ws.cell(row=r, column=1).font = Font(bold=True, size=10)
        ws.cell(row=r, column=2, value=v).border = BORDER
        r += 1

    ws.cell(row=r + 1, column=1, value="Sanity: deployed synthesis mismatches").font = Font(bold=True)
    for ds in DATASETS:
        r += 1
        ws.cell(row=r + 1, column=1, value=PRETTY[ds])
        ws.cell(row=r + 1, column=2,
                value=f"{data['deployed_synthesis_mismatches'][ds]} / 1000")

    ws.cell(row=r + 3, column=1,
            value="Source: results/tlsum_fixA_step_gate_final.json").font = NOTE_FONT
    autosize(ws, {1: 30, 2: 72})


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    wb = Workbook()
    wb.remove(wb.active)
    sheet_protocol(wb, data)
    sheet_main(wb)
    sheet_threshold(wb, data)
    sheet_routing(wb, data)
    sheet_candidate(wb, data)
    wb.save(OUT)
    print(f"wrote {OUT}")
    for ws in wb.worksheets:
        print(f"  sheet: {ws.title}  ({ws.max_row} rows x {ws.max_column} cols)")


if __name__ == "__main__":
    main()
