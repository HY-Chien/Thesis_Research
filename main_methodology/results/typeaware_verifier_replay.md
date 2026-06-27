# Type-aware Verifier Offline Replay

Date: 2026-06-23

This replay tests a conservative verifier tweak without rerunning the LLM. It
reads existing `run_seek_dual.py` traces, keeps the saved direct/decompose
answers, and only changes the final direct-vs-decompose choice.

## Rule

Only applies when the original verifier reason is `auto_fallback_decomp`.

The rule switches to the direct answer when the direct path is sufficiently
preferred by a small score:

- low direct entropy (`skip_probe_ent < 0.3`)
- high decompose post-answer entropy (`max post_signal_value > 0.8`)
- decompose answer type mismatches the question type while direct does not
- decompose step answer looks like an entity fragment copied from the sub-query
- decompose final answer equals an intermediate step answer
- decompose final answer appears in the original question when the question is
  not an entity-selection question

Conservative parameters used by default:

```text
direct_entropy_tau = 0.3
decomp_post_tau    = 0.8
margin             = 2.5
```

Script:

```bash
python scripts/replay_typeaware_verifier.py \
  --input results/hotpotqa_1000_verify_v2_pathaware_directctx.jsonl \
  --out results/hotpotqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl
```

## Main Result: Direct-context Union Runs

Metrics below are recomputed with the original dev-file aliases.

| Dataset | Base verifier EM/F1 | Type-aware replay EM/F1 | Delta | Changed | Rescued / Lost |
|---|---:|---:|---:|---:|---:|
| HotpotQA directctx | 34.3 / 44.3 | **35.5 / 46.1** | **+1.2 / +1.9** | 142 | +26 / -14 |
| 2Wiki directctx | 49.9 / 56.4 | **50.0 / 56.6** | **+0.1 / +0.2** | 17 | +1 / -0 |
| MuSiQue directctx | 24.8 / 32.9 | **24.7 / 33.1** | -0.1 / +0.2 | 109 | +5 / -6 |
| PopQA directctx | 52.0 / 54.8 | **51.9 / 54.7** | -0.1 / -0.1 | 7 | +0 / -1 |
| **Average** | **40.3 / 47.1** | **40.5 / 47.6** | **+0.3 / +0.5** | - | - |

HotpotQA oracle headroom after direct-context union is 41.6 EM. The type-aware
replay reduces oracle-missed direct-only cases from 62 to 36, but increases
decomp-only misses from 11 to 25. This is a useful but conservative verifier
refinement; it improves the four-dataset average while keeping non-HotpotQA
regressions at about 0.1 EM.

## Regression Check: Canonical Path-aware Runs

These are offline replays on existing canonical path-aware traces, without
direct-context union.

| Dataset | Base EM | Type-aware replay EM | Delta |
|---|---:|---:|---:|
| HotpotQA | 31.8 | **32.7** | **+0.9** |
| 2Wiki | 46.9 | 46.9 | +0.0 |
| MuSiQue | 24.6 | 24.5 | -0.1 |
| PopQA | 51.8 | 51.7 | -0.1 |

Note: replay F1 uses the primary `gold` field only and is not authoritative for
alias-heavy datasets such as MuSiQue and PopQA. EM is the reliable comparison
for this regression check.

## Interpretation

The aggressive HotpotQA-only replay can reach about 35.9 EM, but hurts 2Wiki
badly. The conservative rule is safer: it improves HotpotQA by 0.9 to 1.2 EM
depending on whether direct-context union is used, while keeping 2Wiki stable.

Recommended next step:

1. Integrate the conservative rule as an optional verifier mode, not as the
   default canonical verifier yet.
2. Run formal no-LLM replay outputs for all four datasets.
3. If accepted, expose it in `run_seek_dual.py` as a flag such as
   `--typeaware-fallback`.
