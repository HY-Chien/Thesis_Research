# Canonical evaluation questions

This directory contains the exact 1,000-question subsets used to align SEEK
with DeepRAG. Each file is a byte-for-byte copy of the corresponding tracked
file under `deeprag_compare/data/top1000/`.

| File | Official source and split | Selection | Upstream terms |
|---|---|---|---|
| `hotpotqa.jsonl` | HotpotQA distractor `dev` | first 1,000 | dataset: [CC BY-SA 4.0](https://github.com/hotpotqa/hotpot#license) |
| `2wikimultihopqa.jsonl` | 2WikiMultiHopQA `dev` | first 1,000 | official repository: [Apache-2.0](https://github.com/Alab-NII/2wikimultihop); consult upstream for dataset terms |
| `musique.jsonl` | MuSiQue-Ans `dev` | first 1,000 | dataset: [CC BY 4.0](https://github.com/StonyBrookNLP/musique#data) |
| `popqa.jsonl` | PopQA `test` | first 1,000 | official repository: [MIT](https://github.com/AlexTMallen/adaptive-retrieval); consult upstream for dataset terms |

The selection is positional and deterministic: no shuffle, seed, or sampling
with replacement is involved. This matches DeepRAG's `--sample 1000` logic,
which slices the first 1,000 rows when `--shuffle` is absent.

Every line has this compact schema:

```json
{"qid": "string", "question": "string", "answer": "string or list of strings", "answer_id": "string or null"}
```

`run_seek.py` and `run_seek_dual.py` accept this schema as well as the older
SEEK schema (`id`, scalar `answer`, optional `answer_aliases`).

These compact files intentionally contain no retrieved paragraphs. The final
thesis protocol uses the external 21M-passage Wikipedia Elasticsearch index,
so live inference must pass `--corpus wiki`. Running with the default
`--corpus paragraphs` is a different protocol and requires paragraph-bearing
inputs.

Verify the released copies:

```bash
(cd datasets/evaluation && sha256sum -c SHA256SUMS)
wc -l datasets/evaluation/*.jsonl
```

From the `main_methodology` directory, rebuild the files from official
downloads without author-specific paths:

```bash
python scripts/prepare_eval_subsets.py \
  --hotpotqa /path/to/hotpot_dev_distractor_v1.json \
  --twowiki /path/to/2wiki/dev.json \
  --musique /path/to/musique_ans_v1.0_dev.jsonl \
  --popqa /path/to/popQA.tsv \
  --output-dir /tmp/seek-evaluation \
  --force
```

The source revisions used in the original workstation were not pinned. The
released files and `SHA256SUMS` are therefore the authoritative fixed split.
Please retain the upstream citations and license notices when redistributing
the data.
