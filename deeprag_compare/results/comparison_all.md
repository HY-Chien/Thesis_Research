# Full Comparison — All Methods

Model: Llama-3-8B-Instruct (base) + DeepRAG-8b (decomp).  N=1000 dev.

## Table 1: Multi-hop QA vs DeepRAG paper

| Method | hotpotqa EM | hotpotqa F1 | 2wikimultihopqa EM | 2wikimultihopqa F1 | musique EM | musique F1 |
|---|---|---|---|---|---|---|
| Ours CoT (no-RAG) | 25.2 | 35.6 | 31.2 | 39.3 | 11.5 | 20.3 |
| Ours Single-RAG | 32.4 | 44.0 | 37.9 | 47.0 | 19.2 | 29.0 |
| Ours SC-gated@20 | 27.5 | 38.0 | 33.1 | 42.1 | 13.5 | 22.8 |
| Ours SC-gated@50 | 30.6 | 41.4 | 34.8 | 44.2 | 16.8 | 26.1 |
| Ours SC-gated@80 | 32.1 | 43.2 | 37.3 | 46.6 | 18.2 | 28.1 |
| Ours Dynamic-RAG (continue, basic) | 20.6 | 32.2 | 20.6 | 30.1 | 10.0 | 19.1 |
| Ours Dynamic-RAG (restart, basic) | 19.1 | 30.7 | 17.2 | 28.0 | 10.3 | 19.4 |
| Ours Dynamic-RAG (regen+masked, planA) | 20.2 | 31.3 | 19.5 | 29.2 | 10.1 | 19.0 |
| DeepRAG-8b + learned gate | 39.6 | 51.1 | 50.2 | 54.8 | 23.7 | 30.9 |
| DeepRAG-8b + SC gate (th=17) | 32.0 | 43.5 | 39.0 | 44.1 | 18.8 | 27.2 |
| DeepRAG-8b + SC gate (th=19) | 36.7 | 48.5 | 47.3 | 52.0 | 22.7 | 30.6 |
| DeepRAG-8b + SC gate (th=21) | 39.5 | 50.4 | 50.2 | 54.9 | 23.0 | 30.4 |
| DeepRAG-8b + SC gate (th=23) | 39.7 | 50.4 | 51.1 | 55.9 | 22.8 | 30.2 |
|  |  |  |  |  |  |  |
| CoT (paper) | 27.2 | 37.8 | 28.2 | 34.9 | 13.7 | 23.0 |
| FLARE (paper) | 23.4 | 32.9 | 30.5 | 37.5 | 17.6 | 23.6 |
| DRAGIN (paper) | 27.6 | 38.1 | 29.1 | 35.7 | 11.8 | 20.0 |
| UAR (paper) | 29.7 | 40.7 | 34.8 | 42.4 | 19.1 | 28.4 |
| DeepRAG (paper) | 40.7 | 51.5 | 48.1 | 53.3 | 22.5 | 30.4 |

## Table 2: PopQA (single-hop factoid)

| Method | EM | F1 |
|---|---|---|
| Ours CoT (no-RAG) | 20.4 | 23.1 |
| Ours Single-RAG | 41.6 | 51.3 |
| Ours SC-gated@20 | 22.6 | 27.5 |
| Ours SC-gated@50 | 29.5 | 35.6 |
| Ours SC-gated@80 | 35.0 | 42.6 |
| Ours Dynamic-RAG (continue, basic) | 7.7 | 17.8 |
| Ours Dynamic-RAG (restart, basic) | 8.5 | 18.8 |
| Ours Dynamic-RAG (regen+masked, planA) | 7.7 | 17.6 |
