## Table 1: Multi-hop QA vs DeepRAG

| Method | hotpotqa EM | hotpotqa F1 | 2wikimultihopqa EM | 2wikimultihopqa F1 | musique EM | musique F1 |
|---|---|---|---|---|---|---|
| Ours CoT (no-RAG) | 25.2 | 35.6 | 31.2 | 39.3 | 11.5 | 20.3 |
| Ours Single-RAG | 32.4 | 44.0 | 37.9 | 47.0 | 19.2 | 29.0 |
| Ours Dynamic SC-RAG | 20.8 | 32.3 | 20.6 | 30.2 | 10.1 | 19.1 |
| Ours SC-gated@20 | 27.5 | 38.0 | 33.1 | 42.1 | 13.5 | 22.8 |
| Ours SC-gated@50 | 30.6 | 41.4 | 34.8 | 44.2 | 16.8 | 26.1 |
| Ours SC-gated@80 | 32.1 | 43.2 | 37.3 | 46.6 | 18.2 | 28.1 |
| CoT (paper) | 27.2 | 37.8 | 28.2 | 34.9 | 13.7 | 23.0 |
| FLARE (paper) | 23.4 | 32.9 | 30.5 | 37.5 | 17.6 | 23.6 |
| DRAGIN (paper) | 27.6 | 38.1 | 29.1 | 35.7 | 11.8 | 20.0 |
| UAR (paper) | 29.7 | 40.7 | 34.8 | 42.4 | 19.1 | 28.4 |
| DeepRAG (paper) | 40.7 | 51.5 | 48.1 | 53.3 | 22.5 | 30.4 |

## Table 2: PopQA

| Method | EM | F1 |
|---|---|---|
| CoT (no-RAG) | 20.4 | 23.1 |
| Single-RAG | 41.6 | 51.3 |
| Dynamic SC-RAG | 7.7 | 18.3 |
| SC-gated@20 | 22.6 | 27.5 |
| SC-gated@50 | 29.5 | 35.6 |
| SC-gated@80 | 35.0 | 42.6 |
