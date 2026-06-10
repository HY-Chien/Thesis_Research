"""BM25 retrieval back-ends.

Two scopes:
  PerQuestionBM25  per-question candidate-paragraph pool (distractor setting).
  OpenDomainBM25   shared Elasticsearch corpus (DeepRAG-equivalent open domain).

NOTE: DeepRAG does NOT use the per-question pool — its `--es_index_name`
defaults to "wiki" (psgs_w100, 21M passages) with `--retrieve_topk` 3, i.e.
full open-domain BM25. Use OpenDomainBM25 for a retrieval-equivalent comparison.
"""
from __future__ import annotations

import re
from typing import Iterable

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"\b\w+\b")


def _tok(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]


class PerQuestionBM25:
    """Tokenise + index a question's paragraphs once, then rank queries."""

    def __init__(self, paragraphs: list[dict]):
        # title prepended to body — titles are very informative for BM25
        self.paragraphs = paragraphs
        corpus = [_tok(f"{p.get('title','')} {p.get('paragraph_text','')}")
                  for p in paragraphs]
        # rank_bm25 dies on an empty doc; pad with a dummy token
        corpus = [c or ["_"] for c in corpus]
        self.bm25 = BM25Okapi(corpus)

    def topk(self, query: str, k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(_tok(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.paragraphs[i] for i in ranked[:k]]


def format_context(paragraphs: Iterable[dict]) -> str:
    """Render top-k paragraphs as a numbered context block for the prompt."""
    return "\n\n".join(
        f"[{i+1}] {p.get('title','')}\n{p.get('paragraph_text','')}"
        for i, p in enumerate(paragraphs)
    )


# --- Open-domain retrieval (DeepRAG-equivalent setting) -------------------
# Reuses Adaptive-RAG's existing ElasticsearchRetriever against the shared ES
# `wiki` index (psgs_w100, 21M passages) — the exact corpus DeepRAG retrieves
# from. Same .topk() interface as PerQuestionBM25 so the pipeline is unchanged.
import sys as _sys

_ADAPTIVE_RETRIEVER = "/home/S113065528/Adaptive-RAG/retriever_server"
if _ADAPTIVE_RETRIEVER not in _sys.path:
    _sys.path.insert(0, _ADAPTIVE_RETRIEVER)


class OpenDomainBM25:
    """BM25 over a shared Elasticsearch corpus (e.g. the 21M-passage `wiki`
    index). Constructed once and shared across all questions."""

    def __init__(self, corpus_name: str = "wiki",
                 host: str = "localhost", port: int = 9200):
        from elasticsearch_retriever import ElasticsearchRetriever
        self._r = ElasticsearchRetriever(host=host, port=port)
        self.corpus_name = corpus_name

    def topk(self, query: str, k: int = 3) -> list[dict]:
        # Retry with backoff: under concurrent GPUs the shared ES can 429 /
        # time out. A single failed query must NOT crash a 1000-row run — retry
        # a few times, then degrade gracefully to empty context.
        import time
        last = None
        for attempt in range(4):
            try:
                return self._r.retrieve_paragraphs(
                    self.corpus_name, query_text=query,
                    max_hits_count=k, max_buffer_count=max(100, k))[:k]
            except Exception as e:  # noqa: BLE001 — ES 429/timeout under load
                last = e
                time.sleep(2 ** attempt)
        print(f"[OpenDomainBM25] retrieval gave up after retries: {last}",
              flush=True)
        return []
