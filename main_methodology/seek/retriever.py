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

    def topk(self, query: str, k: int = 5, title_field=None) -> list[dict]:
        # title_field accepted for a uniform interface with OpenDomainBM25;
        # ignored here — the per-question pool already prepends title to body.
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
# Queries the shared Elasticsearch `wiki` index (psgs_w100, 21M passages) —
# the exact corpus DeepRAG retrieves from. The small client is kept here so a
# checkout of SEEK does not depend on Adaptive-RAG's Python source tree.


class OpenDomainBM25:
    """BM25 over a shared Elasticsearch corpus (e.g. the 21M-passage `wiki`
    index). Constructed once and shared across all questions."""

    def __init__(self, corpus_name: str = "wiki",
                 host: str = "localhost", port: int = 9200,
                 query_title_field_too: bool = True):
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise ImportError(
                "Open-domain retrieval requires the 'elasticsearch' package. "
                "Install the project dependencies before using an ES corpus."
            ) from exc
        self._es = Elasticsearch([f"http://{host}:{port}"], request_timeout=30)
        self.corpus_name = corpus_name
        # Match the `title` field as well as `paragraph_text`. Titles are highly
        # informative for finding the right ARTICLE (DeepRAG's BM25 does this);
        # leaving it off handicaps open-domain multi-hop recall. Default ON.
        self.query_title_field_too = query_title_field_too

    def topk(self, query: str, k: int = 3, title_field=None) -> list[dict]:
        # title_field: per-call override of title matching. None -> instance
        # default. PATH-AWARE use: True for short decompose sub-queries (≈ an
        # entity/title, title match surfaces the right article), False for the
        # long single-step original question (title match injects stray hits).
        tf = self.query_title_field_too if title_field is None else title_field
        # Retry with backoff: under concurrent GPUs the shared ES can 429 /
        # time out. A single failed query must NOT crash a 1000-row run — retry
        # a few times, then degrade gracefully to empty context.
        import time
        last = None
        for attempt in range(4):
            try:
                should = [{"match": {"paragraph_text": query}}]
                if tf:
                    should.append({"match": {"title": query}})
                body = {
                    "size": max(100, k),
                    "_source": [
                        "id", "title", "paragraph_text", "url",
                        "is_abstract", "paragraph_index",
                    ],
                    "query": {"bool": {"should": should}},
                }
                result = self._es.search(index=self.corpus_name, body=body)

                # Match Adaptive-RAG's behavior: remove duplicate paragraph
                # text, sort by ES score, and return plain source dictionaries.
                by_text = {}
                for hit in result.get("hits", {}).get("hits", []):
                    source = hit.get("_source", {})
                    text = (source.get("paragraph_text") or "").strip().lower()
                    by_text[text] = hit
                hits = sorted(
                    by_text.values(),
                    key=lambda hit: hit.get("_score") or 0.0,
                    reverse=True,
                )[:k]
                paragraphs = []
                for hit in hits:
                    paragraph = dict(hit.get("_source", {}))
                    paragraph["score"] = hit.get("_score")
                    paragraph["corpus_name"] = self.corpus_name
                    paragraphs.append(paragraph)
                return paragraphs
            except Exception as e:  # noqa: BLE001 — ES 429/timeout under load
                last = e
                time.sleep(2 ** attempt)
        print(f"[OpenDomainBM25] retrieval gave up after retries: {last}",
              flush=True)
        return []
