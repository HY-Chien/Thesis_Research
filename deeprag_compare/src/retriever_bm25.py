"""
BM25 retrieval over the Wikipedia `psgs_w100` Elasticsearch index.

This matches DeepRAG's retrieval setup: `psgs_w100.tsv.gz` indexed into
Elasticsearch (see https://github.com/gxy-gxy/DeepRAG#wikipedia-indexing).
We query the local `wiki` index already built at port 9200 (~21M docs).
"""

from typing import List, Dict
from elasticsearch import Elasticsearch


class BM25Retriever:
    def __init__(self, index: str = "wiki", host: str = "localhost", port: int = 9200):
        self.index = index
        self.es = Elasticsearch([f"http://{host}:{port}"], request_timeout=30)

    # The `wiki` index (~21M psgs_w100 passages) stores passage text under `txt`;
    # the dataset-specific indices (hotpotqa / 2wikimultihopqa / musique) use
    # `paragraph_text`. We auto-detect below.
    _TEXT_FIELD = {
        "wiki": "paragraph_text",
        "hotpotqa": "paragraph_text",
        "2wikimultihopqa": "paragraph_text",
        "musique": "paragraph_text",
    }

    def _tf(self) -> str:
        return self._TEXT_FIELD.get(self.index, "paragraph_text")

    def search(self, query: str, topk: int = 3) -> List[Dict]:
        tf = self._tf()
        body = {
            "size": topk,
            "query": {
                "bool": {
                    "should": [
                        {"match": {tf: query}},
                        {"match": {"title": query}},
                    ]
                }
            },
        }
        res = self.es.search(index=self.index, body=body)
        hits = res["hits"]["hits"]
        return [
            {
                "title": h["_source"].get("title", ""),
                "text": h["_source"].get(tf, ""),
                "score": h["_score"],
            }
            for h in hits
        ]

    @staticmethod
    def format_context(docs: List[Dict]) -> str:
        """Format retrieved docs as a prompt-friendly context block."""
        lines = []
        for i, d in enumerate(docs, 1):
            lines.append(f"[{i}] {d['title']}: {d['text']}")
        return "\n".join(lines)


if __name__ == "__main__":
    r = BM25Retriever()
    docs = r.search("Who is the director of the film Nosferatu?", topk=3)
    for d in docs:
        print(f"[{d['score']:.2f}] {d['title']}: {d['text'][:120]}...")
