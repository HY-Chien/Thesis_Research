# p43 revision — Path-aware Title Retrieval (詳細版)

Feedback addressed: (1) 明確說「比對標題」= 比對 **document（Wikipedia 文章）的標題欄位**；
(2) 直覺化「為什麼拆了開標題、不拆關標題」— 不拆的長問題用標題找會撈進更多 noise。

All retrieval examples below are **real**, re-run live against the ES `wiki` index (21M passages,
BM25, k=5) via `OpenDomainBM25.topk(query, title_field=True/False)` on 2026-07-04.

---

## Slide content (English, paste-ready)

### Title: Path-aware Title Retrieval

**What "title matching" means** (definition box, top of slide):

> Each Wikipedia passage in the index has two fields: **article title** + passage text.
> **Title ON** = the BM25 query scores against **both** the title field and the text.
> **Title OFF** = the query scores against the passage text only.

**One principle, two cases** (replace the current Observation/Strategy bullets):

| | Query looks like | What title matching does | Setting |
|---|---|---|---|
| **Sub-question** (after decomposition) | ≈ a single **entity name** (e.g. "Adriana Trigiani >> work location") | Entity name ≈ the article's title → title match lands **directly on the right article** | **Title ON** |
| **Original question** (no decomposition) | **Long**, many content words (entities + constraints + relation words) | **Every word** can accidentally match some unrelated article's **title** → stray articles crowd out the gold evidence | **Title OFF** |

> Key sentence for the slide: *The switch is not about the path itself — it is about **query length**.
> Decomposition produces short entity-like queries, and only those are safe to match against titles.*

**Real example** (one HotpotQA item, BM25 top-5 article titles, wiki 21M):

Q: *"The director of the romantic comedy 'Big Stone Gap' is based in what New York city?"*
(evidence chain: Big Stone Gap (film) → Adriana Trigiani)

| Query | Title ON — top-5 articles | Title OFF — top-5 articles |
|---|---|---|
| **Original question** (13 words) | Big Stone Gap (film) · Big Stone Gap, Virginia · ⚠️ **Romantic Comedy (play)** · ⚠️ **Romantic Comedy (1983 film)** · Big Stone Gap, Virginia | ✅ Big Stone Gap (film) · ✅ **Adriana Trigiani** ×3 · Roberto Benabib |
| **Sub-question**: "Adriana Trigiani >> work location" | ✅ **Adriana Trigiani** ×5 (exact article, all 5 hits) | Adriana Trigiani ×3 · Big Stone Gap (film) · Big Stone Gap, Virginia |

⚠️ = stray title hit: the words **"romantic comedy"** in the long question literally match articles
*titled* "Romantic Comedy" — a play and a 1983 film that have **nothing to do with the question** —
and they push the actual evidence (the director's page) out of the top-5.

(Second supporting example, optional / backup slide — MuSiQue:
*"What province shares a border with the province where Lago District is located?"* with Title ON
retrieves **Provinces of Finland**, **Badghis Province** (Afghanistan), **Oro Province** (Papua New
Guinea) — the generic word "province" alone hijacks the title match.)

**Results table: keep as-is** (Title-off / Title-on / Path-aware, PopQA · HotpotQA · MuSiQue · 2Wiki).
Path-aware = Title OFF for the direct path's original-question retrieval, Title ON for decompose-path
sub-question retrieval — i.e. take the better column of each row *per query type*, which is why it is
the best or tied on every dataset.

---

## Speaker notes（中文，回答當場被問的問題）

「比對標題」指的是：索引裡每篇 Wikipedia 文章都有「標題」和「內文」兩個欄位，
Title ON 就是 BM25 檢索時把 query 同時去比對標題欄位，Title OFF 就是只比內文。

為什麼拆了才開標題、不拆反而要關？關鍵不是路徑本身，而是 **query 的長度**。
拆解後的子問題幾乎就是一個實體名（例如「Adriana Trigiani 的工作地點」），
實體名跟正確文章的標題幾乎一模一樣，所以用標題比對一撈就中——實測 top-5 五篇全是正確文章。

反過來，不拆的原始問題很長，除了實體還有一堆限制詞和關係詞；
這些詞每一個都可能剛好撞到某篇不相干文章的「標題」。
實例：問「romantic comedy《Big Stone Gap》的導演住在紐約哪裡」，開了標題比對後，
top-5 裡撈進了一齣叫《Romantic Comedy》的舞台劇和一部 1983 年的同名電影——
跟問題完全無關，卻因為標題撞字被排到前面，反而把真正需要的導演頁面擠出去。
所以「不拆就用標題來找」只會引入更多 noise，這就是 path-aware 設計的原因。

---

## Reproduction

```bash
# ES wiki index (port 9200) must be running
conda run -n adaptive_rag python <scratchpad>/title_probe.py
# core call: OpenDomainBM25("wiki").topk(query, k=5, title_field=True|False)
# code: main_methodology/seek/retriever.py:80 (per-call title_field override)
#       main_methodology/seek/pipeline.py:375-381 (title_mode off/global/pathaware)
```
