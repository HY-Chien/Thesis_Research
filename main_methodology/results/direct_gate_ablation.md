# Direct-only Confidence-Gated RAG vs. Decompose+Gate（凸顯拆解的貢獻）

> 2026-07-04。回答的問題:**「不拆解、只用信心度決定要不要 RAG」能走多遠?**
> 若這個 baseline 追不上 SEEK(decompose + per-step gate),就證明拆解的貢獻
> 無法被 confidence gating 取代——slide p39–41(never/gate/always、gating
> efficiency、τ sensitivity)缺的那塊對照。

## Setup

- **Direct-gate baseline**:不拆解。先無 context 直接答原問題並量 `max_ent`;
  `max_ent < τ` → 採 parametric 答案(跳過檢索),`max_ent ≥ τ` → 檢索原問題
  (honest wiki-21M BM25)後重答。與 canonical per-step gate 同一條規則、同一個
  訊號,只是套在原問題而非 sub-question 上。
- 模型/設定與主表完全一致:Llama-3-8B-Instruct + 12K three-source adapter、
  mode C、greedy、1000Q(dev 前 1000)、HotpotQA/2Wiki/MuSiQue k=8、PopQA k=3。
- 實作 = 離線 replay:parametric 端 `scripts/run_direct_norag_probe.py` →
  `results/{ds}_1000_direct_norag.jsonl`(新生成);RAG 端重用現有
  direct+always-RAG traces(`{ds}_1000_single_k8.jsonl` /
  `popqa_1000_wiki_single_top3.jsonl`)。greedy 且兩候選互相獨立 → replay 與
  live gated run 等價,τ 整條曲線免費。`scripts/replay_direct_gate.py`
  內建 sanity check:τ=0 精確重現主表 direct-always EM(30.3/18.0/10.2/37.1)✓。
- decompose 三欄沿用 §4.6 3-way ablation(slide 39 數字),同 pool 同題集。

## 主表:2×3 矩陣(EM,honest wiki-21M,1000Q)

| Dataset | Dir-never | Dir-gate τ=0.3 | Dir-gate best-τ | Dir-always | Dec-never | **Dec-gate(SEEK)** | Dec-always | Δ(Dec-gate − Dir best) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 18.2 | 30.7 (91%) | 30.7 @τ=0.3 | 30.3 | 15.7 | 25.0 | 25.1 | −5.7 * |
| 2Wiki    | 12.6 | 18.5 (95%) | 18.6 @τ=0.4 | 18.0 | 29.4 | **47.2** | 47.1 | **+28.6** |
| MuSiQue  |  5.9 | 10.4 (98%) | 10.8 @τ=0.8 | 10.2 |  9.6 | **24.7** | 25.0 | **+13.9** |
| PopQA    | 22.4 | 37.1 (99%) | 39.2 @τ=2.0 (76%) | 37.1 | 24.2 | **51.5** | 51.4 | **+12.3** |

(括號 = retrieval rate。* HotpotQA 是已知例外,見下。)

## 結論

1. **拆解的貢獻無法被 confidence gating 取代。** 就算讓 direct-gate baseline
   事後挑最好的 τ(對 baseline 最寬容的比較),2Wiki/MuSiQue/PopQA 仍輸
   decompose+gate **+12.3 ~ +28.6 EM**。不拆解時,gate 只能在「parametric 答案」
   和「單一原問題檢索的答案」之間二選一;多跳題兩個候選常常都錯
   (single-query 檢索 cover 不了第二跳,§4.4 M3 已量化),再好的 gate 也救不了。
2. **不是訊號不好,是天花板太低。** AUROC(max_ent → parametric 答錯)=
   HotpotQA 0.813 / 2Wiki 0.781 / MuSiQue 0.716 / PopQA 0.748——訊號本身可用,
   但 direct-gate 相對 direct-always 只擠出 ≤ +0.6 EM(PopQA best-τ +2.1)。
   gate 的價值被「兩個候選的品質」限制住;把候選變好的是**拆解**
   (sub-question 檢索 full-coverage 大幅提升,§4.3/§4.4),不是 gate。
3. **拆解連 parametric-only 都幫。** Dec-never vs Dir-never:2Wiki 29.4 vs 12.6、
   MuSiQue 9.6 vs 5.9、PopQA 24.2 vs 22.4(HotpotQA 15.7 vs 18.2 例外)。
   逐步答小問題比一口氣答多跳原問題更能把 parametric 知識引出來。
4. **HotpotQA 例外與既有敘事一致**(§3.3/§4.2/§4.6):HotpotQA 的第二跳常與
   原問題共享 surface entity,單 query 檢索已 cover(M3:decomposition 對
   HotpotQA recall 幫助最小),direct 路徑本來就強(dirOnly 180 > decOnly 97)。
   注意 full SEEK(verifier 路由)= **31.8 ≥ direct-gate 30.7**——完整系統
   透過 per-query 路由把 direct 的優勢吃回來,並不輸這個 baseline。
5. 2Wiki 小註:direct 路徑 parametric F1(31.9)> RAG F1(27.6)但 EM 相反
   (12.6 < 18.0)——單 query 檢索在 2Wiki 常拉進撞名實體的干擾文章,
   partial-credit token 反而變少;這是 direct 檢索品質差的另一個症狀,
   不影響 EM 結論。

**Slide 一句話:** Confidence-gated retrieval *without decomposition* plateaus at
the single-query retrieval ceiling (2Wiki 18.6 / MuSiQue 10.8 / PopQA 39.2 EM even
with the best τ); decomposition, not the gate, is what lifts multi-hop QA
(47.2 / 24.7 / 51.5).

## 附錄 A:direct-gate 全 τ sweep(EM / F1 / retrieval rate)

| τ | HotpotQA | 2Wiki | MuSiQue | PopQA |
|---|---|---|---|---|
| 0 (=always) | 30.3 / 41.5 / 100% | 18.0 / 27.6 / 100% | 10.2 / 17.6 / 100% | 37.1 / 47.7 / 100% |
| 0.1 | 30.4 / 41.5 / 95% | 18.2 / 27.7 / 99% | 10.3 / 17.7 / 99% | 37.1 / 47.7 / 100% |
| 0.2 | 30.4 / 41.7 / 93% | 18.2 / 27.6 / 97% | 10.4 / 17.7 / 99% | 37.2 / 47.7 / 99% |
| **0.3 (matched)** | **30.7 / 41.8 / 91%** | **18.5 / 28.0 / 95%** | **10.4 / 17.7 / 98%** | **37.1 / 47.6 / 99%** |
| 0.4 | 30.5 / 41.6 / 88% | 18.6 / 28.2 / 93% | 10.4 / 17.7 / 97% | 37.2 / 47.6 / 98% |
| 0.5 | 30.4 / 41.3 / 86% | 18.5 / 28.5 / 90% | 10.6 / 17.8 / 96% | 37.3 / 47.8 / 98% |
| 0.6 | 30.2 / 41.1 / 83% | 18.5 / 28.8 / 88% | 10.6 / 17.8 / 95% | 37.4 / 47.9 / 97% |
| 0.8 | 30.3 / 41.2 / 75% | 17.4 / 30.4 / 79% | 10.8 / 17.9 / 92% | 38.0 / 48.0 / 94% |
| 1.0 | 29.6 / 40.5 / 70% | 17.0 / 31.1 / 74% | 10.6 / 17.7 / 90% | 38.0 / 47.7 / 92% |
| 1.2 | 29.3 / 39.9 / 66% | 17.3 / 32.0 / 70% | 10.5 / 17.6 / 86% | 38.7 / 47.8 / 89% |
| 1.5 | 28.4 / 39.0 / 59% | 16.6 / 32.9 / 65% | 10.3 / 17.5 / 81% | 39.0 / 47.4 / 83% |
| 2.0 | 27.4 / 38.1 / 49% | 16.4 / 34.0 / 54% |  9.7 / 16.8 / 71% | 39.2 / 46.8 / 76% |
| 3.0 | 23.8 / 33.7 / 27% | 14.2 / 32.7 / 34% |  8.6 / 15.3 / 46% | 36.3 / 42.1 / 53% |
| ∞ (=never) | 18.2 / 26.9 / 0% | 12.6 / 31.9 / 0% |  5.9 / 12.4 / 0% | 22.4 / 24.1 / 0% |

## 附錄 B:AUROC(max_ent → parametric 答錯)

| Dataset | AUROC | 對照 §4.6 step-level AUROC(entropy→never-RAG 錯) |
|---|---:|---|
| HotpotQA | 0.813 | 0.696 |
| 2Wiki | 0.781 | 0.540(step-level 近隨機;query-level 訊號好很多) |
| MuSiQue | 0.716 | 0.809 |
| PopQA | 0.748 | 0.749 |

註:本表是 **query-level**(原問題 parametric 答案的 entropy 對「該答案錯」),
§4.6 是 **step-level**(decompose 之後 sub-answer entropy)。2Wiki 在 query-level
訊號其實不差(0.781)——direct-gate 還是輸,再次說明瓶頸在候選品質不在訊號。

## 重現

```bash
# parametric probe(4×1000,新生成;雙 GPU 各半)
python scripts/run_direct_norag_probe.py \
  --adapter checkpoints/llama3-8b-decomposer-musique-2wiki-popqa \
  --dev datasets/2wiki/raw/dev_first1000.jsonl \
  --out results/2wiki_1000_direct_norag.jsonl
# (hotpotqa/musique 同;popqa 用 datasets/popqa/raw/dev.jsonl --limit 1000)

# 離線 gate replay + τ sweep + AUROC(含 τ=0 sanity check)
python scripts/replay_direct_gate.py
```

檔:`results/{ds}_1000_direct_norag.jsonl` ×4、`results/direct_gate_replay_dump.json`。
