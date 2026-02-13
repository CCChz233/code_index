# Cross-Encoder 重排（Reranking）设计文档

适用范围：

1. 一阶段检索：`method/retrieval/run_with_index.py`
2. 多步收敛检索：`method/retrieval/run_with_index_convergent.py`

目标：在现有 Dense 检索结果上增加 Cross-Encoder 精排，提高 Top-K 排名质量，尤其是前 10~50 个候选的命中率。

## 1. 背景与问题

当前检索器以 bi-encoder 向量相似度为主，优点是快，缺点是对 query 与代码块的细粒度匹配能力有限。  
Cross-Encoder 直接对 `(query, code_snippet)` 成对打分，通常更适合做“候选集重排”。

关键约束：

1. Dense 索引 `metadata.jsonl` 只保存结构化信息，不保存 block 原文。
2. block 定位依赖 `file_path + start_line + end_line`（0-based 行号）。
3. 最终输出契约必须保持 `loc_outputs.jsonl` 兼容。

## 2. 总体方案

采用“两阶段检索”：

1. Stage-1（召回）：沿用现有 `off/prf/global_local` 检索逻辑，得到 Top-N block 候选。
2. Stage-2（精排）：用 Cross-Encoder 对 Top-N 候选重打分，输出重排后的 Top-K block。
3. Stage-3（映射）：沿用现有文件/模块/实体映射逻辑。

推荐只在候选集上 rerank，而不是全量 block（成本过高）。

## 3. 数据流设计

### 3.1 候选生成

1. 输入：Stage-1 的 `block_scores`（`[(block_idx, dense_score)]`）。
2. 截断：保留 `rerank_top_k_in`（例如 100）。
3. 输出：待重排候选列表。

### 3.2 代码片段回源

由于 metadata 无原文，需从仓库文件按行回源：

1. 使用 metadata 原始 `file_path`（不要用 `clean_file_path` 后的路径回源）。
2. 读取 `repos_root / repo_name / file_path`。
3. 截取行区间：`[start_line, end_line]`（闭区间，0-based）。
4. 可选上下文扩展：`rerank_context_lines`（前后各 N 行）。
5. 文本模板建议：

```text
File: {file_path}
Lines: {start_line+1}-{end_line+1}
Code:
{snippet}
```

### 3.3 成对打分

1. 输入 pair：`(query_text, snippet_text)`。
2. 批量推理：`rerank_batch_size`，动态 padding。
3. 分数来源：
   - 单 logit：直接作为 rerank 分数。
   - 二分类 logits：使用正类 logit 或 softmax 概率。
4. 可选归一化：`none/minmax/sigmoid`。

### 3.4 分数融合策略

两种策略：

1. `replace`：仅用 rerank 分数排序（默认）。
2. `linear`：`final = w_rerank * rerank + w_dense * dense`。

建议默认 `replace`，并保留 `linear` 便于冷启动。

## 4. 组件设计

建议新增模块：

1. `method/retrieval/rerank/cross_encoder.py`
2. `method/retrieval/rerank/common.py`

核心接口：

1. `load_reranker(model_name, device, trust_remote_code) -> (model, tokenizer)`
2. `build_rerank_pairs(query_text, candidates, metadata, repo_root, cfg) -> pairs`
3. `score_pairs(pairs, model, tokenizer, cfg) -> scores`
4. `rerank_candidates(candidates, dense_scores, rerank_scores, cfg) -> reranked_block_scores`

## 5. 与现有脚本的接入点

在 `run_with_index_convergent.py` 中：

1. 先执行 `iterative_block_retrieval(...)` 获取 Stage-1 `block_scores`。
2. 在 `rank_files(...)` 前插入 `maybe_rerank(...)`。
3. `maybe_rerank` 返回最终 `block_scores`，后续映射逻辑不变。

接入位置建议：

1. 候选输出点：`iterative_block_retrieval` 返回后。
2. 文件聚合点：`rank_files(...)` 调用前。

## 6. CLI 参数草案

```text
--enable_rerank
--rerank_model_name
--rerank_trust_remote_code
--rerank_top_k_in                # 进入重排的候选数，默认 100
--rerank_top_k_out               # 重排后保留数，默认与 top_k_blocks 一致
--rerank_batch_size              # 默认 8 或 16
--rerank_max_length              # pair 编码长度，默认 512
--rerank_context_lines           # 代码上下文扩展，默认 0 或 5
--rerank_snippet_max_lines       # 代码片段最大行数，默认 120
--rerank_score_mode              # logit / prob
--rerank_score_norm              # none / minmax / sigmoid
--rerank_fusion                  # replace / linear
--rerank_weight                  # linear 时 rerank 权重，默认 0.8
--dense_weight                   # linear 时 dense 权重，默认 0.2
--rerank_fail_open               # 失败时回退 stage-1 结果（默认 true）
```

## 7. 性能与工程约束

### 7.1 复杂度

总成本约为：`O(num_instances * rerank_top_k_in * CE_forward_cost)`，显著高于纯向量相似度。

### 7.2 降本策略

1. 控制 `rerank_top_k_in`（建议 50~150）。
2. 使用 fp16/bf16 推理（可配置）。
3. 文件内容缓存（按 `file_path` LRU）。
4. pair 构造前先做长度裁剪。
5. 仅对前几轮结果 rerank（若后续扩展到轮内重排）。

### 7.3 稳定性策略

1. 文件读取失败：跳过该候选或退回 dense 分数。
2. 模型推理异常/OOM：按 `rerank_fail_open` 回退到 Stage-1。
3. 强制输出兼容：无论 rerank 成功与否，均输出标准 `loc_outputs.jsonl`。

## 8. 评测设计

### 8.1 对照组

同一批数据、同一索引、同一 Stage-1 参数下比较：

1. Baseline：`off`（无 rerank）
2. Baseline + Rerank
3. PRF（无 rerank）
4. PRF + Rerank
5. Global-Local（无 rerank）
6. Global-Local + Rerank

### 8.2 指标

1. `file/module/entity` Hit@K
2. MRR / nDCG（如已有脚本支持）
3. 空结果比例（`found_files == []`）
4. 每实例耗时（总耗时、rerank 耗时）
5. 显存峰值（可选）

### 8.3 日志与可观测性

建议新增可选落盘：

1. `rerank_trace.jsonl`（每实例 top 候选的 dense/rerank/final 分数）
2. `timing.jsonl`（stage1_ms/rerank_ms/total_ms）

## 9. 推荐默认配置（v1）

1. `enable_rerank=true`
2. `rerank_top_k_in=100`
3. `rerank_top_k_out=50`
4. `rerank_batch_size=8`
5. `rerank_max_length=512`
6. `rerank_context_lines=0`
7. `rerank_snippet_max_lines=120`
8. `rerank_fusion=replace`
9. `rerank_fail_open=true`

## 10. 落地计划

Phase 1（最小可用）：

1. 新增 rerank 模块与 CLI。
2. 仅接入 `run_with_index_convergent.py`。
3. 支持 `replace` 策略与 fail-open。

Phase 2（增强）：

1. 支持 `linear` 融合与分数归一化。
2. 新增 trace/timing 落盘。
3. 增加并行优化与缓存。

Phase 3（扩展）：

1. 接入 hybrid/summary 检索流程。
2. 按查询类型动态启停 rerank。
3. 加入蒸馏或轻量 reranker 方案。

## 11. 风险与开放问题

1. 代码片段回源会引入 I/O 开销，可能成为瓶颈。
2. 不同 Cross-Encoder 模型输出标度差异大，融合策略需调参。
3. 对超长函数的截断策略会影响重排效果，需要实验验证。
4. 若 `file_path` 与本地 repo 不一致，回源失败率会升高，需要路径校验与告警。
