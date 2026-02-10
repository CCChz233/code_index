# 混合检索与多路融合设计文档

## 1. 背景与目标

当前系统已有三类异构索引：Sparse（BM25）、Dense（向量）、Summary（摘要语义）。它们的数据结构、召回粒度和评分范围不一致，直接在检索逻辑中硬编码融合会导致耦合度高、难以扩展。  
目标是提供一套可插拔的多路融合框架，支持并行召回、评分归一化、粒度对齐与融合算法替换。

## 2. 设计原则

1. 插件化优先。新增索引或新融合算法时不改核心引擎逻辑。
2. 粒度可对齐。允许 block 级、文件级、摘要文档级的结果统一到目标粒度。
3. 评分可比较。每一路结果必须经过标准化或 rank 化，避免尺度不一致。
4. 可复现。融合配方和索引路径要落盘为 manifest。

## 3. 术语与范围

1. 索引源（Source）。一个可检索的数据源，如 dense、sparse、summary。
2. 粒度（Granularity）。召回结果的最小单位，如 block、file、doc。
3. 对齐（Alignment）。将不同粒度的结果统一到目标粒度的过程。
4. 融合（Fusion）。多路结果经过标准化后合并成单一排序列表。

本设计仅覆盖“检索期融合”。索引构建仍由各自管线完成。

## 4. 总体架构

```
Query
  -> [Query Analyzer] (可选：实验特性，v1.0 默认关闭)
  -> SourceRetrievers (dense / sparse / summary / ...)
  -> RetrievalResult[]
  -> Aligner (统一到 target granularity + 保留最佳片段信息)
  -> Normalizer (统一分数尺度)
  -> FusionStrategy (RRF / Weighted / LTR)
  -> [Re-ranker] (可选：Cross-Encoder 精排)
  -> Ranked Items (file/module/entity)
  -> Mapper -> loc_outputs.jsonl
```

## 5. 核心数据契约

### 5.1 RetrievalItem

统一结果项，适配不同索引的输出。

字段建议：

1. `item_id`：全局唯一 ID。block 级可用 `file_path:start:end`。
2. `score`：原始分数或相似度。
3. `payload`：最小必要元数据，至少包含 `file_path`，可选 `start_line`、`end_line`、`doc_id`。
4. `best_provenance`：可选。记录聚合前的最佳来源，解决“对齐导致丢失定位”的问题。

`best_provenance` 建议字段：

1. `original_id`：原始 item_id，例如 `file.py:10:20`。
2. `original_score`：原始分数。
3. `snippet`：可选，展示用的简短片段。

### 5.2 RetrievalResult

字段建议：

1. `source`：索引源名称，例如 `dense`、`sparse`、`summary`。
2. `granularity`：`block` / `file` / `doc`。
3. `items`：`List[RetrievalItem]`。
4. `meta`：可选统计信息，如 `top_k`、`elapsed_ms`。

### 5.3 FusionInput

字段建议：

1. `target_granularity`：期望融合粒度，默认 `file`。
2. `results`：对齐后的 `RetrievalResult[]`。
3. `normalize`：标准化策略配置。
4. `weights`：融合权重配置。

## 6. 组件设计

### 6.1 SourceRetriever 接口

职责：读对应索引并返回 `RetrievalResult`。  
每个索引源独立实现，注册到 `method/core/registry.py` 的 `retriever` 命名空间。

接口建议：

1. `retrieve(query_text, repo_name, top_k) -> RetrievalResult`
2. `supports(granularity) -> bool`

并发模型建议：

1. 对 I/O 密集（如外部向量库、搜索服务）使用 async/await。
2. 对本地推理密集（如本地 Dense 模型）使用线程池或进程池隔离，避免阻塞整体请求。

### 6.2 Aligner（粒度对齐）

职责：把 `block` / `doc` 结果统一到 `file` 或 `module` 级。

常用策略：

1. Block -> File：按文件聚合分数，默认 `sum(top_n)`，避免长文件偏置。
2. Doc -> File：根据 `file_path` 直接映射。
3. Function/Class -> Entity：保留 `start_line/end_line` 供 mapper 使用。

推荐增强策略（避免信息丢失）：

1. Block -> File (Max-Decay)：按衰减权重聚合 Top N 分数，并保留 Top1 Block 作为 `best_provenance`。

示例聚合公式：

```
score(file) = s1 + 0.5*s2 + 0.25*s3 + ...
```

延迟对齐方案（可选）：

1. 若目标是精准定位，可在 Block 级融合，Summary 的 File 结果通过映射扩展到 Block 参与融合。

建议提供可配置聚合策略：

1. `sum`
2. `max`
3. `mean`
4. `sum_top_n`（默认，n 可配置）

### 6.3 Normalizer（评分标准化）

职责：将各路分数映射到可比较区间，减少尺度偏差。

建议支持：

1. `rank`：只保留排序，不使用原始分数。
2. `minmax`：映射到 `[0, 1]`。
3. `zscore`：均值方差标准化。
4. `softmax`：将分数变成概率分布。
5. `sigmoid`：更鲁棒的压缩映射，避免极值拉大差距。

默认建议：RRF 使用 `rank`，加权和优先 `sigmoid` 或 `zscore`。  
注意：`minmax` 对单次查询的分数分布非常敏感，容易放大微小差距。

### 6.4 FusionStrategy（融合器）

职责：接收多路 `RetrievalResult`，输出融合后的排序列表。

建议支持：

1. `rrf`：默认，稳健且无需分数对齐。
2. `weighted_sum`：对标准化分数加权。
3. `ltr`：学习排序模型，作为可选扩展。

缺失值处理：

1. 在 RRF 中，若某文档未出现在某路结果中，则视为 `rank = +inf`，得分贡献为 0。
2. 在加权融合中，缺失路的分数贡献为 0，不影响其他路结果。

融合器注册到 `method/core/registry.py` 的 `fusion` 命名空间。

### 6.5 Recipe 与 Manifest

为保证可复现，需要保存融合配方。

建议新增 `manifest.json` 字段：

1. `recipe_id`
2. `sources`：索引路径、top_k、权重、归一化方式
3. `fusion`：算法与参数
4. `target_granularity`
5. `created_at`

落盘路径建议：

`index_root/hybrid/{recipe_id}/manifest.json`

## 7. 粒度对齐策略细节

### 7.1 Block -> File 聚合

推荐默认策略：

```
score(file) = sum(top_n block_score in file)
```

这样兼顾高分块和避免长文件累积分数过大。  
可选增强：使用衰减因子（Max-Decay）抑制重复命中噪声。

### 7.2 Summary 文档与文件的映射

`summary_index.py` 的 doc 通常包含 `file_path`、`module_path`，可直接映射到 file 或 module 级。

### 7.3 目标粒度与 mapper

最终输出仍以 `found_files` / `found_modules` / `found_entities` 为主。  
对于 entity 级，保留 function/class 的 `start_line/end_line` 供 mapper 精确定位。

## 8. 配置方案

推荐新增 YAML 结构，示例如下：

```yaml
target_granularity: file
sources:
  - name: dense
    type: dense
    index_dir: /path/to/dense_index
    top_k: 100
    normalize: minmax
    weight: 0.4
  - name: sparse
    type: sparse
    index_dir: /path/to/sparse_index
    top_k: 200
    normalize: minmax
    weight: 0.3
  - name: summary
    type: summary
    index_dir: /path/to/summary_index
    top_k: 80
    normalize: sigmoid
    weight: 0.3
fusion:
  type: weighted_sum
query_analyzer:
  enabled: false
  # Experimental: v1.0 默认关闭。后续可用规则或模型动态调整权重。
  rules:
    - match: "error_code"
      boost:
        sparse: 1.2
        dense: 0.9
```

CLI 优先级覆盖 YAML。`run_eval.py` 可扩展为 `--fusion_config`。

## 9. 兼容与迁移

1. 保留现有 `hybrid_retriever.py` 作为单一实现，逐步迁移到新引擎。
2. 先引入抽象接口与 registry，不改变现有索引格式。
3. 增量支持 `summary` 融合，最后切换默认入口到新引擎。
4. 若需要 Block 级融合，需保证各路索引共享一致的切分策略与 ID 生成规则；否则强制回退到 file 级对齐。

## 10. 里程碑建议

1. Phase 1：抽象 `SourceRetriever` 与 `FusionStrategy`，接入 dense + sparse。
2. Phase 2：接入 summary，并实现粒度对齐与标准化。
3. Phase 3：加入 recipe manifest 与可复现评测。
4. Phase 4：开放 Re-ranker、LTR 与 Query Analyzer 扩展接口。

## 12. 备注与实现细节

### 12.1 v1.0 默认黄金配置

1. Aligner：`sum_decay`（Top1 + 0.5*Top2 + 0.25*Top3 ...）。
2. Fusion：`rrf`，`k=60`。
3. Normalizer：当使用 `weighted_sum` 时优先 `sigmoid`，避免 `minmax` 放大微小差距。

### 12.2 Query Analyzer 的落地阶段

1. v1.0 仅支持静态权重。
2. Query Analyzer 作为 Phase 4/Experimental 特性，后续通过规则或模型逐步启用。

## 11. 风险与开放问题

1. 粒度不一致可能导致“文件级融合”和“实体级输出”之间信息损失，需要 `best_provenance` 或延迟对齐策略兜底。
2. 标准化方式选择不当会压制某一路优势，需要基于验证集调参。
3. 并行检索会带来 GPU 与 CPU 混合资源争用，需要控制线程池策略。
