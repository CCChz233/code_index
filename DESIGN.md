# IR-Base 设计文档（索引构建与评估一体化）

本文档用于**确定最终项目架构**，目标是支持多种索引类型（稠密 / 稀疏 / 摘要 / 混合）并保持可扩展与可复现。

---

## 1. 背景与目标

### 1.1 背景
当前代码已完成 v2 重构（`method/indexing` + `method/retrieval`），具备完整的稠密索引与评估流程。未来需要在同一项目内纳入稀疏索引（BM25/TF-IDF 等）、摘要索引以及混合检索。

### 1.2 仓库画像（LocBench Repos）
基于当前本地仓库（`/Users/chz/code/locbench/locbench_repos`）的快速统计：
- **仓库数量**：165 个
- **代码文件规模**：约 101,830 个源码文件
- **语言分布（Top10）**：
  - `.py` ~81,816
  - `.ts` ~5,515
  - `.go` ~4,297
  - `.js` ~2,473
  - `.h` ~2,005
  - `.c` ~1,604
  - `.cs` ~1,412
  - `.php` ~970
  - `.cpp` ~790
  - `.java` ~549
- **极端文件大小**：存在超过 20MB 的单文件（如 docs/search.js 一类的构建产物）

**本项目只对 Python 代码建索引**，因此实际扫描将只保留 `.py` 文件，其余语言统一跳过。系统需要重点处理**大文件跳过**、**非 UTF-8 容错**、**可控的遍历规则**，并确保排序稳定与复现一致。

### 1.3 目标
- **统一入口**：一套 CLI 支持索引构建与评估。
- **统一数据契约**：不同索引类型共享一致的 Block 与 metadata schema。
- **可插拔**：新增策略只需新增模块与注册，不改核心流程。
- **可组合**：混合检索通过融合器组合多个检索器。
- **可复现**：索引构建过程可追溯，结果可对比。

### 1.4 非目标
- 不在本阶段重写 mapper / dependency_graph / repo_index。
- 不引入完全新的评估指标体系。
- 不强制统一所有第三方依赖版本。

---

## 2. 设计原则

1. **稳定契约优先**：Block 与 metadata 的字段名优先稳定。
2. **最小耦合**：Chunker、Encoder、Indexer、Retriever、Fusion、Store 解耦。
3. **可配置可复用**：YAML/CLI 双入口，CLI 覆盖 YAML。
4. **依赖分层**：核心依赖最小化，策略依赖按需加载。
5. **可观测**：日志与 manifest 输出帮助定位问题。
6. **语言过滤**：仅索引 Python（`.py`），其余语言文件全部跳过。
7. **大文件保护**：对超大文件/二进制文件进行跳过或降级处理。

---

## 3. 核心数据契约

### 3.1 Block
- `file_path`（str）
- `start_line`（int, 0-based）
- `end_line`（int, 0-based）
- `content`（str）
- `block_type`（str）
- 可选字段：`context_text`, `function_text`, `summary_text`, `original_content`

> 说明：AST 提供的行号是 **1-based**，在进入 Block 前需统一转换到 **0-based** 的 `start_line/end_line`，避免 Off-by-one。

### 3.2 ChunkResult
- `blocks: List[Block]`
- `metadata: Dict[str, Dict]`（结构化信息）
- `aux: Dict[str, Dict]`（临时信息，默认不落盘）

### 3.3 IndexArtifact（建议新增）
- `index_type`（dense/sparse/summary/hybrid）
- `strategy`（chunker 策略）
- `schema_version`
- `repo_name`
- `block_count`
- `build_time`
- `config_hash`

---

## 4. 架构与组件

### 4.1 组件职责
- **Core**：稳定契约与共享能力（Block / ChunkResult / IndexArtifact / registry / store）
- **Chunker**：从仓库生成 Block
- **Encoder**：把 Block 编码成特征（稠密向量 / 稀疏向量 / summary）
- **Indexer**：把特征写入索引存储
- **Retriever**：查询索引，返回 block scores
- **Fusion**：融合多个检索结果

### 4.2 流水线示意
```
Repo -> Chunker -> Blocks -> Encoder -> Features -> Indexer -> Index Store
Bug  -> Retriever -> Block Scores -> (Fusion) -> Ranked Blocks -> Mapper -> Output
```

### 4.3 语言过滤（Python-only）
仅保留 Python 文件作为索引输入：

- `.py` → 允许使用 AST/function chunker（ir_function / function_level）
- 非 `.py` → 一律跳过（不参与索引）

这样可以显著降低解析失败与行为漂移风险，并保证结果复现一致。

### 4.4 AST 解析安全带（必选）
为避免异常文件拖垮批量构建，AST 解析必须具备以下保护：

- **超时机制**：单文件 AST 解析超时则跳过该文件
- **递归深度限制**：防止极端深度语法树导致崩溃
- **失败降级策略**：解析失败 → 记录日志并跳过（不影响主流程）

### 4.5 索引范围与过滤规则（推荐默认）
为保证性能与复现稳定，建议固定以下过滤规则（可通过 CLI 覆盖）：

- **语言过滤**：仅 `.py`
- **目录跳过**：`**/.git/**`, `**/node_modules/**`, `**/dist/**`, `**/build/**`, `**/target/**`, `**/__pycache__/**`, `**/.venv/**`, `**/venv/**`
- **大文件阈值**：默认跳过超过 `1~2 MB` 的源文件（可调）
- **编码容错**：非 UTF-8 文件允许 `errors=ignore` 读取

---

## 5. 目录结构（最终建议）

```
method/
  core/                 # Block / ChunkResult / IndexArtifact / registry / store
    store/
      filesystem.py

  indexing/             # 仅负责“构建索引”
    chunker/            # 已有 chunker 模块
    encoder/
      dense/
      sparse/
      summary/
    indexer/
      dense_indexer.py
      sparse_indexer.py
      summary_indexer.py

  retrieval/            # 仅负责“检索与评估”
    dense_retriever.py
    sparse_retriever.py
    summary_retriever.py
    hybrid_retriever.py
    fusion/
      rrf.py
      weighted.py

  cli/                  # 统一入口
    build_index.py
    run_eval.py
```

> 当前已落地 `method/indexing` 与 `method/retrieval` 的基本结构，后续按 Phase 继续细化组件拆分。

---

## 6. 索引类型设计

> 说明：以下所有索引类型的输入均为 **Python 代码块（`.py`）**。

### 6.1 稠密索引（Dense）
- 输入：Block
- 编码：CodeRankEmbed / SFR
- 索引输出：`embeddings.pt` + `metadata.jsonl`

### 6.2 稀疏索引（Sparse）
- 输入：Block
- 编码：BM25/TF-IDF
- 索引输出：稀疏矩阵（`.npz`）+ `vocab.json` + `metadata.jsonl`

> 约定：使用 `scipy.sparse` 的 `.npz` 存储矩阵，`vocab.json` 存储词表与索引映射；不使用 pickle。

### 6.3 摘要索引（Summary）
- 输入：Block
- 编码：LLM summary
- 索引输出：summary.jsonl + `metadata.jsonl`

### 6.4 混合索引（Hybrid）
- 不单独构建新索引
- 运行时融合 Dense/Sparse/Summary 的检索结果

---

## 7. 索引存储与目录约定

建议统一结构：

```
index_root/
  dense/{strategy}/{repo}/embeddings.pt + metadata.jsonl
  sparse/{strategy}/{repo}/index.* + metadata.jsonl
  summary/{strategy}/{repo}/summary.jsonl + metadata.jsonl
  hybrid/{recipe}/{repo}/manifest.json
```

并在每个索引目录输出 `manifest.json`：
- schema_version
- component versions
- config hash
- build time
- language allowlist / skip patterns
- max file size / binary policy

---

## 8. 配置与注册表

### 8.1 配置优先级
1. CLI 参数
2. YAML 配置
3. 默认值

### 8.2 注册表示例
```python
registry.register("chunker", "ir_function", IRFunctionChunker)
registry.register("encoder", "coderank", CodeRankEncoder)
registry.register("indexer", "dense", DenseIndexer)
registry.register("retriever", "dense", DenseRetriever)
registry.register("fusion", "rrf", RRFusion)
```

---

## 9. CLI 设计（统一入口）

### 9.1 build_index
- `--index_type dense|sparse|summary`
- `--strategy`（chunker 策略）
- `--encoder`（dense/sparse/summary）
- `--config` YAML
- `--lang_allowlist`（按扩展名过滤，默认只允许 .py）
- `--max_file_size_mb`（超大文件跳过阈值）
- `--skip_patterns`（glob 列表，例如 **/node_modules/**）

### 9.2 run_eval
- `--index_type dense|sparse|hybrid`
- `--fusion`（rrf/weighted）
- `--top_k_*`

---

## 10. 兼容性与复现

- metadata 关键字段保持稳定：
  `block_id, file_path, start_line, end_line, block_type, strategy`
- 输出 schema 稳定：`loc_outputs.jsonl`
- 黄金样例对比：block 数量、起止行、Top-K 文件列表

---

## 11. 测试策略

- **Smoke Test**：快速构建单仓库索引
- **Golden Sample**：小型仓库 + 固定输出
- **Regression**：抽样策略 + Top-K 对比
- **Large-file**：验证超大文件跳过/降级逻辑是否稳定
- **Non-Python skip**：验证非 `.py` 文件是否被稳定跳过

---

## 12. 迁移路线（建议）

1. **Phase 1**：整合 `method/indexing` 到 `method/indexing/`
2. **Phase 2**：引入 BM25 稀疏索引模块
3. **Phase 3**：新增统一 CLI
4. **Phase 4**：混合检索融合器

---

## 13. 未决问题

- Summary 索引是否需要二次压缩或去重？
- 混合检索结果的权重策略默认值如何确定？

---

如确认本设计方向，我可以按 Phase 逐步落地并同步更新实现代码。
