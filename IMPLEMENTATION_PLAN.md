# IR-Base 实现计划（Python-only 索引体系）

本文档回答三个问题：
1) 目前已经实现了什么功能；
2) 计划要实现什么功能；
3) 代码框架如何设计，以及分阶段落地步骤。

> 约定：只对 **Python 代码（.py）** 建索引。

---

## 1. 已实现功能（当前现状）

### 1.1 索引构建（Dense v2）
入口：
- `method/indexing/batch_build_index.py`（批量）
- `method/indexing/build_index.py`（单仓库）
- `method/indexing/batch_build_index_sfr.py`（SFR 批量）

支持策略：
- 基础：`fixed / sliding / rl_fixed / rl_mini`
- 函数级：`ir_function / function_level`
- 摘要：`summary`
- LlamaIndex：`llamaindex_code / sentence / token / semantic`
- LangChain：`langchain_fixed / recursive / token`
- Epic：`epic`

输出格式：
- `embeddings.pt` + `metadata.jsonl`
- function/summary 元信息已写入 metadata

### 1.2 检索评估（Dense v2）
入口：
- `method/retrieval/run_with_index.py`（CodeRankEmbed）
- `method/retrieval/run_with_index_sfr.py`（SFR）

输出：
- `loc_outputs.jsonl`

### 1.3 映射与图能力
- `method/mapping/`（AST/Graph mapper）
- `dependency_graph/`（图索引/遍历）
- `repo_index/`（EpicSplitter 等）

### 1.4 已具备的“Python-only 基础”
- `ir_function/function_level` 本身只处理 `.py`
- llamaindex/langchain/epic 已限定 `required_exts=[".py"]`

> 但基础 chunker 仍会扫描多语言（见“待做”）。

---

## 2. 计划实现功能（目标能力）

### 2.1 索引类型
- **Dense**：保持现有能力
- **Sparse**：BM25 / TF-IDF
- **Summary**：LLM 摘要索引
- **Hybrid**：Dense + Sparse + Summary 运行时融合

### 2.2 系统能力
- 统一 CLI（build_index / run_eval）
- 统一 Store / manifest（可复现）
- 注册表机制（可插拔组件）
- Python-only 强制过滤（全路径一致）
- 大文件策略（可跳过/降级）

---

## 3. 目标框架（最终目录结构）

```
method/
  core/                 # Block / ChunkResult / IndexArtifact / registry / store
    store/
      filesystem.py

  indexing/             # 仅负责“构建索引”
    chunker/
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

> 说明：`core` 放在顶层，避免 retrieval 依赖 indexing 的内部路径。

---

## 4. 需要补齐 / 新增的“占位文件”清单

以下是**计划创建或迁移**的文件骨架（包含来源）：

### 4.1 core（新增）
- `method/core/__init__.py`  （NEW）
- `method/core/contract.py`  （NEW：Block/ChunkResult/IndexArtifact）
- `method/core/registry.py`  （NEW）
- `method/core/store/filesystem.py`（NEW）

### 4.2 indexing（迁移）
- `method/indexing/chunker/*`  （MOVE from method/indexing/chunker）
- `method/indexing/encoder/dense/*`（NEW）
- `method/indexing/encoder/sparse/*`（NEW）
- `method/indexing/encoder/summary/*`（NEW）
- `method/indexing/indexer/dense_indexer.py`（NEW）
- `method/indexing/indexer/sparse_indexer.py`（NEW）
- `method/indexing/indexer/summary_indexer.py`（NEW）

### 4.3 retrieval（迁移 + 新增）
- `method/retrieval/dense_retriever.py`（MOVE from method/retrieval/run_with_index.py）
- `method/retrieval/sparse_retriever.py`（NEW）
- `method/retrieval/summary_retriever.py`（NEW）
- `method/retrieval/hybrid_retriever.py`（NEW）
- `method/retrieval/fusion/rrf.py`（NEW）
- `method/retrieval/fusion/weighted.py`（NEW）

### 4.4 cli（新增）
- `method/cli/build_index.py`（NEW）
- `method/cli/run_eval.py`（NEW）

---

## 5. 分阶段实现步骤（按顺序落地）

### Phase 0：冻结现状 + Python-only 过滤
- [x] 把 `iter_files()` 统一限制为 `.py`
- [x] 增加 `--lang_allowlist` / `--max_file_size_mb` / `--skip_patterns`
- [x] 更新测试：非 `.py` 文件必须被稳定跳过

### Phase 1：Core 抽离
- [x] 新建 `method/core/`
- [x] 迁移 `Block / ChunkResult / IndexArtifact`
- [x] 引入 registry 与 store（先空壳）

### Phase 2：Indexing 迁移
- [x] `method/indexing/chunker` 已落地（由 `index_v2` 迁移）
- [x] 编码逻辑拆到 `indexing/encoder/dense`
- [x] 写入逻辑拆到 `indexing/indexer/dense_indexer.py`

### Phase 3：Retrieval 迁移
- [x] `method/retrieval/run_with_index.py` 已落地（由 `dense_v2` 迁移）
- [x] 抽离公共数据加载/排序逻辑（`method/retrieval/common.py`）

### Phase 4：Sparse Index
- [x] BM25 Indexer + Retriever
- [x] 保存倒排索引（.npz）与 vocab.json + metadata.jsonl

### Phase 5：Hybrid Fusion
- [x] RRF / Weighted 融合
- [x] 统一输出 schema

### Phase 6：统一 CLI
- [x] `build_index.py` 支持 dense/sparse（summary 待扩展）
- [x] `run_eval.py` 支持 dense/sparse/hybrid

### Phase 7：文档与回归
- [ ] 更新 README/TESTING
- [ ] Golden Sample 对齐

---

## 6. 里程碑验收标准

- **M1**（Phase 0-2）：dense 构建可用 + Python-only 生效
- **M2**（Phase 3）：dense 检索可用 + 输出稳定
- **M3**（Phase 4）：BM25 索引与检索可用
- **M4**（Phase 5-6）：混合检索与统一 CLI 完成

---

如果你认可这份计划，我可以按 Phase 0 开始实施，保持“最小改动 + 稳定输出”。
