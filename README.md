# IR-Base（Python-only Indexing & Retrieval）

IR-Base 是一个**面向 Python 代码仓库的索引与检索框架**，支持稠密索引、稀疏索引以及混合检索。该项目已完成结构性重构，索引构建与检索评估分离，核心契约统一。

> 仅对 `.py` 文件建立索引，其他语言文件默认跳过。

---

## 功能概览

### 已实现
- **Dense 索引**：CodeRankEmbed / SFR
- **Sparse 索引**：BM25（`scipy.sparse` `.npz` + `vocab.json`）
- **Hybrid 检索**：Dense + Sparse 融合（RRF / Weighted）
- **多策略切块**：fixed / sliding / rl_fixed / rl_mini / ir_function / function_level / llamaindex / langchain / epic / summary

### 未实现（预留）
- Summary 索引统一入口（当前仍使用 summary chunker 作为 dense 流程的一部分）

---

## 目录结构（核心）

```
IR-base/
  method/
    core/                 # Block/ChunkResult/IndexArtifact/registry/store
    indexing/             # 索引构建
    retrieval/            # 检索评估
    cli/                  # 统一入口
    mapping/              # AST/Graph 映射器
  dependency_graph/       # Graph 索引
  repo_index/             # EpicSplitter 等
```

---

## 环境与依赖

```bash
conda activate locagent
export PYTHONPATH="$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
```

建议依赖：
- Dense：`torch`, `transformers`
- SFR：`sentence-transformers`
- Sparse：`scipy`
- Summary / LlamaIndex / LangChain：按策略安装对应依赖

---

## 索引构建

### Dense（批量）
```bash
python method/indexing/batch_build_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --model_name /path/to/embedding_model \
  --strategy ir_function \
  --num_processes 4 \
  --skip_existing
```

### Dense（单仓库）
```bash
python method/indexing/build_index.py \
  --repo_path /path/to/repo \
  --output_dir /path/to/output_index \
  --model_name /path/to/embedding_model \
  --strategy ir_function
```

### Sparse（BM25，批量）
```bash
python method/indexing/batch_build_sparse_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --strategy ir_function \
  --skip_existing
```

### Sparse（BM25，单仓库）
```bash
python method/indexing/build_sparse_index.py \
  --repo_path /path/to/repo \
  --output_dir /path/to/output_index \
  --strategy ir_function
```

---

## 检索评估

### Dense 检索
```bash
python method/retrieval/run_with_index.py \
  --index_dir /path/to/index_root/dense_index_ir_function \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --output_folder /path/to/output_eval \
  --model_name /path/to/embedding_model \
  --repos_root /path/to/repos_root
```

### Sparse 检索（BM25）
```bash
python method/retrieval/sparse_retriever.py \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --index_dir /path/to/index_root/sparse_index_ir_function \
  --output_folder /path/to/output_eval_sparse \
  --repos_root /path/to/repos_root
```

### Hybrid 检索（Dense + Sparse）
```bash
python method/retrieval/hybrid_retriever.py \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --dense_index_dir /path/to/index_root/dense_index_ir_function \
  --sparse_index_dir /path/to/index_root/sparse_index_ir_function \
  --output_folder /path/to/output_eval_hybrid \
  --model_name /path/to/embedding_model \
  --repos_root /path/to/repos_root \
  --fusion rrf
```

---

## 统一 CLI（可选）

### build_index
```bash
python method/cli/build_index.py \
  --index_type dense \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --model_name /path/to/embedding_model \
  --strategy ir_function
```

### run_eval
```bash
python method/cli/run_eval.py \
  --index_type hybrid \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --dense_index_dir /path/to/index_root/dense_index_ir_function \
  --sparse_index_dir /path/to/index_root/sparse_index_ir_function \
  --output_folder /path/to/output_eval_hybrid \
  --model_name /path/to/embedding_model \
  --repos_root /path/to/repos_root
```

---

## Python-only 文件过滤

默认仅索引 `.py`。可通过参数覆盖：
- `--lang_allowlist`（默认 `.py`）
- `--skip_patterns`（默认跳过 `.git/node_modules/dist/build/...`）
- `--max_file_size_mb`

---

## 索引格式

### Dense
```
index_root/dense_index_{strategy}/{repo}/embeddings.pt
index_root/dense_index_{strategy}/{repo}/metadata.jsonl
```

### Sparse
```
index_root/sparse_index_{strategy}/{repo}/index.npz
index_root/sparse_index_{strategy}/{repo}/vocab.json
index_root/sparse_index_{strategy}/{repo}/metadata.jsonl
```

---

## 文档入口
- `DESIGN.md`：总体架构设计
- `IMPLEMENTATION_PLAN.md`：实现计划与阶段状态
- `TESTING.md`：测试与验证

---

如需继续补 Summary 索引或扩展更多融合策略，请基于 `DESIGN.md` 的约定继续实现。
