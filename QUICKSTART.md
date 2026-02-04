# IR-base 快速开始

本指南面向 **Python 代码仓库索引与检索**，覆盖 Dense / Sparse / Hybrid 的最短路径。

## 1. 环境准备

```bash
conda activate locagent
cd /path/to/IR-base

export PYTHONPATH="$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
```

可选（中国网络）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

依赖提示：
- Dense：`torch`, `transformers`
- SFR：`sentence-transformers`
- Sparse：`scipy`
- Summary / LlamaIndex / LangChain：按策略安装对应依赖（summary 需可用 LLM 配置）

---

## 2. 准备数据

- 代码仓库根目录（多仓库）：`/path/to/repos_root`
- 数据集路径：`/path/to/Loc-Bench_V1_dataset.jsonl`

> 默认只索引 `.py` 文件；其他语言会被跳过。

---

## 3. 构建索引（Dense）

### 批量构建
```bash
python method/indexing/batch_build_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --model_name /path/to/embedding_model \
  --strategy ir_function \
  --num_processes 4 \
  --skip_existing
```

### 单仓库构建
```bash
python method/indexing/build_index.py \
  --repo_path /path/to/repo \
  --output_dir /path/to/output_index \
  --model_name /path/to/embedding_model \
  --strategy ir_function
```

---

## 4. 构建索引（Sparse / BM25）

### 批量构建
```bash
python method/indexing/batch_build_sparse_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --strategy ir_function \
  --skip_existing
```

### 单仓库构建
```bash
python method/indexing/build_sparse_index.py \
  --repo_path /path/to/repo \
  --output_dir /path/to/output_index \
  --strategy ir_function
```

说明：Sparse 支持与 Dense 相同的分块策略（含 `llamaindex_*` / `langchain_*` / `summary`），使用前确保依赖与 LLM 配置齐全。

---

## 5. 检索评估

### Dense 检索
```bash
python method/retrieval/run_with_index.py \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --index_dir /path/to/index_root/dense_index_ir_function \
  --output_folder /path/to/output_eval \
  --model_name /path/to/embedding_model \
  --repos_root /path/to/repos_root
```

### Sparse 检索
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

## 6. 统一 CLI（可选）

```bash
python method/cli/build_index.py \
  --index_type dense \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --model_name /path/to/embedding_model \
  --strategy ir_function
```

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

## 7. 常用参数（Python-only 过滤）

- `--lang_allowlist`：语言白名单（默认 `.py`）
- `--skip_patterns`：跳过路径模式（默认过滤 `.git/node_modules/dist/build/...`）
- `--max_file_size_mb`：单文件大小上限

如需更多细节，见 `README.md` 与 `TESTING.md`。
