# 所有方法的命令清单（索引构建 + 测评）

本清单覆盖 **当前已实现** 的索引构建方式与测评入口（Dense / Sparse / Hybrid），并列出所有可用的 **chunk 策略**。

> 约定：以下命令均在 `IR-base/` 目录下执行。

---

## 0. 环境变量与路径

```bash
conda activate locagent
cd /path/to/IR-base

export PYTHONPATH="$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
```

建议先定义以下变量，后续命令可直接复制：
```bash
export REPOS_ROOT=/path/to/repos_root
export INDEX_ROOT=/path/to/index_root
export DATASET_PATH=/path/to/Loc-Bench_V1_dataset.jsonl
export MODEL_NAME=/path/to/embedding_model        # CodeRankEmbed
export MODEL_NAME_SFR=/path/to/sfr_model           # SFR (sentence-transformers)
```

---

## 1. Dense 索引构建（CodeRankEmbed）

### 1.1 可用 chunk 策略（批量脚本支持）
**batch_build_index.py** 支持以下策略：
```
ir_function
function_level
summary
llamaindex_code
llamaindex_sentence
llamaindex_token
llamaindex_semantic
langchain_fixed
langchain_recursive
langchain_token
epic
```

### 1.2 单次构建（批量）
```bash
python method/indexing/batch_build_index.py \
  --repo_path $REPOS_ROOT \
  --index_dir $INDEX_ROOT \
  --model_name $MODEL_NAME \
  --strategy ir_function \
  --num_processes 4 \
  --skip_existing
```

### 1.3 跑完所有策略（批量）
```bash
for STRATEGY in \
  ir_function function_level summary \
  llamaindex_code llamaindex_sentence llamaindex_token llamaindex_semantic \
  langchain_fixed langchain_recursive langchain_token \
  epic; do

  python method/indexing/batch_build_index.py \
    --repo_path $REPOS_ROOT \
    --index_dir $INDEX_ROOT \
    --model_name $MODEL_NAME \
    --strategy $STRATEGY \
    --num_processes 4 \
    --skip_existing

done
```

---

## 2. Dense 索引构建（SFR）

### 2.1 可用 chunk 策略
与 CodeRankEmbed 批量脚本一致：
```
ir_function
function_level
summary
llamaindex_code
llamaindex_sentence
llamaindex_token
llamaindex_semantic
langchain_fixed
langchain_recursive
langchain_token
epic
```

### 2.2 批量构建
```bash
python method/indexing/batch_build_index_sfr.py \
  --repo_path $REPOS_ROOT \
  --index_dir $INDEX_ROOT \
  --model_name $MODEL_NAME_SFR \
  --strategy ir_function \
  --num_processes 4 \
  --skip_existing
```

### 2.3 跑完所有策略（批量）
```bash
for STRATEGY in \
  ir_function function_level summary \
  llamaindex_code llamaindex_sentence llamaindex_token llamaindex_semantic \
  langchain_fixed langchain_recursive langchain_token \
  epic; do

  python method/indexing/batch_build_index_sfr.py \
    --repo_path $REPOS_ROOT \
    --index_dir $INDEX_ROOT \
    --model_name $MODEL_NAME_SFR \
    --strategy $STRATEGY \
    --num_processes 4 \
    --skip_existing

done
```

---

## 3. Dense 索引构建（单仓库，支持 Basic 策略）

**build_index.py** 额外支持：`fixed / sliding / rl_fixed / rl_mini`。

```bash
python method/indexing/build_index.py \
  --repo_path /path/to/single_repo \
  --output_dir $INDEX_ROOT/dense_index_fixed/single_repo \
  --model_name $MODEL_NAME \
  --strategy fixed
```

如果你想对多仓库跑 Basic 策略（目前无批量脚本），可用 bash 循环：
```bash
for REPO in $(ls $REPOS_ROOT); do
  python method/indexing/build_index.py \
    --repo_path $REPOS_ROOT/$REPO \
    --output_dir $INDEX_ROOT/dense_index_fixed/$REPO \
    --model_name $MODEL_NAME \
    --strategy fixed

done
```

---

## 4. Sparse 索引构建（BM25）

### 4.1 可用 chunk 策略
**batch_build_sparse_index.py** 支持以下策略：
```
fixed
sliding
rl_fixed
rl_mini
function_level
ir_function
epic
```
> 注意：`summary / llamaindex / langchain` 暂不支持稀疏构建。

### 4.2 批量构建
```bash
python method/indexing/batch_build_sparse_index.py \
  --repo_path $REPOS_ROOT \
  --index_dir $INDEX_ROOT \
  --strategy ir_function \
  --skip_existing
```

### 4.3 跑完所有策略（批量）
```bash
for STRATEGY in fixed sliding rl_fixed rl_mini function_level ir_function epic; do

  python method/indexing/batch_build_sparse_index.py \
    --repo_path $REPOS_ROOT \
    --index_dir $INDEX_ROOT \
    --strategy $STRATEGY \
    --skip_existing

done
```

---

## 5. Summary 相关（可选）

Summary 需要 LLM 配置。推荐使用模板：
- `method/indexing/summary_config_template.yaml`

示例：
```bash
export OPENAI_API_KEY=your_key
export OPENAI_API_BASE=https://api.your-llm-provider/v1

python method/indexing/batch_build_index.py \
  --config method/indexing/summary_config_template.yaml \
  --repo_path $REPOS_ROOT \
  --index_dir $INDEX_ROOT \
  --model_name $MODEL_NAME \
  --strategy summary \
  --summary_llm_api_key $OPENAI_API_KEY \
  --summary_llm_api_base $OPENAI_API_BASE
```

---

## 6. 测评命令（Dense / Sparse / Hybrid）

### 6.1 Dense 测评（CodeRankEmbed）
```bash
python method/retrieval/run_with_index.py \
  --dataset_path $DATASET_PATH \
  --index_dir $INDEX_ROOT/dense_index_ir_function \
  --output_folder $INDEX_ROOT/outputs/dense_ir_function \
  --model_name $MODEL_NAME \
  --repos_root $REPOS_ROOT
```

### 6.2 Dense 测评（SFR）
```bash
python method/retrieval/run_with_index_sfr.py \
  --dataset_path $DATASET_PATH \
  --index_dir $INDEX_ROOT/dense_index_ir_function \
  --output_folder $INDEX_ROOT/outputs/sfr_ir_function \
  --model_name $MODEL_NAME_SFR \
  --repos_root $REPOS_ROOT
```

### 6.3 Sparse 测评（BM25）
```bash
python method/retrieval/sparse_retriever.py \
  --dataset_path $DATASET_PATH \
  --index_dir $INDEX_ROOT/sparse_index_ir_function \
  --output_folder $INDEX_ROOT/outputs/sparse_ir_function \
  --repos_root $REPOS_ROOT
```

### 6.4 Hybrid 测评（Dense + Sparse）
```bash
python method/retrieval/hybrid_retriever.py \
  --dataset_path $DATASET_PATH \
  --dense_index_dir $INDEX_ROOT/dense_index_ir_function \
  --sparse_index_dir $INDEX_ROOT/sparse_index_ir_function \
  --output_folder $INDEX_ROOT/outputs/hybrid_ir_function \
  --model_name $MODEL_NAME \
  --repos_root $REPOS_ROOT \
  --fusion rrf
```

---

## 7. Python-only 过滤（可选）

所有索引构建脚本支持以下参数：
- `--lang_allowlist`（默认 `.py`）
- `--skip_patterns`（默认跳过 `.git/node_modules/dist/build/...`）
- `--max_file_size_mb`

示例：
```bash
python method/indexing/batch_build_index.py \
  --repo_path $REPOS_ROOT \
  --index_dir $INDEX_ROOT \
  --model_name $MODEL_NAME \
  --strategy ir_function \
  --lang_allowlist .py \
  --skip_patterns ".git,node_modules,dist,build" \
  --max_file_size_mb 1
```

---

## 8. 输出位置

- Dense：`$INDEX_ROOT/dense_index_{strategy}/{repo}/`  
  `embeddings.pt` + `metadata.jsonl`

- Sparse：`$INDEX_ROOT/sparse_index_{strategy}/{repo}/`  
  `index.npz` + `vocab.json` + `metadata.jsonl`

---

如需我帮你生成“按策略自动跑全流程”的脚本（构建 + 测评），告诉我你要跑的策略列表与硬件资源（CPU/GPU）。
