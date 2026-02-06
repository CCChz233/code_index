# Summary 实验操作文档（Decoupled 流水线）
版本：v1.0  
适用：LocBench Summary Index 实验（Generator/Indexer 解耦）

---

## 0. 前置依赖
必须安装：
```
pip install chromadb rank_bm25 networkx
```

如需运行 Summary LLM（vLLM/OpenAI 兼容）：
```
export OPENAI_API_KEY="dummy_key"
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
```

---

## 1. 推荐的实验方式（脚本一键）
使用 `eval_summary.sh`（已集成构建 + 评测）。

### 1.1 全流程（构建 + 评测）
```
bash eval_summary.sh
```

### 1.2 仅构建（跳过评测）
```
RUN_BUILD=1 bash eval_summary.sh
```

### 1.3 仅评测（跳过构建）
```
RUN_BUILD=0 bash eval_summary.sh
```

### 1.4 分阶段运行（推荐）
```
STAGE=generator bash eval_summary.sh   # 只生成摘要→SQLite
STAGE=indexer  bash eval_summary.sh   # 只构建索引
STAGE=both     bash eval_summary.sh   # 默认
```

---

## 2. 手动运行（更可控）

### 2.1 仅生成摘要（SQLite）
```
python method/cli/run_generator.py \
  --repo_path /path/to/repos_root/<repo> \
  --sqlite_path /path/to/index_root/summary_index_function_level/<repo>/sqlite/summary.db \
  --graph_index_dir /path/to/graph_index \
  --summary_llm_provider openai \
  --summary_llm_model GPT-OSS-120B \
  --summary_llm_api_base http://127.0.0.1:8000/v1 \
  --summary_llm_api_key dummy_key \
  --summary_cache_dir /path/to/summary_cache \
  --summary_language Chinese
```

### 2.2 仅构建索引（SQLite → Dense/BM25/Graph）
```
python method/cli/run_indexer.py \
  --sqlite_path /path/to/index_root/summary_index_function_level/<repo>/sqlite/summary.db \
  --output_dir /path/to/index_root/summary_index_function_level/<repo> \
  --chroma_dir /path/to/index_root/summary_index_function_level/<repo>/chroma \
  --bm25_dir /path/to/index_root/summary_index_function_level/<repo>/bm25 \
  --graph_path /path/to/index_root/summary_index_function_level/<repo>/graph/graph.gpickle \
  --embed_model_name /path/to/CodeRankEmbed \
  --trust_remote_code \
  --gpu_id 0
```

---

## 3. 评测 Summary 检索
```
python method/cli/run_eval.py \
  --index_type summary \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --index_dir /path/to/index_root/summary_index_function_level \
  --output_folder /path/to/output_eval/summary_index \
  --model_name /path/to/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /path/to/locbench_repos \
  --top_k_docs 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50 \
  --max_length 512 \
  --batch_size 8 \
  --mapper_type ast \
  --gpu_id 0
```

输出：
```
/path/to/output_eval/summary_index/summary_results.jsonl
```

---

## 4. 结果检查（健康状态）
1. 报错检查：
```
rg -n "\"event\": \"error\"" logs/summary_index/metrics.jsonl
rg -n "ERROR|Traceback" logs/summary_index/summary_worker_*.log
```

2. 成果检查：
```
ls /path/to/index_root/summary_index_function_level/<repo>
```
应包含：
```
summary.jsonl
summary_ast/
dense/embeddings.pt
dense/index_map.json
bm25/
sqlite/summary.db
```

---

## 5. 常见参数
1. `STAGE=generator|indexer|both`
2. `NUM_PROCESSES` / `EMBED_GPU_IDS`
3. `TOP_K_*` / `MAX_LENGTH` / `BATCH_SIZE`
4. `summary_levels`（function/class/file/module）

---

## 6. 推荐实验顺序
1. 小规模（1-2 个 repo） → 验证链路  
2. 中规模（10-20 repo） → 观察吞吐  
3. 全量（LocBench 全库）  
