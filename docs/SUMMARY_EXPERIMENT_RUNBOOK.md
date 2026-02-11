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

---

## 7. 使用 summary_v2 / summary_runs 跑测评

若索引构建在 `index_v2/summary_v2/summary_runs/<run_id>` 下（例如用 `run_summary.sh` 构建），测评时需让 `--index_dir` 指向该 run 的 **summary_index_function_level** 目录（每 repo 一个子目录，内含 `summary.jsonl`、`dense/` 等）。

### 7.1 确认索引已完整
索引目录应为：`index_v2/summary_v2/summary_runs/<run_id>/summary_index_function_level/`，其下每个 repo 目录内需有：
- `summary.jsonl`
- `dense/embeddings.pt`、`dense/index_map.json`

若只有 `repos/<repo>/intermediate/sqlite/summary.db` 而没有 `repos/<repo>/output/summary.jsonl`，说明只跑了 Generator，需先跑一次 Indexer（同一 `run_id`）再测评：
```bash
STAGE=indexer bash run_summary.sh
# 或显式指定 run_id（与构建时一致）：
# python method/indexing/batch_build_summary_index.py ... --index_dir .../index_v2/summary_v2 --run_id 20260211T044455Z --stage indexer ...
```

### 7.2 仅测评（索引已构建好）
在 `code_index` 目录下执行（将 `<run_id>` 换成你的 run，如 `20260211T044455Z`）：

```bash
cd /home/chaihongzheng/workspace/locbench/code_index
export PYTHONPATH="$(pwd):$PYTHONPATH"

python method/cli/run_eval.py \
  --index_type summary \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/summary_v2/summary_runs/<run_id>/summary_index_function_level \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/summary_v2 \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_docs 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50 \
  --max_length 512 \
  --batch_size 8 \
  --mapper_type ast \
  --gpu_id 0
```

结果会写入：`output_eval/summary_v2/summary_results.jsonl`（或脚本打印的路径）。

### 7.3 用 eval_summary.sh 跑 summary_v2 测评
`eval_summary.sh` 支持通过环境变量 `INDEX_DIR` 指定测评用的索引目录（覆盖默认的 `INDEX_ROOT/summary_index_function_level`）。将下面 `<run_id>` 换成实际 run（如 `20260211T044455Z`）：

```bash
cd /home/chaihongzheng/workspace/locbench/code_index
RUN_BUILD=0 INDEX_DIR="$(pwd)/index_v2/summary_v2/summary_runs/<run_id>/summary_index_function_level" bash eval_summary.sh
```

评测结果仍写入脚本中配置的 `OUTPUT_EVAL`（默认 `output_eval/summary_index`）。
