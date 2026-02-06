# Summary Index 操作文档

本文件用于维护 **Summary Index 构建** 与 **Summary 检索评估** 的操作命令与流程。  
适用版本：summary index v2.1（`summary.md` 为唯一事实来源）

---

## 1. 环境准备

建议环境：
```
conda activate locagent
export PYTHONPATH="$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
```

必要依赖（最小集合）：
1. Dense embedding：`torch`, `transformers`
2. Sparse BM25：`scipy`
3. Summary LLM：`llama-index` + 对应 provider 包  
4. 可视化（可选）：`rich`

OpenAI 示例：
```
pip install llama-index llama-index-llms-openai
export OPENAI_API_KEY="..."
```

---

## 2. 构建 Summary Index（单仓库）

命令：
```
python method/indexing/build_summary_index.py \
  --repo_path /path/to/repo \
  --output_dir /path/to/index_root/summary_index_function_level/repo_name \
  --summary_levels function,class,file,module \
  --graph_index_dir /path/to/graph_index_dir \
  --model_name /path/to/embedding_model \
  --summary_llm_provider openai \
  --summary_llm_model gpt-3.5-turbo
```

### 使用本地 vLLM 生成摘要（推荐你的配置）
说明：
1. Dense embedding **继续使用与 Code Index 相同的模型**（`--model_name`）
2. Summary LLM 使用本地 vLLM 模型（`--summary_llm_provider vllm`）

命令：
```
python method/indexing/build_summary_index.py \
  --repo_path /path/to/repo \
  --output_dir /path/to/index_root/summary_index_function_level/repo_name \
  --summary_levels function,class,file,module \
  --graph_index_dir /path/to/graph_index_dir \
  --model_name /path/to/embedding_model \
  --summary_llm_provider vllm \
  --summary_llm_model /path/to/local_vllm_model
```

可选常用参数：
1. `--summary_cache_dir /path/to/cache`  
2. `--summary_cache_policy read_write|read_only|disabled`  
3. `--summary_prompt_file /path/to/prompt.txt`  
4. `--summary_language English|Chinese`  
5. `--summary_embed_max_tokens 512`  
6. `--filter_min_lines 3`  
7. `--filter_min_complexity 2`  
8. `--filter_skip_patterns "^get_,^set_,^__.*__$"`  
9. `--top_k_callers 5`  
10. `--top_k_callees 5`  
11. `--context_similarity_threshold 0.8`  
12. `--lang_allowlist .py`  
13. `--skip_patterns "**/node_modules/**,**/.git/**"`  
14. `--max_file_size_mb 5`  
15. `--progress_style rich|tqdm|simple`（终端可视化）  

---

## 3. 构建 Summary Index（批量）

命令：
```
python method/indexing/batch_build_summary_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --summary_levels function,class,file,module \
  --graph_index_dir /path/to/graph_index_dir \
  --skip_existing \
  --model_name /path/to/embedding_model \
  --summary_llm_provider openai \
  --summary_llm_model gpt-3.5-turbo
```

### 使用本地 vLLM 生成摘要（批量）
```
python method/indexing/batch_build_summary_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --summary_levels function,class,file,module \
  --graph_index_dir /path/to/graph_index_dir \
  --skip_existing \
  --model_name /path/to/embedding_model \
  --summary_llm_provider vllm \
  --summary_llm_model /path/to/local_vllm_model
```

### 多进程构建（推荐 8 卡服务器）
说明：
1. vLLM 建议通过 OpenAI 兼容 server 提供服务
2. embedding GPU 与 vLLM GPU 分开，使用 `--gpu_ids` 控制
3. 失败断点续跑：将 `RESUME_FAILED=1` 或直接传 `--resume_failed`
4. 可选阶段控制：`--stage generator|indexer|both`（默认 both）

```
python method/indexing/batch_build_summary_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --summary_levels function,class,file,module \
  --graph_index_dir /path/to/graph_index_dir \
  --model_name /path/to/embedding_model \
  --summary_llm_provider openai \
  --summary_llm_model GPT-OSS-120B \
  --summary_llm_api_base http://127.0.0.1:8000/v1 \
  --summary_llm_api_key dummy_key \
  --num_processes 4 \
  --gpu_ids 4,5,6,7 \
  --llm_timeout 300 \
  --max_retries 3 \
  --log_dir /path/to/logs \
  --progress_style rich \
  --stage both \
  --skip_existing
```

### Worker 日志文件
每个进程会输出独立日志：
```
logs/summary_index/summary_worker_0.log
logs/summary_index/summary_worker_1.log
...
```

### Metrics 输出（JSONL）
批量构建会生成结构化指标日志：
```
logs/summary_index/metrics.jsonl
```
包含 `run_start/run_end`、`repo_start/repo_end`、`llm_finish`（含 token 使用量）、`cache_hit/cache_miss`、`error` 等事件，便于离线汇总与排障。

如需断点续跑：
```
python method/indexing/batch_build_summary_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --summary_llm_provider openai \
  --summary_llm_model GPT-OSS-120B \
  --summary_llm_api_base http://127.0.0.1:8000/v1 \
  --summary_llm_api_key dummy_key \
  --resume_failed
```

输出目录结构（每个 repo）：
```
summary_index_function_level/{repo}/
  summary.jsonl
  dense/embeddings.pt
  dense/index_map.json
  bm25/index.npz
  bm25/vocab.json
  bm25/metadata.jsonl
  summary_ast/<path>.json
  manifest.json
```

---

## 4. AST 摘要快速查看（调试/验收）
```
python scripts/inspect_summary.py \
  --repo flask \
  --file app.py \
  --root summary_index_function_level
```
说明：
1. 读取 `summary_ast/<file>.json` 并以树状结构展示  
2. 若未安装 `rich`，自动降级为纯文本输出  
3. 支持模糊搜索（输入 `app`、`app.py` 或完整路径均可）

---

## 4. Summary 检索评估

命令（使用统一 CLI）：
```
python method/cli/run_eval.py \
  --index_type summary \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --index_dir /path/to/index_root/summary_index_function_level \
  --output_folder /path/to/output_eval_summary \
  --model_name /path/to/embedding_model \
  --repos_root /path/to/repos_root \
  --top_k_docs 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50
```

说明：
1. `summary_retriever.py` 使用 **dense embeddings + index_map.json** 做 doc_id 映射。  
2. `raw_output_loc` 会记录 doc_id / type / score，便于调试。  
3. `mapper_type` 可选 `ast` 或 `graph`（默认 `ast`）。  

---

## 5. 常见问题

1. embeddings.pt 与 summary.jsonl 行数不一致  
   - 正常现象，原因是 `is_placeholder=true` 的文档不进入 Dense Index  
   - 必须使用 `dense/index_map.json` 反查 doc_id  

2. 缓存命中但摘要为空  
   - 检查 `summary_cache_policy` 与 `summary_cache_miss` 设置  
   - `read_only` + cache miss 会导致空摘要  

3. Graph 上下文缺失  
   - 确认 `--graph_index_dir` 下存在 `{repo}.pkl`  
   - 无 graph 会自动降级为无上下文摘要  

---

如需扩展 Hybrid 或 Rerank，请以 `summary.md` 为规范更新。  
