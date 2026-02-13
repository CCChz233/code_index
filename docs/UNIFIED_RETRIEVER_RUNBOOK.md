# 大一统检索器操作文档（Runbook）

适用脚本：`method/retrieval/run_with_index_convergent.py`  
目标：用一个脚本通过参数组合完成 `Baseline / PRF / Rerank / 完全体` 对照实验。

## 1. 环境准备

```bash
cd /Users/chz/code/locbench/code_index
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
```

## 2. 配置公共变量

按你的实际路径修改：

```bash
export DATASET_PATH="/path/to/Loc-Bench_V1_dataset.jsonl"
export INDEX_DIR="/path/to/dense_index_ir_function"
export MODEL_NAME="/path/to/CodeRankEmbed"
export REPOS_ROOT="/path/to/locbench_repos"

export RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
export OUTPUT_ROOT="/path/to/output_unified_retriever"
mkdir -p "$OUTPUT_ROOT"
```

公共参数（四组实验保持一致）：

```bash
COMMON_ARGS="\
  --dataset_path $DATASET_PATH \
  --index_dir $INDEX_DIR \
  --model_name $MODEL_NAME \
  --trust_remote_code \
  --repos_root $REPOS_ROOT \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50 \
  --max_length 1024 \
  --pooling first_non_pad \
  --file_score_agg sum \
  --gpu_id 0"
```

## 3. 四种核心场景命令

## 3.1 Baseline（单轮、无重排）

```bash
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "$OUTPUT_ROOT/baseline_off" \
  --convergence_mode off
```

## 3.2 仅 PRF（多步收敛、无重排）

```bash
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "$OUTPUT_ROOT/prf_only" \
  --convergence_mode prf \
  --max_steps 3 \
  --top_k_blocks_expand 100 \
  --feedback_top_m 8 \
  --feedback_file_cap 2 \
  --query_update_alpha 0.35 \
  --query_anchor_beta 0.15 \
  --feedback_temp 0.05 \
  --round_fusion rrf \
  --rrf_k 60 \
  --converge_jaccard_k 50 \
  --converge_jaccard_threshold 0.90 \
  --converge_min_improve 0.002 \
  --patience 1 \
  --min_cos_to_q0 0.75
```

## 3.3 仅 Rerank（单轮 + CE 重排）

```bash
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "$OUTPUT_ROOT/rerank_only" \
  --convergence_mode off \
  --enable_rerank \
  --rerank_model "$RERANK_MODEL" \
  --rerank_top_k 50 \
  --rerank_batch_size 8 \
  --rerank_max_length 512 \
  --rerank_context_lines 0 \
  --rerank_snippet_max_lines 120 \
  --rerank_fail_open
```

## 3.4 完全体（PRF + CE 重排）

```bash
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "$OUTPUT_ROOT/prf_rerank_full" \
  --convergence_mode prf \
  --max_steps 3 \
  --top_k_blocks_expand 100 \
  --feedback_top_m 8 \
  --feedback_file_cap 2 \
  --query_update_alpha 0.35 \
  --query_anchor_beta 0.15 \
  --feedback_temp 0.05 \
  --round_fusion rrf \
  --rrf_k 60 \
  --converge_jaccard_k 50 \
  --converge_jaccard_threshold 0.90 \
  --converge_min_improve 0.002 \
  --patience 1 \
  --min_cos_to_q0 0.75 \
  --enable_rerank \
  --rerank_model "$RERANK_MODEL" \
  --rerank_top_k 50 \
  --rerank_batch_size 8 \
  --rerank_max_length 512 \
  --rerank_context_lines 0 \
  --rerank_snippet_max_lines 120 \
  --rerank_fail_open
```

## 4. 可选扩展场景

`global_local + rerank`：

```bash
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "$OUTPUT_ROOT/global_local_rerank" \
  --convergence_mode global_local \
  --top_k_seed_files 20 \
  --max_steps 3 \
  --top_k_blocks_expand 100 \
  --feedback_top_m 8 \
  --feedback_file_cap 2 \
  --query_update_alpha 0.35 \
  --query_anchor_beta 0.15 \
  --feedback_temp 0.05 \
  --round_fusion rrf \
  --rrf_k 60 \
  --enable_rerank \
  --rerank_model "$RERANK_MODEL" \
  --rerank_top_k 50 \
  --rerank_batch_size 8 \
  --rerank_max_length 512 \
  --rerank_fail_open
```

## 5. 输出位置

每次运行会输出：

1. `OUTPUT_FOLDER/loc_outputs.jsonl`

示例：

1. `.../baseline_off/loc_outputs.jsonl`
2. `.../prf_only/loc_outputs.jsonl`
3. `.../rerank_only/loc_outputs.jsonl`
4. `.../prf_rerank_full/loc_outputs.jsonl`

## 6. 快速结果检查

检查各实验空结果率：

```bash
python - <<'PY'
import glob, json, os

root = os.path.expanduser(os.environ.get("OUTPUT_ROOT", ""))
paths = sorted(glob.glob(os.path.join(root, "*", "loc_outputs.jsonl")))
for p in paths:
    total, empty = 0, 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if not rec.get("found_files"):
                empty += 1
    ratio = (empty / total * 100) if total else 0.0
    print(f"{p}: empty_found_files={empty}/{total} ({ratio:.2f}%)")
PY
```

## 7. 常见问题

1. `ModuleNotFoundError: method`  
确保先执行：`export PYTHONPATH="$(pwd):${PYTHONPATH:-}"`。

2. Rerank 模型加载失败  
默认 `--rerank_fail_open` 会自动回退到仅召回排序；若你希望失败即中断，可用 `--no-rerank_fail_open`。

3. 显存不足  
优先降低：`--rerank_batch_size`、`--rerank_top_k`、`--top_k_blocks_expand`。

4. 运行很慢  
先跑 Baseline 确认链路，再逐步开启 PRF 与 Rerank。
