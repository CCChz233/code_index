#!/usr/bin/env bash
# 简化版两阶段评估脚本 - 在 code_index 目录下运行

cd "$(dirname "$0")"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ===== 只需修改这几个路径 =====
DATASET_PATH="${DATASET_PATH:-/home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl}"
INDEX_DIR="${INDEX_DIR:-/home/chaihongzheng/workspace/locbench/IR-based/new_index_data/llamaindex_code_custom_40_15_800/dense_index_llamaindex_code}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${PWD}/output_eval/two_stage_yuanban_v5}"
MODEL_NAME="${MODEL_NAME:-/home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed}"
REPOS_ROOT="${REPOS_ROOT:-/home/chaihongzheng/workspace/locbench/repos/locbench_repos}"

# ===== 常用参数（可选覆盖） =====
# 例如: STAGE1_TOP_K_BLOCKS=200 STAGE2_MIN_ACCEPT_BLOCKS=3 bash eval_two_stage_simple.sh
STAGE1_TOP_K_BLOCKS="${STAGE1_TOP_K_BLOCKS:-200}"
TOP_K_BLOCKS="${TOP_K_BLOCKS:-40}"
STAGE2_MIN_ACCEPT_BLOCKS="${STAGE2_MIN_ACCEPT_BLOCKS:-5}"
GPU_ID="${GPU_ID:-0}"

python method/retrieval/run_with_index_two_stage.py \
  --dataset_path "$DATASET_PATH" \
  --index_dir "$INDEX_DIR" \
  --output_folder "$OUTPUT_FOLDER" \
  --model_name "$MODEL_NAME" \
  --repos_root "$REPOS_ROOT" \
  --strategy llamaindex_code \
  --trust_remote_code \
  --stage1_top_k_blocks "$STAGE1_TOP_K_BLOCKS" \
  --top_k_blocks "$TOP_K_BLOCKS" \
  --stage2_min_accept_blocks "$STAGE2_MIN_ACCEPT_BLOCKS" \
  --gpu_id "$GPU_ID"

echo "完成。结果输出: ${OUTPUT_FOLDER}/loc_outputs.jsonl"
