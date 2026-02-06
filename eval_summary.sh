#!/usr/bin/env bash
set -euo pipefail

# Summary Index build + evaluation (LocBench)
# Usage:
#   bash eval_summary.sh
#   RUN_BUILD=0 bash eval_summary.sh   # skip build, only eval

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"

# ==== 配置区（按需修改） ====
REPOS_ROOT="/home/chaihongzheng/workspace/locbench/repos/locbench_repos"
INDEX_ROOT="/home/chaihongzheng/workspace/locbench/code_index/index_v2/summary"
GRAPH_INDEX_DIR="/home/chaihongzheng/workspace/locbench/code_index/graph_index"
SUMMARY_CACHE_DIR="/home/chaihongzheng/workspace/locbench/code_index/summary_cache"
LOG_DIR="/home/chaihongzheng/workspace/locbench/code_index/logs/summary_index"

DATASET_PATH="/home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl"
OUTPUT_EVAL="/home/chaihongzheng/workspace/locbench/code_index/output_eval/summary_index"

EMBED_MODEL="/home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed"

# vLLM OpenAI-compatible server
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy_key}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
SUMMARY_LLM_MODEL="${SUMMARY_LLM_MODEL:-GPT-OSS-120B}"

# 构建配置
NUM_PROCESSES="${NUM_PROCESSES:-4}"
EMBED_GPU_IDS="${EMBED_GPU_IDS:-4,5,6,7}"
RESUME_FAILED="${RESUME_FAILED:-0}"
PROGRESS_STYLE="${PROGRESS_STYLE:-rich}"
STAGE="${STAGE:-both}"

# 评测配置
GPU_ID="${GPU_ID:-0}"
TOP_K_DOCS="${TOP_K_DOCS:-50}"
TOP_K_FILES="${TOP_K_FILES:-20}"
TOP_K_MODULES="${TOP_K_MODULES:-20}"
TOP_K_ENTITIES="${TOP_K_ENTITIES:-50}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAPPER_TYPE="${MAPPER_TYPE:-ast}"

RUN_BUILD="${RUN_BUILD:-1}"

if [[ "${RUN_BUILD}" == "1" ]]; then
  EXTRA_FLAGS=""
  if [[ "${RESUME_FAILED}" == "1" ]]; then
    EXTRA_FLAGS="--resume_failed"
  fi

  python method/indexing/batch_build_summary_index.py \
    --repo_path "${REPOS_ROOT}" \
    --index_dir "${INDEX_ROOT}" \
    --summary_levels function,class,file,module \
    --graph_index_dir "${GRAPH_INDEX_DIR}" \
    --model_name "${EMBED_MODEL}" \
    --trust_remote_code \
    --summary_llm_provider openai \
    --summary_llm_model "${SUMMARY_LLM_MODEL}" \
    --summary_llm_api_base "${OPENAI_API_BASE}" \
    --summary_llm_api_key "${OPENAI_API_KEY}" \
    --summary_cache_dir "${SUMMARY_CACHE_DIR}" \
    --summary_cache_policy read_write \
    --summary_cache_miss empty \
    --summary_language Chinese \
    --num_processes "${NUM_PROCESSES}" \
    --gpu_ids "${EMBED_GPU_IDS}" \
    --llm_timeout 300 \
    --max_retries 3 \
    --log_dir "${LOG_DIR}" \
    --progress_style "${PROGRESS_STYLE}" \
    --stage "${STAGE}" \
    --skip_existing \
    ${EXTRA_FLAGS}
fi

# === 评测 ===
python method/cli/run_eval.py \
  --index_type summary \
  --dataset_path "${DATASET_PATH}" \
  --index_dir "${INDEX_ROOT}/summary_index_function_level" \
  --output_folder "${OUTPUT_EVAL}" \
  --model_name "${EMBED_MODEL}" \
  --trust_remote_code \
  --repos_root "${REPOS_ROOT}" \
  --top_k_docs "${TOP_K_DOCS}" \
  --top_k_files "${TOP_K_FILES}" \
  --top_k_modules "${TOP_K_MODULES}" \
  --top_k_entities "${TOP_K_ENTITIES}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --mapper_type "${MAPPER_TYPE}" \
  --gpu_id "${GPU_ID}"
