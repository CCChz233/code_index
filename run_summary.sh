#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"

# ==== 配置区（按需修改） ====
REPOS_ROOT="/home/chaihongzheng/workspace/locbench/repos/locbench_repos"
INDEX_ROOT="/home/chaihongzheng/workspace/locbench/code_index/index_v2/summary_v2"
GRAPH_INDEX_DIR="/home/chaihongzheng/workspace/locbench/code_index/graph_index"
EMBED_MODEL="/home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed"
SUMMARY_CACHE_DIR="/home/chaihongzheng/workspace/locbench/code_index/summary_cache_v2"
LOG_DIR="/home/chaihongzheng/workspace/locbench/code_index/logs/summary_index_v2"

# vLLM OpenAI-compatible server
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy_key}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
SUMMARY_LLM_MODEL="${SUMMARY_LLM_MODEL:-GPT-OSS-120B}"

# 多进程与 GPU 绑定（若 vLLM 占用部分 GPU，请在这里排除）
NUM_PROCESSES="${NUM_PROCESSES:-4}"
EMBED_GPU_IDS="${EMBED_GPU_IDS:-0,1,2,3}"
RESUME_FAILED="${RESUME_FAILED:-0}"
STAGE="${STAGE:-generator}"
PROGRESS_STYLE="${PROGRESS_STYLE:-rich}"

EXTRA_FLAGS=""
if [ "$RESUME_FAILED" = "1" ]; then
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
  --summary_language English \
  --num_processes "${NUM_PROCESSES}" \
  --gpu_ids "${EMBED_GPU_IDS}" \
  --log_dir "${LOG_DIR}" \
  --progress_style "${PROGRESS_STYLE}" \
  --skip_existing \
  --stage "${STAGE}" \
  ${EXTRA_FLAGS}

