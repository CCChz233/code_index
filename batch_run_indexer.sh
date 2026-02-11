#!/usr/bin/env bash
set -euo pipefail

# 批量从 summary_index_function_level/<repo>/sqlite/summary.db 构建索引
# 用法：
#   bash batch_run_indexer.sh                       # 默认 GPU 0
#   GPU_ID=2 bash batch_run_indexer.sh              # 指定 GPU
#   FORCE_CPU=1 bash batch_run_indexer.sh            # 强制 CPU
#   SKIP_EXISTING=1 bash batch_run_indexer.sh        # 跳过已有 summary.jsonl 的 repo

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"

# ==== 配置 ====
INDEX_ROOT="${INDEX_ROOT:-/home/chaihongzheng/workspace/locbench/code_index/index_v2/summary_v2/summary_index_function_level}"
EMBED_MODEL="${EMBED_MODEL:-/home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_CPU="${FORCE_CPU:-0}"

CPU_FLAG=""
if [[ "${FORCE_CPU}" == "1" ]]; then
  CPU_FLAG="--force_cpu"
fi

total=0
skipped=0
built=0
failed=0

for repo_dir in "${INDEX_ROOT}"/*/; do
  repo_name="$(basename "${repo_dir}")"
  sqlite_path="${repo_dir}sqlite/summary.db"
  output_dir="${repo_dir}"

  if [[ ! -f "${sqlite_path}" ]]; then
    echo "[SKIP] ${repo_name}: no sqlite/summary.db"
    ((skipped++)) || true
    ((total++)) || true
    continue
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${repo_dir}summary.jsonl" ]]; then
    echo "[SKIP] ${repo_name}: summary.jsonl already exists"
    ((skipped++)) || true
    ((total++)) || true
    continue
  fi

  echo "[BUILD] ${repo_name} ..."
  ((total++)) || true

  if python method/cli/run_indexer.py \
    --sqlite_path "${sqlite_path}" \
    --output_dir "${output_dir}" \
    --bm25_dir "${repo_dir}bm25" \
    --embed_model_name "${EMBED_MODEL}" \
    --trust_remote_code \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --gpu_id "${GPU_ID}" \
    ${CPU_FLAG} 2>&1; then
    echo "[DONE] ${repo_name}"
    ((built++)) || true
  else
    echo "[FAIL] ${repo_name}"
    ((failed++)) || true
  fi
done

echo ""
echo "===== Summary ====="
echo "Total:   ${total}"
echo "Built:   ${built}"
echo "Skipped: ${skipped}"
echo "Failed:  ${failed}"
