#!/usr/bin/env bash
# 使用 run_with_index_convergent.py 跑齐所有测评方式：
#  - baseline / PRF / global_local（收敛模式）
#  - 无 rerank / 有 rerank
# 从 code_index 目录运行

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# 公共参数
COMMON_ARGS="\
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/IR-based/new_index_data/llamaindex_code_custom_40_15_800/dense_index_llamaindex_code \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_blocks 50 \
  --top_k_files 50 \
  --top_k_modules 50 \
  --top_k_entities 50 \
  --max_length 1024 \
  --pooling first_non_pad \
  --file_score_agg sum \
  --gpu_id 0"

# 收敛/PRF 用参数（prf 与 global_local 共用一套超参，仅 mode 不同）
CONVERGE_ARGS="\
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
  --min_cos_to_q0 0.75"

# Rerank 用参数
RERANK_MODEL="/home/chaihongzheng/workspace/model/cross-encoder__ms-marco-MiniLM-L6-v2/cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_ARGS="\
  --enable_rerank \
  --rerank_model ${RERANK_MODEL} \
  --rerank_top_k 50 \
  --rerank_batch_size 8 \
  --rerank_max_length 512 \
  --rerank_context_lines 0 \
  --rerank_snippet_max_lines 120 \
  --rerank_fail_open"

OUT_BASE="/home/chaihongzheng/workspace/locbench/code_index/output_eval"

echo "========== 1/6 baseline（单轮检索，无收敛、无 rerank）=========="
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "${OUT_BASE}/dense_llamaindex_code_v2_baseline" \
  --convergence_mode off

echo "========== 2/6 PRF only（多轮 PRF 收敛，无 rerank）=========="
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "${OUT_BASE}/dense_llamaindex_code_v2_prf_only" \
  --convergence_mode prf \
  $CONVERGE_ARGS

echo "========== 3/6 global_local only（多轮 global_local 收敛，无 rerank）=========="
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "${OUT_BASE}/dense_llamaindex_code_v2_global_local_only" \
  --convergence_mode global_local \
  --top_k_seed_files 20 \
  $CONVERGE_ARGS

echo "========== 4/6 rerank only（单轮检索 + cross-encoder rerank）=========="
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "${OUT_BASE}/dense_llamaindex_code_v2_rerank_only" \
  --convergence_mode off \
  $RERANK_ARGS

echo "========== 5/6 PRF + rerank（PRF 收敛后再 rerank）=========="
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "${OUT_BASE}/dense_llamaindex_code_v2_prf_rerank" \
  --convergence_mode prf \
  $CONVERGE_ARGS \
  $RERANK_ARGS

echo "========== 6/6 global_local + rerank（global_local 收敛后再 rerank）=========="
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder "${OUT_BASE}/dense_llamaindex_code_v2_global_local_rerank" \
  --convergence_mode global_local \
  --top_k_seed_files 20 \
  $CONVERGE_ARGS \
  $RERANK_ARGS

echo "========== 全部 6 种测评方式已完成 =========="
echo "结果目录: ${OUT_BASE}"
echo "  - dense_llamaindex_code_v2_baseline"
echo "  - dense_llamaindex_code_v2_prf_only"
echo "  - dense_llamaindex_code_v2_global_local_only"
echo "  - dense_llamaindex_code_v2_rerank_only"
echo "  - dense_llamaindex_code_v2_prf_rerank"
echo "  - dense_llamaindex_code_v2_global_local_rerank"
