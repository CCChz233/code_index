export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
COMMON_ARGS="\
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_code_v2_40_15_800/dense_index_llamaindex_code \
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

python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_llamaindex_code_v2_40_15_800_prf_only \
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

# python method/retrieval/run_with_index_convergent.py \
#   $COMMON_ARGS \
#   --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_llamaindex_code_v2_40_15_800_rerank_only \
#   --convergence_mode off \
#   --enable_rerank \
#   --rerank_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
#   --rerank_top_k 50 \
#   --rerank_batch_size 8 \
#   --rerank_max_length 512 \
#   --rerank_context_lines 0 \
#   --rerank_snippet_max_lines 120 \
#   --rerank_fail_open