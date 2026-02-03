# 从 code_index 目录运行，确保 method 包可被找到
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
python method/retrieval/run_with_index.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/dense_index_ir_function \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_index_ir_function \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50