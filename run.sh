# 从 code_index 目录运行，确保 method 包可被找到
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
python method/indexing/batch_build_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/langchain_recursive \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --strategy langchain_recursive \
  --num_processes 4 \
  --gpu_ids 4,5,6,7 \
  --trust_remote_code \
  --skip_existing

python method/indexing/batch_build_sparse_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/BM25_ir_function \
  --strategy ir_function \
  --skip_existing

python method/indexing/batch_build_sparse_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/BM25_llamaindex_code \
  --strategy llamaindex_code \
  --skip_existing

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
python method/indexing/batch_build_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_code_v2_40_15_800_first_non_pad \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --strategy llamaindex_code \
  --num_processes 3 \
  --gpu_ids 0,2,3 \
  --trust_remote_code \
  --llamaindex_chunk_lines 40 \
  --llamaindex_chunk_lines_overlap 15 \
  --llamaindex_max_chars 800 \
  --pooling first_non_pad