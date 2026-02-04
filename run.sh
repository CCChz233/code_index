# 从 code_index 目录运行，确保 method 包可被找到
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
python method/indexing/batch_build_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/dense_index_ir_function \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --strategy llamaindex_code \
  --num_processes 4 \
  --trust_remote_code \
  --skip_existing

python method/indexing/batch_build_sparse_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/BM25_ir_function \
  --strategy ir_function \
  --skip_existing

python method/indexing/batch_build_sparse_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/BM25_ir_function \
  --strategy ir_function \
  --skip_existing