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

python method/retrieval/run_with_index.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/langchain_recursive/dense_index_langchain_recursive \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_langchain_recursive \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50

python method/retrieval/run_with_index.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_code/dense_index_llamaindex_code \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_llamaindex_code \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50

python method/retrieval/sparse_retriever.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/BM25_ir_function/sparse_index_ir_function \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/sparse_index \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50 \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos

python method/retrieval/sparse_retriever.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/BM25_llamaindex_code/sparse_index_llamaindex_code \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/sparse_index_llamaindex_code \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50 \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos

cd /home/chaihongzheng/workspace/locbench/code_index

python method/cli/run_eval.py \
  --index_type summary \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/summary/summary_index_function_level \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval_summary \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --trust_remote_code \
  --top_k_docs 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
python method/retrieval/run_with_index.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_code_v2/dense_index_llamaindex_code \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_llamaindex_code_v2 \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_blocks 50 \
  --top_k_files 50 \
  --top_k_modules 50 \
  --top_k_entities 50