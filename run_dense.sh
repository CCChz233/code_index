export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
python method/indexing/batch_build_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_cls \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --strategy llamaindex_code \
  --num_processes 3 \
  --gpu_ids 0,2,3 \
  --trust_remote_code \
  --llamaindex_chunk_lines 40 \
  --llamaindex_chunk_lines_overlap 15 \
  --llamaindex_max_chars 800 \
  --pooling cls 

python method/indexing/batch_build_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_first_non_pad \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --strategy llamaindex_code \
  --num_processes 3 \
  --gpu_ids 0,2,3 \
  --trust_remote_code \
  --llamaindex_chunk_lines 40 \
  --llamaindex_chunk_lines_overlap 15 \
  --llamaindex_max_chars 800 \
  --pooling first_non_pad

python method/indexing/batch_build_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_last_token \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --strategy llamaindex_code \
  --num_processes 3 \
  --gpu_ids 0,2,3 \
  --trust_remote_code \
  --llamaindex_chunk_lines 40 \
  --llamaindex_chunk_lines_overlap 15 \
  --llamaindex_max_chars 800 \
  --pooling last_token

python method/indexing/batch_build_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_mean \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --strategy llamaindex_code \
  --num_processes 3 \
  --gpu_ids 0,2,3 \
  --trust_remote_code \
  --llamaindex_chunk_lines 40 \
  --llamaindex_chunk_lines_overlap 15 \
  --llamaindex_max_chars 800 \
  --pooling mean

python method/retrieval/run_with_index.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_cls/dense_index_llamaindex_code \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_llamaindex_cls \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_blocks 50 \
  --top_k_files 50 \
  --top_k_modules 50 \
  --top_k_entities 50

python method/retrieval/run_with_index.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_first_non_pad/dense_index_llamaindex_code \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_llamaindex_first_non_pad \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_blocks 50 \
  --top_k_files 50 \
  --top_k_modules 50 \
  --top_k_entities 50

python method/retrieval/run_with_index.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_last_token/dense_index_llamaindex_code \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_llamaindex_last_token \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_blocks 50 \
  --top_k_files 50 \
  --top_k_modules 50 \
  --top_k_entities 50

python method/retrieval/run_with_index.py \
  --dataset_path /home/chaihongzheng/workspace/locbench/data/Loc-Bench_V1_dataset.jsonl \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/llamaindex_mean/dense_index_llamaindex_code \
  --output_folder /home/chaihongzheng/workspace/locbench/code_index/output_eval/dense_llamaindex_mean \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --repos_root /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --top_k_blocks 50 \
  --top_k_files 50 \
  --top_k_modules 50 \
  --top_k_entities 50