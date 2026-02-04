export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"

# 如果使用 vLLM OpenAI-compatible server：
export OPENAI_API_KEY=dummy_key   # 只需非空
export OPENAI_API_BASE=http://127.0.0.1:8000/v1

python method/indexing/batch_build_summary_index.py \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/summary \
  --summary_levels function,class,file,module \
  --graph_index_dir /home/chaihongzheng/workspace/locbench/code_index/graph_index \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --summary_llm_provider openai \
  --summary_llm_model GPT-OSS-120B \
  --summary_llm_api_base http://127.0.0.1:8000/v1 \
  --summary_llm_api_key dummy_key \
  --summary_cache_dir /home/chaihongzheng/workspace/locbench/code_index/summary_cache \
  --summary_cache_policy read_write \
  --summary_cache_miss empty \
  --summary_language Chinese \
  --skip_existing
