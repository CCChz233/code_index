export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"

# ==== 配置区（按需修改） ====
REPOS_ROOT="/home/chaihongzheng/workspace/locbench/repos/locbench_repos"
INDEX_ROOT="/home/chaihongzheng/workspace/locbench/code_index/index_v2/summary"
GRAPH_INDEX_DIR="/home/chaihongzheng/workspace/locbench/code_index/graph_index"
EMBED_MODEL="/home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed"
SUMMARY_CACHE_DIR="/home/chaihongzheng/workspace/locbench/code_index/summary_cache"
LOG_DIR="/home/chaihongzheng/workspace/locbench/code_index/logs/summary_index"

# vLLM OpenAI-compatible server
export OPENAI_API_KEY=dummy_key   # 只需非空
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
SUMMARY_LLM_MODEL="GPT-OSS-120B"

# 多进程与 GPU 绑定
# 注意：若 vLLM 占用部分 GPU，请在这里排除
NUM_PROCESSES=4
EMBED_GPU_IDS="0,1,2,3"
RESUME_FAILED=0

EXTRA_FLAGS=""
if [ "$RESUME_FAILED" = "1" ]; then
  EXTRA_FLAGS="--resume_failed"
fi

python method/indexing/batch_build_summary_index.py \
  --repo_path "/home/chaihongzheng/workspace/locbench/repos/locbench_repos" \
  --index_dir "/home/chaihongzheng/workspace/locbench/code_index/index_v2/summary" \
  --summary_levels function,class,file,module \
  --graph_index_dir "/home/chaihongzheng/workspace/locbench/code_index/graph_index" \
  --model_name "/home/chaihongzheng/workspace/locbench/IR-based/models/CodeRankEmbed" \
  --trust_remote_code \
  --summary_llm_provider openai \
  --summary_llm_model "GPT-OSS-120B" \
  --summary_llm_api_base "http://127.0.0.1:8000/v1" \
  --summary_llm_api_key "dummy_key" \
  --summary_cache_dir "/home/chaihongzheng/workspace/locbench/code_index/summary_cache" \
  --summary_cache_policy read_write \
  --summary_cache_miss empty \
  --summary_language Chinese \
  --num_processes "4" \
  --gpu_ids "0,1,2,3" \
  --log_dir "/home/chaihongzheng/workspace/locbench/code_index/logs/summary_index" 
 
