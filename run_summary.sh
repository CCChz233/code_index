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
EMBED_GPU_IDS="4,5,6,7"
RESUME_FAILED=0

EXTRA_FLAGS=""
if [ "$RESUME_FAILED" = "1" ]; then
  EXTRA_FLAGS="--resume_failed"
fi

python method/indexing/batch_build_summary_index.py \
  --repo_path "$REPOS_ROOT" \
  --index_dir "$INDEX_ROOT" \
  --summary_levels function,class,file,module \
  --graph_index_dir "$GRAPH_INDEX_DIR" \
  --model_name "$EMBED_MODEL" \
  --trust_remote_code \
  --summary_llm_provider openai \
  --summary_llm_model "$SUMMARY_LLM_MODEL" \
  --summary_llm_api_base "$OPENAI_API_BASE" \
  --summary_llm_api_key "$OPENAI_API_KEY" \
  --summary_cache_dir "$SUMMARY_CACHE_DIR" \
  --summary_cache_policy read_write \
  --summary_cache_miss empty \
  --summary_language Chinese \
  --num_processes "$NUM_PROCESSES" \
  --gpu_ids "$EMBED_GPU_IDS" \
  --llm_timeout 300 \
  --max_retries 3 \
  --log_dir "$LOG_DIR" \
  --progress_style rich \
  --show_progress \
  --log_level INFO \
  --skip_existing \
  $EXTRA_FLAGS
