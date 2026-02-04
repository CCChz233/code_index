export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"
export OPENAI_API_KEY=dummy_key   # 随便一个非空字符串
export OPENAI_API_BASE=http://127.0.0.1:8000/v1

python method/indexing/batch_build_index.py \
  --config method/indexing/summary_config_template.yaml \
  --repo_path /home/chaihongzheng/workspace/locbench/repos/locbench_repos \
  --index_dir /home/chaihongzheng/workspace/locbench/code_index/index_v2/summary \
  --model_name /home/chaihongzheng/workspace/locbench/LocAgent/models/CodeRankEmbed \
  --strategy summary \
  --trust_remote_code 