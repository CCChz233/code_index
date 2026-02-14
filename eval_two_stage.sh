#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Usage:
  bash eval_two_stage.sh [OPTIONS]

Description:
  Run two-stage dense retrieval evaluation with fully configurable parameters.

Options:
  --dataset_path PATH
  --index_dir PATH
  --output_folder PATH
  --model_name PATH
  --trust_remote_code / --no_trust_remote_code
  --strategy STR

  --stage1_top_k_blocks INT
  --top_k_blocks INT
  --stage1_max_files INT
  --stage2_min_accept_blocks INT
  --top_k_files INT
  --top_k_modules INT
  --top_k_entities INT
  --file_score_agg {sum|max}
  --max_length INT
  --pooling {first_non_pad|cls|last_token|mean}
  --batch_size INT

  --mapper_type {ast|graph}
  --repos_root PATH
  --graph_index_dir PATH

  --gpu_id INT
  --force_cpu / --no_force_cpu

  --help, -h

Example:
  bash eval_two_stage.sh \
    --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
    --index_dir /path/to/index_v2/llamaindex_code_v2_40_15_800_first_non_pad \
    --output_folder /path/to/output_eval/two_stage \
    --model_name /path/to/CodeRankEmbed \
    --repos_root /path/to/repos/locbench_repos \
    --strategy llamaindex_code \
    --stage1_top_k_blocks 100 \
    --top_k_blocks 20 \
    --stage2_min_accept_blocks 5 \
    --gpu_id 0
EOF
}

# ===== Defaults (can be overridden by env vars or CLI args) =====
DATASET_PATH="${DATASET_PATH:-/path/to/Loc-Bench_V1_dataset.jsonl}"
INDEX_DIR="${INDEX_DIR:-/path/to/index_root}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${SCRIPT_DIR}/output_eval/two_stage}"
MODEL_NAME="${MODEL_NAME:-/path/to/CodeRankEmbed}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
STRATEGY="${STRATEGY:-}"

STAGE1_TOP_K_BLOCKS="${STAGE1_TOP_K_BLOCKS:-100}"
TOP_K_BLOCKS="${TOP_K_BLOCKS:-20}"
STAGE1_MAX_FILES="${STAGE1_MAX_FILES:-0}"
STAGE2_MIN_ACCEPT_BLOCKS="${STAGE2_MIN_ACCEPT_BLOCKS:-5}"
TOP_K_FILES="${TOP_K_FILES:-20}"
TOP_K_MODULES="${TOP_K_MODULES:-20}"
TOP_K_ENTITIES="${TOP_K_ENTITIES:-50}"
FILE_SCORE_AGG="${FILE_SCORE_AGG:-sum}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
POOLING="${POOLING:-first_non_pad}"
BATCH_SIZE="${BATCH_SIZE:-8}"

MAPPER_TYPE="${MAPPER_TYPE:-ast}"
REPOS_ROOT="${REPOS_ROOT:-/path/to/repos/locbench_repos}"
GRAPH_INDEX_DIR="${GRAPH_INDEX_DIR:-}"

GPU_ID="${GPU_ID:-0}"
FORCE_CPU="${FORCE_CPU:-0}"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_path) DATASET_PATH="$2"; shift 2 ;;
    --index_dir) INDEX_DIR="$2"; shift 2 ;;
    --output_folder) OUTPUT_FOLDER="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --trust_remote_code) TRUST_REMOTE_CODE=1; shift ;;
    --no_trust_remote_code) TRUST_REMOTE_CODE=0; shift ;;
    --strategy) STRATEGY="$2"; shift 2 ;;

    --stage1_top_k_blocks) STAGE1_TOP_K_BLOCKS="$2"; shift 2 ;;
    --top_k_blocks) TOP_K_BLOCKS="$2"; shift 2 ;;
    --stage1_max_files) STAGE1_MAX_FILES="$2"; shift 2 ;;
    --stage2_min_accept_blocks) STAGE2_MIN_ACCEPT_BLOCKS="$2"; shift 2 ;;
    --top_k_files) TOP_K_FILES="$2"; shift 2 ;;
    --top_k_modules) TOP_K_MODULES="$2"; shift 2 ;;
    --top_k_entities) TOP_K_ENTITIES="$2"; shift 2 ;;
    --file_score_agg) FILE_SCORE_AGG="$2"; shift 2 ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --pooling) POOLING="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;

    --mapper_type) MAPPER_TYPE="$2"; shift 2 ;;
    --repos_root) REPOS_ROOT="$2"; shift 2 ;;
    --graph_index_dir) GRAPH_INDEX_DIR="$2"; shift 2 ;;

    --gpu_id) GPU_ID="$2"; shift 2 ;;
    --force_cpu) FORCE_CPU=1; shift ;;
    --no_force_cpu) FORCE_CPU=0; shift ;;

    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$MAPPER_TYPE" == "graph" && -z "$GRAPH_INDEX_DIR" ]]; then
  echo "Error: --mapper_type graph requires --graph_index_dir" >&2
  exit 1
fi

mkdir -p "$OUTPUT_FOLDER"

CMD=(
  python method/retrieval/run_with_index_two_stage.py
  --dataset_path "$DATASET_PATH"
  --index_dir "$INDEX_DIR"
  --output_folder "$OUTPUT_FOLDER"
  --model_name "$MODEL_NAME"
  --stage1_top_k_blocks "$STAGE1_TOP_K_BLOCKS"
  --top_k_blocks "$TOP_K_BLOCKS"
  --stage1_max_files "$STAGE1_MAX_FILES"
  --stage2_min_accept_blocks "$STAGE2_MIN_ACCEPT_BLOCKS"
  --top_k_files "$TOP_K_FILES"
  --top_k_modules "$TOP_K_MODULES"
  --top_k_entities "$TOP_K_ENTITIES"
  --file_score_agg "$FILE_SCORE_AGG"
  --max_length "$MAX_LENGTH"
  --pooling "$POOLING"
  --batch_size "$BATCH_SIZE"
  --mapper_type "$MAPPER_TYPE"
  --repos_root "$REPOS_ROOT"
)

if [[ -n "$STRATEGY" ]]; then
  CMD+=(--strategy "$STRATEGY")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi

if [[ "$MAPPER_TYPE" == "graph" ]]; then
  CMD+=(--graph_index_dir "$GRAPH_INDEX_DIR")
fi

if [[ "$FORCE_CPU" == "1" ]]; then
  CMD+=(--force_cpu)
else
  CMD+=(--gpu_id "$GPU_ID")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Done. Retrieval output: ${OUTPUT_FOLDER}/loc_outputs.jsonl"
