# Dense 索引效果提升实验文档（Pooling 对齐版）

适用范围：`method/indexing/batch_build_index.py` + `method/retrieval/run_with_index.py`  
目标：验证并提升 Dense 检索效果，重点排查/比较 pooling 策略。

## 1. 改动背景（你现在要对齐的点）

本轮改造后，Dense 侧新增了 `--pooling`（`cls / first_non_pad / last_token / mean`），并把默认 query `max_length` 调整为 `1024`。  
为了保证效果可比，构建与检索必须严格对齐以下三项：

1. `model_name`
2. `max_length`
3. `pooling`

只要 `pooling` 改了，就必须重建 Dense 索引。

## 2. 环境准备

```bash
conda activate locagent
cd /path/to/code_index
export PYTHONPATH="$(pwd):$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
```

## 3. 设置实验变量

按你的实际路径修改：

```bash
export REPOS_ROOT=/path/to/repos_root
export DATASET_PATH=/path/to/Loc-Bench_V1_dataset.jsonl
export MODEL_NAME=/path/to/embedding_model

export STRATEGY=ir_function
export MAX_LENGTH=1024
export NPROC=4
export GPUS=0,1,2,3

export EXP_ROOT=/path/to/exp_dense_pooling
mkdir -p "$EXP_ROOT"
```

## 4. 单次推荐配置（先跑这个）

推荐先用 `first_non_pad` 做一轮基线：

```bash
export POOLING=first_non_pad
export INDEX_ROOT="$EXP_ROOT/index_${STRATEGY}_${POOLING}"
export OUTPUT_ROOT="$EXP_ROOT/output_${STRATEGY}_${POOLING}"
```

### 4.1 构建 Dense 索引

```bash
python method/indexing/batch_build_index.py \
  --repo_path "$REPOS_ROOT" \
  --index_dir "$INDEX_ROOT" \
  --model_name "$MODEL_NAME" \
  --strategy "$STRATEGY" \
  --max_length "$MAX_LENGTH" \
  --pooling "$POOLING" \
  --num_processes "$NPROC" \
  --gpu_ids "$GPUS" \
  --trust_remote_code
```

### 4.2 Dense 测评

```bash
python method/retrieval/run_with_index.py \
  --dataset_path "$DATASET_PATH" \
  --index_dir "$INDEX_ROOT/dense_index_${STRATEGY}" \
  --output_folder "$OUTPUT_ROOT" \
  --model_name "$MODEL_NAME" \
  --trust_remote_code \
  --repos_root "$REPOS_ROOT" \
  --max_length "$MAX_LENGTH" \
  --pooling "$POOLING" \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50
```

结果文件在：

```bash
$OUTPUT_ROOT/loc_outputs.jsonl
```

## 5. Pooling 消融实验（建议）

一次性比较 `first_non_pad / mean / cls / last_token`：

```bash
for POOLING in first_non_pad mean cls last_token; do
  INDEX_ROOT="$EXP_ROOT/index_${STRATEGY}_${POOLING}"
  OUTPUT_ROOT="$EXP_ROOT/output_${STRATEGY}_${POOLING}"

  python method/indexing/batch_build_index.py \
    --repo_path "$REPOS_ROOT" \
    --index_dir "$INDEX_ROOT" \
    --model_name "$MODEL_NAME" \
    --strategy "$STRATEGY" \
    --max_length "$MAX_LENGTH" \
    --pooling "$POOLING" \
    --num_processes "$NPROC" \
    --gpu_ids "$GPUS" \
    --trust_remote_code

  python method/retrieval/run_with_index.py \
    --dataset_path "$DATASET_PATH" \
    --index_dir "$INDEX_ROOT/dense_index_${STRATEGY}" \
    --output_folder "$OUTPUT_ROOT" \
    --model_name "$MODEL_NAME" \
    --trust_remote_code \
    --repos_root "$REPOS_ROOT" \
    --max_length "$MAX_LENGTH" \
    --pooling "$POOLING" \
    --top_k_blocks 50 \
    --top_k_files 20 \
    --top_k_modules 20 \
    --top_k_entities 50
done
```

## 6. 快速检查输出是否异常

检查每个实验输出是否大量空结果：

```bash
python - <<'PY'
import json, glob, os

for path in sorted(glob.glob(os.path.expanduser("$EXP_ROOT/output_*/*loc_outputs.jsonl"))):
    total = 0
    empty = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            if not rec.get("found_files"):
                empty += 1
    ratio = (empty / total * 100) if total else 0.0
    print(f"{path}: empty_found_files={empty}/{total} ({ratio:.2f}%)")
PY
```

## 7. 常见问题

1. 效果没有变化或变差  
确保“索引构建”和“检索评测”完全使用同一组 `model_name + max_length + pooling`，并且在切换 pooling 后已重建索引。

2. 索引目录找不到  
`run_with_index.py --index_dir` 必须指向 `dense_index_{strategy}` 的上一级目录，命令里已按这个结构给出。

3. 多进程 OOM  
先降低 `NPROC` 或减小 `batch_build_index.py --batch_size`。

## 8. 推荐执行顺序

1. 先跑第 4 节（`first_non_pad`）拿到一版稳定结果。
2. 再跑第 5 节做 pooling 消融。
3. 选最优 pooling 后，再固定该配置扩展到你要比较的 chunk strategy。
