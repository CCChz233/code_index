# 测试文档（索引/检索一体化）

以下步骤覆盖索引构建与检索评估的主流程，并给出必要的验证点。

## 0. 环境准备

```bash
conda activate locagent
export PYTHONPATH="$(pwd):../LocAgent:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
```

可选依赖（按策略需要安装）：
- summary / llamaindex / langchain / epic 等策略需对应依赖
- summary 需要有效的 LLM API Key 与 API Base

> 注意：`method/indexing/summary_config_minimal.yaml` 中可能包含示例 key，务必替换为你自己的配置，且不要提交密钥。

## 1. 快速自检（不依赖外部服务）

目标：验证入口脚本可运行、参数解析正确。

```bash
python -m py_compile method/indexing/batch_build_index.py
python -m py_compile method/indexing/batch_build_index_sfr.py
python -m py_compile method/indexing/build_index.py
python -m py_compile method/retrieval/run_with_index.py
python -m py_compile method/retrieval/run_with_index_sfr.py
```

## 2. 单仓库索引（build_index）

目标：单仓库生成 `embeddings.pt` 与 `metadata.jsonl`。

```bash
python method/indexing/build_index.py \
  --repo_path /path/to/repo \
  --output_dir /path/to/output_index \
  --model_name /path/to/embedding_model \
  --strategy fixed \
  --block_size 15 \
  --batch_size 8 \
  --max_length 512
```

检查点：
- `/path/to/output_index/embeddings.pt` 存在
- `/path/to/output_index/metadata.jsonl` 存在且非空

可选：带 span_ids
```bash
python method/indexing/build_index.py \
  --repo_path /path/to/repo \
  --output_dir /path/to/output_index \
  --model_name /path/to/embedding_model \
  --strategy fixed \
  --graph_index_file /path/to/graph_index.pkl
```

## 3. 批量索引（batch_build_index）

目标：批量构建 dense_index_{strategy} 目录结构。

```bash
python method/indexing/batch_build_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --model_name /path/to/embedding_model \
  --strategy ir_function \
  --num_processes 1 \
  --skip_existing
```

检查点：
- 输出路径：`/path/to/index_root/dense_index_ir_function/{repo_name}/embeddings.pt`
- 输出路径：`/path/to/index_root/dense_index_ir_function/{repo_name}/metadata.jsonl`

## 3.1 批量稀疏索引（BM25）

```bash
python method/indexing/batch_build_sparse_index.py \
  --repo_path /path/to/repos_root \
  --index_dir /path/to/index_root \
  --strategy ir_function \
  --skip_existing
```

检查点：
- 输出路径：`/path/to/index_root/sparse_index_ir_function/{repo_name}/index.npz`
- 输出路径：`/path/to/index_root/sparse_index_ir_function/{repo_name}/vocab.json`
- 输出路径：`/path/to/index_root/sparse_index_ir_function/{repo_name}/metadata.jsonl`

## 4. Summary 策略（batch_build_index + config）

目标：验证 summary 流程可运行（依赖 LLM）。

```bash
python method/indexing/batch_build_index.py \
  --config method/indexing/summary_config_minimal.yaml
```

检查点：
- 日志中出现 summary 相关提示
- `index_dir` 中生成对应索引

> 如果想覆盖配置文件，请在命令行追加同名参数（CLI 会覆盖 config）。

## 5. 检索评估（v2 run_with_index）

目标：读取已构建索引并输出 `loc_outputs.jsonl`。

```bash
python method/retrieval/run_with_index.py \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --index_dir /path/to/index_root/dense_index_ir_function \
  --output_folder /path/to/output_eval \
  --model_name /path/to/embedding_model \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50
```

检查点：
- `/path/to/output_eval/loc_outputs.jsonl` 存在
- 控制台输出成功率与缺失索引统计

## 5.1 稀疏检索评估（BM25）

```bash
python method/retrieval/sparse_retriever.py \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --index_dir /path/to/index_root/sparse_index_ir_function \
  --output_folder /path/to/output_eval_sparse \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50 \
  --repos_root /path/to/repos_root
```

## 5.2 混合检索评估（Dense + Sparse）

```bash
python method/retrieval/hybrid_retriever.py \
  --dataset_path /path/to/Loc-Bench_V1_dataset.jsonl \
  --dense_index_dir /path/to/index_root/dense_index_ir_function \
  --sparse_index_dir /path/to/index_root/sparse_index_ir_function \
  --output_folder /path/to/output_eval_hybrid \
  --model_name /path/to/embedding_model \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50 \
  --repos_root /path/to/repos_root \
  --fusion rrf
```

## 6. span_ids 提取测试（可选）

```bash
python method/indexing/test_span_ids.py
```

依赖：本地存在 graph 索引（脚本内写死路径，必要时请自行修改）。

## 7. Python-only 过滤验证（推荐）

目标：确认非 `.py` 文件不会进入 metadata。

方法：
1. 选择一个包含其他语言文件的仓库（如含 `.js` / `.go`）
2. 构建索引后检查 `metadata.jsonl` 中的 `file_path` 后缀

示例检查命令（仅演示思路）：
```bash
python - <<'PY'
import json, re
from pathlib import Path

meta = Path(\"/path/to/index/metadata.jsonl\")
bad = 0
with meta.open() as f:
    for line in f:
        if not line.strip():
            continue
        path = json.loads(line)[\"file_path\"]
        if not path.endswith(\".py\"):
            bad += 1
print(\"non-py count:\", bad)
PY
```

## 7. 黄金样例对比（建议）

如果已有黄金样例：
- 运行旧入口与 v2 入口
- 对比 block 数量、起止行号、Top-K 文件列表
- 允许 metadata 字段顺序与浮点微小误差

---

若你希望我补一份“自动化对比脚本”（例如对比两个 metadata.jsonl 的 block 统计），告诉我目标格式与比较口径即可。
