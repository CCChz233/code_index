# Dense 多步收敛测评文档（已实现版）

适用范围：

1. 原始单轮：`method/retrieval/run_with_index.py`
2. 新版收敛：`method/retrieval/run_with_index_convergent.py`

目标：在保留原始脚本的前提下，提供可复现的 `baseline vs prf vs global_local` 对照测评，并明确新版测评的执行逻辑。

## 1. 入口与兼容性

1. 原脚本保持不变，仍可作为单轮 baseline。
2. 新脚本是独立入口，不影响原流程。
3. 新脚本输出仍为 `loc_outputs.jsonl`，字段兼容：
   `instance_id/found_files/found_modules/found_entities/raw_output_loc`。
4. 因为输出结构兼容，现有评测/统计脚本可直接复用。

## 2. 新版测评逻辑（run_with_index_convergent.py）

### 2.1 单实例主流程

1. 读取数据集实例，按 `instance_id -> repo_name` 找对应索引。
2. 从 `embeddings.pt + metadata.jsonl` 取该仓库 block 向量和元数据。
3. 对 query 做一次编码得到 `q0`。
4. 根据 `convergence_mode` 执行检索：
   `off`：单轮检索（等价 baseline）。
   `prf`：全局多轮 PRF 收敛。
   `global_local`：首轮全局，后续在 seed files 内局部收敛。
5. 产出最终 `block_scores` 后，沿用原有映射逻辑得到 `found_files/found_modules/found_entities`。
6. 写入 `loc_outputs.jsonl`。

### 2.2 PRF 收敛逻辑

1. 每轮计算 `scores = q_t @ E^T`，取 `top_k_blocks_expand`。
2. 从当前轮高分块中选反馈集合（`feedback_top_m`），并限制单文件反馈上限（`feedback_file_cap`）。
3. 用 softmax(score/temp) 计算反馈权重，得到反馈中心向量。
4. 更新 query：
   `q_{t+1} = normalize((1-alpha-beta)*q_t + alpha*p_t + beta*q0)`。
5. 可选多轮融合：
   `last`：只用最后一轮。
   `rrf`：对各轮排序做 RRF 融合（默认）。

### 2.3 Global-Local 逻辑

1. 第 0 轮先全局召回。
2. 用第 0 轮结果聚合文件分数，取 `top_k_seed_files`。
3. 构建候选 block 子集，仅在该子集做后续 PRF 迭代。
4. 保留与 PRF 相同的早停与融合机制。

### 2.4 早停条件

每轮比较当前轮与上一轮：

1. Top-K Jaccard（`converge_jaccard_k`）。
2. Top-K 均值分数增益（`converge_min_improve`）。
3. `cos(q_t, q0)` 漂移约束（`min_cos_to_q0`）。

停止条件：

1. 达到 `max_steps`。
2. 连续 `patience` 轮满足“高重叠 + 低增益”。
3. 触发漂移阈值（`cos < min_cos_to_q0`）。

### 2.5 运行统计输出

新脚本除原有统计外，额外输出：

1. `Convergence Statistics`
2. `Instances with retrieval rounds`
3. `Average rounds used`

用于判断收敛是否有效和是否过慢。

## 3. 参数说明（核心）

与收敛相关的关键参数：

1. `--convergence_mode {off,prf,global_local}`
2. `--max_steps`
3. `--top_k_blocks_expand`
4. `--feedback_top_m`
5. `--feedback_file_cap`
6. `--query_update_alpha`
7. `--query_anchor_beta`
8. `--feedback_temp`
9. `--round_fusion {last,rrf}`
10. `--rrf_k`
11. `--converge_jaccard_k`
12. `--converge_jaccard_threshold`
13. `--converge_min_improve`
14. `--patience`
15. `--min_cos_to_q0`
16. `--top_k_seed_files`（仅 global_local 有效）

## 4. 同批数据对照命令模板

先设置公共变量（3 组实验必须共用同一套输入）：

```bash
cd /Users/chz/code/locbench/code_index
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export DATASET_PATH="/path/to/Loc-Bench_V1_dataset.jsonl"
export INDEX_DIR="/path/to/dense_index_ir_function"
export MODEL_NAME="/path/to/CodeRankEmbed"
export REPOS_ROOT="/path/to/locbench_repos"

COMMON_ARGS="\
  --dataset_path $DATASET_PATH \
  --index_dir $INDEX_DIR \
  --model_name $MODEL_NAME \
  --trust_remote_code \
  --repos_root $REPOS_ROOT \
  --top_k_blocks 50 \
  --top_k_files 20 \
  --top_k_modules 20 \
  --top_k_entities 50 \
  --max_length 1024 \
  --pooling first_non_pad \
  --file_score_agg sum \
  --gpu_id 0"
```

### 4.1 baseline（单轮）

```bash
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder /path/to/output_compare/baseline_off \
  --convergence_mode off
```

### 4.2 prf（全局多轮收敛）

```bash
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder /path/to/output_compare/prf \
  --convergence_mode prf \
  --max_steps 3 \
  --top_k_blocks_expand 100 \
  --feedback_top_m 8 \
  --feedback_file_cap 2 \
  --query_update_alpha 0.35 \
  --query_anchor_beta 0.15 \
  --feedback_temp 0.05 \
  --round_fusion rrf \
  --rrf_k 60 \
  --converge_jaccard_k 50 \
  --converge_jaccard_threshold 0.90 \
  --converge_min_improve 0.002 \
  --patience 1 \
  --min_cos_to_q0 0.75
```

### 4.3 global_local（两阶段收敛）

```bash
python method/retrieval/run_with_index_convergent.py \
  $COMMON_ARGS \
  --output_folder /path/to/output_compare/global_local \
  --convergence_mode global_local \
  --max_steps 3 \
  --top_k_seed_files 20 \
  --top_k_blocks_expand 100 \
  --feedback_top_m 8 \
  --feedback_file_cap 2 \
  --query_update_alpha 0.35 \
  --query_anchor_beta 0.15 \
  --feedback_temp 0.05 \
  --round_fusion rrf \
  --rrf_k 60 \
  --converge_jaccard_k 50 \
  --converge_jaccard_threshold 0.90 \
  --converge_min_improve 0.002 \
  --patience 1 \
  --min_cos_to_q0 0.75
```

## 5. 推荐评测记录项

每组实验至少记录：

1. 输出路径与命令（完整可复现）。
2. `file/module/entity` Hit@K。
3. 空结果比例（`found_files == []`）。
4. Convergence Statistics（仅新脚本）。
5. 运行耗时与 GPU 配置。

## 6. 推荐默认配置（v1）

如果先跑一版稳定实验，建议：

1. `convergence_mode=prf`
2. `max_steps=3`
3. `top_k_blocks_expand=100`
4. `feedback_top_m=8`
5. `feedback_file_cap=2`
6. `query_update_alpha=0.35`
7. `query_anchor_beta=0.15`
8. `round_fusion=rrf`
9. `converge_jaccard_threshold=0.90`
10. `converge_min_improve=0.002`
11. `patience=1`
12. `min_cos_to_q0=0.75`
