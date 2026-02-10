# Fusion Search 运行文档（v1.0）

本说明针对新引擎 `scripts/search_fusion.py`，通过 YAML 配置并行使用 Dense / Sparse / Summary 三路索引进行融合检索。

## 1. 先决条件

1. 已完成三类索引构建（Dense / Sparse / Summary 独立索引）。
2. 安装 PyYAML（否则无法加载配置）：

```bash
pip install pyyaml
```

## 2. 配置文件

配置文件示例见：`configs/fusion_config.yaml`  
你只需要修改路径和模型名即可使用。

**必须字段**：

1. `repos_root`：代码仓库根目录。
2. `output_folder`：输出目录。
3. `dataset_path` 或 `dataset + split`：数据集来源。
4. `sources`：至少包含 dense / sparse / summary 三路。

## 3. 索引目录结构要求

Dense 索引目录（每个 repo 目录内）：

```
{dense_index_dir}/{repo_name}/embeddings.pt
{dense_index_dir}/{repo_name}/metadata.jsonl
```

Sparse 索引目录（每个 repo 目录内）：

```
{sparse_index_dir}/{repo_name}/index.npz
{sparse_index_dir}/{repo_name}/vocab.json
{sparse_index_dir}/{repo_name}/metadata.jsonl
```

Summary 独立索引目录（每个 repo 目录内）：

```
{summary_index_dir}/{repo_name}/summary.jsonl
{summary_index_dir}/{repo_name}/dense/embeddings.pt
{summary_index_dir}/{repo_name}/dense/index_map.json
```

## 4. 运行方式

```bash
python scripts/search_fusion.py --config_path /path/to/fusion_config.yaml
```

## 5. 输出

输出文件：

```
{output_folder}/loc_outputs.jsonl
```

每条记录包含：

1. `found_files`：融合后的文件级结果。
2. `found_modules` / `found_entities`：通过 mapper 从 block 命中推导。
3. `raw_output_loc`：包含 `best_provenance`，用于定位最相关代码片段。

## 6. 常见问题

1. **找不到索引文件**  
   检查 `index_dir` 是否指到正确的根目录，且目录结构符合上面的要求。

2. **报错缺少 PyYAML**  
   安装 `pyyaml`：`pip install pyyaml`

3. **Dense / Summary 模型不一致**  
   `model_name` 必须与索引构建时使用的模型一致，否则相似度分数不可比。

