# 扩展指南（Chunk / Summary / Hybrid）

本项目当前已经支持 Dense / Sparse / Hybrid。下面给出未来扩展的**标准路径**与**最少改动点**，便于你在现有代码基础上迭代。

---

## 0. 扩展前置原则（务必遵守）

1. **Python-only**：默认只索引 `.py`，新逻辑必须复用 `method/indexing/utils/file.py` 的过滤机制。
2. **稳定的 Block Schema**：`Block` 使用 `start_line/end_line`（1-based），不可改字段含义。
3. **可复现**：同一仓库 + 同一策略，Block 顺序应稳定；Dense/Sparse/Hybrid 必须共享相同的 Block 顺序。
4. **兼容旧数据**：`metadata.jsonl` 的字段允许扩展，但不应破坏已有字段语义。

---

## 1. 扩展 Chunker（新增切块策略）

### 1.1 新建 Chunker
路径：`method/indexing/chunker/<your_chunker>.py`

最小骨架：
```python
from method.indexing.chunker.base import BaseChunker
from method.indexing.core.block import Block, ChunkResult
from method.indexing.utils.file import iter_files

class MyChunker(BaseChunker):
    block_type = "my_chunk"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks = []
        for file in iter_files(Path(repo_path)):
            # TODO: 读取文件 -> 切块 -> 生成 Block
            blocks.append(Block(file_path="...", start_line=1, end_line=10, content="...", block_type=self.block_type))
        metadata = {"chunker": self.get_metadata()}
        aux = {}
        return ChunkResult(blocks=blocks, metadata=metadata, aux=aux)
```

### 1.2 注册策略
在 `method/indexing/chunker/__init__.py` 中：
- import 新类
- 在 `CHUNKER_REGISTRY` 增加 key（策略名）
- 加入 `__all__`

### 1.3 验证
- 运行 `build_index.py`（单仓库）
- 检查 `metadata.jsonl` 的 `file_path` 是否只包含 `.py`
- 确认 `start_line/end_line` 连续且合法

---

## 2. 扩展“基于摘要”的索引

你有两种路线：

### 方案 A：复用 Summary Chunker + Dense Index
**适合快速落地**。直接使用现有 `summary` chunker（`method/indexing/chunker/summary.py`），与 Dense encoder 结合。

- 优点：改动最小
- 缺点：summary 与原始块共享同一 Dense 索引，不易区分

### 方案 B：独立 Summary Index（推荐）
**目标：单独维护 summary_index_{strategy}**

建议改造点：
1. 在 `method/indexing/encoder/summary/` 实现摘要向量编码器（如 `summarizer.py` / `summary_embedder.py`）。
2. 新增 `method/indexing/indexer/summary_indexer.py`：保存 `embeddings.pt` + `metadata.jsonl`。
3. 修改 `method/cli/build_index.py`，添加 `index_type=summary` 分支。
4. 约定输出结构：
   ```
   index_root/summary_index_{strategy}/{repo}/embeddings.pt
   index_root/summary_index_{strategy}/{repo}/metadata.jsonl
   ```
5. `metadata.jsonl` 建议保留字段：
   - `summary_text`（用于检索）
   - `original_content`（可选，用于调试/验证）

配置建议：
- 复用 `summary_config_template.yaml` / `summary_config_minimal.yaml`
- 使用 `method/indexing/core/llm_factory.py` 管理 LLM 初始化
- 缓存建议：summary 生成应开启本地 cache（已有实现）

---

## 3. 扩展 Sparse Index（新算法）

以 BM25 为模板：

1. 新增 `method/indexing/encoder/sparse/<algo>.py`
2. 新增 `method/indexing/indexer/<algo>_indexer.py`
3. 增加对应构建脚本（或扩展 `batch_build_sparse_index.py`）
4. 增加对应检索脚本（或扩展 `method/retrieval/sparse_retriever.py`）

建议输出格式：
- 稀疏矩阵：`index.npz`
- 词表：`vocab.json`
- 元数据：`metadata.jsonl`

---

## 4. 扩展 Hybrid（混合检索 / 融合策略）

当前混合检索入口：`method/retrieval/hybrid_retriever.py`

### 4.1 新增融合策略
- 在 `method/retrieval/fusion/` 新建 `<fusion>.py`
- 函数签名建议与 `fuse_rrf` / `fuse_weighted` 一致
- 在 `hybrid_retriever.py` 增加 `choices` 与调用

### 4.2 新增混合索引（可选）
原则：**不建议把融合结果预计算**，混合建议在运行时完成。

如果你必须实现“混合索引”，可用如下目录规范：
```
index_root/hybrid_index_{strategy}/{repo}/dense/...
index_root/hybrid_index_{strategy}/{repo}/sparse/...
index_root/hybrid_index_{strategy}/{repo}/metadata.jsonl
```

---

## 5. 扩展 checklist（最少改动）

- [ ] 新增 Chunker / Encoder / Indexer 文件
- [ ] 注册到对应 registry / __init__.py
- [ ] 更新 CLI（build_index / run_eval）
- [ ] 更新 README / QUICKSTART / TESTING
- [ ] 跑一次单仓库验证（索引 + 检索）

---

如需我帮你“按步骤拆分任务”或“直接改代码”，告诉我你要扩展哪一块即可。
