# 代码重构设计文档

## 背景情况

### 项目定位
这是一个 **IR-based（Information Retrieval）代码定位框架**，用于 LocBench 基准测试。主要功能是通过稠密向量检索（Dense Retrieval）从 bug 报告中定位相关代码文件、模块和函数。

### 与 LocAgent 的关系
- **IR-based**: 纯索引检索方法，无交互式 Agent，专注于基于索引的代码定位
- **LocAgent**: 交互式 LLM Agent 方法，使用图引导的多轮搜索

两者共享部分基础设施（评估模块、图构建模块等），但 IR-based 作为独立项目运行，有自己的代码组织和依赖。

### 工作流程
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   代码仓库       │ ──> │   索引构建       │ ──> │   索引文件       │
│ (repos/)        │     │ (batch_build)   │     │ (embeddings.pt  │
└─────────────────┘     └─────────────────┘     │  metadata.jsonl)│
                                                 └─────────────────┘
                                                          │
┌─────────────────┐     ┌─────────────────┐              │
│   Bug 报告      │ ──> │   检索评估       │ <─────────────┘
│ (dataset.jsonl) │     │ (run_with_index)│
└─────────────────┘     └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   定位结果       │
                       │ (loc_outputs    │
                       │  .jsonl)        │
                       └─────────────────┘
```

### 使用场景
1. **索引构建**: 对代码仓库进行切块和编码，生成向量索引
2. **检索评估**: 使用 bug 报告作为查询，检索最相关的代码块
3. **结果评估**: 计算定位准确率（Recall@K）

### 实际使用情况
根据分析，项目主要使用以下两个入口：

| 场景 | 使用的脚本 | 常用参数 |
|------|-----------|---------|
| 索引构建 | `batch_build_index.py` | `--strategy ir_function/summary`, `--num-processes 4` |
| 检索评估 | `run_with_index.py` | `--top-k-files 10`, `--top-k-entities 50` |

---

## 0. 兼容性冻结与验收基线（新增）

**目标**：先把“什么算兼容”定义清楚，避免重构后结果细微漂移。
**原则**：旧代码保持原样，所有改动只发生在 `method/index_v2/`，兼容性冻结面向 v2 产物。

### 0.1 需要冻结的行为
- 索引目录命名与文件名规则（`dense_index_{strategy}/{repo_name}/...`）
- `metadata.jsonl` 的字段集合、字段命名（字段顺序允许变化）
- block 排序规则（跨文件、跨函数的稳定排序）
- block ID 的生成方式（若存在）
- CLI 参数名称、默认值、类型（保持不变）
- 断点续跑与多进程分片策略（保持不变）

### 0.2 最小“黄金样例”
- 准备一个极小 repo（2-3 个文件）+ 2-3 条样本 bug 报告
- 固化输出：
  - blocks 数量
  - 每个 block 的 (file, start, end, content hash)
  - metadata 关键字段样例
  - 检索 TopK 结果

### 0.3 验收标准
- 黄金样例上 **功能一致（Functional Consistency）**
- Block 数量必须一致
- Block 起止行号与核心内容文本无逻辑差异（允许空白/换行细微差异）
- metadata 字段集合与语义一致（顺序不要求）
- Top-K 检索结果（File ID 列表）必须完全一致

### 0.4 不作为失败条件（新增）
- Embedding 浮点误差在 1e-6 量级内
- `metadata.jsonl` 字段顺序变化
- 轻微日志/打印差异

---

## 1. 现状分析

### 1.1 当前代码结构

```
method/index/
├── batch_build_index.py        # 索引构建主入口 (~780行)
├── batch_build_index_common.py # 共享代码 (~2850行)
└── build_index.py              # 单仓库索引构建 (~340行)

method/dense/
└── run_with_index.py           # 检索评估入口 (~380行)

method/
├── base.py                     # 基类定义 (~170行)
├── utils.py                    # 工具函数 (~390行)
└── mapping/                    # 映射器模块
    ├── ast_based/              # AST映射器
    └── graph_based/            # Graph映射器
```

### 1.2 支持的切块策略（全部保留）

| 策略类别 | 策略名称 | 说明 | 依赖 |
|---------|---------|------|------|
| **基础策略** | `fixed` | 固定行切块（不重叠） | 无 |
| | `sliding` | 滑动窗口切块（可重叠） | 无 |
| | `rl_fixed` | RLCoder 固定块（12非空行） | 无 |
| | `rl_mini` | RLCoder mini块（空行分段） | 无 |
| **函数级策略** | `ir_function` | IR论文函数级切块（带上下文） | AST |
| | `function_level` | 函数级切块（带fallback） | AST |
| **摘要策略** | `summary` | LLM摘要切块 | LLM API |
| **LlamaIndex策略** | `llamaindex_code` | CodeSplitter（AST-based） | llama-index |
| | `llamaindex_sentence` | SentenceSplitter | llama-index |
| | `llamaindex_token` | TokenTextSplitter | llama-index |
| | `llamaindex_semantic` | SemanticSplitter（语义分割） | llama-index+HF |
| **LangChain策略** | `langchain_fixed` | CharacterTextSplitter | llama-index+langchain |
| | `langchain_recursive` | RecursiveCharacterTextSplitter | llama-index+langchain |
| | `langchain_token` | TokenTextSplitter | llama-index+langchain |
| **Epic策略** | `epic` | EpicSplitter（与BM25一致） | llama-index |

### 1.3 主要问题

1. **代码冗余**: `batch_build_index_common.py` 包含所有切块策略，2850行难以维护
2. **职责不清**: 单个文件混杂了切块策略、编码逻辑、索引保存等多种功能
3. **策略分散**: 每个策略函数独立实现，没有统一的抽象接口
4. **依赖管理**: 可选依赖（llama-index、langchain）没有清晰隔离

### 1.4 重构原则

**重要**: 重构不改变任何功能，只改善代码组织
- ✅ 保留所有15种切块策略
- ✅ 保留所有命令行参数
- ✅ 保留索引格式和输出格式
- ✅ 保留多GPU并行功能
- ✅ 向后兼容现有索引

---

## 2. 重构目标

### 2.1 代码组织优化（新增平行目录，不动旧代码）

- **旧代码完全不动**：`method/index/` 保持原样
- **新代码进 v2**：新增 `method/index_v2/` 作为重构区
- **最终交付方式**：如需替换，整体复制 `method/index_v2/` 到目标位置即可

在现有仓库结构内拆分：

| 原文件（Legacy） | v2 位置 | 说明 |
|-------|--------|------|
| `method/index/batch_build_index_common.py` | `method/index_v2/chunker/*.py` | 各类切块策略模块 |
|  | `method/index_v2/core/block.py` | Block 数据类 |
|  | `method/index_v2/core/dataset.py` | BlockDataset 类 |
|  | `method/index_v2/core/encoding.py` | 编码/嵌入逻辑 |
|  | `method/index_v2/core/index_writer.py` | 索引保存 |
|  | `method/index_v2/core/index_builder.py` | 流程编排（chunk→encode→write） |
|  | `method/index_v2/utils/file.py` | 文件工具函数 |

### 2.2 切块策略模块化

按策略类别分组，每组独立模块：

```
method/index_v2/chunker/
├── __init__.py
├── base.py              # BaseChunker 抽象类
├── basic.py             # fixed, sliding, rl_fixed, rl_mini
├── function.py          # ir_function, function_level
├── summary.py           # summary (LLM摘要)
├── llamaindex.py        # llamaindex系列策略
├── langchain.py         # langchain系列策略
└── epic.py              # epic策略
```

### 2.3 依赖分层管理

```
核心依赖 (必需)
├── torch
├── transformers
├── datasets
└── tqdm

可选依赖 (按需安装)
├── llama-index      # llamaindex系列策略
├── langchain        # langchain系列策略
└── openai/vllm      # summary策略
```

---

## 3. 新代码结构（在现有仓库内）

```
method/index/                       # Legacy 旧代码，保持原样
└── ...

method/index_v2/
├── batch_build_index.py           # 新入口（CLI 保持一致）
├── batch_build_index_common.py    # 兼容层/桥接
├── batch_build_index_sfr.py       # SFR 入口（保留）
│
├── core/                          # 核心数据结构与流程
│   ├── __init__.py
│   ├── block.py                   # Block / ChunkResult 数据类
│   ├── dataset.py                 # BlockDataset (编码用)
│   ├── encoding.py                # 编码/嵌入逻辑
│   ├── index_builder.py           # 流程编排
│   └── index_writer.py            # 索引保存
│
├── chunker/                        # 切块策略
│   ├── __init__.py
│   ├── base.py
│   ├── basic.py
│   ├── function.py
│   ├── summary.py
│   ├── llamaindex.py
│   ├── langchain.py
│   └── epic.py
│
└── utils/
    ├── __init__.py
    └── file.py

method/dense/
└── run_with_index.py               # 旧入口保持不动，直接复用 v2 产物

method/mapping/                     # 原模块保持不动
```

---

## 4. 核心模块设计

### 4.1 统一切块返回结构（新增）

为避免策略差异导致分支膨胀，统一返回 `ChunkResult`：

```python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Block:
    file_path: str
    start: int
    end: int
    content: str
    block_type: str

    # 可选字段
    context_text: Optional[str] = None
    function_text: Optional[str] = None
    summary_text: Optional[str] = None
    original_content: Optional[str] = None

@dataclass
class ChunkResult:
    blocks: List[Block]
    metadata: Dict[str, Dict]  # 额外结构化信息，如 function_level
    aux: Dict[str, Dict]       # 临时辅助信息，序列化前可丢弃
```

### 4.2 BaseChunker 抽象类 (`chunker/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict

class BaseChunker(ABC):
    block_type: str

    @abstractmethod
    def chunk(self, repo_path: str) -> ChunkResult:
        """将仓库代码切块，返回 ChunkResult"""
        pass

    def get_metadata(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "block_type": self.block_type,
        }
```

### 4.3 各类策略接口调整
- 所有策略统一返回 `ChunkResult`
- 原本返回 `(blocks, function_metadata)` 的策略改为 `ChunkResult(blocks, metadata, aux)`
- 原本直接返回 blocks 的策略返回 `ChunkResult(blocks, {}, {})`

### 4.4 依赖惰性加载与注册表
- 注册表允许“延迟构造”，避免导入触发依赖错误
- 未安装依赖时给出明确提示，而不是导入时报错

---

## 5. 索引构建脚本设计

### 5.1 双入口并行（旧入口不动）
- `method/index/batch_build_index.py` **保持原样**
- 新增 `method/index_v2/batch_build_index.py`，CLI 参数、默认值、行为与旧入口一致
- `method/index_v2/batch_build_index_common.py` 作为兼容层/桥接

### 5.2 关键保持行为
- v2 的多进程分片策略保持不变（块顺序一致）
- v2 的断点续跑逻辑保持不变
- v2 的索引文件名与 metadata 结构保持不变

---

## 6. 检索评估脚本设计

- `method/dense/run_with_index.py` **保持原样**
- v2 输出格式与旧索引一致，因此旧检索脚本可直接复用
- 如需评估侧重构，另行新增 `method/dense_v2/`（不在本阶段）

---

## 7. 迁移计划（更新后）

迁移顺序原则：**先 Core，再 Chunker，最后 Flow（流程与并行逻辑）**，且仅在 v2 目录内进行。

### Phase -1: v2 沙盒搭建
1. 新增 `method/index_v2/` 目录与骨架
2. 复制入口脚本到 v2，保持 CLI 一致
3. 确保旧代码完全不动

### Phase 0: 兼容性冻结与验收基线
1. 冻结索引/metadata/schema/排序/CLI 参数
2. 生成黄金样例（blocks + metadata + TopK）
3. 写出验证脚本或人工步骤

### Phase 1: 基础结构抽象（v2 内完成）
1. 在 `method/index_v2/` 下新增 `core/` 与 `chunker/`
2. 实现 `Block`、`ChunkResult`、`BaseChunker`
3. v2 入口保持与旧 CLI 对齐

### Phase 2: 分策略迁移（每步跑黄金样例）
1. 迁移基础策略：`basic.py`
2. 迁移函数级策略：`function.py`
3. 迁移 LlamaIndex、LangChain、Summary、Epic
4. 为每类策略增加依赖检查和错误提示

### Phase 3: 索引构建与并行逻辑抽离
1. 抽离 IndexBuilder/Encoding/IndexWriter
2. 保持 shard 顺序与 resume 行为一致
3. 跑黄金样例对比

### Phase 4: （可选）检索评估迁移
1. 如需迁移，新增 `method/dense_v2/` 作为评估侧沙盒
2. mapper 逻辑保持不变（优先适配层）
3. 跑黄金样例对比

### Phase 5: 回归测试与文档
1. 全量策略抽样测试
2. README/QUICKSTART 标注兼容性与依赖
3. 清理过期代码（保留兼容导入）

---

## 8. 向后兼容

### 8.1 索引格式保持不变
```
index_dir/
└── dense_index_{strategy}/
    └── {repo_name}/
        ├── embeddings.pt       # torch.Tensor
        └── metadata.jsonl      # JSONL格式
```

### 8.2 输出格式保持不变
```json
{
    "instance_id": "UXARRAY__uxarray-1117",
    "found_files": ["uxarray/grid/grid.py"],
    "found_modules": ["uxarray/grid/grid.py:Grid"],
    "found_entities": ["uxarray/grid/grid.py:Grid.construct_face_centers"],
    "raw_output_loc": []
}
```

### 8.3 命令行参数保持兼容
- CLI 参数名称、默认值、类型全部保持不变
- 旧脚本路径仍可直接运行
- v2 入口与旧 CLI 对齐，便于对比与切换

---

## 9. 依赖管理

### 9.1 核心依赖 (requirements.txt)
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0
numpy>=1.24.0
```

### 9.2 可选依赖 (requirements-optional.txt)
```
# LlamaIndex策略
llama-index-core>=0.10.0
llama-index-readers>=0.10.0
llama-index-embeddings-huggingface>=0.10.0

# LangChain策略
langchain-text-splitters>=0.2.0

# Summary策略
openai>=1.0.0
# 或使用 vLLM
vllm>=0.4.0
```

### 9.3 依赖检查机制
- 在策略构造时检查依赖，而不是 import 时直接失败
- 注册表中记录 `required_deps`，用于 CLI 提示

---

## 10. 预期成果

### 10.1 代码组织改善
| 指标 | 原代码 | 重构后 |
|------|--------|--------|
| 最大单文件行数 | ~2850行 | ~400行 |
| 模块职责划分 | 混乱 | 清晰 |
| 策略扩展性 | 困难 | 简单（注册表模式） |
| 依赖隔离 | 无 | 分层管理 |

### 10.2 功能保持
- ✅ 全部15种切块策略
- ✅ 多GPU并行构建
- ✅ 检索评估功能
- ✅ 实体映射功能
- ✅ 断点续跑功能
- ✅ 结果可比（黄金样例功能一致）

### 10.3 用户体验提升
- ✅ 更清晰的错误提示（缺少依赖时）
- ✅ 更好的代码可读性
- ✅ 更容易扩展新策略
- ✅ 命令行参数完全兼容
