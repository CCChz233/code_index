# Graph-Augmented Hierarchical Summary Index 设计文档
版本: v2.1 (Industrial Edition)  
状态: Draft (Design)  
范围: **仅定义“摘要索引（Summary Index）”的构建与存储**。混合检索将在后续文档单独定义。  
唯一事实来源（Single Source of Truth）：本文件 `summary.md`

---

## 1. 目标与设计哲学

### 1.1 目标
1. 提供独立的 Summary Index 检索管线，与现有 `dense` / `sparse` 代码索引并列可评估。
2. 在摘要中注入调用上下文，但不牺牲可维护性与增量效率。
3. 采用层次化摘要（Function -> File -> Module），降低 Token 爆炸。
4. 输出结构化摘要文档，兼容 Dense 与 Sparse 检索。
5. 支持可复现构建与可控增量更新。

### 1.2 设计哲学
1. Context-Aware: 摘要应体现“角色与意图”，而非仅代码语法。
2. Hierarchical: 上层摘要来自下层摘要，不重复读取原始代码。
3. Intent-Oriented: 摘要与业务意图分离，支持语义检索。

---

## 2. 决策记录 (Decision Record)

1. Summary Index 是独立检索管线，需可单独评估基线。
2. Function/Class/File/Module 存于同一索引库，用 `type` 过滤。
3. ID 稳定性以路径为准，Rename = New ID。
4. Dependencies 来源混合：function 使用 Call Graph；file/module 使用 Static Imports。
5. Summary 文本禁止出现具体 caller/callee 名称，仅保留语义描述。
6. 过滤掉的简单函数生成占位摘要，且不进入向量索引。
7. 默认基础分块策略为 `function_level`（AST-based）。
8. Class 级摘要必须生成，作为 File 级摘要的输入之一。

---

## 3. 核心数据结构

### 3.1 Summary Document Schema (JSON)
摘要索引最小单元为 **Summary Document**。

```json
{
  "id": "src/auth.py::LoginHandler::authenticate",
  "type": "function",
  "file_path": "src/auth.py",
  "module_path": "src/auth",
  "qualified_name": "LoginHandler.authenticate",

  "summary": "处理用户登录的认证逻辑，校验凭据并生成访问令牌。",
  "business_intent": "完成身份验证并维护安全访问。",

  "keywords": ["Login", "JWT", "OAuth", "Security"],
  "dependencies": ["db.query_user", "utils.hash_password"],

  "context": {
    "called_by": ["api.routes.login_endpoint", "test.test_auth"],
    "calls": ["db_connector", "logger"]
  },

  "content_hash": "a1b2c3d4e5f6",
  "graph_hash": "b2c3d4e5f6a1",
  "summary_hash": "c3d4e5f6a1b2",
  "hash": "c3d4e5f6a1b2",

  "language": "python",
  "last_updated": "2026-02-04T12:00:00Z",
  "is_placeholder": false,

  "start_line": 45,
  "end_line": 80
}
```

### 3.2 字段约束
1. 必须字段：`id` `type` `file_path` `module_path` `summary` `business_intent` `keywords` `content_hash` `graph_hash` `summary_hash` `hash` `language` `last_updated` `is_placeholder`
2. 可选字段：`qualified_name` `dependencies` `context` `start_line` `end_line`
3. `hash` 为兼容字段，默认等于 `summary_hash`
4. `context` 中的 caller/callee 列表不会进入摘要文本

### 3.3 字段适用范围（按 type）
1. `function` 必填 `qualified_name` `start_line` `end_line`，`dependencies` 来自 Call Graph
2. `class` 必填 `qualified_name`，`dependencies` 可为空
3. `file` 不要求 `start_line` / `end_line`，`dependencies` 来自 Static Imports
4. `module` 不要求 `start_line` / `end_line`，`dependencies` 来自 Static Imports

### 3.4 ID 规则
1. `function/class`: `{rel_path}::{qualified_name}`
2. `file`: `{rel_path}`
3. `module`: `{module_path}`
4. Rename = New ID，不维护 alias

---

## 4. Hash 规则与归一化规范

### 4.1 Hash 定义
1. `content_hash`: `sha1(normalized_source)`
2. `graph_hash`: `sha1(normalized_graph_context)`
3. `summary_hash`: `sha1(summary_prompt + normalized_source)`

### 4.2 normalize(code)
1. 移除所有注释
2. 保留 Docstring（Docstring 变化应触发摘要更新）
3. 移除空行
4. 连续空白压缩为单空格
5. 行尾空格去除，统一换行符为 `\n`

### 4.3 normalize(context)
1. 对 `called_by` / `calls` 列表去重
2. 按字母序排序
3. 序列化为 `called_by=a|b|c\ncalls=d|e|f`

---

## 5. 构建流水线 (Pipeline)

### 5.1 阶段 0: Graph 预构建
1. 优先复用 `dependency_graph`（AST-based）
2. 缺失 Graph 时降级为无上下文摘要

**调用上下文采样规则（确定性降噪）**
1. 同模块优先
2. 非测试代码优先（排除 `tests/`、`test_*.py`）
3. PageRank / 调用频次高者优先
4. 同分采用字母序裁决，确保稳定

### 5.2 阶段 1: 原子摘要生成 (Atomic)
1. 基础分块默认 `function_level`
2. 过滤规则：行数 < N (默认 3) 或 Cyclomatic Complexity < M (默认 2)
3. 被过滤函数生成占位摘要：`summary` 使用模板，`is_placeholder=true`，仅供上层聚合使用

**Prompt 约束**
1. 禁止在 `summary` / `business_intent` 中输出具体 caller/callee 名称
2. caller/callee 只允许出现在 `context` 字段中
3. 输出严格 JSON

### 5.3 阶段 2: 层次化聚合
1. Class 级摘要输入：Class Docstring + Method Summaries
2. 文件级摘要输入：Class Summaries + Standalone Function Summaries + File Docstring + Imports + Global Variables
3. 模块级摘要输入：文件摘要列表 + 目录 README（若存在）
4. 仅基于摘要与元数据，不再读取原始代码

### 5.4 阶段 3: 索引构建
1. Doc Store：`summary.jsonl` 保存完整文档
2. Dense Index：输入文本为 `qualified_name + business_intent + summary`，占位摘要不入向量库
3. Sparse Index：输入文本为 `keywords + qualified_name + file_path`

**Embedding 长度与截断规则**
1. `summary_embed_max_tokens` 默认 512
2. `qualified_name + business_intent` 永远保留
3. `summary` 超长时截断，保持确定性

---

## 6. 存储与目录结构 (Updated)

```
index_root/
  summary_index_{base_strategy}/
    {repo}/
      summary.jsonl         # 完整文档 (包含占位符)
      
      # Dense 索引
      dense/
        embeddings.pt       # 向量文件 (仅包含非占位符文档)
        index_map.json      # [List] 向量行号 -> doc_id 映射
      
      # Sparse 索引
      bm25/
        index.npz
        vocab.json
        metadata.jsonl
      
      manifest.json
```

约束：由于 `is_placeholder=true` 的文档不进入 Dense Index，`summary.jsonl` 与 `dense/embeddings.pt` 行数不对齐。必须读取 `dense/index_map.json` 通过向量行号反查 `doc_id`。

说明：`base_strategy` 为基础分块策略（默认 `function_level`），`summary_levels` 由 manifest 记录。

---

## 7. Manifest 规范

`manifest.json` 必须包含以下字段：
1. `schema_version`
2. `summary_levels`
3. `base_strategy`
4. `prompt_template_hash`
5. `summary_llm_provider`
6. `summary_llm_model`
7. `summary_llm_temperature`
8. `summary_llm_seed`（若不可控则记录为 null）
9. `graph_version`
10. `hash_policy`（content/graph/summary 规则）
11. `lang_allowlist` `skip_patterns` `max_file_size_mb`

---

## 8. 增量更新与缓存

1. `content_hash` 变化必须重算摘要与向量
2. `graph_hash` 变化只更新 `context`，不重算摘要
3. graph 变化若超过阈值（新增跨模块调用或新增调用者数量 > T）可触发重算
4. `summary_cache_dir` 以 `{id}:{summary_hash}` 缓存
5. 若新旧 context 的 Jaccard 相似度 > `context_similarity_threshold`，视为未变化

---

## 9. 失败模式与降级策略

1. Graph 不存在：跳过上下文注入
2. LLM 输出非法 JSON：解析失败则降级为 `summary=raw_text`
3. 文件过大或解析失败：记录日志并跳过
4. 过度更新：触发阈值保护并输出告警

---

## 10. 评估路径

1. Summary Index 检索返回 `doc_id`
2. 评估时需将 `doc_id` 映射到 GT
3. `function/class` 使用 `file_path + start_line + end_line`
4. `file` 使用文件级匹配
5. `module` 使用路径前缀匹配

---

## 11. 实施路线（建议）

1. Phase 1: 复用 `summary` chunker + Graph 上下文注入
2. Phase 2: 文件 / 模块级层次摘要
3. Phase 3: Summary Index 独立 CLI 与存储
4. Phase 4: 与 Dense / Sparse 统一评估

---

## 12. 与现有代码的对齐点

1. `method/indexing/chunker/summary.py` 可复用/扩展
2. `dependency_graph` 提供调用关系
3. `method/cli/build_index.py` 已预留 `summary` 入口

---

## Appendix A: Implementation Specifications

### A.1 复杂度过滤 (Complexity Filter)
1. 选型：使用 Python 标准库 `ast` 自行实现轻量级计数
2. 简化算法：统计 `if/elif/for/while/except/with/assert` 的数量 + 1
3. 配置：
4. `filter.min_lines`: 3
5. `filter.min_complexity`: 2
6. `filter.skip_patterns`: `["^get_", "^set_", "^__.*__$"]`

### A.2 层次化信息抽取 (Hierarchical Extraction)
1. Docstring：使用 `ast.get_docstring(node)`
2. Imports：遍历 Module 级 `ast.Import` 与 `ast.ImportFrom`
3. Globals：遍历 Module 级 `ast.Assign` 与 `ast.AnnAssign`，仅保留全大写变量名

### A.3 Graph 权重计算 (Graph Importance)
1. 选型：优先 `networkx`，否则内存计数
2. In-Degree：统计被引用次数
3. PageRank（可选）：全量构建时计算，阻尼系数 `alpha=0.85`
4. 缓存：写入 `graph_index/stats.json`，结构 `{node_id: {in_degree: int, pagerank: float}}`

### A.4 默认阈值与参数 (Default Configurations)
1. `summary_embed_max_tokens`: 512
2. `embedding_model`: `text-embedding-3-small` 或同等轻量级模型
3. `context_similarity_threshold`: 0.8
4. `top_k_callers`: 5
