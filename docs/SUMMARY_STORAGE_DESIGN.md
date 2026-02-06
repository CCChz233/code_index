# 设计文档：解耦摘要生成 + 混合存储（MVP）
版本：v0.1  
状态：Draft  
范围：仓库级代码索引系统

---

## 1. 目标（Goals）
1. **解耦摘要生成与索引构建**（Producer–Consumer）
2. **SQLite 作为黄金数据源**（Single Source of Truth）
3. **混合存储架构**：SQLite + ChromaDB + BM25 + NetworkX
4. **支持增量更新**（基于 stable_id + hash）
5. MVP：保证数据流清晰、可落地、可扩展

---

## 2. 非目标（Non‑Goals）
- 不做最终检索策略或 Web UI
- 不引入分布式调度框架（Celery/Beam）
- 图存储仅用 NetworkX（非图数据库）
- 不实现跨语言统一的 ID（先聚焦 Python）

---

## 3. 总体架构

```
Generator（重任务）         SQLite（黄金数据源）          Indexer（轻任务）
─────────────────         ───────────────────         ────────────────
扫描源码                  nodes 表                    读取 nodes/edges
AST 解析                  edges 表                    计算向量 → Chroma
静态依赖图                                            构建 BM25
LLM 生成摘要                                            构建图 → gpickle
写入 SQLite                                            写入索引产物
```

**原则**：  
Generator 只负责“生成 + 写入 SQLite”；Indexer 只负责“读取 SQLite + 构建索引”。

---

## 4. 数据模型（Pydantic）

- `NodeModel`
  - stable_id
  - file_path
  - type（function/class/file/module）
  - summary_json（结构化 JSON）
  - content_hash
  - summary_hash
  - updated_at

- `EdgeModel`
  - src_id
  - dst_id
  - edge_type
  - updated_at

- `SummaryPayload`
  - summary
  - business_intent
  - keywords
  - metadata

---

## 5. SQLite Schema

**nodes 表**
- stable_id (PRIMARY KEY)
- file_path
- type
- summary_json (TEXT JSON)
- content_hash
- summary_hash
- updated_at

**edges 表**
- src_id
- dst_id
- edge_type
- updated_at

约束建议：  
`UNIQUE(src_id, dst_id, edge_type)`。

---

## 6. Generator（生产流）

**职责**
1. 扫描代码 → AST 解析 → 函数/类级节点
2. 构建静态调用边（edges）
3. LLM 生成摘要
4. 写入 SQLite（nodes + edges）

**增量逻辑**
- content_hash 未变 → 跳过 LLM
- summary_hash 变 → 更新摘要
- 更新 updated_at

---

## 7. Indexer（消费流）

**职责**
1. 从 SQLite 读取 nodes/edges
2. 计算向量 → 写入 ChromaDB（upsert）
3. 构建 BM25 → 序列化保存
4. 构建 NetworkX 图 → gpickle

**增量逻辑**
- summary_hash 改变 → 更新向量 + BM25
- edges 改变 → 重建图（MVP 可全量）

---

## 8. StorageManager（MVP 接口设计）

**职责统一管理四个存储：**
- SQLite（nodes/edges）
- ChromaDB（向量）
- NetworkX（图）
- BM25（关键词）

**接口建议**
- `upsert_nodes(nodes)`
- `upsert_edges(edges)`
- `load_nodes(updated_since=None)`
- `load_edges(updated_since=None)`
- `upsert_vectors(id->vector)`
- `build_bm25(docs)`
- `save_graph(path)`
- `load_graph(path)`

---

## 9. 数据流契约（Contract）

**Generator → SQLite**  
写入结构化 summary_json，hash 计算后保存

**Indexer → Stores**
- Chroma: upsert by stable_id
- BM25: rebuild or partial update（MVP 先全量）
- Graph: gpickle

---

## 10. 稳定 ID & Hash 策略

**stable_id 格式（Python）**  
```
file_path::ClassName::method_name
file_path::function_name
file_path (file)
module_path (module)
```

**hash 策略**  
- content_hash = hash(normalized_source)
- summary_hash = hash(prompt + normalized_source + model_name)

---

## 11. 产物目录（MVP）
```
index_root/
  sqlite/
    summary.db
  chroma/
    ...
  bm25/
    index.pkl
  graph/
    graph.gpickle
  manifest.json
```

---

## 12. 风险与权衡
- BM25 增量更新复杂 → MVP 全量重建
- Graph 增量更新复杂 → MVP 全量重建
- Summary/Index 解耦后调度复杂度上升 → 以 CLI 分两步执行

---

## 13. 下一步
1. 确认 stable_id 格式  
2. 固定 SQLite schema  
3. 实现 Generator/Indexer 骨架  
4. 实现 StorageManager
