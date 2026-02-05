# Summary Index 可视化设计文档
版本: v1.0  
状态: Draft / For Review  
范围: Summary Index 构建与评估的运行可视化与离线报告

---

## 1. 目标
1. 让构建过程“可感知”：进度、速度、耗时、错误、缓存命中一目了然
2. 让排障“可追溯”：每个 repo 的关键指标可查询
3. 让评估“可复盘”：检索指标与构建指标可关联

---

## 2. 非目标
1. 不构建完整 Web 控制台（仅预留扩展点）
2. 不实时展示源码或 LLM prompt（避免泄露与噪音）
3. 不在运行时增加重型依赖（优先轻量实现）

---

## 3. 当前痛点
1. 多进程日志与进度条互相干扰，输出噪音大
2. 只有“仓库级进度”，无法感知单仓库内部进度
3. 缺少结构化指标，无法做统计或趋势分析
4. 失败和重试信息分散，排障成本高

---

## 4. 总体方案
分三层实现，可独立启用，逐步增强。

1. 终端实时视图（TUI）
2. 结构化指标日志（JSONL）
3. 离线报告与可视化（HTML 或 PNG）

---

## 5. 结构化指标设计
统一输出到 `logs/summary_index/metrics.jsonl`，由主进程写入。

事件类型：
1. `run_start` / `run_end`
2. `repo_start` / `repo_end`
3. `llm_call` / `llm_retry` / `llm_finish`
4. `cache_hit` / `cache_miss`
5. `embedding_batch`
6. `status`（worker 心跳/状态刷新）
7. `error`

字段规范：
1. `ts`: ISO 时间戳
2. `event`: 事件类型
3. `repo`: 仓库名
4. `worker_id`: 进程编号
5. `duration_ms`: 耗时
6. `counts`: 计数器字典
7. `payload`: 动态字段（事件相关）

当 `event=llm_finish` 时：
1. `payload.latency_ms`
2. `payload.usage.prompt_tokens`
3. `payload.usage.completion_tokens`
4. `payload.usage.total_tokens`
5. `payload.usage.model`（实际调用模型，防止配置漂移）

当 `event=status` 时：
1. `payload.state`（Idle/Busy/Retrying/Error）
2. `payload.current_repo`
3. `payload.current_file`（可选）
4. `payload.stage`（Embed/Sum/Map）

当 `event=error` 时：
1. `payload.error_type`
2. `payload.error_msg`
3. `payload.stack_trace`（可选）

约束：
1. 不记录源码和原始 prompt
2. 仅记录统计与识别信息（doc_id 可哈希化）
3. Token 统计必须全量记录，不允许采样

---

## 6. 采集与管道
1. Worker 通过 `metrics_queue` 上报事件
2. 主进程 `metrics_thread` 统一写入 JSONL
3. 主进程同步维护聚合计数器，供 TUI 使用

容错：
1. `metrics_queue` 队列满时允许丢弃低优先级事件
2. 遇到异常仍要刷写 `run_end`

---

## 7. 终端可视化设计
提供两种模式：

模式 A：Simple（tqdm）
1. 仅显示总进度条 + 完成计数
2. 适合 CI 或日志重定向环境

模式 B：Live Table（rich）
1. 使用 `rich.live.Live` 刷新控制台
2. Header：总进度、累计 Tokens、总耗时
3. Body：Worker 状态表（每行一个 worker）
4. Footer：最近 5 条 Warning/Error

日志输出：
1. 每个 repo 完成时输出一条持久化日志
2. 日志包含 repo 名称与本 repo 的 Token 用量

示例布局：
```
Status: Running | Progress: 45/120 | Tokens: 1.2M

Worker ID | State    | Current Repo     | Stage | Last Event
----------|----------|------------------|-------|-----------
Worker-0  | BUSY     | torvalds/linux   | Sum   | LLM Generating...
Worker-1  | IDLE     | -                | -     | Waiting queue
Worker-2  | ERROR    | pytorch/pytorch  | Sum   | ContextLengthExceeded
Worker-3  | BUSY     | flask/flask      | Embed | Embedding batch
```

实现参数：
1. `--progress_style` 取值 `simple|tqdm|rich`（默认 `rich`）
2. `--refresh_interval` 控制刷新频率
3. `--show_repo_progress` 仅在单进程时显示函数级进度

日志降噪：
1. 默认将 `httpx/httpcore/openai/urllib3` 设为 WARNING
2. 保留 `method.*` 日志

---

## 8. 离线报告与图表
运行结束后生成：
1. `logs/summary_index/report.json`
2. `logs/summary_index/report.html`（单文件）

报告内容：
1. Repo 耗时分布
2. Blocks 数量分布
3. LLM 成功率与重试率
4. 缓存命中率分布
5. Token 统计（总量、均值、分位数）

生成方式：
1. 新增脚本 `scripts/plot_summary_metrics.py`
2. 使用 `jinja2` + `plotly` 生成单文件 HTML
3. 所有图表 JSON 直接内嵌，便于分享与归档

---

## 9. 评估可视化（可选）
当运行 `summary` 检索评估时：
1. 产出 `eval_metrics.jsonl`
2. 汇总 `Hit@k`、`MRR`、`Recall@k`
3. 把评估与构建报告关联到同一次 `run_id`

---

## 10. 性能与稳定性
1. 事件采样，避免每次 LLM 请求都写日志
2. JSONL 写入采用批量 flush
3. 指标聚合在主进程完成，避免锁竞争
4. Error 事件需分类，便于自动重试与统计

Error 类型建议：
1. `LLM_API_ERROR`（500/503/Timeout，可重试）
2. `CONTEXT_LENGTH_ERROR`（Prompt 过长，不可重试）
3. `PARSING_ERROR`（JSON 格式错，可重试）
4. `SYSTEM_ERROR`（OOM/DiskFull，需人工介入）

---

## 11. 与现有代码对齐
优先修改以下入口：
1. `method/indexing/batch_build_summary_index.py`
2. `run_summary.sh`

最小可行改动：
1. 增加 `metrics_queue` + 写入线程
2. 在关键节点发送事件
3. 增加 CLI 参数与默认值

---

## 12. 里程碑
Phase 1 (MVP):
1. JSONL 指标落地
2. 现有 tqdm 增强
3. 日志降噪

Phase 2 (增强):
1. rich TUI
2. 离线图表输出

Phase 3 (扩展):
1. 可选 Web Dashboard
2. 与评估指标联动展示
