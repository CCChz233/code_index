# Summary Index 健康检查与排障说明
版本: v1.0  
范围: 实验运行健康确认、vLLM 变慢排查、缓存与中间产物说明

---

**1. 健康检查（是否有报错）**
1. 运行结束标志：终端输出 `All processes completed`。
2. 失败列表：检查 `logs/summary_index/failed_repos.jsonl` 是否为空。
3. 指标日志：检查 `logs/summary_index/metrics.jsonl` 中是否存在 `event=error`。
4. Worker 日志：查看 `logs/summary_index/summary_worker_{i}.log` 是否有 ERROR。
5. 产物完整性：确认 `summary_index_function_level/{repo}/summary.jsonl` 存在。

常用检查命令：
```bash
rg -n "\"event\": \"error\"" logs/summary_index/metrics.jsonl
wc -l logs/summary_index/failed_repos.jsonl
rg -n "ERROR|Traceback" logs/summary_index/summary_worker_*.log
```

---

**2. vLLM 变慢的定位路径**
优先判断是“上下文过长”还是“服务拥塞”。

2.1 上下文过长的信号  
1. `metrics.jsonl` 中 `llm_finish` 的 `usage.total_tokens` 普遍偏大。  
2. `llm_retry` 明显增多，或出现 `CONTEXT_LENGTH_ERROR`。  

2.2 服务拥塞的信号  
1. `llm_retry` 大量出现，且 error_type 多为 `LLM_API_ERROR`（超时/429/5xx）。  
2. vLLM server 日志显示队列积压、吞吐下降。  
3. 服务端存活检查：
```bash
curl http://127.0.0.1:8000/health
nvidia-smi
```

2.3 快速验证命令（按 repo 聚合 token）
```bash
python - <<'PY'
import json
from collections import defaultdict

path = "logs/summary_index/metrics.jsonl"
repo_tokens = defaultdict(int)
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("event") != "llm_finish":
            continue
        repo = obj.get("repo") or "-"
        payload = obj.get("payload") or {}
        usage = payload.get("usage") or obj.get("usage") or {}
        repo_tokens[repo] += int(usage.get("total_tokens", 0) or 0)

top = sorted(repo_tokens.items(), key=lambda x: x[1], reverse=True)[:10]
for repo, tok in top:
    print(f"{repo}: {tok}")
PY
```

2.4 可能的原因（按影响排序）
1. 文件/模块级摘要使用大量下层 summary 导致上下文变长。
2. 超大函数或超长 docstring 导致 prompt 变长。
3. vLLM 并发压力大，排队导致超时重试。

2.5 可验证的试验建议（不改代码）
1. 仅开 `function` 层级，关闭 `file/module` 观察速度差异。  
2. 降低并发：减小 `--num_processes`。  
3. 检查 `llm_retry` 数量，判断是否过载。

---

**3. 缓存逻辑说明**
3.1 缓存位置  
`summary_cache_dir/` 下按 `cache_key.json` 保存。

3.2 cache_key  
`md5(doc_id + summary_hash)`  
其中 `summary_hash = sha1(prompt + normalized_source + model_name)`。  
注：加入 `model_name` 确保切换模型时自动失效旧缓存。

3.3 复用逻辑  
1. 若 `summary.jsonl` 中已有 doc 且 `content_hash + summary_hash` 匹配：直接复用。  
2. 若缓存命中：读取 cache json 复用。  
3. 若缓存未命中：调用 LLM 生成，并写回（read_write 模式）。  

3.4 cache 策略  
1. `read_write`: 读 + 写  
2. `read_only`: 只读，未命中不生成  
3. `disabled`: 不使用缓存  

3.5 未命中处理  
1. `summary_cache_miss=empty`: 输出空摘要  
2. `summary_cache_miss=skip`: 跳过该 doc  
3. `summary_cache_miss=error`: 直接报错  

---

**4. 如何查看 vLLM 生成的摘要**
4.1 主输出（最终摘要）  
`summary_index_function_level/{repo}/summary.jsonl`  
字段包含 `summary`, `business_intent`, `keywords`, `id`, `type` 等。

4.2 缓存输出（单条摘要）  
`summary_cache_dir/{cache_key}.json`  
包含 `summary`, `business_intent`, `keywords`, `summary_hash`, `cached_at`。

4.3 快速查看某个函数摘要
```bash
rg -n "\"id\": \"src/auth.py::LoginHandler::authenticate\"" summary_index_function_level/<repo>/summary.jsonl
```

---

**5. 可视化与健康信号**
5.1 终端可视化  
`--progress_style rich` 会展示：
1. 全局进度条（已完成/总数/ETA）  
2. Worker 状态（IDLE/BUSY/ERROR）  
3. 当前 repo + 累计 token  
4. 完成日志（含 repo token）

5.2 无法判断是否卡住时  
1. Rich 表格中如果 worker 长时间停在同一 repo，可打开对应 `summary_worker_{i}.log`。  
2. 查看 metrics.jsonl 是否持续写入（`llm_finish` 事件持续出现则未卡死）。

---

**6. 中间产物保存逻辑（现状）**
每个 repo 输出：
```
summary_index_function_level/{repo}/
  summary.jsonl
  dense/embeddings.pt
  dense/index_map.json
  bm25/index.npz
  bm25/vocab.json
  bm25/metadata.jsonl
  manifest.json
```
全局日志：
```
logs/summary_index/metrics.jsonl
logs/summary_index/failed_repos.jsonl
logs/summary_index/summary_worker_{i}.log
```

---

**7. 是否需要重设计中间产物（建议）**
实验阶段一般不需要重构存储结构。  
如果你希望更强可恢复性或调试能力，可考虑以下“可选增强”：
1. 以 `run_id/` 作为日志与产物的顶层目录，避免多次运行混淆。  
2. 在 summary 生成阶段落地 `function_docs.jsonl`，便于复盘函数级摘要。  
3. 追加 `repo_metrics.json`（repo 级汇总：token、耗时、命中率）。  

这些增强不影响现有索引结构，但会增加 I/O 与存储。
