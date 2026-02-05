# Summary Index 多进程改造设计文档
版本: v1.2  
状态: Approved / Ready for Dev  
范围: Summary Index 构建流程的并行化改造（批量模式）

---

## 1. 目标
1. 提升 Summary Index 批量构建吞吐量
2. 利用 8 卡服务器资源
3. 保持可复现与稳定性

---

## 2. 现状瓶颈
1. Summary 生成依赖 LLM，串行调用耗时高
2. Dense embedding 每仓库单独加载，整体吞吐不足
3. 批量构建为单进程，repo 间无法并行

---

## 3. 改造方案（推荐）

### 3.1 Repo 级多进程并行（参考 batch_build_index.py）
1. 使用 `repo_queue` 动态分发 repo（负载均衡优于静态切片）
2. 每个进程循环消费队列，避免大 repo 阻塞总进度
3. 每个进程绑定一张 GPU（仅用于 embedding）
4. Summary LLM 统一走 vLLM server（OpenAI 兼容 API 并发）

### 3.2 优点
1. 实现简单，风险低
2. 吞吐接近线性提升（受 GPU 数限制）
3. 与现有 vLLM 部署兼容

---

## 4. 设计细节（对齐 batch_build_index.py）

### 4.1 任务分发（动态队列）
1. repos = list(repo_root)
2. repo_queue.put((repo_path, repo_name))
3. worker_i 轮询 `repo_queue` 直到耗尽

理由：动态队列能均衡长尾 repo，不会因固定分片拖慢整体。

### 4.2 GPU 绑定
1. 解析 `--gpu_ids`（如 `0,1,2,3`），不足时截断
2. worker_i 使用 `gpu_ids[rank % len(gpu_ids)]`
3. 只在子进程内加载 embedding 模型（每进程一次）
4. vLLM server 独立管理 GPU（Summary LLM 走 API）
5. Embedding 模型显存校验：`单进程显存占用 * 进程数 < 剩余显存`
6. 若 vLLM 占用部分 GPU，必须通过 `--gpu_ids` 明确排除

### 4.3 进度与日志（与 batch_build_index.py 同步）
1. 主进程创建 `completed_repos` (mp.Value) + 总进度条  
2. 子进程通过 `log_queue` 回传日志，避免多进程打印互相覆盖  
3. 使用 `_QueueWriter` 重定向 stdout/stderr  
4. `monitor_progress()` 线程刷新总进度  
5. 可选：每进程日志落地 `log_dir/summary_worker_{i}.log`  
6. `use_tqdm` 仅在 TTY 启用，子进程可设置 `TQDM_DISABLE=1` 避免嵌套进度条  
7. `failed_repos.jsonl` 由主进程统一写入（子进程通过 queue 上报）

### 4.4 容错与恢复
1. 单 repo 失败不影响其他 repo
2. 失败 repo 写入 `failed_repos.jsonl`
3. 提供 `--resume_failed` 重新跑失败列表

### 4.5 LLM Provider 约束（与 batch_build_index.py 保持一致）
1. 若 `summary_llm_provider=vllm`（本地进程），强制 `num_processes=1`
2. 多进程场景应使用 `summary_llm_provider=openai` + vLLM server

### 4.6 vLLM 并发与超时策略
1. LLM client timeout 建议 >= 300s（默认 60s 不足）
2. 增加 `max_retries` + 指数退避，处理短时拥塞与超时
3. 若 vLLM 压力过大，调低 `num_processes` 或限制并发

### 4.7 优雅退出
1. 主进程捕获 SIGINT / SIGTERM
2. 发送 sentinel 或 terminate 子进程，避免僵尸进程占用显存
3. 清理 log_queue 与进度线程
4. 使用 `spawn` 上下文启动子进程（避免 CUDA fork 问题）

---

## 5. CLI 设计（对齐 batch_build_index.py）

新增参数：
1. `--num_processes`（默认 1）
2. `--gpu_ids`（例如 `0,1,2,3,4,5,6,7`）
3. `--force_cpu`（强制 CPU）
4. `--skip_existing`（跳过已完成 repo）
5. `--log_dir`（默认 `logs/summary_index`）
6. `--resume_failed`（布尔开关）
7. `--llm_timeout`（默认 300s）
8. `--max_retries`（默认 3）

---

## 6. 进程模型（伪代码，对齐 batch_build_index.py）

主进程：
```
repos = list_repos()
repo_queue = Queue(repos)
spawn workers(repo_queue, gpu_ids)
monitor_progress(completed_repos)
collect results
```

子进程：
```
set CUDA_VISIBLE_DEVICES
while repo_queue not empty:
  build_summary_index(repo)
  report progress
```

---

## 7. 风险与规避

1. GPU 显存不足  
   - 方案：一进程一 GPU，避免共享
2. vLLM 限流  
   - 方案：调低 num_workers 或在 vLLM 侧限流
3. 输出冲突  
   - 方案：每 repo 独立输出目录
4. 进度丢失  
   - 方案：失败列表 + resume
5. 请求超时  
   - 方案：提高 timeout + 重试退避

---

## 8. 参考实现（对标 batch_build_index.py）
实现多进程部分时，直接复用以下成熟结构：
1. `mp.Queue` 作为 repo 任务队列  
2. `completed_repos` (mp.Value) + `monitor_progress()`  
3. `log_queue` + `_QueueWriter` 汇总日志  
4. `gpu_ids[rank % len(gpu_ids)]` 绑定 GPU  
5. `num_processes==1` 时保留单进程路径  

建议抽离公共工具模块：
1. `setup_logging_queue()`
2. `worker_init_process()`（GPU 绑定 + TQDM 设置）
3. `DynamicQueueDispatcher`
4. `GracefulShutdownManager`

---

## 9. 实施路线

1. Phase 1: 增加多进程与 repo 分片
2. Phase 2: GPU 绑定 + 失败恢复
3. Phase 3: 日志与进度管理完善

---

如确认方案，后续在 `batch_build_summary_index.py` 中落地实现。  

---

## 10. 实现避坑提示（Python 多进程）

1. 强制使用 `spawn` 启动方式  
2. 重量级 import 与模型加载必须放在 worker 内部  
3. Graceful shutdown 必须在主进程统一处理  
