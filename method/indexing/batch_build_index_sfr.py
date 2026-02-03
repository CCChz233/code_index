#!/usr/bin/env python
"""
SFR 专用索引构建工具
使用 SentenceTransformer 后端
"""

import argparse
import json
import os
import os.path as osp
import queue
import random
import sys
import time
import threading
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from multiprocessing import Value

from method.indexing.encoder.dense import embed_blocks_sfr as _embed_blocks_sfr
from method.indexing.core.index_builder import IndexBuilder
from method.indexing.utils.file import set_file_scan_options


def _parse_allowlist(value: str):
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    normalized = set()
    for p in parts:
        if not p.startswith("."):
            p = f".{p}"
        normalized.add(p.lower())
    return normalized or None


def _parse_skip_patterns(value: str):
    if not value:
        return None
    return [p.strip() for p in value.split(",") if p.strip()] or None


def _apply_scan_options(args) -> None:
    set_file_scan_options(
        suffixes=_parse_allowlist(args.lang_allowlist),
        skip_patterns=_parse_skip_patterns(args.skip_patterns),
        max_file_size_mb=args.max_file_size_mb,
    )

# Optional YAML config support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


def _safe_print(message: str, use_tqdm: bool) -> None:
    print(message, flush=True)


class _QueueWriter:
    def __init__(self, log_queue):
        self._queue = log_queue

    def write(self, msg: str) -> None:
        if not msg:
            return
        for line in msg.replace("\r", "").splitlines():
            line = line.strip()
            if line:
                self._queue.put(line)

    def flush(self) -> None:
        return


def _emit_log(message: str, use_tqdm: bool, log_queue=None) -> None:
    if log_queue is not None:
        log_queue.put(message)
    else:
        _safe_print(message, use_tqdm)


def should_skip_repo(repo_name: str, index_dir: str, strategy: str) -> bool:
    """检查仓库索引是否已存在且完整"""
    index_path = Path(index_dir) / f"dense_index_{strategy}" / repo_name
    embeddings_file = index_path / "embeddings.pt"
    metadata_file = index_path / "metadata.jsonl"
    return embeddings_file.exists() and metadata_file.exists()

# 导入共享代码
from batch_build_index_common import (
    Block,
    # 工具函数
    resolve_model_name,
    list_folders,
    instance_id_to_repo_name,
)

# 尝试导入 datasets，如果失败则只支持本地模式
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def embed_blocks(
    blocks: List[Block],
    model: SentenceTransformer,  # 明确类型
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """SFR 专用的 embedding 函数，使用 SentenceTransformer 后端"""
    return _embed_blocks_sfr(
        blocks=blocks,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )


def monitor_progress(completed_repos: Value, total_pbar: tqdm, total_repos: int):
    """监控总进度并更新进度条"""
    import time
    last = -1

    builder = IndexBuilder(args, use_direct_llamaindex_params=False)
    while True:
        current = completed_repos.value
        if current != last:
            total_pbar.n = current
            total_pbar.refresh()
            last = current
        if current >= total_repos:
            break
        time.sleep(1)


def run(
    rank: int,
    repo_queue,
    args,
    gpu_ids: list,
    total_repos: int,
    completed_repos: Value,
    use_tqdm: bool,
    log_queue,
):
    """单个进程的工作函数"""
    import signal
    import atexit

    _apply_scan_options(args)

    if args.strategy == "summary":
        os.environ.setdefault("TQDM_DISABLE", "1")

    if use_tqdm and log_queue is not None:
        sys.stdout = _QueueWriter(log_queue)
        sys.stderr = _QueueWriter(log_queue)

    actual_gpu_id = None
    if gpu_ids and not args.force_cpu:
        actual_gpu_id = gpu_ids[rank % len(gpu_ids)]
        device = torch.device(f'cuda:{actual_gpu_id}')
        _emit_log(f'[Process {rank}] Using GPU {actual_gpu_id}', use_tqdm, log_queue)
    else:
        device = torch.device('cpu')
        _emit_log(f'[Process {rank}] Using CPU', use_tqdm, log_queue)

    def cleanup_resources():
        """清理资源"""
        try:
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            torch.cuda.empty_cache()
        except:
            pass

    atexit.register(cleanup_resources)

    builder = IndexBuilder(args, use_direct_llamaindex_params=False)

    while True:
        try:
            repo_info = repo_queue.get(timeout=1)
        except:
            break

        if repo_info is None:
            break

        repo_path, repo_name = repo_info

        # 检查是否需要跳过已处理的仓库
        if args.skip_existing and should_skip_repo(repo_name, args.index_dir, args.strategy):
            _emit_log(
                f'[Process {rank}] Skipping already processed repo: {repo_name}',
                use_tqdm,
                log_queue,
            )
            # 即使跳过也要更新计数器，避免进度条卡住
            with completed_repos.get_lock():
                completed_repos.value += 1
            continue

        _emit_log(f'[Process {rank}] Processing repo: {repo_name}', use_tqdm, log_queue)

        try:
            artifacts = builder.collect_blocks(repo_path)
            blocks = artifacts.blocks

            if not blocks:
                _emit_log(
                    f"[Process {rank}] No blocks collected from {repo_name}, skipping",
                    use_tqdm,
                    log_queue,
                )
                continue

            _emit_log(
                f"[Process {rank}] Collected {len(blocks)} blocks from {repo_name}",
                use_tqdm,
                log_queue,
            )
            # 加载 SFR 模型（固定使用 SentenceTransformer）
            if device.type == "cuda":
                _emit_log(
                    f'[Process {rank}] Loading SFR model on GPU {actual_gpu_id}...',
                    use_tqdm,
                    log_queue,
                )
            else:
                _emit_log(f'[Process {rank}] Loading SFR model on CPU...', use_tqdm, log_queue)
            trust_remote_code = args.trust_remote_code
            model_name = resolve_model_name(args.model_name)

            try:
                model = SentenceTransformer(
                    model_name,
                    device=device,
                    trust_remote_code=trust_remote_code
                )
                tokenizer = model.tokenizer
                if device.type == "cuda":
                    _emit_log(
                        f'[Process {rank}] Model loaded on GPU {actual_gpu_id}.',
                        use_tqdm,
                        log_queue,
                    )
                else:
                    _emit_log(f'[Process {rank}] Model loaded on CPU.', use_tqdm, log_queue)
            except Exception as e:
                _emit_log(f'[Process {rank}] Failed to load model: {e}', use_tqdm, log_queue)
                cleanup_resources()
                return

            block_count = builder.build_index(
                repo_path=repo_path,
                repo_name=repo_name,
                model=model,
                tokenizer=tokenizer,
                device=device,
                embed_fn=embed_blocks,
                embed_kwargs={
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                },
                index_dir=args.index_dir,
                artifacts=artifacts,
            )

            if block_count == 0:
                _emit_log(
                    f"[Process {rank}] No blocks collected from {repo_name}, skipping",
                    use_tqdm,
                    log_queue,
                )
                continue
            _emit_log(
                f'[Process {rank}] Saved index for {repo_name} with {block_count} blocks',
                use_tqdm,
                log_queue,
            )

            # 更新完成计数器
            with completed_repos.get_lock():
                completed_repos.value += 1

        except Exception as e:
            _emit_log(f'[Process {rank}] Error processing {repo_name}: {e}', use_tqdm, log_queue)
            import traceback
            traceback.print_exc()

            # 即使出错也更新计数器，避免进度条卡住
            with completed_repos.get_lock():
                completed_repos.value += 1

        finally:
            cleanup_resources()


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="",
                            help="Path to YAML config file")
    pre_args, _ = pre_parser.parse_known_args()

    config_data = {}
    if pre_args.config:
        if not HAS_YAML:
            raise ImportError("PyYAML is required for --config. Install with: pip install pyyaml")
        with open(pre_args.config, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        if not isinstance(config_data, dict):
            raise ValueError("Config file must be a YAML mapping at top level.")

    parser = argparse.ArgumentParser(
        description="SFR 专用索引构建工具（使用 SentenceTransformer）"
    )
    parser.add_argument("--config", type=str, default=pre_args.config,
                        help="Path to YAML config file")

    # 数据源参数
    parser.add_argument("--dataset", type=str, default="",
                       help="HuggingFace dataset name (e.g., 'czlll/Loc-Bench_V1')")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split")
    parser.add_argument("--repo_path", type=str,
                       default="/workspace/locbench/repos/locbench_repos",
                       help="Local repository directory")
    parser.add_argument("--index_dir", type=str,
                       default="/workspace/locbench/IR-based/new_index_data",
                       help="Output directory for indexes")
    parser.add_argument("--instance_id_path", type=str, default="",
                       help="Path to instance_id file for filtering")

    # 文件扫描过滤（Python-only 仍可覆盖）
    parser.add_argument("--lang_allowlist", type=str, default=".py",
                       help="Comma-separated file suffix allowlist, e.g. .py")
    parser.add_argument("--skip_patterns", type=str, default="",
                       help="Comma-separated glob patterns to skip")
    parser.add_argument("--max_file_size_mb", type=float, default=None,
                       help="Skip files larger than this size (MB)")

    # 模型参数（SFR 专用）
    parser.add_argument("--model_name", type=str, required=True,
                       help="SFR model path")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow execution of model repository code")

    # 切块策略参数
    parser.add_argument("--strategy", type=str, required=True,
                       choices=[
                           "ir_function", "function_level", "summary",
                           "llamaindex_code", "llamaindex_sentence", "llamaindex_token", "llamaindex_semantic",
                           "langchain_fixed", "langchain_recursive", "langchain_token",
                           "epic"
                       ],
                       help="Chunking strategy")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for embedding")
    parser.add_argument("--block_size", type=int, default=15,
                       help="Block size for fixed chunking")
    parser.add_argument("--window_size", type=int, default=20,
                       help="Sliding window size for summary base strategies")
    parser.add_argument("--slice_size", type=int, default=2,
                       help="Sliding step factor for summary base strategies")
    parser.add_argument("--ir_function_context_tokens", type=int, default=256,
                       help="Context tokens for IR function strategy")

    # Summary strategy parameters
    parser.add_argument(
        "--summary_base_strategy",
        type=str,
        default="ir_function",
        choices=[
            "ir_function", "function_level", "epic",
            "fixed", "sliding", "rl_fixed", "rl_mini",
            "langchain_fixed", "langchain_recursive", "langchain_token",
            "llamaindex_code", "llamaindex_sentence", "llamaindex_token", "llamaindex_semantic",
        ],
        help="Base strategy for summary blocks",
    )
    parser.add_argument("--summary_llm_provider", type=str, default="openai",
                       choices=["openai", "vllm"],
                       help="Summary LLM provider")
    parser.add_argument("--summary_llm_model", type=str, default="gpt-3.5-turbo",
                       help="Summary LLM model name or path")
    parser.add_argument("--summary_llm_api_key", type=str, default=None,
                       help="OpenAI-compatible API key for summary LLM")
    parser.add_argument("--summary_llm_api_base", type=str, default=None,
                       help="OpenAI-compatible API base URL for summary LLM")
    parser.add_argument("--summary_llm_max_new_tokens", type=int, default=512,
                       help="Summary LLM max_new_tokens")
    parser.add_argument("--summary_llm_temperature", type=float, default=0.1,
                       help="Summary LLM temperature")
    parser.add_argument("--summary_prompt", type=str, default=None,
                       help="Summary prompt template")
    parser.add_argument("--summary_prompt_file", type=str, default=None,
                       help="Path to summary prompt template file")
    parser.add_argument("--summary_language", type=str, default="English",
                       help="Language placeholder for summary prompt")
    parser.add_argument("--summary_cache_dir", type=str, default="",
                       help="Directory for summary cache")
    parser.add_argument("--summary_cache_policy", type=str, default="read_write",
                       choices=["read_only", "read_write", "disabled"],
                       help="Summary cache policy")
    parser.add_argument("--summary_cache_miss", type=str, default="empty",
                       choices=["empty", "skip", "error"],
                       help="Behavior when summary cache misses")
    parser.add_argument(
        "--summary_store_pure_summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store pure summary in metadata",
    )
    parser.add_argument(
        "--summary_store_pure_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store pure code in metadata",
    )
    parser.add_argument("--summary_max_tokens", type=int, default=None,
                       help="Deprecated alias for --summary_llm_max_new_tokens")
    parser.add_argument("--summary_temperature", type=float, default=None,
                       help="Deprecated alias for --summary_llm_temperature")

    # LangChain splitter parameters (for summary base strategy)
    parser.add_argument("--langchain_chunk_size", type=int, default=1000,
                       help="LangChain splitters chunk size")
    parser.add_argument("--langchain_chunk_overlap", type=int, default=200,
                       help="LangChain splitters chunk overlap")

    # LlamaIndex splitter parameters (for summary base strategy)
    parser.add_argument("--llamaindex_language", type=str, default="python",
                       help="LlamaIndex CodeSplitter language")
    parser.add_argument("--llamaindex_chunk_lines", type=int, default=40,
                       help="LlamaIndex CodeSplitter chunk lines")
    parser.add_argument("--llamaindex_chunk_lines_overlap", type=int, default=15,
                       help="LlamaIndex CodeSplitter overlap lines")
    parser.add_argument("--llamaindex_max_chars", type=int, default=1500,
                       help="LlamaIndex CodeSplitter max chars")
    parser.add_argument("--llamaindex_chunk_size", type=int, default=1024,
                       help="LlamaIndex splitter chunk size")
    parser.add_argument("--llamaindex_chunk_overlap", type=int, default=200,
                       help="LlamaIndex splitter chunk overlap")
    parser.add_argument("--llamaindex_separator", type=str, default=" ",
                       help="LlamaIndex TokenTextSplitter separator")
    parser.add_argument("--llamaindex_buffer_size", type=int, default=3,
                       help="LlamaIndex SemanticSplitter buffer size")
    parser.add_argument("--llamaindex_embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="LlamaIndex SemanticSplitter embedding model")

    # 并行处理参数
    parser.add_argument("--num_processes", type=int, default=4,
                       help="Number of parallel processes")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3",
                       help="Comma-separated GPU IDs")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU usage")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip repositories that already have complete indexes")

    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = [k for k in config_data.keys() if k not in valid_keys]
    if unknown_keys:
        print(f"Warning: Unknown config keys ignored: {sorted(unknown_keys)}")

    config_filtered = {k: v for k, v in config_data.items() if k in valid_keys}
    if config_filtered:
        parser.set_defaults(**config_filtered)

    args = parser.parse_args()

    _apply_scan_options(args)

    use_tqdm = sys.stderr.isatty()
    if not use_tqdm:
        os.environ.setdefault("TQDM_DISABLE", "1")

    if args.summary_max_tokens is not None:
        args.summary_llm_max_new_tokens = args.summary_max_tokens
    if args.summary_temperature is not None:
        args.summary_llm_temperature = args.summary_temperature

    if not args.summary_cache_dir:
        args.summary_cache_dir = osp.join(args.index_dir, "_summary_cache")

    if (
        args.strategy == "summary"
        and args.summary_llm_provider == "vllm"
        and args.num_processes > 1
    ):
        raise ValueError(
            "summary strategy with provider=vllm requires num_processes=1. "
            "Use provider=openai with a vLLM server for multi-process runs."
        )

    # 设置 GPU
    gpu_ids = []  # 初始化为空列表
    if not args.force_cpu:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
        if not gpu_ids:
            _safe_print("Warning: No valid GPU IDs provided, falling back to CPU", use_tqdm)
            args.force_cpu = True
        else:
            # 检测实际可用的 GPU 数量
            available_gpus = torch.cuda.device_count()
            if available_gpus == 0:
                _safe_print("Warning: No CUDA devices available, falling back to CPU", use_tqdm)
                args.force_cpu = True
                gpu_ids = []
            else:
                # 限制使用的 GPU 数量不超过实际可用数
                max_gpu = max(gpu_ids) + 1
                if max_gpu > available_gpus:
                    _safe_print(
                        f"Warning: Requested GPUs {gpu_ids} but only {available_gpus} available",
                        use_tqdm,
                    )
                    _safe_print(f"Limiting to use GPUs 0-{available_gpus-1}", use_tqdm)
                    gpu_ids = [i for i in range(available_gpus)]

    # 获取仓库列表
    if args.dataset and HAS_DATASETS:
        _safe_print(f"Loading dataset {args.dataset}", use_tqdm)
        ds = load_dataset(args.dataset, split=args.split)
        repos = []
        for item in ds:
            repo_name = item.get('repo_name') or item.get('instance_id', '').split('-')[0]
            if repo_name:
                repos.append(repo_name)
    else:
        _safe_print(f"Using local repos from {args.repo_path}", use_tqdm)
        repos = list_folders(args.repo_path)

    if args.instance_id_path:
        try:
            with open(args.instance_id_path, 'r') as f:
                instance_ids = [line.strip() for line in f if line.strip()]
            repos = [instance_id_to_repo_name(iid) for iid in instance_ids if iid]
        except Exception as e:
            _safe_print(
                f"Warning: Failed to load instance_ids from {args.instance_id_path}: {e}",
                use_tqdm,
            )

    repos = list(set(repos))  # 去重
    _safe_print(f"Processing {len(repos)} repositories", use_tqdm)

    if not repos:
        _safe_print("No repositories to process", use_tqdm)
        return

    # 创建多进程共享计数器和总进度条
    completed_repos = Value('i', 0)
    tqdm_lock = mp.RLock()
    tqdm.set_lock(tqdm_lock)
    log_queue = mp.Queue() if use_tqdm else None
    log_stop = threading.Event()
    log_thread = None
    if log_queue is not None:
        def _log_worker():
            while not log_stop.is_set():
                try:
                    msg = log_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if msg is None:
                    break
                _safe_print(msg, use_tqdm)

        log_thread = threading.Thread(target=_log_worker, daemon=True)
        log_thread.start()
    total_position = 0
    total_pbar = tqdm(
        total=len(repos),
        desc="总进度",
        ncols=80,
        disable=not use_tqdm,
        position=total_position,
        leave=True,
        mininterval=1.0,
        maxinterval=5.0,
        dynamic_ncols=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
    )
    if args.strategy == "summary":
        api_base = args.summary_llm_api_base or "default"
        base_display = api_base
        try:
            parsed = urlparse(api_base)
            if parsed.scheme and parsed.netloc:
                base_display = f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            base_display = api_base
        summary_info = (
            f"summary_llm={args.summary_llm_model} "
            f"provider={args.summary_llm_provider} "
            f"base={base_display} "
            f"base_strategy={args.summary_base_strategy}"
        )
        if use_tqdm:
            total_pbar.set_description(f"总进度 | {summary_info}")
            total_pbar.set_postfix_str("")
            total_pbar.refresh()
        else:
            _safe_print(f"Summary LLM: {summary_info}", use_tqdm)

    # 启动进度监控线程
    monitor_thread = threading.Thread(
        target=monitor_progress,
        args=(completed_repos, total_pbar, len(repos)),
        daemon=True
    )
    monitor_thread.start()

    # 创建仓库队列
    repo_queue = mp.Queue()
    for repo_name in repos:
        if args.dataset and HAS_DATASETS:
            repo_path = osp.join(args.repo_path, repo_name)
        else:
            repo_path = osp.join(args.repo_path, repo_name)
        repo_queue.put((repo_path, repo_name))

    # 启动多进程
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(
            target=run,
            args=(rank, repo_queue, args, gpu_ids, len(repos), completed_repos, use_tqdm, log_queue),
        )
        p.start()
        processes.append(p)

    # 等待完成
    for p in processes:
        p.join()

    # 确保进度条完成
    total_pbar.n = len(repos)
    total_pbar.refresh()
    total_pbar.close()
    if log_queue is not None:
        log_stop.set()
        log_queue.put(None)
        if log_thread is not None:
            log_thread.join(timeout=2.0)
    _safe_print("All processes completed", use_tqdm)


if __name__ == "__main__":
    main()
