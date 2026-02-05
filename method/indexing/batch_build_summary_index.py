#!/usr/bin/env python
import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _safe_print(message: str, use_tqdm: bool) -> None:
    if use_tqdm and tqdm is not None:
        try:
            tqdm.write(message)
            return
        except Exception:
            pass
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

    def isatty(self) -> bool:
        return False


class _QueueAndFileWriter:
    def __init__(self, log_queue, file_obj):
        self._queue = log_queue
        self._file = file_obj

    def write(self, msg: str) -> None:
        if not msg:
            return
        for line in msg.replace("\r", "").splitlines():
            line = line.strip()
            if not line:
                continue
            if self._queue is not None:
                self._queue.put(line)
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def isatty(self) -> bool:
        return False


class _TeeWriter:
    def __init__(self, primary, file_obj):
        self._primary = primary
        self._file = file_obj

    def write(self, msg: str) -> None:
        if self._primary is not None:
            self._primary.write(msg)
        if self._file is not None:
            self._file.write(msg)
            self._file.flush()

    def flush(self) -> None:
        if self._primary is not None:
            self._primary.flush()
        if self._file is not None:
            self._file.flush()

    def isatty(self) -> bool:
        return False

def _emit_log(message: str, use_tqdm: bool, log_queue=None) -> None:
    if log_queue is not None:
        log_queue.put(message)
    else:
        _safe_print(message, use_tqdm)


def _list_repos(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.is_dir()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary index for multiple repos (multiprocess).")
    parser.add_argument("--repo_path", required=True, help="Root directory containing repos.")
    parser.add_argument("--index_dir", required=True, help="Root output directory for indexes.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip repos with existing summary.jsonl.")
    parser.add_argument("--resume_failed", action="store_true", help="Resume from failed_repos.jsonl.")

    parser.add_argument("--model_name", default="nov3630/RLRetriever")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--summary_embed_max_tokens", type=int, default=512)

    parser.add_argument("--summary_levels", type=str, default="function,class,file,module")
    parser.add_argument("--graph_index_dir", type=str, default="")

    parser.add_argument("--summary_llm_provider", type=str, default="openai", choices=["openai", "vllm"])
    parser.add_argument("--summary_llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--summary_llm_api_key", type=str, default=None)
    parser.add_argument("--summary_llm_api_base", type=str, default=None)
    parser.add_argument("--summary_llm_max_new_tokens", type=int, default=512)
    parser.add_argument("--summary_llm_temperature", type=float, default=0.1)
    parser.add_argument("--summary_prompt", type=str, default=None)
    parser.add_argument("--summary_prompt_file", type=str, default=None)
    parser.add_argument("--summary_language", type=str, default="English")
    parser.add_argument("--summary_cache_dir", type=str, default="")
    parser.add_argument(
        "--summary_cache_policy",
        type=str,
        default="read_write",
        choices=["read_only", "read_write", "disabled"],
    )
    parser.add_argument(
        "--summary_cache_miss",
        type=str,
        default="empty",
        choices=["empty", "skip", "error"],
    )
    parser.add_argument("--llm_timeout", type=float, default=300)
    parser.add_argument("--max_retries", type=int, default=3)

    parser.add_argument("--filter_min_lines", type=int, default=3)
    parser.add_argument("--filter_min_complexity", type=int, default=2)
    parser.add_argument("--filter_skip_patterns", type=str, default="^get_,^set_,^__.*__$")

    parser.add_argument("--top_k_callers", type=int, default=5)
    parser.add_argument("--top_k_callees", type=int, default=5)
    parser.add_argument("--context_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--show_progress", action="store_true", help="Show per-repo function summary progress.")

    parser.add_argument("--lang_allowlist", type=str, default=".py")
    parser.add_argument("--skip_patterns", type=str, default="")
    parser.add_argument("--max_file_size_mb", type=float, default=None)

    parser.add_argument("--num_processes", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage.")
    parser.add_argument("--log_dir", type=str, default="logs/summary_index", help="Log directory.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG/INFO/WARNING).")

    return parser.parse_args()


def _parse_gpu_ids(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def monitor_progress(completed_repos, total_pbar, total_repos):
    last = -1
    while True:
        current = completed_repos.value
        if current != last:
            total_pbar.n = current
            total_pbar.refresh()
            last = current
        if current >= total_repos:
            break
        time.sleep(1)


def run_worker(
    rank: int,
    repo_queue,
    args,
    gpu_ids: List[int],
    completed_repos,
    use_tqdm: bool,
    log_queue,
    failed_queue,
    stop_event,
):
    if args.num_processes > 1:
        os.environ.setdefault("TQDM_DISABLE", "1")

    actual_gpu_id = None
    if gpu_ids and not args.force_cpu:
        actual_gpu_id = gpu_ids[rank % len(gpu_ids)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(actual_gpu_id)

    # Per-worker log file
    log_path = Path(args.log_dir) / f"summary_worker_{rank}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")

    if log_queue is not None:
        sys.stdout = _QueueAndFileWriter(log_queue, log_file)
        sys.stderr = _QueueAndFileWriter(log_queue, log_file)
    else:
        sys.stdout = _TeeWriter(sys.stdout, log_file)
        sys.stderr = _TeeWriter(sys.stderr, log_file)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    # Lazy imports inside worker
    import torch
    from method.indexing.summary_index import build_summary_index

    device_label = f"GPU {actual_gpu_id}" if actual_gpu_id is not None else "CPU"
    _emit_log(f"[Process {rank}] Using {device_label}", use_tqdm, log_queue)

    summary_levels = [s.strip() for s in args.summary_levels.split(",") if s.strip()]
    skip_patterns = [p.strip() for p in args.filter_skip_patterns.split(",") if p.strip()]
    show_progress = args.show_progress and args.num_processes == 1

    while not stop_event.is_set():
        try:
            item = repo_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            break
        repo_path, repo_name = item

        try:
            output_dir = Path(args.index_dir) / "summary_index_function_level" / repo_name
            summary_jsonl = output_dir / "summary.jsonl"
            if args.skip_existing and summary_jsonl.exists():
                with completed_repos.get_lock():
                    completed_repos.value += 1
                continue

            start_ts = time.time()
            build_summary_index(
                repo_path=repo_path,
                output_dir=str(output_dir),
                model_name=args.model_name,
                trust_remote_code=args.trust_remote_code,
                batch_size=args.batch_size,
                summary_levels=summary_levels,
                graph_index_dir=args.graph_index_dir or None,
                summary_llm_provider=args.summary_llm_provider,
                summary_llm_model=args.summary_llm_model,
                summary_llm_api_key=args.summary_llm_api_key,
                summary_llm_api_base=args.summary_llm_api_base,
                summary_llm_max_new_tokens=args.summary_llm_max_new_tokens,
                summary_llm_temperature=args.summary_llm_temperature,
                llm_timeout=args.llm_timeout,
                llm_max_retries=args.max_retries,
                summary_prompt=args.summary_prompt,
                summary_prompt_file=args.summary_prompt_file,
                summary_language=args.summary_language,
                summary_cache_dir=args.summary_cache_dir or None,
                summary_cache_policy=args.summary_cache_policy,
                summary_cache_miss=args.summary_cache_miss,
                min_lines=args.filter_min_lines,
                min_complexity=args.filter_min_complexity,
                skip_patterns=skip_patterns,
                top_k_callers=args.top_k_callers,
                top_k_callees=args.top_k_callees,
                context_similarity_threshold=args.context_similarity_threshold,
                summary_embed_max_tokens=args.summary_embed_max_tokens,
                lang_allowlist=args.lang_allowlist,
                skip_file_patterns=args.skip_patterns,
                max_file_size_mb=args.max_file_size_mb,
                show_progress=show_progress,
            )
            elapsed = time.time() - start_ts
            _emit_log(f"[Process {rank}] Finished {repo_name} in {elapsed:.1f}s", use_tqdm, log_queue)
        except Exception as exc:
            failed_queue.put({"repo_name": repo_name, "error": str(exc)})
            _emit_log(f"[Process {rank}] Error processing {repo_name}: {exc}", use_tqdm, log_queue)
        finally:
            with completed_repos.get_lock():
                completed_repos.value += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    try:
        log_file.close()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.summary_llm_provider == "vllm" and args.num_processes > 1:
        raise ValueError("summary_llm_provider=vllm requires num_processes=1. Use openai+vLLM server.")

    use_tqdm = sys.stderr.isatty()
    if not use_tqdm:
        os.environ.setdefault("TQDM_DISABLE", "1")

    repo_root = Path(args.repo_path)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    failed_path = log_dir / "failed_repos.jsonl"

    repos: List[str] = []
    if args.resume_failed and failed_path.exists():
        with failed_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    repo_name = payload.get("repo_name")
                    if repo_name:
                        repos.append(repo_name)
                except Exception:
                    continue
        _safe_print(f"Resuming {len(repos)} failed repos", use_tqdm)
    else:
        repos = [p.name for p in _list_repos(repo_root)]
        _safe_print(f"Using local repos from {repo_root}", use_tqdm)

    repos = sorted(set(repos))
    if not repos:
        _safe_print("No repositories to process", use_tqdm)
        return

    gpu_ids = [] if args.force_cpu else _parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        args.force_cpu = True

    if args.skip_existing:
        filtered = []
        for repo_name in repos:
            output_dir = Path(args.index_dir) / "summary_index_function_level" / repo_name
            if not (output_dir / "summary.jsonl").exists():
                filtered.append(repo_name)
        repos = filtered

    ctx = __import__("multiprocessing").get_context("spawn")
    completed_repos = ctx.Value("i", 0)
    repo_queue = ctx.Queue()
    failed_queue = ctx.Queue()
    stop_event = ctx.Event()

    for repo_name in repos:
        repo_queue.put((str(repo_root / repo_name), repo_name))

    total_pbar = tqdm(
        total=len(repos),
        desc="Summary Index Total",
        disable=not use_tqdm,
        dynamic_ncols=True,
        mininterval=1.0,
        maxinterval=5.0,
    ) if tqdm else None

    log_queue = ctx.Queue() if use_tqdm and args.num_processes > 1 else None
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

    def _drain_failed():
        with failed_path.open("w", encoding="utf-8") as f:
            while True:
                try:
                    payload = failed_queue.get(timeout=0.5)
                except queue.Empty:
                    if completed_repos.value >= len(repos):
                        break
                    continue
                if payload is None:
                    break
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f.flush()

    failed_thread = threading.Thread(target=_drain_failed, daemon=True)
    failed_thread.start()

    def _handle_signal(signum, frame):
        _safe_print("[Main] Received stop signal, shutting down...", use_tqdm)
        stop_event.set()
        for _ in range(args.num_processes):
            repo_queue.put(None)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    monitor_thread = None
    if total_pbar is not None:
        monitor_thread = threading.Thread(
            target=monitor_progress,
            args=(completed_repos, total_pbar, len(repos)),
            daemon=True,
        )
        monitor_thread.start()

    processes = []
    if args.num_processes == 1:
        run_worker(
            0,
            repo_queue,
            args,
            gpu_ids,
            completed_repos,
            use_tqdm,
            log_queue,
            failed_queue,
            stop_event,
        )
    else:
        for rank in range(args.num_processes):
            p = ctx.Process(
                target=run_worker,
                args=(rank, repo_queue, args, gpu_ids, completed_repos, use_tqdm, log_queue, failed_queue, stop_event),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    if total_pbar is not None:
        total_pbar.n = len(repos)
        total_pbar.refresh()
        total_pbar.close()

    if log_queue is not None:
        log_stop.set()
        log_queue.put(None)
        if log_thread is not None:
            log_thread.join(timeout=2.0)

    failed_queue.put(None)
    failed_thread.join(timeout=2.0)

    _safe_print("All processes completed", use_tqdm)


if __name__ == "__main__":
    main()
