#!/usr/bin/env python
import argparse
import os

from method.indexing.decoupled_pipeline import Generator, GeneratorConfig, StorageManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summaries and write to SQLite (decoupled).")
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--sqlite_path", required=True)
    parser.add_argument("--graph_index_dir", type=str, default="")

    parser.add_argument("--summary_levels", type=str, default="function,class,file,module")
    parser.add_argument("--summary_llm_provider", type=str, default="openai")
    parser.add_argument("--summary_llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--summary_llm_api_key", type=str, default=None)
    parser.add_argument("--summary_llm_api_base", type=str, default=None)
    parser.add_argument("--summary_llm_max_new_tokens", type=int, default=512)
    parser.add_argument("--summary_llm_temperature", type=float, default=0.1)
    parser.add_argument("--llm_timeout", type=float, default=300)
    parser.add_argument("--llm_max_retries", type=int, default=3)
    parser.add_argument("--llm_backoff", type=float, default=1.0)

    parser.add_argument("--summary_prompt", type=str, default=None)
    parser.add_argument("--summary_prompt_file", type=str, default=None)
    parser.add_argument("--summary_language", type=str, default="English")

    parser.add_argument("--summary_cache_dir", type=str, default="")
    parser.add_argument("--summary_cache_policy", type=str, default="read_write")
    parser.add_argument("--summary_cache_miss", type=str, default="empty")

    parser.add_argument("--filter_min_lines", type=int, default=3)
    parser.add_argument("--filter_min_complexity", type=int, default=2)
    parser.add_argument("--filter_skip_patterns", type=str, default="^get_,^set_,^__.*__$")

    parser.add_argument("--top_k_callers", type=int, default=5)
    parser.add_argument("--top_k_callees", type=int, default=5)
    parser.add_argument("--context_similarity_threshold", type=float, default=0.8)

    parser.add_argument("--lang_allowlist", type=str, default=".py")
    parser.add_argument("--skip_patterns", type=str, default="")
    parser.add_argument("--max_file_size_mb", type=float, default=None)

    args = parser.parse_args()

    cfg = GeneratorConfig(
        summary_levels=[s.strip() for s in args.summary_levels.split(",") if s.strip()],
        graph_index_dir=args.graph_index_dir or None,
        summary_llm_provider=args.summary_llm_provider,
        summary_llm_model=args.summary_llm_model,
        summary_llm_api_key=args.summary_llm_api_key or os.environ.get("OPENAI_API_KEY"),
        summary_llm_api_base=args.summary_llm_api_base or os.environ.get("OPENAI_API_BASE"),
        summary_llm_max_new_tokens=args.summary_llm_max_new_tokens,
        summary_llm_temperature=args.summary_llm_temperature,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        llm_backoff=args.llm_backoff,
        summary_prompt=args.summary_prompt,
        summary_prompt_file=args.summary_prompt_file,
        summary_language=args.summary_language,
        summary_cache_dir=args.summary_cache_dir or None,
        summary_cache_policy=args.summary_cache_policy,
        summary_cache_miss=args.summary_cache_miss,
        filter_min_lines=args.filter_min_lines,
        filter_min_complexity=args.filter_min_complexity,
        filter_skip_patterns=[p.strip() for p in args.filter_skip_patterns.split(",") if p.strip()],
        top_k_callers=args.top_k_callers,
        top_k_callees=args.top_k_callees,
        context_similarity_threshold=args.context_similarity_threshold,
        lang_allowlist=args.lang_allowlist,
        skip_patterns=args.skip_patterns,
        max_file_size_mb=args.max_file_size_mb,
    )

    storage = StorageManager(sqlite_path=args.sqlite_path)
    Generator(storage=storage, config=cfg).run(args.repo_path)


if __name__ == "__main__":
    main()
