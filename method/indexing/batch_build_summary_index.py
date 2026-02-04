import argparse
from pathlib import Path
from typing import List, Optional

from method.indexing.summary_index import build_summary_index


def _list_repos(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.is_dir()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary index for multiple repos.")
    parser.add_argument("--repo_path", required=True, help="Root directory containing repos.")
    parser.add_argument("--index_dir", required=True, help="Root output directory for indexes.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip repos with existing summary.jsonl.")

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

    parser.add_argument("--filter_min_lines", type=int, default=3)
    parser.add_argument("--filter_min_complexity", type=int, default=2)
    parser.add_argument("--filter_skip_patterns", type=str, default="^get_,^set_,^__.*__$")

    parser.add_argument("--top_k_callers", type=int, default=5)
    parser.add_argument("--top_k_callees", type=int, default=5)
    parser.add_argument("--context_similarity_threshold", type=float, default=0.8)

    parser.add_argument("--lang_allowlist", type=str, default=".py")
    parser.add_argument("--skip_patterns", type=str, default="")
    parser.add_argument("--max_file_size_mb", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_path)
    index_root = Path(args.index_dir)

    summary_levels = [s.strip() for s in args.summary_levels.split(",") if s.strip()]
    skip_patterns = [p.strip() for p in args.filter_skip_patterns.split(",") if p.strip()]

    for repo_dir in _list_repos(repo_root):
        repo_name = repo_dir.name
        output_dir = index_root / "summary_index_function_level" / repo_name
        summary_jsonl = output_dir / "summary.jsonl"

        if args.skip_existing and summary_jsonl.exists():
            continue

        build_summary_index(
            repo_path=str(repo_dir),
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
        )


if __name__ == "__main__":
    main()
