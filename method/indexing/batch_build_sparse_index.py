"""
Batch sparse BM25 index builder (single-process).
"""

import argparse
import os
from pathlib import Path
from typing import List

from method.indexing.core.index_builder import IndexBuilder
from method.indexing.encoder.sparse import build_bm25_matrix
from method.indexing.indexer import save_bm25_index
from method.indexing.utils.file import set_file_scan_options

from batch_build_index_common import list_folders, instance_id_to_repo_name

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


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


def should_skip_repo(repo_name: str, index_dir: str, strategy: str) -> bool:
    index_path = Path(index_dir) / f"sparse_index_{strategy}" / repo_name
    index_file = index_path / "index.npz"
    metadata_file = index_path / "metadata.jsonl"
    vocab_file = index_path / "vocab.json"
    return index_file.exists() and metadata_file.exists() and vocab_file.exists()


def _collect_repo_list(args) -> List[str]:
    if args.dataset:
        if not HAS_DATASETS:
            raise ImportError("datasets is required for --dataset")
        dataset = load_dataset(args.dataset, split=args.split)
        repo_names = [instance_id_to_repo_name(x["instance_id"]) for x in dataset]
        return sorted(set(repo_names))

    if args.repo_path:
        return list_folders(args.repo_path)

    raise ValueError("Either --dataset or --repo_path must be provided")


def parse_args():
    parser = argparse.ArgumentParser(description="Build sparse BM25 indexes (batch)")
    parser.add_argument("--dataset", type=str, default="", help="HF dataset name")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--repo_path", type=str, default="", help="Local repo root")
    parser.add_argument("--index_dir", type=str, required=True, help="Output index root")
    parser.add_argument("--instance_id_path", type=str, default="")

    parser.add_argument(
        "--strategy",
        type=str,
        default="ir_function",
        choices=[
            "fixed",
            "sliding",
            "rl_fixed",
            "rl_mini",
            "function_level",
            "ir_function",
            "summary",
            "llamaindex_code",
            "llamaindex_sentence",
            "llamaindex_token",
            "llamaindex_semantic",
            "langchain_fixed",
            "langchain_recursive",
            "langchain_token",
            "epic",
        ],
    )
    parser.add_argument("--block_size", type=int, default=15)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--slice_size", type=int, default=2)

    parser.add_argument(
        "--summary_base_strategy",
        type=str,
        default="ir_function",
        choices=[
            "ir_function",
            "function_level",
            "epic",
            "fixed",
            "sliding",
            "rl_fixed",
            "rl_mini",
            "langchain_fixed",
            "langchain_recursive",
            "langchain_token",
            "llamaindex_code",
            "llamaindex_sentence",
            "llamaindex_token",
            "llamaindex_semantic",
        ],
        help="Base strategy for summary blocks.",
    )
    parser.add_argument(
        "--summary_llm_provider",
        type=str,
        default="openai",
        choices=["openai", "vllm"],
        help="Summary LLM provider",
    )
    parser.add_argument(
        "--summary_llm_model",
        type=str,
        default="gpt-3.5-turbo",
        help="Summary LLM model name or path",
    )
    parser.add_argument("--summary_llm_api_key", type=str, default=None, help="LLM API key")
    parser.add_argument("--summary_llm_api_base", type=str, default=None, help="LLM API base URL")
    parser.add_argument("--summary_llm_max_new_tokens", type=int, default=512, help="LLM max_new_tokens")
    parser.add_argument("--summary_llm_temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--summary_prompt", type=str, default=None, help="Summary prompt template")
    parser.add_argument("--summary_prompt_file", type=str, default=None, help="Summary prompt template file")
    parser.add_argument("--summary_language", type=str, default="English", help="Summary language")
    parser.add_argument("--summary_cache_dir", type=str, default="", help="Summary cache directory")
    parser.add_argument(
        "--summary_cache_policy",
        type=str,
        default="read_write",
        choices=["read_only", "read_write", "disabled"],
        help="Summary cache policy",
    )
    parser.add_argument(
        "--summary_cache_miss",
        type=str,
        default="empty",
        choices=["empty", "skip", "error"],
        help="Behavior when summary cache misses",
    )
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
    parser.add_argument(
        "--summary_max_tokens",
        type=int,
        default=None,
        help="Deprecated alias for --summary_llm_max_new_tokens",
    )
    parser.add_argument(
        "--summary_temperature",
        type=float,
        default=None,
        help="Deprecated alias for --summary_llm_temperature",
    )

    parser.add_argument("--langchain_chunk_size", type=int, default=1000, help="LangChain chunk size")
    parser.add_argument("--langchain_chunk_overlap", type=int, default=200, help="LangChain chunk overlap")

    parser.add_argument("--llamaindex_language", type=str, default="python", help="Language for llamaindex_code")
    parser.add_argument("--llamaindex_chunk_lines", type=int, default=40, help="CodeSplitter chunk lines")
    parser.add_argument("--llamaindex_chunk_lines_overlap", type=int, default=15, help="CodeSplitter overlap")
    parser.add_argument("--llamaindex_max_chars", type=int, default=800, help="CodeSplitter max chars")

    parser.add_argument("--llamaindex_chunk_size", type=int, default=1024, help="LlamaIndex chunk size")
    parser.add_argument("--llamaindex_chunk_overlap", type=int, default=200, help="LlamaIndex chunk overlap")
    parser.add_argument("--llamaindex_separator", type=str, default=" ", help="TokenTextSplitter separator")
    parser.add_argument("--llamaindex_buffer_size", type=int, default=3, help="SemanticSplitter buffer size")
    parser.add_argument(
        "--llamaindex_embed_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SemanticSplitter embedding model",
    )

    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--bm25_b", type=float, default=0.75)
    parser.add_argument("--bm25_min_df", type=int, default=1)
    parser.add_argument("--bm25_max_df_ratio", type=float, default=1.0)

    parser.add_argument("--lang_allowlist", type=str, default=".py")
    parser.add_argument("--skip_patterns", type=str, default="")
    parser.add_argument("--max_file_size_mb", type=float, default=None)
    parser.add_argument("--skip_existing", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    _apply_scan_options(args)

    if args.summary_max_tokens is not None:
        args.summary_llm_max_new_tokens = args.summary_max_tokens
    if args.summary_temperature is not None:
        args.summary_llm_temperature = args.summary_temperature
    if not args.summary_cache_dir:
        args.summary_cache_dir = os.path.join(args.index_dir, "_summary_cache")

    repo_names = _collect_repo_list(args)
    if args.instance_id_path:
        with open(args.instance_id_path, "r", encoding="utf-8") as f:
            allowed = {line.strip() for line in f if line.strip()}
        repo_names = [r for r in repo_names if r in allowed]

    repo_root = args.repo_path
    if not repo_root and args.dataset:
        raise ValueError("--repo_path is required when using --dataset")

    builder = IndexBuilder(args)

    for repo_name in repo_names:
        if args.skip_existing and should_skip_repo(repo_name, args.index_dir, args.strategy):
            continue

        repo_path = os.path.join(repo_root, repo_name)
        if not os.path.isdir(repo_path):
            print(f"Skipping missing repo: {repo_name}")
            continue

        print(f"Processing repo: {repo_name}")
        artifacts = builder.collect_blocks(repo_path)
        blocks = artifacts.blocks
        if not blocks:
            print(f"No blocks produced for {repo_name}")
            continue

        texts = [b.content for b in blocks]
        tf_matrix, vocab, idf, doc_len, avgdl = build_bm25_matrix(
            texts,
            min_df=args.bm25_min_df,
            max_df_ratio=args.bm25_max_df_ratio,
        )

        output_dir = Path(args.index_dir) / f"sparse_index_{args.strategy}" / repo_name
        save_bm25_index(
            output_dir,
            tf_matrix,
            vocab,
            idf,
            doc_len,
            avgdl,
            strategy=args.strategy,
            blocks=blocks,
            k1=args.bm25_k1,
            b=args.bm25_b,
            function_metadata=artifacts.function_metadata if args.strategy in {"ir_function", "function_level"} else None,
            summary_metadata=artifacts.summary_metadata if args.strategy == "summary" else None,
        )


if __name__ == "__main__":
    main()
