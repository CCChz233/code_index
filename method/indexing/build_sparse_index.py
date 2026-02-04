"""
Single-repo sparse BM25 index builder.
"""

import argparse
from pathlib import Path

from method.indexing.core.index_builder import IndexBuilder
from method.indexing.encoder.sparse import build_bm25_matrix
from method.indexing.indexer import save_bm25_index
from method.indexing.utils.file import set_file_scan_options


def parse_args():
    parser = argparse.ArgumentParser(description="Build a single-repo sparse BM25 index.")
    parser.add_argument("--repo_path", required=True, help="Repository root to index.")
    parser.add_argument("--output_dir", required=True, help="Where to save index + metadata.")

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
        help="Chunking strategy",
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

    return parser.parse_args()


def main():
    args = parse_args()

    suffixes = None
    if args.lang_allowlist:
        parts = [p.strip() for p in args.lang_allowlist.split(",") if p.strip()]
        suffixes = {p if p.startswith(".") else f".{p}" for p in parts} or None
    skip_patterns = [p.strip() for p in args.skip_patterns.split(",") if p.strip()] if args.skip_patterns else None

    set_file_scan_options(
        suffixes=suffixes,
        skip_patterns=skip_patterns,
        max_file_size_mb=args.max_file_size_mb,
    )

    if args.summary_max_tokens is not None:
        args.summary_llm_max_new_tokens = args.summary_max_tokens
    if args.summary_temperature is not None:
        args.summary_llm_temperature = args.summary_temperature
    if not args.summary_cache_dir:
        args.summary_cache_dir = str(Path(args.output_dir) / "_summary_cache")

    builder = IndexBuilder(args)
    artifacts = builder.collect_blocks(args.repo_path)

    blocks = artifacts.blocks
    if not blocks:
        raise RuntimeError(f"No blocks produced under {args.repo_path}")

    texts = [b.content for b in blocks]
    tf_matrix, vocab, idf, doc_len, avgdl = build_bm25_matrix(
        texts,
        min_df=args.bm25_min_df,
        max_df_ratio=args.bm25_max_df_ratio,
    )

    output_dir = Path(args.output_dir)
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
