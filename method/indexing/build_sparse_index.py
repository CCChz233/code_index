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
            "epic",
        ],
        help="Chunking strategy",
    )
    parser.add_argument("--block_size", type=int, default=15)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--slice_size", type=int, default=2)

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
