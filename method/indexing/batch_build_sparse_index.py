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
            "epic",
        ],
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
    parser.add_argument("--skip_existing", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    _apply_scan_options(args)

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
