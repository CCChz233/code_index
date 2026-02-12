"""
Single-repo dense index builder (v2).
Supports the same chunking strategies as batch_build_index.py,
while keeping a simple single-repo workflow for debugging.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModel

from dependency_graph import RepoEntitySearcher
from dependency_graph.build_graph import NODE_TYPE_FUNCTION, NODE_TYPE_CLASS

from method.core.embedding import POOLING_CHOICES
from method.indexing.core.block import Block
from method.indexing.encoder.dense import embed_blocks_coderank
from method.indexing.core.index_builder import IndexBuilder
from method.indexing.indexer import save_index
from method.indexing.chunker.collect import collect_blocks as collect_basic_blocks
from method.indexing.utils.file import set_file_scan_options


_BASIC_STRATEGIES = {"fixed", "sliding", "rl_fixed", "rl_mini"}


def extract_span_ids_from_graph(
    blocks: List[Block],
    graph_index_file: str,
    repo_path: str,
) -> Dict[int, List[str]]:
    """
    从 Graph 索引中为代码块提取 span_ids。

    Args:
        blocks: 代码块列表
        graph_index_file: Graph 索引文件路径（.pkl）
        repo_path: 仓库根目录（用于路径匹配）

    Returns:
        Dict[int, List[str]]: 代码块索引 -> span_ids 列表的映射
    """
    if not os.path.exists(graph_index_file):
        print(f"Warning: Graph index file not found: {graph_index_file}")
        return {}

    try:
        with open(graph_index_file, "rb") as f:
            graph = pickle.load(f)
        searcher = RepoEntitySearcher(graph)
    except Exception as e:
        print(f"Warning: Failed to load graph index {graph_index_file}: {e}")
        return {}

    blocks_by_file: Dict[str, List[Tuple[int, Block]]] = {}
    for idx, block in enumerate(blocks):
        file_path = block.file_path
        blocks_by_file.setdefault(file_path, []).append((idx, block))

    span_ids_map: Dict[int, List[str]] = {}

    for file_path, file_blocks in blocks_by_file.items():
        file_nodes = [node_id for node_id in searcher.G if node_id.startswith(f"{file_path}:")]

        for block_idx, block in file_blocks:
            span_ids = set()
            block_start_1based = block.start + 1
            block_end_1based = block.end + 1

            for node_id in file_nodes:
                if not searcher.has_node(node_id):
                    continue
                try:
                    node_data = searcher.get_node_data([node_id], return_code_content=False)[0]
                    node_start = node_data.get("start_line")
                    node_end = node_data.get("end_line")
                    if node_start is None or node_end is None:
                        continue
                    if not (block_end_1based < node_start or block_start_1based > node_end):
                        node_type = node_data.get("type")
                        if node_type in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS]:
                            if ":" in node_id:
                                entity_name = node_id.split(":", 1)[1]
                                span_ids.add(entity_name)
                except Exception:
                    continue

            if span_ids:
                span_ids_map[block_idx] = list(span_ids)

    return span_ids_map


def _collect_blocks(args):
    if args.strategy in _BASIC_STRATEGIES:
        blocks = collect_basic_blocks(
            args.repo_path,
            args.strategy,
            args.block_size,
            args.window_size,
            args.slice_size,
        )
        return blocks, {}, None

    builder = IndexBuilder(args)
    artifacts = builder.collect_blocks(args.repo_path)
    return artifacts.blocks, artifacts.function_metadata, artifacts.summary_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Build a single-repo dense index (v2).")
    parser.add_argument("--repo_path", required=True, help="Repository root to index.")
    parser.add_argument("--output_dir", required=True, help="Where to save embeddings + metadata.")

    parser.add_argument("--model_name", default="nov3630/RLRetriever", help="HF model id or local path.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow execution of model code.")

    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed",
        choices=[
            "fixed",
            "sliding",
            "rl_fixed",
            "rl_mini",
            "ir_function",
            "function_level",
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

    parser.add_argument("--max_length", type=int, default=512, help="Max tokens for encoding.")
    parser.add_argument(
        "--pooling",
        type=str,
        default="first_non_pad",
        choices=list(POOLING_CHOICES),
        help="Pooling strategy for dense embeddings.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding.")
    parser.add_argument("--block_size", type=int, default=15, help="Used for fixed/function strategies.")
    parser.add_argument("--window_size", type=int, default=20, help="Used for sliding strategy.")
    parser.add_argument("--slice_size", type=int, default=2, help="Used for sliding strategy.")
    parser.add_argument(
        "--ir_function_context_tokens",
        "--ir_context_tokens",
        dest="ir_context_tokens",
        type=int,
        default=256,
        help="Context tokens for IR function strategy.",
    )

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
    parser.add_argument("--llamaindex_max_chars", type=int, default=1500, help="CodeSplitter max chars")

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

    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA available")
    parser.add_argument(
        "--graph_index_file",
        type=str,
        default=None,
        help="Graph 索引文件路径（.pkl），如果提供将在 metadata 中添加 span_ids",
    )

    parser.add_argument("--lang_allowlist", type=str, default=".py",
                        help="Comma-separated file suffix allowlist, e.g. .py")
    parser.add_argument("--skip_patterns", type=str, default="",
                        help="Comma-separated glob patterns to skip")
    parser.add_argument("--max_file_size_mb", type=float, default=None,
                        help="Skip files larger than this size (MB)")

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
        args.summary_cache_dir = os.path.join(args.output_dir, "_summary_cache")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=False,
        device_map=None,
    ).to(device)
    model.eval()

    blocks, function_metadata, summary_metadata = _collect_blocks(args)
    if not blocks:
        raise RuntimeError(f"No blocks produced under {args.repo_path}")

    embeddings = embed_blocks_coderank(
        blocks=blocks,
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        ir_context_tokens=args.ir_context_tokens,
        pooling=args.pooling,
        show_progress=True,
        progress_desc="Embedding blocks",
    )

    span_ids_map = {}
    if args.graph_index_file:
        print(f"Extracting span_ids from graph index: {args.graph_index_file}")
        span_ids_map = extract_span_ids_from_graph(blocks, args.graph_index_file, args.repo_path)
        print(f"Extracted span_ids for {len(span_ids_map)}/{len(blocks)} blocks")

    output_dir = Path(args.output_dir)
    save_index(
        output_dir,
        embeddings,
        blocks,
        args.strategy,
        span_ids_map=span_ids_map,
        function_metadata=function_metadata if args.strategy in {"ir_function", "function_level"} else None,
        summary_metadata=summary_metadata if args.strategy == "summary" else None,
    )


if __name__ == "__main__":
    main()
