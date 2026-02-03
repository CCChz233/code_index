from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from method.indexing.core.block import Block
from method.indexing.indexer import save_index
from method.indexing.chunker.epic import collect_epic_blocks
from method.indexing.chunker.function import collect_function_blocks, collect_ir_function_blocks
from method.indexing.chunker.langchain import (
    collect_langchain_fixed_blocks,
    collect_langchain_recursive_blocks,
    collect_langchain_token_blocks,
)
from method.indexing.chunker.llamaindex import (
    collect_llamaindex_code_blocks,
    collect_llamaindex_sentence_blocks,
    collect_llamaindex_token_blocks,
    collect_llamaindex_semantic_blocks,
)
from method.indexing.chunker.summary import collect_summary_blocks


@dataclass
class BuildArtifacts:
    blocks: List[Block]
    function_metadata: Dict[int, Dict[str, Optional[str]]]
    summary_metadata: Optional[Dict[int, Dict[str, Any]]]


class IndexBuilder:
    def __init__(self, args, use_direct_llamaindex_params: bool = True):
        self._args = args
        self._use_direct_llamaindex_params = use_direct_llamaindex_params

    def collect_blocks(self, repo_path: str) -> BuildArtifacts:
        args = self._args
        summary_metadata = None
        function_metadata: Dict[int, Dict[str, Optional[str]]] = {}

        if args.strategy == "ir_function":
            blocks, function_metadata = collect_ir_function_blocks(repo_path)
        elif args.strategy == "function_level":
            blocks, function_metadata = collect_function_blocks(repo_path, args.block_size)
        elif args.strategy == "summary":
            blocks, summary_metadata = collect_summary_blocks(
                repo_path,
                base_strategy=args.summary_base_strategy,
                block_size=args.block_size,
                window_size=args.window_size,
                slice_size=args.slice_size,
                langchain_chunk_size=args.langchain_chunk_size,
                langchain_chunk_overlap=args.langchain_chunk_overlap,
                llamaindex_language=args.llamaindex_language,
                llamaindex_chunk_lines=args.llamaindex_chunk_lines,
                llamaindex_chunk_lines_overlap=args.llamaindex_chunk_lines_overlap,
                llamaindex_max_chars=args.llamaindex_max_chars,
                llamaindex_chunk_size=args.llamaindex_chunk_size,
                llamaindex_chunk_overlap=args.llamaindex_chunk_overlap,
                llamaindex_separator=args.llamaindex_separator,
                llamaindex_buffer_size=args.llamaindex_buffer_size,
                llamaindex_embed_model=args.llamaindex_embed_model,
                summary_llm_provider=args.summary_llm_provider,
                summary_llm_model=args.summary_llm_model,
                summary_llm_api_key=args.summary_llm_api_key,
                summary_llm_api_base=args.summary_llm_api_base,
                summary_llm_max_new_tokens=args.summary_llm_max_new_tokens,
                summary_llm_temperature=args.summary_llm_temperature,
                summary_prompt=args.summary_prompt,
                summary_prompt_file=args.summary_prompt_file,
                summary_language=args.summary_language,
                summary_cache_dir=args.summary_cache_dir,
                summary_cache_policy=args.summary_cache_policy,
                summary_cache_miss=args.summary_cache_miss,
                summary_store_pure_summary=args.summary_store_pure_summary,
                summary_store_pure_code=args.summary_store_pure_code,
            )
        elif args.strategy == "llamaindex_code":
            if self._use_direct_llamaindex_params:
                blocks = collect_llamaindex_code_blocks(
                    repo_path,
                    language=args.llamaindex_language,
                    chunk_lines=args.llamaindex_chunk_lines,
                    chunk_lines_overlap=args.llamaindex_chunk_lines_overlap,
                    max_chars=args.llamaindex_max_chars,
                )
            else:
                blocks = collect_llamaindex_code_blocks(repo_path)
        elif args.strategy == "llamaindex_sentence":
            if self._use_direct_llamaindex_params:
                blocks = collect_llamaindex_sentence_blocks(
                    repo_path,
                    chunk_size=args.llamaindex_chunk_size,
                    chunk_overlap=args.llamaindex_chunk_overlap,
                )
            else:
                blocks = collect_llamaindex_sentence_blocks(repo_path)
        elif args.strategy == "llamaindex_token":
            if self._use_direct_llamaindex_params:
                blocks = collect_llamaindex_token_blocks(
                    repo_path,
                    chunk_size=args.llamaindex_chunk_size,
                    chunk_overlap=args.llamaindex_chunk_overlap,
                    separator=args.llamaindex_separator,
                )
            else:
                blocks = collect_llamaindex_token_blocks(repo_path)
        elif args.strategy == "llamaindex_semantic":
            if self._use_direct_llamaindex_params:
                blocks = collect_llamaindex_semantic_blocks(
                    repo_path,
                    buffer_size=args.llamaindex_buffer_size,
                    model_name=args.llamaindex_embed_model,
                )
            else:
                blocks = collect_llamaindex_semantic_blocks(repo_path)
        elif args.strategy == "langchain_fixed":
            blocks = collect_langchain_fixed_blocks(repo_path)
        elif args.strategy == "langchain_recursive":
            blocks = collect_langchain_recursive_blocks(repo_path)
        elif args.strategy == "langchain_token":
            blocks = collect_langchain_token_blocks(repo_path)
        elif args.strategy == "epic":
            blocks = collect_epic_blocks(repo_path)
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        return BuildArtifacts(
            blocks=blocks,
            function_metadata=function_metadata,
            summary_metadata=summary_metadata,
        )

    def build_index(
        self,
        repo_path: str,
        repo_name: str,
        *,
        model,
        tokenizer,
        device,
        embed_fn,
        embed_kwargs: Optional[Dict[str, Any]] = None,
        index_dir: str,
        artifacts: Optional[BuildArtifacts] = None,
    ) -> int:
        if artifacts is None:
            artifacts = self.collect_blocks(repo_path)
        blocks = artifacts.blocks
        if not blocks:
            return 0

        embed_kwargs = embed_kwargs or {}
        embeddings = embed_fn(
            blocks=blocks,
            model=model,
            tokenizer=tokenizer,
            device=device,
            **embed_kwargs,
        )

        output_dir = Path(index_dir) / f"dense_index_{self._args.strategy}" / repo_name
        output_dir.mkdir(parents=True, exist_ok=True)

        save_index(
            output_dir,
            embeddings,
            blocks,
            self._args.strategy,
            function_metadata=artifacts.function_metadata if self._args.strategy in {"ir_function", "function_level"} else None,
            summary_metadata=artifacts.summary_metadata if self._args.strategy == "summary" else None,
        )

        return len(blocks)
