import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from method.indexing.core.block import Block, ChunkResult
from method.indexing.core.llm_factory import get_llm
from method.indexing.chunker.base import BaseChunker
from method.indexing.chunker.collect import collect_blocks
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


DEFAULT_SUMMARY_PROMPT = (
    "You are a senior code analyst. Summarize the functionality of the following code.\n"
    "Requirements:\n"
    "1. Start directly with the summary content. No \"This code does...\" or \"Here is...\".\n"
    "2. Focus on the business logic, inputs, and outputs.\n"
    "3. Use {language} (e.g., Chinese/English).\n\n"
    "Code:\n"
    "{context_str}\n\n"
    "Summary:\n"
)

_SUMMARY_LLM_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _load_summary_prompt(
    summary_prompt: Optional[str],
    summary_prompt_file: Optional[str],
    summary_language: str,
) -> str:
    prompt = summary_prompt
    if summary_prompt_file:
        try:
            with open(summary_prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()
        except Exception as e:
            logging.warning("Failed to read summary_prompt_file %s: %s", summary_prompt_file, e)

    if not prompt:
        prompt = DEFAULT_SUMMARY_PROMPT

    if "{language}" in prompt:
        prompt = prompt.replace("{language}", summary_language or "English")

    if "{context_str}" not in prompt:
        prompt = prompt.rstrip() + "\n\nCode:\n{context_str}\n\nSummary:\n"

    return prompt


def _compute_summary_cache_key(original_code: str, prompt_template: str) -> str:
    payload = (original_code + prompt_template).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _load_cached_summary(cache_dir: Optional[str], cache_key: str) -> Optional[str]:
    if not cache_dir:
        return None
    cache_path = Path(cache_dir) / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    except Exception as e:
        logging.warning("Failed to read summary cache %s: %s", cache_path, e)
    return None


def _save_cached_summary(cache_dir: Optional[str], cache_key: str, summary: str, prompt_id: str) -> None:
    if not cache_dir:
        return
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "prompt_id": prompt_id,
        "created_at": datetime.now().isoformat(),
    }
    target = cache_path / f"{cache_key}.json"
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=cache_path) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, target)
    except Exception as e:
        logging.warning("Failed to write summary cache %s: %s", target, e)


def _extract_summary_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    for key in ("summary", "section_summary", "doc_summary", "generated_summary"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key, value in metadata.items():
        if "summary" in key and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_summary_extractor(llm, prompt_template: str):
    from llama_index.core.extractors import SummaryExtractor
    kwargs = {
        "llm": llm,
        "summaries": ["self"],
        "show_progress": False,
    }

    try:
        return SummaryExtractor(prompt_template=prompt_template, **kwargs)
    except Exception:
        pass
    try:
        return SummaryExtractor(summary_template=prompt_template, **kwargs)
    except Exception:
        pass

    try:
        from llama_index.core.prompts import PromptTemplate
        prompt_obj = PromptTemplate(prompt_template)
    except Exception:
        prompt_obj = prompt_template

    try:
        return SummaryExtractor(prompt_template=prompt_obj, **kwargs)
    except Exception:
        return SummaryExtractor(summary_template=prompt_obj, **kwargs)


def _summarize_text(summary_extractor, text: str) -> Optional[str]:
    from llama_index.core.schema import TextNode

    node = TextNode(text=text)
    nodes = [node]
    metadata_list = None
    try:
        metadata_list = summary_extractor.extract(nodes)
    except Exception:
        metadata_list = None

    if isinstance(metadata_list, list) and metadata_list:
        meta = metadata_list[0]
        if isinstance(meta, dict):
            node.metadata.update(meta)
    else:
        try:
            from llama_index.core.ingestion import IngestionPipeline
            pipeline = IngestionPipeline(transformations=[summary_extractor])
            try:
                nodes = pipeline.run(nodes=nodes, show_progress=False)
            except TypeError:
                try:
                    nodes = pipeline.run(nodes=nodes)
                except TypeError:
                    try:
                        nodes = pipeline.run(documents=nodes, show_progress=False)
                    except TypeError:
                        nodes = pipeline.run(documents=nodes)
            if nodes:
                node = nodes[0]
        except Exception:
            return None

    return _extract_summary_from_metadata(node.metadata or {})


def collect_summary_blocks(
    repo_path: str,
    base_strategy: str = "ir_function",
    block_size: int = 15,
    window_size: int = 20,
    slice_size: int = 2,
    langchain_chunk_size: int = 1000,
    langchain_chunk_overlap: int = 200,
    llamaindex_language: str = "python",
    llamaindex_chunk_lines: int = 40,
    llamaindex_chunk_lines_overlap: int = 15,
    llamaindex_max_chars: int = 1500,
    llamaindex_chunk_size: int = 1024,
    llamaindex_chunk_overlap: int = 200,
    llamaindex_separator: str = " ",
    llamaindex_buffer_size: int = 3,
    llamaindex_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    summary_llm_provider: str = "openai",
    summary_llm_model: str = "gpt-3.5-turbo",
    summary_llm_api_key: Optional[str] = None,
    summary_llm_api_base: Optional[str] = None,
    summary_llm_max_new_tokens: int = 512,
    summary_llm_temperature: float = 0.1,
    summary_prompt: Optional[str] = None,
    summary_prompt_file: Optional[str] = None,
    summary_language: str = "English",
    summary_cache_dir: Optional[str] = None,
    summary_cache_policy: str = "read_write",
    summary_cache_miss: str = "empty",
    summary_store_pure_summary: bool = True,
    summary_store_pure_code: bool = False,
) -> Tuple[List[Block], Dict[int, Dict[str, Any]]]:
    logger = logging.getLogger(__name__)

    def _emit_summary_stats(message: str) -> None:
        if sys.stderr.isatty():
            try:
                from tqdm import tqdm
                tqdm.write(message)
                return
            except Exception:
                pass
        print(message, flush=True)

    if summary_cache_policy not in {"read_only", "read_write", "disabled"}:
        raise ValueError(f"Unknown summary_cache_policy: {summary_cache_policy}")
    if summary_cache_miss not in {"empty", "skip", "error"}:
        raise ValueError(f"Unknown summary_cache_miss: {summary_cache_miss}")

    function_metadata = {}
    if base_strategy == "ir_function":
        base_blocks, function_metadata = collect_ir_function_blocks(repo_path)
    elif base_strategy == "function_level":
        base_blocks, function_metadata = collect_function_blocks(repo_path, block_size)
    elif base_strategy == "epic":
        base_blocks = collect_epic_blocks(repo_path)
    elif base_strategy == "langchain_fixed":
        base_blocks = collect_langchain_fixed_blocks(
            repo_path, chunk_size=langchain_chunk_size, chunk_overlap=langchain_chunk_overlap
        )
    elif base_strategy == "langchain_recursive":
        base_blocks = collect_langchain_recursive_blocks(
            repo_path, chunk_size=langchain_chunk_size, chunk_overlap=langchain_chunk_overlap
        )
    elif base_strategy == "langchain_token":
        base_blocks = collect_langchain_token_blocks(
            repo_path, chunk_size=langchain_chunk_size, chunk_overlap=langchain_chunk_overlap
        )
    elif base_strategy == "llamaindex_code":
        base_blocks = collect_llamaindex_code_blocks(
            repo_path,
            language=llamaindex_language,
            chunk_lines=llamaindex_chunk_lines,
            chunk_lines_overlap=llamaindex_chunk_lines_overlap,
            max_chars=llamaindex_max_chars,
        )
    elif base_strategy == "llamaindex_sentence":
        base_blocks = collect_llamaindex_sentence_blocks(
            repo_path,
            chunk_size=llamaindex_chunk_size,
            chunk_overlap=llamaindex_chunk_overlap,
        )
    elif base_strategy == "llamaindex_token":
        base_blocks = collect_llamaindex_token_blocks(
            repo_path,
            chunk_size=llamaindex_chunk_size,
            chunk_overlap=llamaindex_chunk_overlap,
            separator=llamaindex_separator,
        )
    elif base_strategy == "llamaindex_semantic":
        base_blocks = collect_llamaindex_semantic_blocks(
            repo_path,
            buffer_size=llamaindex_buffer_size,
            model_name=llamaindex_embed_model,
        )
    elif base_strategy in {"fixed", "sliding", "rl_fixed", "rl_mini"}:
        base_blocks = collect_blocks(repo_path, base_strategy, block_size, window_size, slice_size)
    else:
        raise ValueError(f"Unknown summary_base_strategy: {base_strategy}")

    if not base_blocks:
        logger.warning("No base blocks collected for summary strategy in %s", repo_path)
        return [], {}

    total_blocks = len(base_blocks)
    cache_hits = 0
    cache_misses = 0
    llm_calls = 0
    llm_success = 0
    llm_empty = 0
    llm_fail = 0
    skipped_blocks = 0
    final_empty = 0
    final_non_empty = 0

    prompt_template = _load_summary_prompt(summary_prompt, summary_prompt_file, summary_language)
    prompt_id = hashlib.md5(prompt_template.encode("utf-8")).hexdigest()

    cache_key = (
        summary_llm_provider,
        summary_llm_model,
        summary_llm_api_base,
        summary_llm_api_key,
        summary_llm_max_new_tokens,
        summary_llm_temperature,
    )
    llm = _SUMMARY_LLM_CACHE.get(cache_key)
    if llm is None:
        try:
            llm = get_llm(
                provider=summary_llm_provider,
                model_name=summary_llm_model,
                api_key=summary_llm_api_key,
                api_base=summary_llm_api_base,
                max_new_tokens=summary_llm_max_new_tokens,
                temperature=summary_llm_temperature,
            )
        except Exception as e:
            logger.error("Failed to initialize summary LLM: %s", e)
            return [], {}
        _SUMMARY_LLM_CACHE[cache_key] = llm

    try:
        summary_extractor = _build_summary_extractor(llm, prompt_template)
    except Exception as e:
        logger.error("Failed to initialize SummaryExtractor: %s", e)
        return [], {}

    summary_blocks: List[Block] = []
    summary_metadata: Dict[int, Dict[str, Any]] = {}
    for base_block in base_blocks:
        original_code = base_block.content
        cache_id = _compute_summary_cache_key(original_code, prompt_template)
        summary_text = None

        if summary_cache_policy != "disabled":
            summary_text = _load_cached_summary(summary_cache_dir, cache_id)
            if summary_text is None:
                cache_misses += 1
            else:
                cache_hits += 1

        if summary_text is None and summary_cache_policy != "read_only":
            try:
                llm_calls += 1
                summary_text = _summarize_text(summary_extractor, original_code)
            except Exception as e:
                logger.warning(
                    "Summary generation failed for %s:%s-%s: %s",
                    base_block.file_path,
                    base_block.start,
                    base_block.end,
                    e,
                )
                llm_fail += 1
                summary_text = None
            else:
                if summary_text is None:
                    llm_fail += 1
                elif summary_text == "":
                    llm_empty += 1
                else:
                    llm_success += 1
            if summary_text and summary_cache_policy == "read_write":
                _save_cached_summary(summary_cache_dir, cache_id, summary_text, prompt_id)

        if summary_text is None:
            if summary_cache_miss == "error":
                raise RuntimeError(
                    f"Summary cache miss for {base_block.file_path}:{base_block.start}-{base_block.end}"
                )
            if summary_cache_miss == "skip":
                logger.warning(
                    "Summary cache miss, skipping block %s:%s-%s",
                    base_block.file_path,
                    base_block.start,
                    base_block.end,
                )
                skipped_blocks += 1
                continue
            summary_text = ""
        if summary_text:
            final_non_empty += 1
        else:
            final_empty += 1

        mixed_content = f"# Summary: {summary_text}\n\n{original_code}"
        block = Block(
            file_path=base_block.file_path,
            start=base_block.start,
            end=base_block.end,
            content=mixed_content,
            block_type="summary",
            context_text=base_block.context_text,
            function_text=base_block.function_text,
            summary_text=summary_text or None,
            original_content=original_code,
        )
        summary_blocks.append(block)
        idx = len(summary_blocks) - 1
        metadata = {
            "summary_model": summary_llm_model,
            "summary_base_strategy": base_strategy,
            "summary_base_block_type": base_block.block_type,
            "summary_prompt_id": prompt_id,
            "summary_length": len(summary_text or ""),
        }
        if summary_store_pure_summary:
            metadata["pure_summary"] = summary_text or ""
        if summary_store_pure_code:
            metadata["pure_code"] = original_code
        summary_metadata[idx] = metadata

    _emit_summary_stats(
        "Summary stats: "
        f"blocks={total_blocks} "
        f"cache_policy={summary_cache_policy} "
        f"cache_hit={cache_hits} cache_miss={cache_misses} "
        f"llm_calls={llm_calls} llm_success={llm_success} "
        f"llm_empty={llm_empty} llm_fail={llm_fail} "
        f"final_empty={final_empty} final_non_empty={final_non_empty} "
        f"skipped={skipped_blocks}"
    )

    return summary_blocks, summary_metadata


class SummaryChunker(BaseChunker):
    block_type = "summary"

    def __init__(
        self,
        base_strategy: str = "ir_function",
        block_size: int = 15,
        window_size: int = 20,
        slice_size: int = 2,
        langchain_chunk_size: int = 1000,
        langchain_chunk_overlap: int = 200,
        llamaindex_language: str = "python",
        llamaindex_chunk_lines: int = 40,
        llamaindex_chunk_lines_overlap: int = 15,
        llamaindex_max_chars: int = 1500,
        llamaindex_chunk_size: int = 1024,
        llamaindex_chunk_overlap: int = 200,
        llamaindex_separator: str = " ",
        llamaindex_buffer_size: int = 3,
        llamaindex_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        summary_llm_provider: str = "openai",
        summary_llm_model: str = "gpt-3.5-turbo",
        summary_llm_api_key: Optional[str] = None,
        summary_llm_api_base: Optional[str] = None,
        summary_llm_max_new_tokens: int = 512,
        summary_llm_temperature: float = 0.1,
        summary_prompt: Optional[str] = None,
        summary_prompt_file: Optional[str] = None,
        summary_language: str = "English",
        summary_cache_dir: Optional[str] = None,
        summary_cache_policy: str = "read_write",
        summary_cache_miss: str = "empty",
        summary_store_pure_summary: bool = True,
        summary_store_pure_code: bool = False,
    ):
        self.base_strategy = base_strategy
        self.block_size = block_size
        self.window_size = window_size
        self.slice_size = slice_size
        self.langchain_chunk_size = langchain_chunk_size
        self.langchain_chunk_overlap = langchain_chunk_overlap
        self.llamaindex_language = llamaindex_language
        self.llamaindex_chunk_lines = llamaindex_chunk_lines
        self.llamaindex_chunk_lines_overlap = llamaindex_chunk_lines_overlap
        self.llamaindex_max_chars = llamaindex_max_chars
        self.llamaindex_chunk_size = llamaindex_chunk_size
        self.llamaindex_chunk_overlap = llamaindex_chunk_overlap
        self.llamaindex_separator = llamaindex_separator
        self.llamaindex_buffer_size = llamaindex_buffer_size
        self.llamaindex_embed_model = llamaindex_embed_model
        self.summary_llm_provider = summary_llm_provider
        self.summary_llm_model = summary_llm_model
        self.summary_llm_api_key = summary_llm_api_key
        self.summary_llm_api_base = summary_llm_api_base
        self.summary_llm_max_new_tokens = summary_llm_max_new_tokens
        self.summary_llm_temperature = summary_llm_temperature
        self.summary_prompt = summary_prompt
        self.summary_prompt_file = summary_prompt_file
        self.summary_language = summary_language
        self.summary_cache_dir = summary_cache_dir
        self.summary_cache_policy = summary_cache_policy
        self.summary_cache_miss = summary_cache_miss
        self.summary_store_pure_summary = summary_store_pure_summary
        self.summary_store_pure_code = summary_store_pure_code

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks, metadata = collect_summary_blocks(
            repo_path=repo_path,
            base_strategy=self.base_strategy,
            block_size=self.block_size,
            window_size=self.window_size,
            slice_size=self.slice_size,
            langchain_chunk_size=self.langchain_chunk_size,
            langchain_chunk_overlap=self.langchain_chunk_overlap,
            llamaindex_language=self.llamaindex_language,
            llamaindex_chunk_lines=self.llamaindex_chunk_lines,
            llamaindex_chunk_lines_overlap=self.llamaindex_chunk_lines_overlap,
            llamaindex_max_chars=self.llamaindex_max_chars,
            llamaindex_chunk_size=self.llamaindex_chunk_size,
            llamaindex_chunk_overlap=self.llamaindex_chunk_overlap,
            llamaindex_separator=self.llamaindex_separator,
            llamaindex_buffer_size=self.llamaindex_buffer_size,
            llamaindex_embed_model=self.llamaindex_embed_model,
            summary_llm_provider=self.summary_llm_provider,
            summary_llm_model=self.summary_llm_model,
            summary_llm_api_key=self.summary_llm_api_key,
            summary_llm_api_base=self.summary_llm_api_base,
            summary_llm_max_new_tokens=self.summary_llm_max_new_tokens,
            summary_llm_temperature=self.summary_llm_temperature,
            summary_prompt=self.summary_prompt,
            summary_prompt_file=self.summary_prompt_file,
            summary_language=self.summary_language,
            summary_cache_dir=self.summary_cache_dir,
            summary_cache_policy=self.summary_cache_policy,
            summary_cache_miss=self.summary_cache_miss,
            summary_store_pure_summary=self.summary_store_pure_summary,
            summary_store_pure_code=self.summary_store_pure_code,
        )
        return ChunkResult(blocks=blocks, metadata=metadata, aux={})
