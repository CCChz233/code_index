import logging
import os
from typing import Dict, List, Optional

import torch

from method.indexing.core.block import Block, ChunkResult
from method.indexing.chunker.base import BaseChunker
from method.indexing.chunker.collect import collect_blocks
from method.indexing.chunker.llama_utils import (
    build_context_enhanced_content,
    _get_or_calculate_line_numbers,
)
from method.indexing.utils.file import get_file_scan_options

try:
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import (
        CodeSplitter,
        SentenceSplitter,
        TokenTextSplitter as LlamaIndexTokenTextSplitter,
        SemanticSplitterNodeParser,
    )
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False
    SimpleDirectoryReader = None
    CodeSplitter = None
    SentenceSplitter = None
    LlamaIndexTokenTextSplitter = None
    SemanticSplitterNodeParser = None

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    HAS_LLAMA_INDEX_HF = True
except ImportError:
    HAS_LLAMA_INDEX_HF = False
    HuggingFaceEmbedding = None


def collect_llamaindex_code_blocks(
    repo_path: str,
    language: str = "python",
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 15,
    max_chars: int = 1500,
) -> List[Block]:
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_code strategy.")
        return []

    required_exts, exclude_patterns, max_file_size_mb = _get_reader_scan_options()
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            filename_as_id=True,
            required_exts=required_exts,
            exclude=exclude_patterns,
            recursive=True,
        )
        docs = _filter_docs_by_size(reader.load_data(), max_file_size_mb, logger)
        if not docs:
            print(f'\n[DEBUG] {repo_path}: No documents loaded')
            logger.warning(f"No documents loaded from {repo_path} (after exclusions)")
            return []
        print(f'\n[DEBUG] {repo_path}: Loaded {len(docs)} documents')
        logger.info(f"Loaded {len(docs)} documents from {repo_path}")
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    try:
        splitter = CodeSplitter(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
        )

        nodes = splitter.get_nodes_from_documents(docs)
        print(f'\n[DEBUG] {repo_path}: CodeSplitter returned {len(nodes) if nodes else 0} nodes')
        if not nodes:
            print(f'\n[DEBUG] {repo_path}: No nodes generated from {len(docs)} documents')
            logger.warning(f"No nodes generated from {len(docs)} documents for {repo_path}")
            return []
        print(f'\n[DEBUG] {repo_path}: Generated {len(nodes)} nodes from {len(docs)} documents')
        logger.info(f"Generated {len(nodes)} nodes from {len(docs)} documents for {repo_path}")
    except Exception as e:
        print(f'\n[DEBUG] {repo_path}: CodeSplitter Exception: {type(e).__name__}: {e}')
        logger.error(f"LlamaIndex CodeSplitter failed for {repo_path}: {e}")
        logger.warning(f"Falling back to fixed chunking strategy for {repo_path}")
        try:
            blocks = collect_blocks(repo_path, "fixed", block_size=40, window_size=50, slice_size=10)
            if blocks:
                print(f'\n[DEBUG] {repo_path}: Fallback to fixed strategy generated {len(blocks)} blocks')
                logger.info(f"Fallback to fixed strategy generated {len(blocks)} blocks for {repo_path}")
                return blocks
        except Exception as fallback_error:
            logger.error(f"Fallback strategy also failed for {repo_path}: {fallback_error}")
        return []

    blocks: List[Block] = []
    file_to_docs: Dict[str, List] = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        original_content = file_contents.get(file_path, content_text)
        prev_end = previous_chunk_end_by_file.get(file_path, 0)
        start_line, end_line = _get_or_calculate_line_numbers(
            node, original_content, previous_chunk_end=prev_end
        )
        chunk_pos = original_content.find(content_text, prev_end)
        if chunk_pos >= 0:
            previous_chunk_end_by_file[file_path] = chunk_pos + len(content_text)
        else:
            previous_chunk_end_by_file[file_path] = max(prev_end, end_line * 80)

        rel_path = os.path.relpath(file_path, repo_path)
        content = build_context_enhanced_content(
            file_path=rel_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=rel_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="llamaindex_code",
            )
        )

    if not blocks:
        print(f'\n[DEBUG] {repo_path}: No blocks created from {len(nodes)} nodes')
        logger.warning(f"No blocks created from {len(nodes)} nodes for {repo_path}")
    else:
        print(f'\n[DEBUG] {repo_path}: Collected {len(blocks)} blocks from {len(nodes)} nodes')
        logger.info(f"Collected {len(blocks)} blocks using LlamaIndex CodeSplitter from {repo_path}")
    return blocks


def collect_llamaindex_sentence_blocks(
    repo_path: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> List[Block]:
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_sentence strategy.")
        return []

    required_exts, exclude_patterns, max_file_size_mb = _get_reader_scan_options()
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            filename_as_id=True,
            required_exts=required_exts,
            exclude=exclude_patterns,
            recursive=True,
        )
        docs = _filter_docs_by_size(reader.load_data(), max_file_size_mb, logger)
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    try:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        nodes = splitter.get_nodes_from_documents(docs)
    except Exception as e:
        logger.error(f"LlamaIndex SentenceSplitter failed for {repo_path}: {e}")
        return []

    blocks: List[Block] = []
    file_to_docs: Dict[str, List] = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        original_content = file_contents.get(file_path, content_text)
        prev_end = previous_chunk_end_by_file.get(file_path, 0)
        start_line, end_line = _get_or_calculate_line_numbers(
            node, original_content, previous_chunk_end=prev_end
        )
        chunk_pos = original_content.find(content_text, prev_end)
        if chunk_pos >= 0:
            previous_chunk_end_by_file[file_path] = chunk_pos + len(content_text)
        else:
            previous_chunk_end_by_file[file_path] = max(prev_end, end_line * 80)

        rel_path = os.path.relpath(file_path, repo_path)
        content = build_context_enhanced_content(
            file_path=rel_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=rel_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="llamaindex_sentence",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LlamaIndex SentenceSplitter from {repo_path}")
    return blocks


def collect_llamaindex_token_blocks(
    repo_path: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 20,
    separator: str = " ",
) -> List[Block]:
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_token strategy.")
        return []

    required_exts, exclude_patterns, max_file_size_mb = _get_reader_scan_options()
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            filename_as_id=True,
            required_exts=required_exts,
            exclude=exclude_patterns,
            recursive=True,
        )
        docs = _filter_docs_by_size(reader.load_data(), max_file_size_mb, logger)
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    try:
        splitter = LlamaIndexTokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
        )

        nodes = splitter.get_nodes_from_documents(docs)
    except Exception as e:
        logger.error(f"LlamaIndex TokenTextSplitter failed for {repo_path}: {e}")
        return []

    blocks: List[Block] = []
    file_to_docs: Dict[str, List] = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        original_content = file_contents.get(file_path, content_text)
        prev_end = previous_chunk_end_by_file.get(file_path, 0)
        start_line, end_line = _get_or_calculate_line_numbers(
            node, original_content, previous_chunk_end=prev_end
        )
        chunk_pos = original_content.find(content_text, prev_end)
        if chunk_pos >= 0:
            previous_chunk_end_by_file[file_path] = chunk_pos + len(content_text)
        else:
            previous_chunk_end_by_file[file_path] = max(prev_end, end_line * 80)

        rel_path = os.path.relpath(file_path, repo_path)
        content = build_context_enhanced_content(
            file_path=rel_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=rel_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="llamaindex_token",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LlamaIndex TokenTextSplitter from {repo_path}")
    return blocks


def collect_llamaindex_semantic_blocks(
    repo_path: str,
    buffer_size: int = 3,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Block]:
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_semantic strategy.")
        return []
    if not HAS_LLAMA_INDEX_HF:
        logger.error("HuggingFaceEmbedding is not available; install LlamaIndex HuggingFace embeddings to use llamaindex_semantic strategy.")
        return []

    required_exts, exclude_patterns, max_file_size_mb = _get_reader_scan_options()
    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            filename_as_id=True,
            required_exts=required_exts,
            exclude=exclude_patterns,
            recursive=True,
        )
        docs = _filter_docs_by_size(reader.load_data(), max_file_size_mb, logger)
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    try:
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            embed_model=embed_model,
        )

        nodes = splitter.get_nodes_from_documents(docs)
    except Exception as e:
        logger.error(f"LlamaIndex SemanticSplitter failed for {repo_path}: {e}")
        return []

    blocks: List[Block] = []
    file_to_docs: Dict[str, List] = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        original_content = file_contents.get(file_path, content_text)
        prev_end = previous_chunk_end_by_file.get(file_path, 0)
        start_line, end_line = _get_or_calculate_line_numbers(
            node, original_content, previous_chunk_end=prev_end
        )
        chunk_pos = original_content.find(content_text, prev_end)
        if chunk_pos >= 0:
            previous_chunk_end_by_file[file_path] = chunk_pos + len(content_text)
        else:
            previous_chunk_end_by_file[file_path] = max(prev_end, end_line * 80)

        rel_path = os.path.relpath(file_path, repo_path)
        content = build_context_enhanced_content(
            file_path=rel_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=rel_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="llamaindex_semantic",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LlamaIndex SemanticSplitter from {repo_path}")
    return blocks


class LlamaIndexCodeChunker(BaseChunker):
    block_type = "llamaindex_code"

    def __init__(self, language: str = "python", chunk_lines: int = 40, chunk_lines_overlap: int = 15, max_chars: int = 1500):
        self.language = language
        self.chunk_lines = chunk_lines
        self.chunk_lines_overlap = chunk_lines_overlap
        self.max_chars = max_chars

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks = collect_llamaindex_code_blocks(
            repo_path,
            language=self.language,
            chunk_lines=self.chunk_lines,
            chunk_lines_overlap=self.chunk_lines_overlap,
            max_chars=self.max_chars,
        )
        return ChunkResult(blocks=blocks, metadata={}, aux={})


class LlamaIndexSentenceChunker(BaseChunker):
    block_type = "llamaindex_sentence"

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks = collect_llamaindex_sentence_blocks(
            repo_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return ChunkResult(blocks=blocks, metadata={}, aux={})


class LlamaIndexTokenChunker(BaseChunker):
    block_type = "llamaindex_token"

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 20, separator: str = " "):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks = collect_llamaindex_token_blocks(
            repo_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator,
        )
        return ChunkResult(blocks=blocks, metadata={}, aux={})


class LlamaIndexSemanticChunker(BaseChunker):
    block_type = "llamaindex_semantic"

    def __init__(self, buffer_size: int = 3, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.buffer_size = buffer_size
        self.model_name = model_name

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks = collect_llamaindex_semantic_blocks(
            repo_path, buffer_size=self.buffer_size, model_name=self.model_name
        )
        return ChunkResult(blocks=blocks, metadata={}, aux={})


def _get_reader_scan_options() -> tuple[list[str], list[str], Optional[float]]:
    scan_options = get_file_scan_options()
    required_exts = scan_options.get("suffixes") or [".py"]
    exclude_patterns = scan_options.get("skip_patterns") or []
    max_file_size_mb = scan_options.get("max_file_size_mb")
    return list(required_exts), list(exclude_patterns), max_file_size_mb


def _filter_docs_by_size(docs: List, max_file_size_mb: Optional[float], logger: logging.Logger) -> List:
    if max_file_size_mb is None:
        return docs

    max_bytes = max_file_size_mb * 1024 * 1024
    filtered: List = []
    skipped = 0
    for doc in docs:
        file_path = doc.metadata.get("file_path", "") or doc.metadata.get("file_name", "")
        if not file_path:
            filtered.append(doc)
            continue
        try:
            if os.path.getsize(file_path) > max_bytes:
                skipped += 1
                continue
        except OSError:
            # If file stat fails, keep the document to avoid data loss.
            pass
        filtered.append(doc)

    if skipped:
        logger.info("Skipped %d documents over max_file_size_mb=%.3f", skipped, max_file_size_mb)
    return filtered
