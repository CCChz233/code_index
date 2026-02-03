import logging
from typing import List

from method.indexing.core.block import Block, ChunkResult
from method.indexing.chunker.base import BaseChunker
from method.indexing.chunker.llama_utils import (
    build_context_enhanced_content,
    _calculate_chunk_line_numbers,
)

try:
    from llama_index.core import SimpleDirectoryReader
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False
    SimpleDirectoryReader = None

try:
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
    )
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    CharacterTextSplitter = None
    RecursiveCharacterTextSplitter = None
    TokenTextSplitter = None


def collect_langchain_recursive_blocks(
    repo_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Block]:
    """
    使用 LangChain RecursiveCharacterTextSplitter 收集代码块（简化版）
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use langchain strategies.")
        return []
    if not HAS_LANGCHAIN:
        logger.error("LangChain text splitters are not available; install langchain-text-splitters to use langchain strategies.")
        return []

    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        for doc in docs:
            original_content = doc.get_content()
            chunks = splitter.split_text(original_content)

            previous_chunk_end = 0
            for chunk in chunks:
                start_line, end_line = _calculate_chunk_line_numbers(
                    original_content, chunk, previous_chunk_end
                )
                chunk_pos = original_content.find(chunk, previous_chunk_end)
                if chunk_pos >= 0:
                    previous_chunk_end = chunk_pos + len(chunk)
                else:
                    previous_chunk_end = end_line * 80

                all_chunks.append({
                    'file_path': doc.metadata.get('file_path', ''),
                    'content': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                })
    except Exception as e:
        logger.error(f"LangChain RecursiveCharacterTextSplitter failed for {repo_path}: {e}")
        return []

    blocks: List[Block] = []
    for chunk_info in all_chunks:
        file_path = chunk_info['file_path']
        content_text = chunk_info['content']
        start_line = chunk_info['start_line']
        end_line = chunk_info['end_line']

        if not file_path:
            continue

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="langchain_recursive",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LangChain RecursiveCharacterTextSplitter from {repo_path}")
    return blocks


def collect_langchain_fixed_blocks(
    repo_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Block]:
    """
    使用 LangChain CharacterTextSplitter 收集代码块（固定字符数分割）
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use langchain strategies.")
        return []
    if not HAS_LANGCHAIN:
        logger.error("LangChain text splitters are not available; install langchain-text-splitters to use langchain strategies.")
        return []

    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    try:
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        for doc in docs:
            original_content = doc.get_content()
            chunks = splitter.split_text(original_content)

            previous_chunk_end = 0
            for chunk in chunks:
                start_line, end_line = _calculate_chunk_line_numbers(
                    original_content, chunk, previous_chunk_end
                )
                chunk_pos = original_content.find(chunk, previous_chunk_end)
                if chunk_pos >= 0:
                    previous_chunk_end = chunk_pos + len(chunk)
                else:
                    previous_chunk_end = end_line * 80

                all_chunks.append({
                    'file_path': doc.metadata.get('file_path', ''),
                    'content': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                })
    except Exception as e:
        logger.error(f"LangChain CharacterTextSplitter failed for {repo_path}: {e}")
        return []

    blocks: List[Block] = []
    for chunk_info in all_chunks:
        file_path = chunk_info['file_path']
        content_text = chunk_info['content']
        start_line = chunk_info['start_line']
        end_line = chunk_info['end_line']

        if not file_path:
            continue

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="langchain_fixed",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LangChain CharacterTextSplitter from {repo_path}")
    return blocks


def collect_langchain_token_blocks(
    repo_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Block]:
    """
    使用 LangChain TokenTextSplitter 收集代码块（按 token 数分割）
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use langchain strategies.")
        return []
    if not HAS_LANGCHAIN:
        logger.error("LangChain text splitters are not available; install langchain-text-splitters to use langchain strategies.")
        return []

    try:
        reader = SimpleDirectoryReader(
            input_dir=repo_path,
            exclude=[
                "**/test/**",
                "**/tests/**",
                "**/test_*.py",
                "**/*_test.py",
            ],
            filename_as_id=True,
            required_exts=[".py"],
            recursive=True,
        )
        docs = reader.load_data()
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    try:
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            disallowed_special=(),
        )

        all_chunks = []
        for doc in docs:
            original_content = doc.get_content()
            chunks = splitter.split_text(original_content)

            previous_chunk_end = 0
            for chunk in chunks:
                start_line, end_line = _calculate_chunk_line_numbers(
                    original_content, chunk, previous_chunk_end
                )
                chunk_pos = original_content.find(chunk, previous_chunk_end)
                if chunk_pos >= 0:
                    previous_chunk_end = chunk_pos + len(chunk)
                else:
                    previous_chunk_end = end_line * 80

                all_chunks.append({
                    'file_path': doc.metadata.get('file_path', ''),
                    'content': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                })
    except Exception as e:
        logger.error(f"LangChain TokenTextSplitter failed for {repo_path}: {e}")
        return []

    blocks: List[Block] = []
    for chunk_info in all_chunks:
        file_path = chunk_info['file_path']
        content_text = chunk_info['content']
        start_line = chunk_info['start_line']
        end_line = chunk_info['end_line']

        if not file_path:
            continue

        content = build_context_enhanced_content(
            file_path=file_path,
            original_content=content_text,
            start_line=start_line,
        )

        blocks.append(
            Block(
                file_path=file_path,
                start=start_line,
                end=end_line,
                content=content,
                block_type="langchain_token",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LangChain TokenTextSplitter from {repo_path}")
    return blocks


class LangChainRecursiveChunker(BaseChunker):
    block_type = "langchain_recursive"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks = collect_langchain_recursive_blocks(
            repo_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return ChunkResult(blocks=blocks, metadata={}, aux={})


class LangChainFixedChunker(BaseChunker):
    block_type = "langchain_fixed"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks = collect_langchain_fixed_blocks(
            repo_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return ChunkResult(blocks=blocks, metadata={}, aux={})


class LangChainTokenChunker(BaseChunker):
    block_type = "langchain_token"

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks = collect_langchain_token_blocks(
            repo_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return ChunkResult(blocks=blocks, metadata={}, aux={})
