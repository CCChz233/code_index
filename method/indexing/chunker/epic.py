import logging
from dataclasses import dataclass
from typing import List, Optional

from method.indexing.core.block import Block, ChunkResult
from method.indexing.chunker.base import BaseChunker
from method.indexing.chunker.collect import collect_blocks
from method.indexing.chunker.llama_utils import build_context_enhanced_content

try:
    from llama_index.core import SimpleDirectoryReader
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False
    SimpleDirectoryReader = None


@dataclass
class EpicSplitterConfig:
    """EpicSplitter 配置，尽量与 BM25 保持一致。"""
    min_chunk_size: int = 512
    chunk_size: int = 1024
    max_chunk_size: int = 2048
    hard_token_limit: int = 4096
    max_chunks: int = 512

    @classmethod
    def from_bm25_config(cls):
        try:
            return cls()
        except Exception:
            return cls()


def collect_epic_blocks(repo_path: str, config: Optional[EpicSplitterConfig] = None) -> List[Block]:
    """
    使用与 BM25 一致的 EpicSplitter 构建块，并做上下文增强。
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use epic strategy.")
        return []

    try:
        from repo_index.index.epic_split import EpicSplitter
    except ImportError as e:
        logger.error(f"Failed to import EpicSplitter: {e}. Please install required dependencies.")
        return []

    if config is None:
        config = EpicSplitterConfig.from_bm25_config()

    logger.info(f"Using EpicSplitter config: {config}")

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
        splitter = EpicSplitter(
            min_chunk_size=config.min_chunk_size,
            chunk_size=config.chunk_size,
            max_chunk_size=config.max_chunk_size,
            hard_token_limit=config.hard_token_limit,
            max_chunks=config.max_chunks,
            repo_path=repo_path,
        )
        nodes = splitter.get_nodes_from_documents(docs, show_progress=False)
    except Exception as e:
        logger.error(f"EpicSplitter failed for {repo_path}: {e}")
        logger.warning("Falling back to rl_mini chunking")
        return collect_blocks(repo_path, "rl_mini", block_size=15, window_size=20, slice_size=2)

    blocks: List[Block] = []
    for i, node in enumerate(nodes):
        try:
            meta = getattr(node, "metadata", None) or {}
            file_path = meta.get("file_path", "")
            if not file_path:
                logger.warning(f"Node {i} missing file_path, skipping")
                continue

            content_text = node.get_content()
            if not content_text:
                continue

            start_line = meta.get("start_line", 0)
            end_line = meta.get("end_line", max(0, start_line + content_text.count("\n")))

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
                    block_type="epic",
                )
            )
        except Exception as e:
            logger.warning(f"Error processing node {i}: {e}")
            continue

    logger.info(f"Collected {len(blocks)} epic blocks from {repo_path}")
    return blocks


class EpicChunker(BaseChunker):
    block_type = "epic"

    def __init__(self, config: Optional[EpicSplitterConfig] = None):
        self.config = config

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks = collect_epic_blocks(repo_path, self.config)
        return ChunkResult(blocks=blocks, metadata={}, aux={})
