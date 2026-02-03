import logging
from pathlib import Path
from typing import Iterable, List, Tuple

from method.indexing.core.block import Block, ChunkResult
from method.indexing.chunker.base import BaseChunker
from method.indexing.utils.file import iter_files


def blocks_fixed_lines(text: str, rel: str, block_size: int) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []
    start = 0
    while start < len(lines):
        end = min(start + block_size, len(lines))
        chunk = lines[start:end]
        if any(line.strip() for line in chunk):
            blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "fixed"))
        start = end
    return blocks


def blocks_sliding(text: str, rel: str, window_size: int, slice_size: int) -> List[Block]:
    lines = text.splitlines()
    delta = window_size // 2
    step = max(1, window_size // slice_size) if slice_size > 0 else window_size
    blocks: List[Block] = []
    for line_no in range(0, len(lines), step):
        start = max(0, line_no - delta)
        end = min(len(lines), line_no + window_size - delta)
        chunk = lines[start:end]
        if any(line.strip() for line in chunk):
            blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "sliding"))
    return blocks


def blocks_rl_fixed(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    lines = text.splitlines()
    non_empty = [(idx, line) for idx, line in enumerate(lines) if line.strip()]
    blocks: List[Block] = []
    for i in range(0, min(len(non_empty), max_lines), 12):
        chunk = non_empty[i:i + 12]
        if not chunk:
            break
        start = chunk[0][0]
        end = chunk[-1][0]
        content = "\n".join(line for _, line in chunk)
        blocks.append(Block(rel, start, end, content, "rl_fixed"))
    return blocks


def blocks_rl_mini(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    mini_blocks = []
    cur = []
    for idx, line in enumerate(text.splitlines()):
        if line.strip():
            cur.append((idx, line))
        else:
            if cur:
                mini_blocks.append(cur)
                cur = []
    if cur:
        mini_blocks.append(cur)

    temp = []
    for mb in mini_blocks:
        if len(mb) > 15:
            for idx in range(0, len(mb), 15):
                temp.append(mb[idx: idx + 15])
        else:
            temp.append(mb)
    mini_blocks = temp

    blocks: List[Block] = []
    current = []
    total = 0
    for block in mini_blocks:
        if total >= max_lines:
            break
        if len(current) + len(block) <= 15:
            current.extend(block)
            total += len(block)
        else:
            if current:
                start = current[0][0]
                end = current[-1][0]
                content = "\n".join(line for _, line in current)
                blocks.append(Block(rel, start, end, content, "rl_mini"))
            current = list(block)
            total += len(block)
    if current:
        start = current[0][0]
        end = current[-1][0]
        content = "\n".join(line for _, line in current)
        blocks.append(Block(rel, start, end, content, "rl_mini"))
    return blocks


def _iter_repo_texts(repo_root: Path, logger: logging.Logger) -> Iterable[Tuple[Path, str]]:
    for p in iter_files(repo_root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Error reading {p}: {e}")
                continue
        except Exception as e:
            logger.warning(f"Error reading {p}: {e}")
            continue
        yield p, text


class FixedChunker(BaseChunker):
    block_type = "fixed"

    def __init__(self, block_size: int = 15):
        self.block_size = block_size
        self._logger = logging.getLogger(__name__)

    def chunk(self, repo_path: str) -> ChunkResult:
        repo_root = Path(repo_path)
        blocks: List[Block] = []
        for p, text in _iter_repo_texts(repo_root, self._logger):
            rel = str(p.relative_to(repo_root))
            blocks.extend(blocks_fixed_lines(text, rel, self.block_size))
        return ChunkResult(blocks=blocks, metadata={}, aux={})


class SlidingChunker(BaseChunker):
    block_type = "sliding"

    def __init__(self, window_size: int = 20, slice_size: int = 2):
        self.window_size = window_size
        self.slice_size = slice_size
        self._logger = logging.getLogger(__name__)

    def chunk(self, repo_path: str) -> ChunkResult:
        repo_root = Path(repo_path)
        blocks: List[Block] = []
        for p, text in _iter_repo_texts(repo_root, self._logger):
            rel = str(p.relative_to(repo_root))
            blocks.extend(blocks_sliding(text, rel, self.window_size, self.slice_size))
        return ChunkResult(blocks=blocks, metadata={}, aux={})


class RLFixedChunker(BaseChunker):
    block_type = "rl_fixed"

    def __init__(self, max_lines: int = 5000):
        self.max_lines = max_lines
        self._logger = logging.getLogger(__name__)

    def chunk(self, repo_path: str) -> ChunkResult:
        repo_root = Path(repo_path)
        blocks: List[Block] = []
        for p, text in _iter_repo_texts(repo_root, self._logger):
            rel = str(p.relative_to(repo_root))
            blocks.extend(blocks_rl_fixed(text, rel, self.max_lines))
        return ChunkResult(blocks=blocks, metadata={}, aux={})


class RLMiniChunker(BaseChunker):
    block_type = "rl_mini"

    def __init__(self, max_lines: int = 5000):
        self.max_lines = max_lines
        self._logger = logging.getLogger(__name__)

    def chunk(self, repo_path: str) -> ChunkResult:
        repo_root = Path(repo_path)
        blocks: List[Block] = []
        for p, text in _iter_repo_texts(repo_root, self._logger):
            rel = str(p.relative_to(repo_root))
            blocks.extend(blocks_rl_mini(text, rel, self.max_lines))
        return ChunkResult(blocks=blocks, metadata={}, aux={})
