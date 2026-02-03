import logging
from pathlib import Path
from typing import List

from method.indexing.core.block import Block
from method.indexing.utils.file import iter_files
from method.indexing.chunker.basic import (
    blocks_fixed_lines,
    blocks_sliding,
    blocks_rl_fixed,
    blocks_rl_mini,
)
from method.indexing.chunker.function import (
    blocks_function_level_with_fallback,
    blocks_ir_function,
)


def collect_blocks(
    repo_path: str,
    strategy: str,
    block_size: int,
    window_size: int,
    slice_size: int,
) -> List[Block]:
    logger = logging.getLogger(__name__)
    repo_root = Path(repo_path)
    blocks: List[Block] = []
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
        rel = str(p.relative_to(repo_root))
        if strategy == "fixed":
            blocks.extend(blocks_fixed_lines(text, rel, block_size))
        elif strategy == "sliding":
            blocks.extend(blocks_sliding(text, rel, window_size, slice_size))
        elif strategy == "rl_fixed":
            blocks.extend(blocks_rl_fixed(text, rel))
        elif strategy == "rl_mini":
            blocks.extend(blocks_rl_mini(text, rel))
        elif strategy == "function_level":
            file_blocks, _ = blocks_function_level_with_fallback(text, rel, block_size)
            blocks.extend(file_blocks)
        elif strategy == "ir_function":
            file_blocks, _ = blocks_ir_function(text, rel)
            blocks.extend(file_blocks)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return blocks
