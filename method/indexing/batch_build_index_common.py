#!/usr/bin/env python
"""
共享的索引构建代码（不包含模型相关逻辑）

包含：
- Block 类定义
- 所有切块策略函数
- BlockDataset 类
- 工具函数（save_index, iter_files 等）
"""

import ast
import hashlib
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
from datetime import datetime

import torch
import logging

from method.indexing.core.dataset import BlockDataset as _CoreBlockDataset
from method.indexing.utils.file import iter_files as _iter_files
from method.core.contract import Block as CoreBlock
from method.indexing.core.index_writer import save_index as _save_index
from method.indexing.chunker.basic import (
    blocks_fixed_lines as _blocks_fixed_lines,
    blocks_sliding as _blocks_sliding,
    blocks_rl_fixed as _blocks_rl_fixed,
    blocks_rl_mini as _blocks_rl_mini,
)
from method.indexing.chunker.function import (
    blocks_ir_function as _blocks_ir_function,
    collect_ir_function_blocks as _collect_ir_function_blocks,
    build_function_context_content as _build_function_context_content,
    blocks_function_level_with_fallback as _blocks_function_level_with_fallback,
    blocks_by_function as _blocks_by_function,
    collect_function_blocks as _collect_function_blocks,
)
from method.indexing.chunker.collect import collect_blocks as _collect_blocks
from method.indexing.chunker.llama_utils import (
    build_context_enhanced_content as _build_context_enhanced_content,
    _calculate_chunk_line_numbers as _calculate_chunk_line_numbers,
    _get_or_calculate_line_numbers as _get_or_calculate_line_numbers,
)
from method.indexing.chunker.epic import (
    EpicSplitterConfig as _EpicSplitterConfig,
    collect_epic_blocks as _collect_epic_blocks,
)
from method.indexing.chunker.langchain import (
    collect_langchain_recursive_blocks as _collect_langchain_recursive_blocks,
    collect_langchain_fixed_blocks as _collect_langchain_fixed_blocks,
    collect_langchain_token_blocks as _collect_langchain_token_blocks,
)
from method.indexing.chunker.llamaindex import (
    collect_llamaindex_code_blocks as _collect_llamaindex_code_blocks,
    collect_llamaindex_sentence_blocks as _collect_llamaindex_sentence_blocks,
    collect_llamaindex_token_blocks as _collect_llamaindex_token_blocks,
    collect_llamaindex_semantic_blocks as _collect_llamaindex_semantic_blocks,
)
from method.indexing.chunker.summary import (
    DEFAULT_SUMMARY_PROMPT as _SUMMARY_DEFAULT_PROMPT,
    _load_summary_prompt as _summary_load_summary_prompt,
    _compute_summary_cache_key as _summary_compute_summary_cache_key,
    _load_cached_summary as _summary_load_cached_summary,
    _save_cached_summary as _summary_save_cached_summary,
    _extract_summary_from_metadata as _summary_extract_summary_from_metadata,
    _build_summary_extractor as _summary_build_summary_extractor,
    _summarize_text as _summary_summarize_text,
    collect_summary_blocks as _collect_summary_blocks,
)

_project_root = Path(__file__).resolve().parents[2]

# ============================================================================
# Block 类和配置
# ============================================================================


Block = CoreBlock


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
        """
        尝试从 BM25 配置读取参数；若不可用则使用本地默认值。
        """
        try:
            # 目前 BM25 构建未暴露默认常量，这里先使用本地默认值。
            # 若后续在 bm25_retriever 中新增常量，可在此导入以保持完全一致。
            return cls()
        except Exception:
            return cls()


# ============================================================================
# 基础切块策略
# ============================================================================


def iter_files(repo_root: Path) -> List[Path]:
    return _iter_files(repo_root)


def blocks_fixed_lines(text: str, rel: str, block_size: int) -> List[Block]:
    return _blocks_fixed_lines(text, rel, block_size)


def blocks_sliding(text: str, rel: str, window_size: int, slice_size: int) -> List[Block]:
    return _blocks_sliding(text, rel, window_size, slice_size)


def blocks_rl_fixed(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    return _blocks_rl_fixed(text, rel, max_lines)


def blocks_rl_mini(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    return _blocks_rl_mini(text, rel, max_lines)


# ============================================================================
# IR-based 函数级分块（论文复现）
# ============================================================================


def _extract_module_level_code(tree: ast.AST, lines: List[str]) -> str:
    """
    提取模块级代码：imports, 全局变量, 顶层赋值等

    按照论文：这些代码会被冗余复制到该文件每个函数的表示中

    Args:
        tree: AST 树
        lines: 源代码行列表

    Returns:
        模块级代码字符串（多行合并为单行，空格分隔）
    """
    module_level_parts = []

    for node in tree.body:
        # import 语句
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])  # 合并为单行
            module_level_parts.append(code.strip())

        # 全局变量赋值
        elif isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            module_level_parts.append(code.strip())

        # 带类型注解的赋值 (e.g., DEBUG: bool = True)
        elif isinstance(node, ast.AnnAssign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            module_level_parts.append(code.strip())

    return " ".join(module_level_parts)


def _extract_class_attributes(class_node: ast.ClassDef, lines: List[str]) -> str:
    """
    提取类属性（非方法的类级别定义）

    按照论文：类属性会被冗余复制到该类每个方法的表示中

    Args:
        class_node: 类的 AST 节点
        lines: 源代码行列表

    Returns:
        类属性代码字符串（多行合并为单行，空格分隔）
    """
    class_attr_parts = []

    for node in class_node.body:
        # 类属性赋值
        if isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            class_attr_parts.append(code.strip())

        # 带类型注解的类属性
        elif isinstance(node, ast.AnnAssign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            class_attr_parts.append(code.strip())

    return " ".join(class_attr_parts)


def _build_ir_function_representation(
    file_path: str,
    class_name: Optional[str],
    function_code: str,
    module_level_code: str,
    class_attributes: str,
) -> str:
    """
    按照论文 IR-based 方法构建函数表示

    论文原文 (Appendix C.1.1):
    "The function's context—its containing file and class—is appended to
    the function representation before embedding"

    格式: {file_path} {class_name} {module_level_code} {class_attributes} {function_code}

    Args:
        file_path: 文件路径
        class_name: 类名（如果是方法）
        function_code: 函数源代码
        module_level_code: 模块级代码（imports, 全局变量等）
        class_attributes: 类属性代码（如果是方法）

    Returns:
        完整的函数表示，用于 embedding
    """
    context_text = _build_ir_function_context(
        file_path=file_path,
        class_name=class_name,
        module_level_code=module_level_code,
        class_attributes=class_attributes,
    )
    if context_text.strip():
        return f"{context_text} {function_code}"
    return function_code


def _build_ir_function_context(
    file_path: str,
    class_name: Optional[str],
    module_level_code: str,
    class_attributes: str,
) -> str:
    parts = [file_path]

    if class_name:
        parts.append(class_name)

    # 添加模块级上下文（冗余复制到每个函数）
    if module_level_code.strip():
        parts.append(module_level_code.strip())

    # 添加类属性（如果是类方法）
    if class_attributes.strip():
        parts.append(class_attributes.strip())

    return " ".join(parts)


def blocks_ir_function(text: str, rel: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    return _blocks_ir_function(text, rel)


def collect_ir_function_blocks(repo_path: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    return _collect_ir_function_blocks(repo_path)


def build_function_context_content(
    file_path: str,
    class_name: Optional[str],
    function_name: str,
    qualified_name: str,
    original_code: str,
    start_line: int,
    end_line: int,
) -> str:
    return _build_function_context_content(
        file_path=file_path,
        class_name=class_name,
        function_name=function_name,
        qualified_name=qualified_name,
        original_code=original_code,
        start_line=start_line,
        end_line=end_line,
    )


def blocks_function_level_with_fallback(
    text: str,
    rel: str,
    block_size: int = 15,
) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    return _blocks_function_level_with_fallback(text, rel, block_size)


def blocks_by_function(text: str, rel: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    return _blocks_by_function(text, rel)


def collect_function_blocks(repo_path: str, block_size: int = 15) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    return _collect_function_blocks(repo_path, block_size)


class BlockDataset(_CoreBlockDataset):
    pass


def collect_blocks(repo_path: str, strategy: str, block_size: int, window_size: int, slice_size: int) -> List[Block]:
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
            # 使用带 fallback 的函数级切块，确保完整覆盖
            file_blocks, _ = blocks_function_level_with_fallback(text, rel, block_size)
            blocks.extend(file_blocks)
        elif strategy == "ir_function":
            # 论文 IR-based 方法：只索引函数，上下文冗余附加
            file_blocks, _ = blocks_ir_function(text, rel)
            blocks.extend(file_blocks)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return blocks


def build_context_enhanced_content(
    file_path: str,
    original_content: str,
    start_line: int = 0,
) -> str:
    """
    为代码块添加上下文信息：
    - 文件路径
    - 尝试性的类名（简易启发式）
    - 行号范围
    - 原始代码
    """
    context_parts = [f"File: {file_path}"]

    # 尝试从前若干行中提取类名（仅作辅助上下文，不做强依赖）
    try:
        import re

        header = "\n".join(original_content.splitlines()[:50])
        class_match = re.search(r"class\s+(\w+)", header)
        if class_match:
            context_parts.append(f"Class: {class_match.group(1)}")
    except Exception:
        # 不因上下文增强失败而中断
        pass

    if start_line > 0:
        context_parts.append(f"Lines: {start_line}-")

    context_parts.append("")  # 空行分隔
    context_parts.append("Code:")
    context_parts.append(original_content)

    return "\n".join(context_parts)


def collect_epic_blocks(repo_path: str, config: Optional[EpicSplitterConfig] = None) -> List[Block]:
    """
    使用与 BM25 一致的 EpicSplitter 构建块，并做上下文增强。
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use epic strategy.")
        return []

    # 延迟导入 EpicSplitter，避免不必要的依赖
    try:
        from repo_index.index.epic_split import EpicSplitter
    except ImportError as e:
        logger.error(f"Failed to import EpicSplitter: {e}. Please install required dependencies.")
        return []

    if config is None:
        config = EpicSplitterConfig.from_bm25_config()

    logger.info(f"Using EpicSplitter config: {config}")

    # 1. 读取仓库为文档
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

    # 2. 使用 EpicSplitter 切块
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
        # Fallback：退回到 rl_mini 行块切分
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

            start_line = meta.get("start_line")
            end_line = meta.get("end_line")
            content_text = node.get_content()

            # 行号缺失时的推断
            if start_line is None or end_line is None:
                lines = content_text.splitlines()
                start_line = 0
                end_line = len(lines)
                logger.debug(
                    f"Block {i} missing line numbers, inferred {start_line}-{end_line}"
                )

            if start_line < 0 or end_line < start_line:
                logger.warning(
                    f"Invalid line numbers for block {i}: {start_line}-{end_line}, skipping"
                )
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
                    block_type="epic",
                )
            )
        except Exception as e:
            logger.error(f"Error processing node {i} in {repo_path}: {e}")
            continue

    if not blocks:
        raise ValueError(f"No valid blocks extracted from {repo_path}")

    logger.info(f"Collected {len(blocks)} epic blocks from {repo_path}")
    return blocks


def _calculate_chunk_line_numbers(
    original_content: str,
    chunk_content: str,
    previous_chunk_end: int = 0
) -> Tuple[int, int]:
    """
    计算 chunk 在原文件中的实际行号

    Args:
        original_content: 原始文件内容
        chunk_content: chunk 内容
        previous_chunk_end: 上一个 chunk 的结束字符位置（用于优化搜索）

    Returns:
        (start_line, end_line) 元组，0-based 行号
    """
    # 策略1: 精确匹配（处理重叠时可能失败）
    chunk_start_pos = original_content.find(chunk_content, previous_chunk_end)

    if chunk_start_pos >= 0:
        # 找到精确匹配，计算行号
        start_line = original_content[:chunk_start_pos].count('\n')
        end_line = start_line + len(chunk_content.splitlines()) - 1
        return start_line, end_line

    # 策略2: 模糊匹配（使用 chunk 的第一行和最后一行）
    chunk_lines = chunk_content.splitlines()
    if not chunk_lines:
        return 0, 0

    first_line = chunk_lines[0].strip()
    last_line = chunk_lines[-1].strip()

    if not first_line and not last_line:
        # 空 chunk，返回估算值
        return previous_chunk_end // 80, previous_chunk_end // 80  # 假设每行80字符

    # 查找第一行在原文件中的位置
    original_lines = original_content.splitlines()
    start_line = None
    end_line = None

    for i, line in enumerate(original_lines):
        if first_line and first_line in line:
            start_line = i
            break

    # 从后往前查找最后一行
    if last_line:
        for i in range(len(original_lines) - 1, -1, -1):
            if last_line in original_lines[i]:
                end_line = i
                break

    # 如果找到了，返回结果；否则使用估算
    if start_line is not None and end_line is not None:
        return start_line, end_line
    elif start_line is not None:
        return start_line, start_line + len(chunk_lines) - 1
    elif end_line is not None:
        return max(0, end_line - len(chunk_lines) + 1), end_line
    else:
        # 完全找不到，使用启发式估算
        estimated_start = previous_chunk_end // 80  # 假设每行80字符
        return estimated_start, estimated_start + len(chunk_lines) - 1


def _get_or_calculate_line_numbers(
    node,
    original_content: str,
    previous_chunk_end: int = 0
) -> Tuple[int, int]:
    """
    从 node metadata 获取行号，如果不存在则计算

    Args:
        node: LlamaIndex node 对象
        original_content: 原始文件内容
        previous_chunk_end: 上一个 chunk 的结束字符位置（用于优化搜索）

    Returns:
        (start_line, end_line) 元组，0-based 行号
    """
    # 检查 metadata 中是否已有行号
    metadata_start = node.metadata.get('start_line')
    metadata_end = node.metadata.get('end_line')

    # 如果 metadata 中已有有效的行号，直接使用
    if metadata_start is not None and metadata_start >= 0:
        if metadata_end is not None and metadata_end >= metadata_start:
            return metadata_start, metadata_end
        else:
            # 只有 start_line，计算 end_line
            chunk_content = node.get_content()
            return metadata_start, metadata_start + len(chunk_content.splitlines()) - 1

    # metadata 中没有行号，使用辅助函数计算
    chunk_content = node.get_content()
    return _calculate_chunk_line_numbers(
        original_content, chunk_content, previous_chunk_end
    )


def collect_langchain_recursive_blocks(
    repo_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Block]:
    """
    使用 LangChain RecursiveCharacterTextSplitter 收集代码块（简化版）

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（字符数）
        chunk_overlap: 块之间的重叠字符数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use langchain strategies.")
        return []
    if not HAS_LANGCHAIN:
        logger.error("LangChain text splitters are not available; install langchain-text-splitters to use langchain strategies.")
        return []

    # 1. 读取仓库为文档
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

    # 2. 使用 LangChain RecursiveCharacterTextSplitter 切块
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        for doc in docs:
            original_content = doc.get_content()  # 保存原始内容
            chunks = splitter.split_text(original_content)

            previous_chunk_end = 0  # 跟踪上一个 chunk 的结束位置
            for chunk in chunks:
                # 计算实际行号
                start_line, end_line = _calculate_chunk_line_numbers(
                    original_content, chunk, previous_chunk_end
                )
                # 更新 previous_chunk_end（估算）
                chunk_pos = original_content.find(chunk, previous_chunk_end)
                if chunk_pos >= 0:
                    previous_chunk_end = chunk_pos + len(chunk)
                else:
                    # 如果找不到，使用行号估算字符位置
                    previous_chunk_end = end_line * 80  # 假设每行80字符

                all_chunks.append({
                    'file_path': doc.metadata.get('file_path', ''),
                    'content': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                })
    except Exception as e:
        logger.error(f"LangChain RecursiveCharacterTextSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    for chunk_info in all_chunks:
        file_path = chunk_info['file_path']
        content_text = chunk_info['content']
        start_line = chunk_info['start_line']  # 使用计算出的行号
        end_line = chunk_info['end_line']       # 使用计算出的行号

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

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（字符数）
        chunk_overlap: 块之间的重叠字符数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use langchain strategies.")
        return []
    if not HAS_LANGCHAIN:
        logger.error("LangChain text splitters are not available; install langchain-text-splitters to use langchain strategies.")
        return []

    # 1. 读取仓库为文档
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

    # 2. 使用 LangChain CharacterTextSplitter 切块
    try:
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        for doc in docs:
            original_content = doc.get_content()  # 保存原始内容
            chunks = splitter.split_text(original_content)

            previous_chunk_end = 0  # 跟踪上一个 chunk 的结束位置
            for chunk in chunks:
                # 计算实际行号
                start_line, end_line = _calculate_chunk_line_numbers(
                    original_content, chunk, previous_chunk_end
                )
                # 更新 previous_chunk_end（估算）
                chunk_pos = original_content.find(chunk, previous_chunk_end)
                if chunk_pos >= 0:
                    previous_chunk_end = chunk_pos + len(chunk)
                else:
                    # 如果找不到，使用行号估算字符位置
                    previous_chunk_end = end_line * 80  # 假设每行80字符

                all_chunks.append({
                    'file_path': doc.metadata.get('file_path', ''),
                    'content': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                })
    except Exception as e:
        logger.error(f"LangChain CharacterTextSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    for chunk_info in all_chunks:
        file_path = chunk_info['file_path']
        content_text = chunk_info['content']
        start_line = chunk_info['start_line']  # 使用计算出的行号
        end_line = chunk_info['end_line']       # 使用计算出的行号

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

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（token数）
        chunk_overlap: 块之间的重叠token数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use langchain strategies.")
        return []
    if not HAS_LANGCHAIN:
        logger.error("LangChain text splitters are not available; install langchain-text-splitters to use langchain strategies.")
        return []

    # 1. 读取仓库为文档
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

    # 2. 使用 LangChain TokenTextSplitter 切块
    try:
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            disallowed_special=(),  # 允许所有特殊 token，避免遇到 '<|endoftext|>' 等特殊 token 时报错
        )

        all_chunks = []
        for doc in docs:
            original_content = doc.get_content()  # 保存原始内容
            chunks = splitter.split_text(original_content)

            previous_chunk_end = 0  # 跟踪上一个 chunk 的结束位置
            for chunk in chunks:
                # 计算实际行号
                start_line, end_line = _calculate_chunk_line_numbers(
                    original_content, chunk, previous_chunk_end
                )
                # 更新 previous_chunk_end（估算）
                chunk_pos = original_content.find(chunk, previous_chunk_end)
                if chunk_pos >= 0:
                    previous_chunk_end = chunk_pos + len(chunk)
                else:
                    # 如果找不到，使用行号估算字符位置
                    previous_chunk_end = end_line * 80  # 假设每行80字符

                all_chunks.append({
                    'file_path': doc.metadata.get('file_path', ''),
                    'content': chunk,
                    'start_line': start_line,
                    'end_line': end_line,
                })
    except Exception as e:
        logger.error(f"LangChain TokenTextSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    for chunk_info in all_chunks:
        file_path = chunk_info['file_path']
        content_text = chunk_info['content']
        start_line = chunk_info['start_line']  # 使用计算出的行号
        end_line = chunk_info['end_line']       # 使用计算出的行号

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


def collect_llamaindex_code_blocks(
    repo_path: str,
    language: str = "python",
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 15,
    max_chars: int = 1500,
) -> List[Block]:
    """
    使用 LlamaIndex CodeSplitter 收集代码块（代码专用，基于AST）

    Args:
        repo_path: 仓库路径
        language: 编程语言（默认python）
        chunk_lines: 每个块的代码行数
        chunk_lines_overlap: 块之间的重叠行数
        max_chars: 每个块的最大字符数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_code strategy.")
        return []

    # 1. 读取仓库为文档（复用现有模式）
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
        if not docs:
            print(f'\n[DEBUG] {repo_path}: No documents loaded (after exclusions)')
            logger.warning(f"No documents loaded from {repo_path} (after exclusions)")
            return []
        print(f'\n[DEBUG] {repo_path}: Loaded {len(docs)} documents')
        logger.info(f"Loaded {len(docs)} documents from {repo_path}")
    except Exception as e:
        logger.error(f"Failed to read repo {repo_path}: {e}")
        return []

    # 2. 使用 LlamaIndex CodeSplitter 切块
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
        # 降级处理：使用 fixed 分块策略作为备用
        try:
            blocks = collect_blocks(repo_path, "fixed", block_size=40, window_size=50, slice_size=10)
            if blocks:
                print(f'\n[DEBUG] {repo_path}: Fallback to fixed strategy generated {len(blocks)} blocks')
                logger.info(f"Fallback to fixed strategy generated {len(blocks)} blocks for {repo_path}")
                return blocks
        except Exception as fallback_error:
            logger.error(f"Fallback strategy also failed for {repo_path}: {fallback_error}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    # 按文件分组处理，以便跟踪 previous_chunk_end
    file_to_docs = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    # 为每个文件建立内容映射
    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        # 合并同一文件的所有文档内容
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        # 获取或计算行号
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
    """
    使用 LlamaIndex SentenceSplitter 收集代码块（按句子分割）

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（字符数）
        chunk_overlap: 块之间的重叠字符数

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_sentence strategy.")
        return []

    # 1. 读取仓库为文档
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

    # 2. 使用 LlamaIndex SentenceSplitter 切块
    try:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        nodes = splitter.get_nodes_from_documents(docs)
    except Exception as e:
        logger.error(f"LlamaIndex SentenceSplitter failed for {repo_path}: {e}")
        return []

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    # 按文件分组处理，以便跟踪 previous_chunk_end
    file_to_docs = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    # 为每个文件建立内容映射
    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        # 合并同一文件的所有文档内容
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        # 获取或计算行号
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
    """
    使用 LlamaIndex TokenTextSplitter 收集代码块（按token分割）

    Args:
        repo_path: 仓库路径
        chunk_size: 目标块大小（token数）
        chunk_overlap: 块之间的重叠token数
        separator: 分隔符

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_token strategy.")
        return []

    # 1. 读取仓库为文档
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

    # 2. 使用 LlamaIndex TokenTextSplitter 切块
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

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    # 按文件分组处理，以便跟踪 previous_chunk_end
    file_to_docs = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    # 为每个文件建立内容映射
    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        # 合并同一文件的所有文档内容
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        # 获取或计算行号
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
    """
    使用 LlamaIndex SemanticSplitterNodeParser 收集代码块（基于语义相似性）

    Args:
        repo_path: 仓库路径
        buffer_size: 缓冲区大小（用于语义分割）
        model_name: HuggingFace embedding 模型名称（默认使用轻量级模型）

    Returns:
        代码块列表
    """
    logger = logging.getLogger(__name__)
    if not HAS_LLAMA_INDEX:
        logger.error("LlamaIndex is not available; install llama-index to use llamaindex_semantic strategy.")
        return []
    if not HAS_LLAMA_INDEX_HF:
        logger.error("HuggingFaceEmbedding is not available; install LlamaIndex HuggingFace embeddings to use llamaindex_semantic strategy.")
        return []

    # 1. 读取仓库为文档
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

    # 2. 使用本地 HuggingFace embedding 模型
    try:
        # 使用本地模型，无需 API key
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

    # 3. 转换为 Block 对象
    blocks: List[Block] = []
    # 按文件分组处理，以便跟踪 previous_chunk_end
    file_to_docs = {}
    for doc in docs:
        file_path = doc.metadata.get('file_path', '') or doc.metadata.get('file_name', '')
        if file_path:
            if file_path not in file_to_docs:
                file_to_docs[file_path] = []
            file_to_docs[file_path].append(doc)

    # 为每个文件建立内容映射
    file_contents = {}
    for file_path, file_docs in file_to_docs.items():
        # 合并同一文件的所有文档内容
        file_contents[file_path] = '\n'.join([doc.get_content() for doc in file_docs])
    previous_chunk_end_by_file = {}

    for node in nodes:
        file_path = node.metadata.get('file_path', '') or node.metadata.get('file_name', '')
        if not file_path:
            continue

        content_text = node.get_content()
        if not content_text:
            continue

        # 获取或计算行号
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
                block_type="llamaindex_semantic",
            )
        )

    logger.info(f"Collected {len(blocks)} blocks using LlamaIndex SemanticSplitter from {repo_path}")
    return blocks


# ============================================================================
# Summary-based strategy
# ============================================================================

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

    # Prefer raw string first (some versions validate prompt_template as str)
    try:
        return SummaryExtractor(prompt_template=prompt_template, **kwargs)
    except Exception:
        pass
    try:
        return SummaryExtractor(summary_template=prompt_template, **kwargs)
    except Exception:
        pass

    # Fallback to PromptTemplate object for versions that require it
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
    """
    Summary-based strategy: generate summaries and build mixed blocks.
    Returns blocks and summary_metadata.
    """
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

    try:
        from core.llm_factory import get_llm
    except Exception:
        from .core.llm_factory import get_llm  # type: ignore

    # 1) collect base blocks
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

    # 2) build LLM and summary extractor
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

    # LlamaIndex IngestionPipeline/SummaryExtractor may fall back to Settings.llm.
    # Set it to our configured LLM so requests use the correct model (e.g. GPT-OSS-120B),
    # not the default gpt-4o-mini.
    try:
        from llama_index.core import Settings
        Settings.llm = llm
    except Exception:
        pass

    try:
        summary_extractor = _build_summary_extractor(llm, prompt_template)
    except Exception as e:
        logger.error("Failed to initialize SummaryExtractor: %s", e)
        return [], {}

    # 3) summarize
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


# ============================================================================
# 工具函数
# ============================================================================


def list_folders(path: str) -> List[str]:
    return [p.name for p in Path(path).iterdir() if p.is_dir()]


def instance_id_to_repo_name(instance_id: str) -> str:
    """将 instance_id 转换为 repo_name（去掉 issue 编号后缀）"""
    import re
    repo_part = re.sub(r'-\d+$', '', instance_id)
    return repo_part.replace('__', '_')


def resolve_model_name(model_name: str) -> str:
    if os.path.isabs(model_name):
        return model_name
    if os.path.exists(model_name):
        return os.path.abspath(model_name)
    project_candidate = _project_root / model_name
    if project_candidate.exists():
        return str(project_candidate.resolve())
    return model_name


def save_index(
    output_dir: Path,
    embeddings: torch.Tensor,
    blocks: List[Block],
    strategy: str,
    span_ids_map: Optional[Dict[int, List[str]]] = None,
    epic_config: Optional[EpicSplitterConfig] = None,
    function_metadata: Optional[Dict[int, Dict[str, Optional[str]]]] = None,
    summary_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
):
    return _save_index(
        output_dir=output_dir,
        embeddings=embeddings,
        blocks=blocks,
        strategy=strategy,
        span_ids_map=span_ids_map,
        epic_config=epic_config,
        function_metadata=function_metadata,
        summary_metadata=summary_metadata,
    )


# ============================================================================
# 外部库依赖检查
# ============================================================================


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


# ============================================================================
# 兼容导出：重定向到新 chunker 模块
# ============================================================================

EpicSplitterConfig = _EpicSplitterConfig

collect_blocks = _collect_blocks
build_context_enhanced_content = _build_context_enhanced_content
_calculate_chunk_line_numbers = _calculate_chunk_line_numbers
_get_or_calculate_line_numbers = _get_or_calculate_line_numbers

collect_epic_blocks = _collect_epic_blocks
collect_langchain_recursive_blocks = _collect_langchain_recursive_blocks
collect_langchain_fixed_blocks = _collect_langchain_fixed_blocks
collect_langchain_token_blocks = _collect_langchain_token_blocks
collect_llamaindex_code_blocks = _collect_llamaindex_code_blocks
collect_llamaindex_sentence_blocks = _collect_llamaindex_sentence_blocks
collect_llamaindex_token_blocks = _collect_llamaindex_token_blocks
collect_llamaindex_semantic_blocks = _collect_llamaindex_semantic_blocks

DEFAULT_SUMMARY_PROMPT = _SUMMARY_DEFAULT_PROMPT
_load_summary_prompt = _summary_load_summary_prompt
_compute_summary_cache_key = _summary_compute_summary_cache_key
_load_cached_summary = _summary_load_cached_summary
_save_cached_summary = _summary_save_cached_summary
_extract_summary_from_metadata = _summary_extract_summary_from_metadata
_build_summary_extractor = _summary_build_summary_extractor
_summarize_text = _summary_summarize_text
collect_summary_blocks = _collect_summary_blocks
