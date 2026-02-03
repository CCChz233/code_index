#!/usr/bin/env python
"""
批量构建稠密索引（仿照 dependency_graph/batch_build_graph.py 的多进程模式）

支持多种切块策略：
- 基础策略: fixed, sliding, rl_fixed, rl_mini
- 函数级策略: function_level, ir_function
- 高级策略: epic
- LangChain策略: langchain_fixed, langchain_recursive, langchain_token
- LlamaIndex策略: llamaindex_code, llamaindex_sentence, llamaindex_token, llamaindex_semantic

用法:
    # 从数据集加载
    python method/index/batch_build_index.py \
        --dataset czlll/Loc-Bench_V1 \
        --split test \
        --num_processes 4 \
        --model_name models/rlretriever \
        --strategy fixed \
        --block_size 15

    # 使用本地仓库目录
    python method/index/batch_build_index.py \
        --repo_path playground/locbench_repos \
        --num_processes 4 \
        --model_name models/rlretriever \
        --strategy fixed

    # 使用 LlamaIndex CodeSplitter
    python method/index/batch_build_index.py \
        --repo_path playground/locbench_repos \
        --strategy llamaindex_code \
        --model_name /path/to/model \
        --num_processes 4
"""

import argparse
import ast
import json
import os
import os.path as osp
import random
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到Python路径
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# LocAgent 依赖已复制到 IR-based 下，不再需要添加路径

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

try:
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import (
        CodeSplitter,
        SentenceSplitter,
        TokenTextSplitter as LlamaIndexTokenTextSplitter,  # Avoid name clash with LangChain
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

# EpicSplitter 延迟导入（只在epic策略时导入，避免不必要的依赖）
# from repo_index.index.epic_split import EpicSplitter
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

# 导入build_index.py中的span_ids提取函数（可选）
try:
    from method.indexing.build_index import extract_span_ids_from_graph
    HAS_GRAPH_INDEX = True
except Exception as e:
    HAS_GRAPH_INDEX = False
    _GRAPH_INDEX_IMPORT_ERROR = e

    def extract_span_ids_from_graph(*_args, **_kwargs):
        return {}

# 尝试导入 datasets，如果失败则只支持本地模式
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ============================================================================
# 切块逻辑（与 build_index.py 一致）
# ============================================================================


class Block:
    def __init__(
        self,
        file_path: str,
        start: int,
        end: int,
        content: str,
        block_type: str,
        context_text: Optional[str] = None,
        function_text: Optional[str] = None,
    ):
        self.file_path = file_path
        self.start = start
        self.end = end
        self.content = content
        self.block_type = block_type
        self.context_text = context_text
        self.function_text = function_text


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


def iter_files(repo_root: Path) -> List[Path]:
    from method.indexing.utils.file import iter_files as _iter_files
    return _iter_files(repo_root)


def blocks_fixed_lines(text: str, rel: str, block_size: int) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []
    start = 0
    while start < len(lines):
        end = min(start + block_size, len(lines))
        chunk = lines[start:end]
        if any(l.strip() for l in chunk):
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
        if any(l.strip() for l in chunk):
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


# ============================================================================
# IR-based 函数级分块（论文复现）
# Paper: Appendix C.1.1
# "The function's context—its containing file and class—is appended to 
# the function representation before embedding, rather than being embedded separately."
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
    """
    按照论文 IR-based 方法提取函数级代码块（Paper Reproduction）
    
    论文关键特点（Appendix C.1.1）：
    1. 每个函数单独作为一个嵌入单元（embedding unit）
    2. 模块级代码（imports, 全局变量）冗余复制到该文件每个函数
    3. 类属性冗余复制到该类每个方法
    4. 使用 Flat Indexing（无层级结构，无单独的全局变量索引）
    5. 非 Python 文件不处理（只索引函数）
    
    与我们的 function_level 的区别：
    - function_level: 只有 qualified_name + 函数代码
    - ir_function: file_path + class_name + module_code + class_attrs + 函数代码
    
    Args:
        text: 文件内容
        rel: 相对文件路径
    
    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典
    """
    blocks: List[Block] = []
    function_metadata: Dict[int, Dict[str, Optional[str]]] = {}
    
    # 只处理 Python 文件（论文只在函数级别索引）
    if not rel.endswith('.py'):
        return blocks, function_metadata
    
    # 检查文件大小
    if len(text) > 10 * 1024 * 1024:  # 10MB
        logging.debug(f"File {rel} is too large, skipping")
        return blocks, function_metadata
    
    try:
        tree = ast.parse(text)
        lines = text.splitlines()
        
        # 1. 提取模块级代码（将冗余复制到每个函数）
        module_level_code = _extract_module_level_code(tree, lines)
        
        def get_function_code(node) -> str:
            """获取函数的完整代码（包括装饰器）"""
            func_start = node.lineno - 1
            func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 10
            
            # 获取装饰器起始行
            decorator_start = func_start
            if node.decorator_list:
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, 'lineno'):
                    decorator_start = first_decorator.lineno - 1
            
            all_lines = lines[decorator_start:func_end]
            return "\n".join(all_lines)
        
        def process_function(
            node,
            class_name: Optional[str] = None,
            class_attributes: str = "",
            function_stack: List[str] = None,
        ):
            """处理单个函数/方法"""
            nonlocal blocks, function_metadata
            
            if function_stack is None:
                function_stack = []
            
            function_name = node.name
            function_stack.append(function_name)  # 添加到栈
            
            # 获取行号范围
            start_line = node.lineno - 1
            end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10
            
            # 包含装饰器的起始行
            if node.decorator_list:
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, 'lineno'):
                    start_line = first_decorator.lineno - 1
            
            # 获取函数代码
            func_code = get_function_code(node)
            if not func_code.strip():
                function_stack.pop()  # 退出时弹出
                return
            
            # 构建完全限定名（用于 metadata）- 考虑嵌套函数
            if class_name:
                if len(function_stack) > 1:
                    # 嵌套函数：使用完整的函数路径
                    qualified_name = f"{rel}::{class_name}::{'::'.join(function_stack)}"
                else:
                    # 顶层方法
                    qualified_name = f"{rel}::{class_name}::{function_name}"
            else:
                if len(function_stack) > 1:
                    # 嵌套函数：使用完整的函数路径
                    qualified_name = f"{rel}::{'::'.join(function_stack)}"
                else:
                    # 顶层函数
                    qualified_name = f"{rel}::{function_name}"
            
            # 按照论文方式构建 IR 表示
            context_text = _build_ir_function_context(
                file_path=rel,
                class_name=class_name,
                module_level_code=module_level_code,
                class_attributes=class_attributes,
            )
            ir_representation = _build_ir_function_representation(
                file_path=rel,
                class_name=class_name,
                function_code=func_code,
                module_level_code=module_level_code,
                class_attributes=class_attributes,
            )
            
            # 创建 Block
            block = Block(
                rel,
                start_line,
                end_line,
                ir_representation,
                "ir_function",
                context_text=context_text,
                function_text=func_code,
            )
            block_index = len(blocks)
            blocks.append(block)
            
            # 保存元数据
            function_metadata[block_index] = {
                "qualified_name": qualified_name,
                "class_name": class_name,
                "function_name": function_name,
            }
            
            # 递归处理嵌套函数（传递函数栈的副本）
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    process_function(child, class_name, class_attributes, function_stack.copy())
            
            function_stack.pop()  # 退出时弹出
        
        # 2. 遍历 AST 顶层节点
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # 提取类属性（将冗余复制到该类每个方法）
                class_attributes = _extract_class_attributes(node, lines)
                
                # 处理类中的方法
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        process_function(child, node.name, class_attributes, [])
                    # 处理嵌套类
                    elif isinstance(child, ast.ClassDef):
                        nested_class_attrs = _extract_class_attributes(child, lines)
                        for nested_child in child.body:
                            if isinstance(nested_child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                process_function(
                                    nested_child, 
                                    f"{node.name}.{child.name}", 
                                    nested_class_attrs,
                                    []  # 嵌套类的方法也使用空栈开始
                                )
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 顶层函数（没有类属性）
                process_function(node, None, "", [])
    
    except SyntaxError as e:
        logging.debug(f"Syntax error in {rel} (line {e.lineno}): {e.msg}. Skipping.")
        return blocks, function_metadata
    except Exception as e:
        logging.error(f"Error processing {rel}: {type(e).__name__}: {e}. Skipping.")
        return blocks, function_metadata
    
    return blocks, function_metadata


def collect_ir_function_blocks(repo_path: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    收集 IR-based 函数级代码块（论文复现版本）
    
    按照论文 Appendix C.1.1：
    - 每个函数单独作为一个嵌入单元
    - 模块级代码冗余复制到该文件每个函数
    - 类属性冗余复制到该类每个方法
    - 只索引 Python 文件中的函数
    
    Args:
        repo_path: 仓库路径
    
    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典
    """
    logger = logging.getLogger(__name__)
    repo_root = Path(repo_path)
    all_blocks: List[Block] = []
    all_function_metadata: Dict[int, Dict[str, Optional[str]]] = {}
    
    # 只处理 Python 文件
    python_files = [p for p in iter_files(repo_root) if p.suffix.lower() == '.py']
    total_files = len(python_files)
    
    if total_files == 0:
        logger.debug(f"No Python files found in {repo_path}")
        return all_blocks, all_function_metadata
    
    logger.info(f"Processing {total_files} Python files in {repo_path} (IR-based paper style)")
    
    stats = {
        'total_files': total_files,
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'total_functions': 0,
        'files_with_functions': 0,
    }
    
    for p in python_files:
        stats['processed'] += 1
        
        # 检查文件大小
        try:
            file_size = p.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.debug(f"Skipping large file {p.name}")
                stats['skipped'] += 1
                continue
        except Exception:
            stats['skipped'] += 1
            continue
        
        # 读取文件
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                stats['skipped'] += 1
                continue
        except Exception:
            stats['skipped'] += 1
            continue
        
        rel = str(p.relative_to(repo_root))
        
        # 使用 IR-based 方法提取
        file_blocks, file_metadata = blocks_ir_function(text, rel)
        
        # 调整 metadata 索引
        current_start_idx = len(all_blocks)
        for local_idx, metadata in file_metadata.items():
            global_idx = current_start_idx + local_idx
            all_function_metadata[global_idx] = metadata
        
        all_blocks.extend(file_blocks)
        
        if file_blocks:
            stats['files_with_functions'] += 1
            stats['total_functions'] += len(file_blocks)
    
    logger.info(
        f"Collected {len(all_blocks)} IR-style function blocks from {repo_path} "
        f"(files: {stats['files_with_functions']}, "
        f"avg per file: {stats['total_functions'] / max(stats['files_with_functions'], 1):.1f})"
    )
    
    return all_blocks, all_function_metadata


# ============================================================================
# 原有的 function_level 实现（保留兼容）
# ============================================================================


def build_function_context_content(
    file_path: str,
    class_name: Optional[str],
    function_name: str,
    qualified_name: str,
    original_code: str,
    start_line: int,  # 保留但不在输出中使用
    end_line: int,    # 保留但不在输出中使用
) -> str:
    """
    构建超简洁的上下文（论文风格）
    
    格式: # qualified_name\n\n{code}
    示例: # src/utils.py::MathUtils::calculate_sum\n\ndef calculate_sum(a, b):\n    return a + b
    
    Args:
        file_path: 文件路径（保留但不在输出中使用）
        class_name: 类名（保留但不在输出中使用）
        function_name: 函数名（保留但不在输出中使用）
        qualified_name: 完全限定名（用于注释）
        original_code: 原始函数代码
        start_line: 起始行号（保留但不在输出中使用）
        end_line: 结束行号（保留但不在输出中使用）
    
    Returns:
        增强后的代码内容（只包含一行注释标识符和代码）
    """
    # 只添加一行注释作为标识符
    header = f"# {qualified_name}\n\n"
    return header + original_code


def blocks_function_level_with_fallback(
    text: str, 
    rel: str, 
    block_size: int = 15
) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    函数级切块 + 未覆盖代码的 fixed 切块作为补充
    
    解决 function_level 策略只索引函数、遗漏大量非函数代码的问题：
    - 模块级代码（全局变量、常量定义、import 语句）
    - 类定义本身（类属性、类级代码）
    - 脚本入口代码（if __name__ == "__main__": 下的非函数代码）
    
    Args:
        text: 文件内容
        rel: 相对文件路径
        block_size: 对未覆盖代码使用的 fixed 切块大小，默认 15 行
    
    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典
    """
    # 1. 先提取所有函数级块
    function_blocks, function_metadata = blocks_by_function(text, rel)
    function_metadata_by_block = {
        function_blocks[idx]: metadata for idx, metadata in function_metadata.items()
    }
    
    # 非 Python 文件或解析失败时，直接使用 fixed 策略
    if not rel.endswith('.py') or not function_blocks:
        fixed_blocks = blocks_fixed_lines(text, rel, block_size)
        # 修改 block_type 以区分
        for b in fixed_blocks:
            b.block_type = "function_level_fallback"
        return fixed_blocks, {}
    
    # 2. 记录被函数覆盖的行号
    lines = text.splitlines()
    total_lines = len(lines)
    covered_lines = set()
    
    for block in function_blocks:
        for line_no in range(block.start, block.end + 1):
            covered_lines.add(line_no)
    
    # 3. 找出未被覆盖的连续行区间
    uncovered_ranges = []
    current_start = None
    
    for i in range(total_lines):
        if i not in covered_lines:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                uncovered_ranges.append((current_start, i - 1))
                current_start = None
    
    # 处理末尾的未覆盖区间
    if current_start is not None:
        uncovered_ranges.append((current_start, total_lines - 1))
    
    # 4. 对未覆盖的区间使用 fixed 策略切分
    fallback_blocks = []
    for range_start, range_end in uncovered_ranges:
        # 提取该区间的文本
        range_lines = lines[range_start:range_end + 1]
        range_text = "\n".join(range_lines)
        
        # 跳过全空区间
        if not any(l.strip() for l in range_lines):
            continue
        
        # 对该区间使用 fixed 切块
        chunk_start = 0
        while chunk_start < len(range_lines):
            chunk_end = min(chunk_start + block_size, len(range_lines))
            chunk_lines = range_lines[chunk_start:chunk_end]
            
            # 跳过全空块
            if any(l.strip() for l in chunk_lines):
                # 计算在原文件中的真实行号
                real_start = range_start + chunk_start
                real_end = range_start + chunk_end - 1
                content = "\n".join(chunk_lines)
                
                fallback_blocks.append(Block(
                    rel, real_start, real_end, content, "function_level_fallback"
                ))
            
            chunk_start = chunk_end
    
    # 5. 合并函数块和补充块
    all_blocks = function_blocks + fallback_blocks
    
    # 按起始行号排序，保持文件顺序
    all_blocks.sort(key=lambda b: b.start)

    sorted_metadata = {}
    for idx, block in enumerate(all_blocks):
        metadata = function_metadata_by_block.get(block)
        if metadata is not None:
            sorted_metadata[idx] = metadata

    return all_blocks, sorted_metadata


def blocks_by_function(text: str, rel: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    使用AST解析提取所有函数和类方法
    
    Args:
        text: 文件内容
        rel: 相对文件路径
    
    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典
        
    注意：如果AST解析失败，返回空blocks和空metadata，不回退到其他策略
    这样可以保持策略的一致性（block_type始终是"function_level"或为空）
    """
    blocks: List[Block] = []
    function_metadata: Dict[int, Dict[str, Optional[str]]] = {}
    
    # 只处理Python文件
    if not rel.endswith('.py'):
        return blocks, function_metadata
    
    # 检查文件大小，避免处理过大的文件
    if len(text) > 10 * 1024 * 1024:  # 10MB
        logging.debug(f"File {rel} is too large ({len(text) / 1024 / 1024:.2f}MB), skipping")
        return blocks, function_metadata
    
    try:
        tree = ast.parse(text)
        lines = text.splitlines()
        
        # 维护类栈和函数栈来追踪嵌套结构
        class_stack: List[str] = []  # 只保存直接父类（用于嵌套类）
        function_stack: List[str] = []  # 用于嵌套函数
        
        def get_function_code(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            """获取函数的完整代码（包括装饰器）"""
            # 获取函数体的行号范围
            func_start = node.lineno - 1  # AST行号从1开始，转换为0-based
            func_end = node.end_lineno - 1 if hasattr(node, 'end_lineno') else func_start + 10
            
            # 获取装饰器：装饰器通常在函数定义之前，需要从函数定义行向上查找
            decorator_start = func_start
            if node.decorator_list:
                # 找到第一个装饰器的起始行
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, 'lineno'):
                    decorator_start = first_decorator.lineno - 1
            
            # 获取从第一个装饰器到函数结束的所有行
            all_lines = lines[decorator_start:func_end + 1]
            return "\n".join(all_lines)
        
        def visit_node(node: ast.AST):
            """递归访问AST节点"""
            nonlocal blocks, function_metadata
            
            # 处理类定义
            if isinstance(node, ast.ClassDef):
                # 只保存直接父类（不包含外层类）
                class_stack.append(node.name)
                
                # 访问类体中的节点
                for child in node.body:
                    visit_node(child)
                
                # 弹出当前类
                class_stack.pop()
            
            # 处理函数定义
            elif isinstance(node, ast.FunctionDef):
                function_name = node.name
                function_stack.append(function_name)
                
                # 获取行号
                start_line = node.lineno - 1  # 转换为0-based
                end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10
                if node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    if hasattr(first_decorator, 'lineno'):
                        start_line = min(start_line, first_decorator.lineno - 1)
                
                # 获取函数完整代码
                func_code = get_function_code(node)
                
                if not func_code.strip():
                    function_stack.pop()
                    # 继续访问嵌套函数
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            visit_node(child)
                    return
                
                # 构建完全限定名
                if class_stack:
                    # 类方法：只包含直接父类
                    class_name = class_stack[-1]
                    qualified_name = f"{rel}::{class_name}::{function_name}"
                else:
                    # 顶层函数
                    qualified_name = f"{rel}::{function_name}"
                
                # 构建上下文增强的代码
                enhanced_content = build_function_context_content(
                    file_path=rel,
                    class_name=class_stack[-1] if class_stack else None,
                    function_name=function_name,
                    qualified_name=qualified_name,
                    original_code=func_code,
                    start_line=start_line,
                    end_line=end_line,
                )
                
                # 创建Block对象
                block = Block(rel, start_line, end_line, enhanced_content, "function_level")
                block_index = len(blocks)
                blocks.append(block)
                
                # 保存元数据
                function_metadata[block_index] = {
                    "qualified_name": qualified_name,
                    "class_name": class_stack[-1] if class_stack else None,
                    "function_name": function_name,
                }
                
                # 访问嵌套函数（函数内部定义的函数）
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit_node(child)
                
                function_stack.pop()
            
            # 处理异步函数（与普通函数处理逻辑相同，可以合并）
            elif isinstance(node, ast.AsyncFunctionDef):
                # 异步函数处理逻辑与普通函数相同
                function_name = node.name
                function_stack.append(function_name)
                
                start_line = node.lineno - 1
                end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10
                if node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    if hasattr(first_decorator, 'lineno'):
                        start_line = min(start_line, first_decorator.lineno - 1)
                
                # 使用get_function_code获取完整代码（包括装饰器）
                func_code = get_function_code(node)
                
                if not func_code.strip():
                    function_stack.pop()
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            visit_node(child)
                    return
                
                if class_stack:
                    class_name = class_stack[-1]
                    qualified_name = f"{rel}::{class_name}::{function_name}"
                else:
                    qualified_name = f"{rel}::{function_name}"
                
                enhanced_content = build_function_context_content(
                    file_path=rel,
                    class_name=class_stack[-1] if class_stack else None,
                    function_name=function_name,
                    qualified_name=qualified_name,
                    original_code=func_code,
                    start_line=start_line,
                    end_line=end_line,
                )
                
                block = Block(rel, start_line, end_line, enhanced_content, "function_level")
                block_index = len(blocks)
                blocks.append(block)
                
                function_metadata[block_index] = {
                    "qualified_name": qualified_name,
                    "class_name": class_stack[-1] if class_stack else None,
                    "function_name": function_name,
                }
                
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit_node(child)
                
                function_stack.pop()
            
            # 对于其他节点，继续访问子节点
            else:
                if hasattr(node, 'body'):
                    for child in node.body:
                        visit_node(child)
        
        # 从顶层节点开始访问
        for node in tree.body:
            visit_node(node)
    
    except SyntaxError as e:
        # 语法错误：文件可能不是有效的Python代码，跳过该文件
        # 使用debug级别，因为可能是正常的（如.py文件包含其他内容）
        logging.debug(f"Syntax error in {rel} (line {e.lineno}): {e.msg}. Skipping file.")
        return blocks, function_metadata
    except (ValueError, AttributeError) as e:
        # AST相关错误：可能是Python版本不兼容或其他问题
        logging.warning(f"AST parsing error in {rel}: {e}. Skipping file.")
        return blocks, function_metadata
    except MemoryError as e:
        # 内存错误：文件太大或AST解析消耗过多内存
        logging.error(f"Memory error processing {rel}: {e}. Skipping file.")
        return blocks, function_metadata
    except Exception as e:
        # 其他未知错误：记录详细信息并跳过
        logging.error(f"Unexpected error processing {rel}: {type(e).__name__}: {e}. Skipping file.")
        return blocks, function_metadata
    
    return blocks, function_metadata


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
        class_match = re.search(r"class\\s+(\\w+)", header)
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


def collect_function_blocks(repo_path: str, block_size: int = 15) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    收集函数级代码块（带 fallback），返回blocks和function_metadata
    
    改进：使用 blocks_function_level_with_fallback，确保完整覆盖所有代码：
    - 函数和方法使用 function_level 切块
    - 未被函数覆盖的代码（imports、全局变量、类属性等）使用 fixed 切块补充
    
    Args:
        repo_path: 仓库路径
        block_size: 对未覆盖代码使用的 fixed 切块大小，默认 15 行
    
    Returns:
        (blocks, function_metadata): 代码块列表和函数元数据字典
        
    注意：如果整个仓库没有提取到任何blocks，返回空列表和空metadata
    不回退到其他策略，让上层决定如何处理
    """
    logger = logging.getLogger(__name__)
    repo_root = Path(repo_path)
    all_blocks: List[Block] = []
    all_function_metadata: Dict[int, Dict[str, Optional[str]]] = {}
    
    # 获取所有支持的文件（不仅是 Python）
    all_files = list(iter_files(repo_root))
    total_files = len(all_files)
    
    if total_files == 0:
        logger.debug(f"No supported files found in {repo_path}")
        return all_blocks, all_function_metadata
    
    logger.info(f"Processing {total_files} files in {repo_path}")
    
    stats = {
        'total_files': total_files,
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'total_blocks': 0,
        'function_blocks': 0,
        'fallback_blocks': 0,
        'files_with_blocks': 0,
    }
    
    for p in all_files:
        stats['processed'] += 1
        if stats['processed'] % 100 == 0:
            logger.debug(f"Processed {stats['processed']}/{total_files} files in {repo_path} (skipped: {stats['skipped']}, errors: {stats['errors']})")
        
        # 检查文件大小，跳过过大的文件
        try:
            file_size = p.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB (提高阈值)
                logger.debug(f"Skipping very large file {p.name} ({file_size / 1024 / 1024:.2f}MB)")
                stats['skipped'] += 1
                continue
        except Exception as e:
            logger.warning(f"Error checking file size for {p}: {e}")
            stats['skipped'] += 1
            continue
        
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning(f"Error reading {p}: {e}")
                stats['skipped'] += 1
                continue
        except Exception as e:
            logger.warning(f"Error reading {p}: {e}")
            stats['skipped'] += 1
            continue
        
        rel = str(p.relative_to(repo_root))
        
        # 使用带 fallback 的函数级切块
        try:
            file_blocks, file_metadata = blocks_function_level_with_fallback(text, rel, block_size)
            
            # 调整metadata的索引（因为要合并到all_blocks中）
            current_start_idx = len(all_blocks)
            for local_idx, metadata in file_metadata.items():
                global_idx = current_start_idx + local_idx
                all_function_metadata[global_idx] = metadata
            
            all_blocks.extend(file_blocks)
            
            # 更新统计
            if file_blocks:
                stats['files_with_blocks'] += 1
                stats['total_blocks'] += len(file_blocks)
                # 分别统计函数块和 fallback 块
                for b in file_blocks:
                    if b.block_type == "function_level":
                        stats['function_blocks'] += 1
                    else:
                        stats['fallback_blocks'] += 1
        except Exception as e:
            logger.error(f"Error processing file {rel}: {e}")
            stats['errors'] += 1
            continue
    
    logger.info(
        f"Collected {len(all_blocks)} blocks from {repo_path} "
        f"(function: {stats['function_blocks']}, fallback: {stats['fallback_blocks']}, "
        f"processed: {stats['processed']}, skipped: {stats['skipped']}, "
        f"errors: {stats['errors']}, files with blocks: {stats['files_with_blocks']})"
    )
    
    # 如果没有提取到任何blocks，返回空（让上层决定是否跳过或标记为失败）
    if not all_blocks:
        logger.warning(f"No blocks extracted from {repo_path}")
        return all_blocks, all_function_metadata
    
    return all_blocks, all_function_metadata


# ============================================================================
# 嵌入与保存
# ============================================================================

class BlockDataset(Dataset):
    def __init__(self, blocks: List[Block], tokenizer, max_length: int, ir_context_tokens: int = 256):
        self.blocks = blocks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ir_context_tokens = ir_context_tokens

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx: int):
        b = self.blocks[idx]
        is_ir_function = b.block_type == "ir_function" and bool(b.function_text or b.context_text)
        if is_ir_function:
            input_ids, attention_mask, orig_len, truncated = self._encode_ir_function(b)
        else:
            input_ids, attention_mask, orig_len, truncated = self._encode_default(b)
        return input_ids, attention_mask, orig_len, truncated, is_ir_function

    def _pad_to_max_length(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.size(0) < self.max_length:
            pad_len = self.max_length - input_ids.size(0)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            pad_ids = torch.full((pad_len,), pad_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros((pad_len,), dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
        elif input_ids.size(0) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        return input_ids, attention_mask

    def _encode_default(self, b: Block) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        text = f"file path: {b.file_path}\nlines: {b.start}-{b.end}\n\n{b.content}"
        enc = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        orig_len = int(input_ids.size(0))
        truncated = orig_len > self.max_length

        if truncated:
            input_ids = input_ids[:self.max_length]
        attention_mask = torch.ones_like(input_ids)
        input_ids, attention_mask = self._pad_to_max_length(input_ids, attention_mask)
        return input_ids, attention_mask, orig_len, truncated

    def _encode_ir_function(self, b: Block) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        context_text = b.context_text or ""
        function_text = b.function_text or b.content

        context_ids = []
        if context_text:
            context_ids = self.tokenizer(
                context_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.ir_context_tokens,
            )["input_ids"]

        function_ids = self.tokenizer(
            function_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        orig_len = len(context_ids) + len(function_ids) + special_tokens

        max_function_len = max(self.max_length - special_tokens, 0)
        if len(function_ids) > max_function_len:
            function_ids = function_ids[:max_function_len]

        remaining_context = max(self.max_length - special_tokens - len(function_ids), 0)
        context_ids = context_ids[:remaining_context]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(context_ids + function_ids)
        attention_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        input_ids, attention_mask = self._pad_to_max_length(input_ids, attention_mask)

        truncated = orig_len > self.max_length
        return input_ids, attention_mask, orig_len, truncated


def embed_blocks(
    blocks: List[Block],
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    ir_context_tokens: int = 256,
    use_sentence_transformer: bool = False,
) -> torch.Tensor:
    ds = BlockDataset(blocks, tokenizer, max_length, ir_context_tokens=ir_context_tokens)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    outs = []
    total_blocks = 0
    truncated_blocks = 0
    ir_blocks = 0
    ir_truncated = 0
    ir_sum_len = 0
    ir_samples = []
    ir_seen = 0
    sample_limit = 10000
    with torch.no_grad():
        for input_ids, attn_mask, orig_len, truncated, is_ir in loader:
            total_blocks += int(orig_len.numel())
            truncated_blocks += int(truncated.sum().item())
            ir_batch = int(is_ir.sum().item())
            if ir_batch:
                ir_blocks += ir_batch
                ir_truncated += int((truncated & is_ir).sum().item())
                orig_len_list = orig_len.tolist()
                is_ir_list = is_ir.tolist()
                for length, flag in zip(orig_len_list, is_ir_list):
                    if not flag:
                        continue
                    ir_sum_len += int(length)
                    ir_seen += 1
                    if len(ir_samples) < sample_limit:
                        ir_samples.append(int(length))
                    else:
                        j = random.randint(0, ir_seen - 1)
                        if j < sample_limit:
                            ir_samples[j] = int(length)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            if use_sentence_transformer:
                outputs = model.forward({
                    "input_ids": input_ids,
                    "attention_mask": attn_mask
                })

                # SentenceTransformer 的输出形式不统一，做兼容处理
                if isinstance(outputs, dict):
                    if "sentence_embedding" in outputs:
                        sent_emb = outputs["sentence_embedding"]
                    elif "embeddings" in outputs:
                        sent_emb = outputs["embeddings"]
                    else:
                        raise RuntimeError(
                            f"SentenceTransformer output missing embedding key. "
                            f"Available keys: {list(outputs.keys())}"
                        )
                elif isinstance(outputs, torch.Tensor):
                    # 极少数模型直接返回 tensor
                    sent_emb = outputs
                else:
                    raise RuntimeError(
                        f"Unexpected SentenceTransformer output type: {type(outputs)}"
                    )

            else:
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                token_embeddings = outputs[0]
                sent_emb = token_embeddings[:, 0]
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outs.append(sent_emb.cpu())
    logger = logging.getLogger(__name__)
    if total_blocks:
        logger.info(
            "Token truncation (>max_length=%d): %d/%d blocks (%.2f%%)",
            max_length,
            truncated_blocks,
            total_blocks,
            100 * truncated_blocks / total_blocks,
        )
    if ir_blocks:
        avg_len = ir_sum_len / max(ir_blocks, 1)
        if ir_samples:
            ir_samples.sort()
            p50 = ir_samples[int((len(ir_samples) - 1) * 0.50)]
            p90 = ir_samples[int((len(ir_samples) - 1) * 0.90)]
            p95 = ir_samples[int((len(ir_samples) - 1) * 0.95)]
            p99 = ir_samples[int((len(ir_samples) - 1) * 0.99)]
            max_len = ir_samples[-1]
            logger.info(
                "ir_function token lengths (context<=%d): avg=%.1f, p50=%d, p90=%d, p95=%d, p99=%d, max=%d (sample=%d)",
                ir_context_tokens,
                avg_len,
                p50,
                p90,
                p95,
                p99,
                max_len,
                len(ir_samples),
            )
        logger.info(
            "ir_function truncation (>max_length=%d): %d/%d blocks (%.2f%%)",
            max_length,
            ir_truncated,
            ir_blocks,
            100 * ir_truncated / ir_blocks,
        )
    return torch.cat(outs, dim=0)


def save_index(
    output_dir: Path,
    embeddings: torch.Tensor,
    blocks: List[Block],
    strategy: str,
    span_ids_map: Optional[Dict[int, List[str]]] = None,
    epic_config: Optional[EpicSplitterConfig] = None,
    function_metadata: Optional[Dict[int, Dict[str, Optional[str]]]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_dir / "embeddings.pt")
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for idx, b in enumerate(blocks):
            metadata = {
                "block_id": idx,
                "file_path": b.file_path,
                "start_line": b.start,
                "end_line": b.end,
                "block_type": b.block_type,
                "strategy": strategy,
            }
            # 如果有span_ids，添加到metadata
            if span_ids_map and idx in span_ids_map:
                metadata["span_ids"] = span_ids_map[idx]

            # epic 策略的额外元数据，便于追溯与复现
            if strategy == "epic":
                cfg = epic_config or EpicSplitterConfig.from_bm25_config()
                metadata.update(
                    {
                        "epic_splitter_version": "1.0",
                        "chunking_config": {
                            "min_chunk_size": cfg.min_chunk_size,
                            "chunk_size": cfg.chunk_size,
                            "max_chunk_size": cfg.max_chunk_size,
                            "hard_token_limit": cfg.hard_token_limit,
                            "max_chunks": cfg.max_chunks,
                        },
                        "context_enhanced": True,
                        "chunk_tokens": len(b.content.split()) if b.content else 0,
                        "created_at": datetime.now().isoformat(),
                    }
                )
            
            # function_level 策略的额外元数据
            if strategy == "function_level" and function_metadata and idx in function_metadata:
                metadata.update({
                    "qualified_name": function_metadata[idx]["qualified_name"],
                    "class_name": function_metadata[idx]["class_name"],
                    "function_name": function_metadata[idx]["function_name"],
                    "function_level_version": "1.0",
                    "created_at": datetime.now().isoformat(),
                })
            
            # ir_function 策略的额外元数据
            if strategy == "ir_function" and function_metadata and idx in function_metadata:
                metadata.update({
                    "qualified_name": function_metadata[idx]["qualified_name"],
                    "class_name": function_metadata[idx]["class_name"],
                    "function_name": function_metadata[idx]["function_name"],
                    "ir_function_version": "1.0",
                    "created_at": datetime.now().isoformat(),
                })

            f.write(json.dumps(metadata) + "\n")


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


def is_sentence_transformer_model(model_name: str) -> bool:
    # HuggingFace Hub 的 ST 命名
    if model_name.startswith("sentence-transformers/"):
        return True

    path = Path(model_name)
    if path.exists():
        if path.is_file():
            path = path.parent

        # 常见的 sentence-transformers 目录标记文件
        markers = [
            "modules.json",
            "config_sentence_transformers.json",
            "_config_sentence_transformers.json",
            "sentence_bert_config.json",
        ]

        # 检查当前目录和父目录
        candidates = [path] + list(path.parents)[:2]
        for c in candidates:
            for m in markers:
                if (c / m).exists():
                    return True

        # 递归搜索子目录（深度2层）
        if path.is_dir():
            for root, dirs, files in os.walk(path):
                for m in markers:
                    if m in files:
                        return True
                # 限制搜索深度
                depth = root.count(os.sep) - str(path).count(os.sep)
                if depth >= 2:
                    dirs.clear()  # 不再深入子目录

    return False


def is_sfr_embedding_model(model_name: str) -> bool:
    """
    检测模型是否是 SFR embedding 模型

    SFR 模型特征：
    - 模型名包含 "SFR" 或 "sfr"
    - 模型名包含 "Salesforce"

    Args:
        model_name: 模型路径或名称

    Returns:
        是否是 SFR embedding 模型
    """
    model_name_lower = model_name.lower()

    # 检查模型名中的关键字
    sfr_keywords = ["sfr", "salesforce"]
    for keyword in sfr_keywords:
        if keyword in model_name_lower:
            return True

    # 如果是本地路径，检查 config.json 中的标识
    path = Path(model_name)
    if path.exists():
        if path.is_file():
            path = path.parent

        # 检查 config.json
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                # 检查模型名称或架构
                model_name_in_config = cfg.get("_name_or_path", "").lower()
                architecture = cfg.get("architectures", [""])[0].lower() if cfg.get("architectures") else ""

                if any(keyword in model_name_in_config or keyword in architecture
                       for keyword in sfr_keywords):
                    return True
            except Exception:
                pass

    return False


def should_enable_trust_remote_code(model_name: str) -> bool:
    path = Path(model_name)
    if not path.exists():
        return False
    if path.is_file():
        path = path.parent
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return "auto_map" in cfg
    except Exception:
        return False


# ============================================================================
# 多进程 Worker
# ============================================================================

def run(rank: int, repo_queue, args, gpu_ids: list, total_repos: int = 0):
    """单个进程的工作函数"""
    import signal
    import atexit
    
    if gpu_ids and not args.force_cpu:
        device_id = gpu_ids[rank % len(gpu_ids)]
        device = torch.device(f"cuda:{device_id}")
        actual_gpu_id = device_id
    else:
        device = torch.device("cpu")
        actual_gpu_id = None
    
    # 资源清理函数
    def cleanup_resources():
        """清理GPU资源"""
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
    
    # 注册清理函数
    atexit.register(cleanup_resources)
    
    # 信号处理：确保进程退出时清理资源
    def signal_handler(signum, frame):
        cleanup_resources()
        exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 设置CUDA设备
    if device.type == 'cuda':
        try:
            torch.cuda.set_device(device)
        except Exception as e:
            print(f'[Process {rank}] Failed to set CUDA device: {e}')
            cleanup_resources()
            return
    
    # 加载模型（每个进程加载一次）
    model = None
    tokenizer = None
    use_sentence_transformer = False
    try:
        print(f'[Process {rank}] Loading model on GPU {actual_gpu_id}...')
        trust_remote_code = getattr(args, 'trust_remote_code', False)
        model_name = resolve_model_name(args.model_name)
        if not trust_remote_code and should_enable_trust_remote_code(model_name):
            trust_remote_code = True
            print(f'[Process {rank}] Enabling trust_remote_code for custom model config.')
        use_sentence_transformer = (is_sentence_transformer_model(model_name) and not getattr(args, 'force_transformers_backend', False)) or getattr(args, 'force_sentence_transformer', False)
        
        # SFR 模型需要特殊处理：强制使用 transformers 后端，避免 SentenceTransformer 的 meta tensor 问题
        # 禁用 SFR 的 SentenceTransformer 路径，统一使用 AutoModel 处理
        use_special_sfr_loading = False
        if use_special_sfr_loading:
            print(f'[Process {rank}] Using transformers backend for SFR model with device_map=None.')
            # 不修改 use_sentence_transformer，让它保持原值（由上面的逻辑决定）

            if not HAS_SENTENCE_TRANSFORMERS:
                raise RuntimeError("sentence-transformers is not installed but model looks like a SentenceTransformer")

            # 关键：不要让 ST 自动把模型搬到 GPU（它会触发 meta -> to 的坑）
            # 先在 CPU 上完整加载，再手动搬到对应 GPU
            try:
                model = SentenceTransformer(model_name, device="cpu", trust_remote_code=trust_remote_code)
            except TypeError as e:
                if "trust_remote_code" not in str(e):
                    raise
                model = SentenceTransformer(model_name, device="cpu")

            tokenizer = model.tokenizer

            # 手动搬运到该进程 GPU，需要处理 meta tensor
            has_meta = any(p.is_meta for p in model.parameters())
            if has_meta:
                print(f'[Process {rank}] SentenceTransformer has meta params; using to_empty().')
                model = model.to_empty(device=device)
            else:
                model = model.to(device)

        else:
            # 在加载 AutoModel 之前，禁用 accelerate 的 meta device，避免 meta tensor 问题
            import os
            os.environ['ACCELERATE_USE_OFFLOAD'] = '0'
            os.environ['ACCELERATE_DEVICE_MAP'] = 'none'
            os.environ['TRANSFORMERS_OFFLOAD'] = '0'
            print(f'[Process {rank}] Disabled accelerate meta device for model loading')
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code,
                    low_cpu_mem_usage=False,  # 禁用 meta tensor 优化
                    device_map=None  # 关键：显式禁用 device_map，避免 SFR 模型内部的 meta device
                )
            except Exception as e:
                if not trust_remote_code and "trust_remote_code" in str(e):
                    trust_remote_code = True
                    print(f'[Process {rank}] Retrying with trust_remote_code=True.')
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    model = AutoModel.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        low_cpu_mem_usage=False,  # 禁用 meta tensor 优化
                        device_map=None  # 关键：显式禁用 device_map，避免 SFR 模型内部的 meta device
                    )
                else:
                    raise
            
            # 立即检查并处理 meta tensor（在加载后立即处理，而不是依赖后面的检查）
            print(f'[Process {rank}] Checking for meta tensors after model loading...')
            has_meta = any(p.is_meta for p in model.parameters())
            if has_meta:
                print(f'[Process {rank}] Meta tensors detected! Using to_empty() to convert to {device}...')
                try:
                    model = model.to_empty(device=device)
                    print(f'[Process {rank}] Successfully converted meta tensors to {device}')
                except Exception as e:
                    print(f'[Process {rank}] to_empty() failed: {e}, trying alternative approach...')
                    # 备选方案：先初始化 meta tensor，再移动
                    try:
                        for name, param in model.named_parameters():
                            if param.is_meta:
                                # 创建一个空 tensor，然后复制到目标设备
                                empty_param = torch.empty_like(param, device='cpu')
                                param.data = empty_param.to(device)
                        model = model.to(device)
                    except Exception as e2:
                        print(f'[Process {rank}] Alternative approach also failed: {e2}')
                        raise RuntimeError(f"Failed to handle meta tensors: {e} (alt: {e2})")
            
            device_map = getattr(model, "hf_device_map", None) or getattr(model, "_hf_device_map", None)
            # 重新检查 meta tensor（可能已经处理了）
            has_meta_after = any(p.is_meta for p in model.parameters())
            if device_map or has_meta_after:
                print(f'[Process {rank}] Model uses device_map/meta params; handling meta tensor...')
                # 处理 meta tensor：使用 to_empty 方法
                if has_meta:
                    try:
                        print(f'[Process {rank}] Converting meta tensors to {device} using to_empty()...')
                        model = model.to_empty(device=device)
                    except Exception as e:
                        print(f'[Process {rank}] Warning: to_empty() failed: {e}, trying .to()')
                        model = model.to(device)
                elif device_map:
                    # 只有 device_map 但没有 meta tensor，跳过移动
                    print(f'[Process {rank}] Model has device_map, skipping .to()')
                else:
                    model = model.to(device)
            else:
                model = model.to(device)
        model.eval()
        print(f'[Process {rank}] Model loaded on GPU {actual_gpu_id}.')
    except Exception as e:
        print(f'[Process {rank}] Failed to load model: {e}')
        cleanup_resources()
        return
    
    # 获取Graph索引目录（如果提供）
    graph_index_dir = getattr(args, 'graph_index_dir', None)

    processed = 0
    skipped = 0
    failed = 0
    
    # 每个进程都有自己的进度条
    pbar = tqdm(
        total=None,  # 动态更新总数
        desc=f"GPU {actual_gpu_id if actual_gpu_id is not None else 'CPU'}",
        position=rank,
        leave=True,
        ncols=120,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )

    while True:
        try:
            repo_name = repo_queue.get_nowait()
        except Exception:
            # Queue is empty
            break

        output_dir = Path(args.index_dir) / repo_name
        output_file = output_dir / "embeddings.pt"
        
        if output_file.exists():
            skipped += 1
            pbar.total = processed + skipped + failed + 1  # 动态更新总数
            pbar.update(1)
            pbar.set_postfix({
                "done": processed, 
                "skip": skipped, 
                "fail": failed, 
                "cur": repo_name[:25]
            })
            continue

        repo_dir = osp.join(args.repo_path, repo_name)
        if not osp.isdir(repo_dir):
            skipped += 1
            pbar.total = processed + skipped + failed + 1
            pbar.update(1)
            pbar.set_postfix({
                "done": processed, 
                "skip": skipped, 
                "fail": failed, 
                "cur": repo_name[:25]
            })
            continue

        pbar.set_postfix({
            "done": processed, 
            "skip": skipped, 
            "fail": failed, 
            "cur": repo_name[:25]
        })
        
        try:
            function_metadata = {}
            if args.strategy == "epic":
                epic_cfg = EpicSplitterConfig.from_bm25_config()
                blocks = collect_epic_blocks(repo_dir, config=epic_cfg)
            elif args.strategy == "function_level":
                blocks, function_metadata = collect_function_blocks(repo_dir)
            elif args.strategy == "ir_function":
                blocks, function_metadata = collect_ir_function_blocks(repo_dir)
            elif args.strategy == "langchain_fixed":
                blocks = collect_langchain_fixed_blocks(
                    repo_dir,
                    chunk_size=getattr(args, 'langchain_chunk_size', 1000),
                    chunk_overlap=getattr(args, 'langchain_chunk_overlap', 200),
                )
            elif args.strategy == "langchain_recursive":
                blocks = collect_langchain_recursive_blocks(
                    repo_dir,
                    chunk_size=getattr(args, 'langchain_chunk_size', 1000),
                    chunk_overlap=getattr(args, 'langchain_chunk_overlap', 200),
                )
            elif args.strategy == "langchain_token":
                blocks = collect_langchain_token_blocks(
                    repo_dir,
                    chunk_size=getattr(args, 'langchain_chunk_size', 1000),
                    chunk_overlap=getattr(args, 'langchain_chunk_overlap', 200),
                )
            elif args.strategy == "llamaindex_code":
                blocks = collect_llamaindex_code_blocks(
                    repo_dir,
                    language=getattr(args, 'llamaindex_language', 'python'),
                    chunk_lines=getattr(args, 'llamaindex_chunk_lines', 40),
                    chunk_lines_overlap=getattr(args, 'llamaindex_chunk_lines_overlap', 15),
                    max_chars=getattr(args, 'llamaindex_max_chars', 1500),
                )
            elif args.strategy == "llamaindex_sentence":
                blocks = collect_llamaindex_sentence_blocks(
                    repo_dir,
                    chunk_size=getattr(args, 'llamaindex_chunk_size', 1024),
                    chunk_overlap=getattr(args, 'llamaindex_chunk_overlap', 200),
                )
            elif args.strategy == "llamaindex_token":
                blocks = collect_llamaindex_token_blocks(
                    repo_dir,
                    chunk_size=getattr(args, 'llamaindex_chunk_size', 1024),
                    chunk_overlap=getattr(args, 'llamaindex_chunk_overlap', 20),
                    separator=getattr(args, 'llamaindex_separator', ' '),
                )
            elif args.strategy == "llamaindex_semantic":
                blocks = collect_llamaindex_semantic_blocks(
                    repo_dir,
                    buffer_size=getattr(args, 'llamaindex_buffer_size', 3),
                    model_name=getattr(args, 'llamaindex_embed_model', 'sentence-transformers/all-MiniLM-L6-v2'),
                )
            else:
                blocks = collect_blocks(
                    repo_dir, 
                    args.strategy, 
                    args.block_size, 
                    args.window_size, 
                    args.slice_size
                )
            if not blocks:
                print(f'\n[Process {rank}] ⚠️  Warning: No blocks generated for {repo_name}, skipping.')
                skipped += 1
                pbar.total = processed + skipped + failed + 1
                pbar.update(1)
                pbar.set_postfix({
                    "done": processed, 
                    "skip": skipped, 
                    "fail": failed, 
                    "cur": repo_name[:25]
                })
                continue
            
            embeddings = embed_blocks(
                blocks, model, tokenizer, 
                args.max_length, args.batch_size, device,
                ir_context_tokens=args.ir_function_context_tokens,
                use_sentence_transformer=use_sentence_transformer,
            )
            
            # 清理中间变量，释放内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 如果提供了Graph索引目录，尝试提取span_ids
            span_ids_map = {}
            if graph_index_dir and not HAS_GRAPH_INDEX:
                print(f'[Process {rank}] Warning: Graph index support unavailable: {_GRAPH_INDEX_IMPORT_ERROR}')
                graph_index_dir = None

            if graph_index_dir:
                graph_index_file = osp.join(graph_index_dir, f"{repo_name}.pkl")
                if osp.exists(graph_index_file):
                    try:
                        span_ids_map = extract_span_ids_from_graph(
                            blocks, graph_index_file, repo_dir
                        )
                        if span_ids_map:
                            print(f'[Process {rank}] Extracted span_ids for {len(span_ids_map)}/{len(blocks)} blocks in {repo_name}')
                    except Exception as e:
                        print(f'[Process {rank}] Warning: Failed to extract span_ids for {repo_name}: {e}')
            
            # 对于 epic 策略，传入 epic 配置以便在 metadata 中记录
            epic_cfg_for_save = epic_cfg if "epic_cfg" in locals() else None
            save_index(output_dir, embeddings, blocks, args.strategy, span_ids_map, 
                      epic_config=epic_cfg_for_save, function_metadata=function_metadata)
            processed += 1
            pbar.total = processed + skipped + failed + 1
            pbar.update(1)
            pbar.set_postfix({
                "done": processed, 
                "skip": skipped, 
                "fail": failed, 
                "cur": repo_name[:25]
            })
        except Exception as e:
            print(f'\n[Process {rank}] ✗ Error {repo_name}: {e}')
            # 清理GPU缓存，避免内存累积
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            failed += 1
            pbar.total = processed + skipped + failed + 1
            pbar.update(1)
            pbar.set_postfix({
                "done": processed, 
                "skip": skipped, 
                "fail": failed, 
                "cur": repo_name[:25]
            })

    pbar.close()
    
    # 显式清理资源
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    cleanup_resources()
    
    print(f'[Process {rank}] Done. Processed: {processed}, Skipped: {skipped}, Failed: {failed}')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="批量构建稠密索引（多进程并行）")
    
    # 数据源参数
    parser.add_argument("--dataset", type=str, default="",
                        help="HuggingFace 数据集名称（如 czlll/Loc-Bench_V1）")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集分片")
    parser.add_argument("--repo_path", type=str, default="/workspace/locbench/repos/locbench_repos",
                        help="本地仓库根目录")
    parser.add_argument("--index_dir", type=str, default="/workspace/locbench/IR-based/new_index_data",
                        help="索引输出基目录")
    parser.add_argument("--instance_id_path", type=str, default="",
                        help="指定实例 ID 列表文件（JSON 数组）")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="models/rlretriever",
                        help="模型路径或 HuggingFace 模型名")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="允许执行模型仓库中的自定义代码（CodeRankEmbed 等模型需要）")
    parser.add_argument("--force_transformers_backend", action="store_true",
                        help="强制使用 transformers 后端（AutoModel）而不是 SentenceTransformer，避免某些模型的 meta tensor 问题")
    parser.add_argument("--force_sentence_transformer", action="store_true",
                        help="强制使用 SentenceTransformer 后端，用于明确指定 SentenceTransformer 模型")

    # 切块策略参数
    parser.add_argument(
        "--strategy",
        choices=["fixed", "sliding", "rl_fixed", "rl_mini", "epic", "function_level", "ir_function",
                 "langchain_fixed", "langchain_recursive", "langchain_token",
                 "llamaindex_code", "llamaindex_sentence", "llamaindex_token", "llamaindex_semantic"],
        default="fixed",
        help="切块策略（epic 使用与 BM25 相同的 EpicSplitter chunking，function_level 使用AST提取函数和类方法，ir_function 使用论文 IR-based 方法（模块级代码和类属性冗余复制到函数），langchain_fixed 使用 CharacterTextSplitter，langchain_recursive 使用 RecursiveCharacterTextSplitter，langchain_token 使用 TokenTextSplitter，llamaindex_code 使用 CodeSplitter，llamaindex_sentence 使用 SentenceSplitter，llamaindex_token 使用 LlamaIndex TokenTextSplitter，llamaindex_semantic 使用 SemanticSplitterNodeParser）",
    )
    parser.add_argument("--block_size", type=int, default=15,
                        help="固定切块大小（strategy=fixed）")
    parser.add_argument("--window_size", type=int, default=20,
                        help="滑动窗口大小（strategy=sliding）")
    parser.add_argument("--slice_size", type=int, default=2,
                        help="滑动步长因子（strategy=sliding）")
    
    # LangChain 分割器通用参数（可选）
    parser.add_argument("--langchain_chunk_size", type=int, default=1000,
                        help="LangChain splitters: 目标块大小（字符数或token数）")
    parser.add_argument("--langchain_chunk_overlap", type=int, default=200,
                        help="LangChain splitters: 块之间的重叠（字符数或token数）")
    
    # LlamaIndex CodeSplitter 参数（可选）
    parser.add_argument("--llamaindex_language", type=str, default="python",
                        help="CodeSplitter: 编程语言")
    parser.add_argument("--llamaindex_chunk_lines", type=int, default=40,
                        help="CodeSplitter: 每个块的代码行数")
    parser.add_argument("--llamaindex_chunk_lines_overlap", type=int, default=15,
                        help="CodeSplitter: 块之间的重叠行数")
    parser.add_argument("--llamaindex_max_chars", type=int, default=1500,
                        help="CodeSplitter: 每个块的最大字符数")
    
    # LlamaIndex SentenceSplitter 和 TokenTextSplitter 通用参数（可选）
    parser.add_argument("--llamaindex_chunk_size", type=int, default=1024,
                        help="LlamaIndex splitters: 目标块大小")
    parser.add_argument("--llamaindex_chunk_overlap", type=int, default=200,
                        help="LlamaIndex splitters: 块之间的重叠")
    parser.add_argument("--llamaindex_separator", type=str, default=" ",
                        help="TokenTextSplitter: 分隔符")
    
    # LlamaIndex SemanticSplitterNodeParser 参数（可选）
    parser.add_argument("--llamaindex_buffer_size", type=int, default=3,
                        help="SemanticSplitterNodeParser: 缓冲区大小")
    parser.add_argument("--llamaindex_embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SemanticSplitterNodeParser: HuggingFace embedding 模型名称")

    # ir_function 上下文控制参数
    parser.add_argument("--ir_function_context_tokens", type=int, default=1024,
                        help="ir_function: 上下文（文件/类/模块级/类属性）token 上限")
    
    # 编码参数
    parser.add_argument("--max_length", type=int, default=2024,
                        help="最大 token 长度")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批量大小")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="日志级别（DEBUG, INFO, WARNING, ERROR）")
    
    # 并行参数
    parser.add_argument("--num_processes", type=int, default=1,
                        help="并行进程数（建议与可用 GPU 数量一致）")
    parser.add_argument("--gpu_ids", type=str, default="",
                        help="指定使用的 GPU ID，逗号分隔（如 '4,5,6,7'）")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用 CPU")
    parser.add_argument(
        "--graph_index_dir",
        type=str,
        default=None,
        help="Graph索引目录，如果提供将在metadata中添加span_ids（从环境变量GRAPH_INDEX_DIR获取默认值）"
    )
    
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    # 规范化模型路径：如果是相对路径且存在，转换为绝对路径
    # 这确保在多进程环境下（spawn模式）所有子进程使用相同的绝对路径
    args.model_name = resolve_model_name(args.model_name)
    
    # 如果未提供graph_index_dir，尝试从环境变量获取
    if not args.graph_index_dir:
        args.graph_index_dir = os.environ.get("GRAPH_INDEX_DIR")
    
    # 解析 GPU ID
    if args.gpu_ids:
        args.gpu_id_list = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        args.gpu_id_list = []

    # 确定索引输出目录
    if args.dataset:
        dataset_name = args.dataset.split('/')[-1]
        args.index_dir = f'{args.index_dir}/{dataset_name}/dense_index_{args.strategy}/'
    else:
        args.index_dir = f'{args.index_dir}/dense_index_{args.strategy}/'
    os.makedirs(args.index_dir, exist_ok=True)

    # 获取仓库列表
    if args.dataset and HAS_DATASETS:
        # 从数据集获取仓库列表
        print(f"Loading dataset: {args.dataset}")
        bench_data = load_dataset(args.dataset, split=args.split)
        
        # 提取唯一的 repo_name
        repo_names = set()
        for instance in bench_data:
            instance_id = instance.get('instance_id', '')
            repo_name = instance_id_to_repo_name(instance_id)
            if repo_name:
                repo_names.add(repo_name)
        repo_folders = sorted(list(repo_names))
        print(f"Found {len(repo_folders)} unique repos from dataset")
    elif args.instance_id_path and osp.exists(args.instance_id_path):
        # 从文件加载实例 ID
        with open(args.instance_id_path, 'r') as f:
            instance_ids = json.loads(f.read())
        repo_folders = sorted(set(instance_id_to_repo_name(iid) for iid in instance_ids))
        print(f"Found {len(repo_folders)} repos from instance_id_path")
    else:
        # 从本地目录列出仓库
        repo_folders = list_folders(args.repo_path)
        print(f"Found {len(repo_folders)} repos in {args.repo_path}")

    if not repo_folders:
        print("No repos found!")
        return

    # 确定使用的 GPU
    if args.gpu_id_list:
        gpu_ids = args.gpu_id_list
    elif torch.cuda.is_available() and not args.force_cpu:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []

    # 打印配置
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.index_dir}")
    print(f"Processes: {args.num_processes}")
    print(f"GPUs: {gpu_ids if gpu_ids else 'CPU'}")
    print(f"Repos: {len(repo_folders)}")
    print("=" * 60)

    # 创建共享队列
    manager = mp.Manager()
    queue = manager.Queue()
    for repo in repo_folders:
        queue.put(repo)

    start_time = time.time()
    total_repos = len(repo_folders)
    
    # 创建总进度条
    total_pbar = tqdm(
        total=total_repos,
        desc="总进度",
        position=args.num_processes,  # 放在所有进程进度条下方
        leave=True,
        ncols=120,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    )
    
    def count_completed_repos():
        """统计已完成的仓库数量"""
        index_path = Path(args.index_dir)
        if not index_path.exists():
            return 0
        count = 0
        for repo_dir in index_path.iterdir():
            if repo_dir.is_dir() and (repo_dir / "embeddings.pt").exists():
                count += 1
        return count
    
    def monitor_progress():
        """监控总进度"""
        while True:
            completed = count_completed_repos()
            total_pbar.n = completed
            total_pbar.set_postfix({
                "已完成": completed, 
                "总计": total_repos,
                "剩余": total_repos - completed
            })
            total_pbar.refresh()
            if completed >= total_repos:
                break
            time.sleep(2)  # 每2秒更新一次
    
    if args.num_processes == 1:
        # 单进程模式
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        run(0, queue, args, gpu_ids, total_repos)
        # 等待监控线程完成
        monitor_thread.join(timeout=1)
    else:
        # 多进程模式
        # 启动总进度监控线程
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        # 使用 spawn 启动多进程，每个进程分配一个 GPU
        try:
            mp.spawn(
                run,
                nprocs=args.num_processes,
                args=(queue, args, gpu_ids, total_repos),
                join=True
            )
        except KeyboardInterrupt:
            print("\n[Main] Interrupted by user, cleaning up...")
            # 子进程会通过信号处理自动清理
        except Exception as e:
            print(f"\n[Main] Error in multiprocessing: {e}")
            raise
        
        # 等待监控线程完成
        monitor_thread.join(timeout=1)
    
    # 最终更新总进度
    final_completed = count_completed_repos()
    total_pbar.n = final_completed
    total_pbar.set_postfix({
        "已完成": final_completed, 
        "总计": total_repos,
        "剩余": total_repos - final_completed
    })
    total_pbar.close()

    end_time = time.time()
    elapsed = end_time - start_time
    print("=" * 60)
    print(f'Total Execution time = {elapsed:.1f}s ({elapsed/60:.1f} min)')


if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    main()
