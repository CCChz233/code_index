import ast
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from method.indexing.core.block import Block, ChunkResult
from method.indexing.chunker.base import BaseChunker
from method.indexing.chunker.basic import blocks_fixed_lines
from method.indexing.utils.file import iter_files


# ============================================================================
# IR-based 函数级分块（论文复现）
# ============================================================================


_AST_TIMEOUT_SEC = 2.0
_AST_MAX_RECURSION = 10000


class _AstTimeoutError(Exception):
    pass


@contextmanager
def _ast_timeout(timeout_sec: float):
    try:
        import signal  # Unix-only
    except Exception:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum, _frame):
        raise _AstTimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _parse_ast_safe(text: str) -> Optional[ast.AST]:
    # ast.parse 不接受含 null 字节的源码，直接跳过并避免刷屏日志
    if "\x00" in text:
        logging.debug("AST parse skipped: source contains null bytes")
        return None
    old_limit = sys.getrecursionlimit()
    if _AST_MAX_RECURSION and _AST_MAX_RECURSION > old_limit:
        sys.setrecursionlimit(_AST_MAX_RECURSION)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            with _ast_timeout(_AST_TIMEOUT_SEC):
                return ast.parse(text)
    except (_AstTimeoutError, RecursionError, SyntaxError) as e:
        logging.warning(f"AST parse failed (skipped): {e}")
        return None
    except Exception as e:
        logging.warning(f"AST parse failed (skipped): {e}")
        return None
    finally:
        if _AST_MAX_RECURSION and _AST_MAX_RECURSION > old_limit:
            sys.setrecursionlimit(old_limit)


def _extract_module_level_code(tree: ast.AST, lines: List[str]) -> str:
    """
    提取模块级代码：imports, 全局变量, 顶层赋值等

    按照论文：这些代码会被冗余复制到该文件每个函数的表示中
    """
    module_level_parts = []

    for node in tree.body:
        # import 语句
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            module_level_parts.append(code.strip())

        # 全局变量赋值
        elif isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            module_level_parts.append(code.strip())

        # 带类型注解的赋值
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
    """
    class_attr_parts = []

    for node in class_node.body:
        if isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
            code = " ".join(lines[start:end])
            class_attr_parts.append(code.strip())
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

    格式: {file_path} {class_name} {module_level_code} {class_attributes} {function_code}
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

    if module_level_code.strip():
        parts.append(module_level_code.strip())

    if class_attributes.strip():
        parts.append(class_attributes.strip())

    return " ".join(parts)


def blocks_ir_function(text: str, rel: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    按照论文 IR-based 方法提取函数级代码块（Paper Reproduction）
    """
    blocks: List[Block] = []
    function_metadata: Dict[int, Dict[str, Optional[str]]] = {}

    if not rel.endswith('.py'):
        return blocks, function_metadata

    if len(text) > 10 * 1024 * 1024:
        logging.debug(f"File {rel} is too large, skipping")
        return blocks, function_metadata

    try:
        tree = _parse_ast_safe(text)
        if tree is None:
            return blocks, function_metadata
        lines = text.splitlines()

        module_level_code = _extract_module_level_code(tree, lines)

        def get_function_code(node) -> str:
            func_start = node.lineno - 1
            func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 10

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
            nonlocal blocks, function_metadata

            if function_stack is None:
                function_stack = []

            function_name = node.name
            function_stack.append(function_name)

            start_line = node.lineno - 1
            end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10

            if node.decorator_list:
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, 'lineno'):
                    start_line = first_decorator.lineno - 1

            func_code = get_function_code(node)
            if not func_code.strip():
                function_stack.pop()
                return

            if class_name:
                if len(function_stack) > 1:
                    qualified_name = f"{rel}::{class_name}::{'::'.join(function_stack)}"
                else:
                    qualified_name = f"{rel}::{class_name}::{function_name}"
            else:
                if len(function_stack) > 1:
                    qualified_name = f"{rel}::{'::'.join(function_stack)}"
                else:
                    qualified_name = f"{rel}::{function_name}"

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

            function_metadata[block_index] = {
                "qualified_name": qualified_name,
                "class_name": class_name,
                "function_name": function_name,
            }

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    process_function(child, class_name, class_attributes, function_stack.copy())

            function_stack.pop()

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_attributes = _extract_class_attributes(node, lines)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        process_function(child, node.name, class_attributes, [])
                    elif isinstance(child, ast.ClassDef):
                        nested_class_attrs = _extract_class_attributes(child, lines)
                        for nested_child in child.body:
                            if isinstance(nested_child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                process_function(
                                    nested_child,
                                    f"{node.name}.{child.name}",
                                    nested_class_attrs,
                                    [],
                                )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
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
    """
    logger = logging.getLogger(__name__)
    repo_root = Path(repo_path)
    all_blocks: List[Block] = []
    all_function_metadata: Dict[int, Dict[str, Optional[str]]] = {}

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

        try:
            file_size = p.stat().st_size
            if file_size > 10 * 1024 * 1024:
                logger.debug(f"Skipping large file {p.name}")
                stats['skipped'] += 1
                continue
        except Exception:
            stats['skipped'] += 1
            continue

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

        file_blocks, file_metadata = blocks_ir_function(text, rel)

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
# function_level 分块（含 fallback）
# ============================================================================


def build_function_context_content(
    file_path: str,
    class_name: Optional[str],
    function_name: str,
    qualified_name: str,
    original_code: str,
    start_line: int,
    end_line: int,
) -> str:
    """
    构建超简洁的上下文（论文风格）

    格式: # qualified_name\n\n{code}
    """
    header = f"# {qualified_name}\n\n"
    return header + original_code


def blocks_function_level_with_fallback(
    text: str,
    rel: str,
    block_size: int = 15,
) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    函数级切块 + 未覆盖代码的 fixed 切块作为补充
    """
    function_blocks, function_metadata = blocks_by_function(text, rel)
    function_metadata_by_block = {
        function_blocks[idx]: metadata for idx, metadata in function_metadata.items()
    }

    if not rel.endswith('.py') or not function_blocks:
        fixed_blocks = blocks_fixed_lines(text, rel, block_size)
        for b in fixed_blocks:
            b.block_type = "function_level_fallback"
        return fixed_blocks, {}

    lines = text.splitlines()
    total_lines = len(lines)
    covered_lines = set()

    for block in function_blocks:
        for line_no in range(block.start, block.end + 1):
            covered_lines.add(line_no)

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

    if current_start is not None:
        uncovered_ranges.append((current_start, total_lines - 1))

    fallback_blocks = []
    for range_start, range_end in uncovered_ranges:
        range_lines = lines[range_start:range_end + 1]
        range_text = "\n".join(range_lines)

        if not any(line.strip() for line in range_lines):
            continue

        chunk_start = 0
        while chunk_start < len(range_lines):
            chunk_end = min(chunk_start + block_size, len(range_lines))
            chunk_lines = range_lines[chunk_start:chunk_end]

            if any(line.strip() for line in chunk_lines):
                real_start = range_start + chunk_start
                real_end = range_start + chunk_end - 1
                content = "\n".join(chunk_lines)

                fallback_blocks.append(
                    Block(rel, real_start, real_end, content, "function_level_fallback")
                )

            chunk_start = chunk_end

    all_blocks = function_blocks + fallback_blocks
    all_blocks.sort(key=lambda b: b.start)

    sorted_metadata: Dict[int, Dict[str, Optional[str]]] = {}
    for idx, block in enumerate(all_blocks):
        metadata = function_metadata_by_block.get(block)
        if metadata is not None:
            sorted_metadata[idx] = metadata

    return all_blocks, sorted_metadata


def blocks_by_function(text: str, rel: str) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    使用AST解析提取所有函数和类方法
    """
    blocks: List[Block] = []
    function_metadata: Dict[int, Dict[str, Optional[str]]] = {}

    if not rel.endswith('.py'):
        return blocks, function_metadata

    if len(text) > 10 * 1024 * 1024:
        logging.debug(f"File {rel} is too large ({len(text) / 1024 / 1024:.2f}MB), skipping")
        return blocks, function_metadata

    try:
        tree = _parse_ast_safe(text)
        if tree is None:
            return blocks, function_metadata
        lines = text.splitlines()

        class_stack: List[str] = []
        function_stack: List[str] = []

        def get_function_code(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            func_start = node.lineno - 1
            func_end = node.end_lineno - 1 if hasattr(node, 'end_lineno') else func_start + 10

            decorator_start = func_start
            if node.decorator_list:
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, 'lineno'):
                    decorator_start = first_decorator.lineno - 1

            all_lines = lines[decorator_start:func_end + 1]
            return "\n".join(all_lines)

        def visit_node(node: ast.AST):
            nonlocal blocks, function_metadata

            if isinstance(node, ast.ClassDef):
                class_stack.append(node.name)
                for child in node.body:
                    visit_node(child)
                class_stack.pop()

            elif isinstance(node, ast.FunctionDef):
                function_name = node.name
                function_stack.append(function_name)

                start_line = node.lineno - 1
                end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10
                if node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    if hasattr(first_decorator, 'lineno'):
                        start_line = min(start_line, first_decorator.lineno - 1)

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

            elif isinstance(node, ast.AsyncFunctionDef):
                function_name = node.name
                function_stack.append(function_name)

                start_line = node.lineno - 1
                end_line = node.end_lineno - 1 if hasattr(node, 'end_lineno') else start_line + 10
                if node.decorator_list:
                    first_decorator = node.decorator_list[0]
                    if hasattr(first_decorator, 'lineno'):
                        start_line = min(start_line, first_decorator.lineno - 1)

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

            else:
                if hasattr(node, 'body'):
                    for child in node.body:
                        visit_node(child)

        for node in tree.body:
            visit_node(node)

    except SyntaxError as e:
        logging.debug(f"Syntax error in {rel} (line {e.lineno}): {e.msg}. Skipping file.")
        return blocks, function_metadata
    except (ValueError, AttributeError) as e:
        logging.warning(f"AST parsing error in {rel}: {e}. Skipping file.")
        return blocks, function_metadata
    except MemoryError as e:
        logging.error(f"Memory error processing {rel}: {e}. Skipping file.")
        return blocks, function_metadata
    except Exception as e:
        logging.error(f"Unexpected error processing {rel}: {type(e).__name__}: {e}. Skipping file.")
        return blocks, function_metadata

    return blocks, function_metadata


def collect_function_blocks(repo_path: str, block_size: int = 15) -> Tuple[List[Block], Dict[int, Dict[str, Optional[str]]]]:
    """
    收集函数级代码块（带 fallback）
    """
    logger = logging.getLogger(__name__)
    repo_root = Path(repo_path)
    all_blocks: List[Block] = []
    all_function_metadata: Dict[int, Dict[str, Optional[str]]] = {}

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
            logger.debug(
                f"Processed {stats['processed']}/{total_files} files in {repo_path} "
                f"(skipped: {stats['skipped']}, errors: {stats['errors']})"
            )

        try:
            file_size = p.stat().st_size
            if file_size > 10 * 1024 * 1024:
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

        try:
            file_blocks, file_metadata = blocks_function_level_with_fallback(text, rel, block_size)

            current_start_idx = len(all_blocks)
            for local_idx, metadata in file_metadata.items():
                global_idx = current_start_idx + local_idx
                all_function_metadata[global_idx] = metadata

            all_blocks.extend(file_blocks)

            if file_blocks:
                stats['files_with_blocks'] += 1
                stats['total_blocks'] += len(file_blocks)
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

    if not all_blocks:
        logger.warning(f"No blocks extracted from {repo_path}")
        return all_blocks, all_function_metadata

    return all_blocks, all_function_metadata


class IRFunctionChunker(BaseChunker):
    block_type = "ir_function"

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks, metadata = collect_ir_function_blocks(repo_path)
        return ChunkResult(blocks=blocks, metadata=metadata, aux={})


class FunctionLevelChunker(BaseChunker):
    block_type = "function_level"

    def __init__(self, block_size: int = 15):
        self.block_size = block_size
        self._logger = logging.getLogger(__name__)

    def chunk(self, repo_path: str) -> ChunkResult:
        blocks, metadata = collect_function_blocks(repo_path, self.block_size)
        return ChunkResult(blocks=blocks, metadata=metadata, aux={})
