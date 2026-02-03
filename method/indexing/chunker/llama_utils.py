from typing import Tuple


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

    try:
        import re

        header = "\n".join(original_content.splitlines()[:50])
        class_match = re.search(r"class\s+(\w+)", header)
        if class_match:
            context_parts.append(f"Class: {class_match.group(1)}")
    except Exception:
        pass

    if start_line > 0:
        context_parts.append(f"Lines: {start_line}-")

    context_parts.append("")
    context_parts.append("Code:")
    context_parts.append(original_content)

    return "\n".join(context_parts)


def _calculate_chunk_line_numbers(
    original_content: str,
    chunk_content: str,
    previous_chunk_end: int = 0,
) -> Tuple[int, int]:
    """
    计算 chunk 在原文件中的行号范围，基于文本匹配
    """
    chunk_pos = original_content.find(chunk_content, previous_chunk_end)
    if chunk_pos >= 0:
        start_line = original_content[:chunk_pos].count("\n")
        end_line = start_line + chunk_content.count("\n")
        return start_line, end_line

    chunk_lines = chunk_content.splitlines()
    if not chunk_lines:
        return 0, 0

    first_line = chunk_lines[0].strip()
    last_line = chunk_lines[-1].strip()

    if not first_line and not last_line:
        return previous_chunk_end // 80, previous_chunk_end // 80

    original_lines = original_content.splitlines()
    start_line = None
    end_line = None

    for i, line in enumerate(original_lines):
        if first_line and first_line in line:
            start_line = i
            break

    if last_line:
        for i in range(len(original_lines) - 1, -1, -1):
            if last_line in original_lines[i]:
                end_line = i
                break

    if start_line is not None and end_line is not None:
        return start_line, end_line
    if start_line is not None:
        return start_line, start_line + len(chunk_lines) - 1
    if end_line is not None:
        return max(0, end_line - len(chunk_lines) + 1), end_line

    estimated_start = previous_chunk_end // 80
    return estimated_start, estimated_start + len(chunk_lines) - 1


def _get_or_calculate_line_numbers(
    node,
    original_content: str,
    previous_chunk_end: int = 0,
) -> Tuple[int, int]:
    """
    从 node metadata 获取行号，如果不存在则计算
    """
    metadata_start = node.metadata.get('start_line')
    metadata_end = node.metadata.get('end_line')

    if metadata_start is not None and metadata_start >= 0:
        if metadata_end is not None and metadata_end >= metadata_start:
            return metadata_start, metadata_end
        chunk_content = node.get_content()
        return metadata_start, metadata_start + len(chunk_content.splitlines()) - 1

    chunk_content = node.get_content()
    return _calculate_chunk_line_numbers(
        original_content, chunk_content, previous_chunk_end
    )
