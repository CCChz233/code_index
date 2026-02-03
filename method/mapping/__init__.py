"""
映射模块

提供两种映射方式：
- GraphBasedMapper: 使用Graph索引+span_ids
- ASTBasedMapper: 使用AST解析（运行时）
"""

from method.mapping.base import BlockMapper
from method.mapping.graph_based.mapper import GraphBasedMapper
from method.mapping.ast_based.mapper import ASTBasedMapper

__all__ = [
    'BlockMapper',
    'GraphBasedMapper',
    'ASTBasedMapper',
]

