"""
基于AST解析的映射器

运行时解析源代码，不依赖Graph索引。
"""

import ast
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from method.mapping.base import BlockMapper
from method.utils import instance_id_to_repo_name

logger = logging.getLogger(__name__)


def build_line_to_entity_map(file_path: str, repo_root: str) -> Dict[int, Tuple[str, Optional[str]]]:
    """
    为单个Python文件构建行号到实体的映射表
    
    使用AST解析，不依赖Graph索引，实现简单的行号到函数/类的映射。
    
    Args:
        file_path: 文件路径（相对于 repo_root）
        repo_root: 仓库根目录
    
    Returns:
        Dict[int, Tuple[str, Optional[str]]]: 
            - key: 行号（1-indexed）
            - value: (entity_id, class_id)
                - entity_id: 函数 ID，格式 "file_path:function_name" 或 "file_path:ClassName.method_name"
                - class_id: 类 ID，格式 "file_path:ClassName"（如果函数在类中），否则 None
    
    Example:
        {
            10: ("file.py:my_function", None),
            25: ("file.py:MyClass.method", "file.py:MyClass"),
            30: ("file.py:MyClass.another_method", "file.py:MyClass"),
        }
    """
    full_path = os.path.join(repo_root, file_path)
    if not os.path.exists(full_path):
        return {}
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        tree = ast.parse(source_code, filename=full_path)
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
        logger.debug(f"Failed to parse {full_path}: {e}")
        return {}
    
    line_map = {}
    
    class EntityCollector(ast.NodeVisitor):
        def __init__(self):
            self.class_stack = []  # 存储嵌套的类名
            self.function_stack = []  # 存储嵌套的函数名（当前未使用，但保留以备将来扩展）
        
        def visit_ClassDef(self, node):
            class_name = node.name
            full_class_name = '.'.join(self.class_stack + [class_name])
            class_id = f"{file_path}:{full_class_name}"
            
            self.class_stack.append(class_name)
            
            # 记录类的所有行号（类作为模块，但不作为实体）
            # 使用 None 作为 entity_id，表示这不是一个函数实体
            end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else node.lineno
            for line_num in range(node.lineno, end_line + 1):
                line_map[line_num] = (None, class_id)  # 类作为模块，但不是实体
            
            self.generic_visit(node)
            self.class_stack.pop()
        
        def visit_FunctionDef(self, node):
            self._visit_function(node)
        
        def visit_AsyncFunctionDef(self, node):
            self._visit_function(node)
        
        def _visit_function(self, node):
            function_name = node.name
            full_function_name = '.'.join(self.function_stack + [function_name])
            
            # 构建函数 ID
            if self.class_stack:
                # 类方法
                class_name = '.'.join(self.class_stack)
                full_name = f"{class_name}.{function_name}"
                entity_id = f"{file_path}:{full_name}"
                class_id = f"{file_path}:{class_name}"
            else:
                # 顶级函数
                entity_id = f"{file_path}:{full_function_name}"
                class_id = None
            
            # 记录函数的所有行号
            end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else node.lineno
            for line_num in range(node.lineno, end_line + 1):
                line_map[line_num] = (entity_id, class_id)
            
            self.function_stack.append(function_name)
            self.generic_visit(node)
            self.function_stack.pop()
    
    collector = EntityCollector()
    try:
        collector.visit(tree)
    except RecursionError as e:
        logger.warning(f"RecursionError when parsing AST for {full_path}. "
                      f"File may have very deep nesting (likely in expressions/comprehensions). "
                      f"Returning partial mapping. Error: {e}")
        # 返回已收集的部分映射，而不是空字典
        return line_map
    
    return line_map


class ASTBasedMapper(BlockMapper):
    """基于AST解析的映射器"""
    
    def __init__(self, repos_root: str):
        """
        初始化AST映射器
        
        Args:
            repos_root: 代码仓库根目录
        """
        self.repos_root = repos_root
        self._line_map_cache = {}  # 缓存映射表，避免重复解析
    
    def _get_line_map(self, file_path: str, repo_root: str):
        """获取或缓存行号映射表"""
        cache_key = f"{repo_root}:{file_path}"
        if cache_key not in self._line_map_cache:
            self._line_map_cache[cache_key] = build_line_to_entity_map(file_path, repo_root)
        return self._line_map_cache[cache_key]
    
    def map_blocks_to_entities(
        self,
        blocks: List[Dict[str, Any]],
        instance_id: str,
        top_k_modules: int = 10,
        top_k_entities: int = 50,
    ) -> Tuple[List[str], List[str]]:
        """
        使用AST解析映射代码块到实体
        
        Args:
            blocks: 代码块列表，每个包含 'file_path', 'start_line', 'end_line'
            instance_id: 实例ID
            top_k_modules: 返回的最大模块数
            top_k_entities: 返回的最大实体数
        
        Returns:
            (found_modules, found_entities): 模块ID列表和实体ID列表
        """
        repo_name = instance_id_to_repo_name(instance_id)
        repo_root = os.path.join(self.repos_root, repo_name)
        
        # 按文件分组
        blocks_by_file = defaultdict(list)
        for block in blocks:
            file_path = block.get('file_path')
            if file_path:
                blocks_by_file[file_path].append(block)
        
        found_modules = []
        found_entities = []
        seen_modules = set()
        seen_entities = set()
        
        # 为每个文件构建映射表并处理代码块
        for file_path, file_blocks in blocks_by_file.items():
            # 构建该文件的映射表
            line_map = self._get_line_map(file_path, repo_root)
            if not line_map:
                continue
            
            # 处理该文件的每个代码块
            for block in file_blocks:
                block_start = block.get('start_line', 0)
                block_end = block.get('end_line', 0)
                
                if block_start < 0 or block_end < 0:
                    continue
                
                # 代码块使用 0-indexed，AST 使用 1-indexed，需要转换
                # 将代码块行号转换为 1-indexed（AST 格式）
                ast_start = block_start + 1
                ast_end = block_end + 1
                
                # 收集代码块范围内的所有实体
                block_entities = set()
                block_classes = set()
                
                for line_num in range(ast_start, ast_end + 1):
                    if line_num in line_map:
                        entity_id, class_id = line_map[line_num]
                        
                        # 收集实体（只有函数，不包括类）
                        if entity_id and entity_id not in seen_entities:
                            block_entities.add(entity_id)
                        
                        # 收集模块：
                        # - 如果函数在类中，模块是类名
                        # - 如果函数是顶级函数，模块是函数名本身
                        # - 如果只有类（entity_id 为 None），模块是类名
                        if class_id:
                            # 类或类方法：模块是类名
                            if class_id not in seen_modules:
                                block_classes.add(class_id)
                        elif entity_id:
                            # 顶级函数：模块是函数名本身
                            if entity_id not in seen_modules:
                                block_classes.add(entity_id)
                
                # 添加到结果列表（保持顺序）
                for entity_id in sorted(block_entities):
                    if len(found_entities) >= top_k_entities:
                        break
                    found_entities.append(entity_id)
                    seen_entities.add(entity_id)
                
                for class_id in sorted(block_classes):
                    if len(found_modules) >= top_k_modules:
                        break
                    found_modules.append(class_id)
                    seen_modules.add(class_id)
                
                if len(found_entities) >= top_k_entities and len(found_modules) >= top_k_modules:
                    break
        
        return found_modules[:top_k_modules], found_entities[:top_k_entities]


    def map_to_line_numbers(self, instance_id: str, file_path: str, relevant_blocks: List[Dict[str, Any]]) -> List[int]:
        """
        将文件映射到相关的行号列表
        
        Args:
            instance_id: 实例ID
            file_path: 文件路径
            relevant_blocks: 该文件相关的代码块列表
            
        Returns:
            行号列表（1-indexed，从1开始）
        """
        line_numbers = set()
        
        for block in relevant_blocks:
            if block.get('file_path') == file_path:
                start_line = block.get('start_line', 0)
                end_line = block.get('end_line', 0)
                
                # 确保行号有效
                if start_line >= 0 and end_line >= 0:
                    # 添加代码块范围内的所有行号（转换为1-indexed）
                    for line_num in range(start_line + 1, end_line + 2):  # +2 因为range不包含结束值，需要+1
                        line_numbers.add(line_num)
        
        return sorted(list(line_numbers))
