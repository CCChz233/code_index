"""
基于Graph索引+span_ids的映射器

使用预构建的Graph索引和代码块的span_ids进行映射。
"""

from typing import List, Dict, Any, Tuple
import logging

from method.mapping.base import BlockMapper
from method.utils import (
    instance_id_to_repo_name,
    get_entity_searcher,
    clean_file_path,
    dedupe_append,
)
from dependency_graph.build_graph import NODE_TYPE_FUNCTION, NODE_TYPE_CLASS

logger = logging.getLogger(__name__)


def _module_id(entity_id: str) -> str:
    """从实体ID提取模块ID"""
    file_path, name = entity_id.split(":", 1)
    if "." in name:
        name = name.split(".")[0]
    return f"{file_path}:{name}"


class GraphBasedMapper(BlockMapper):
    """基于Graph索引的映射器"""
    
    def __init__(self, graph_index_dir: str):
        """
        初始化Graph映射器
        
        Args:
            graph_index_dir: Graph索引目录
        """
        self.graph_index_dir = graph_index_dir
        self._searcher_cache = {}  # 缓存searcher，避免重复加载
    
    def _get_searcher(self, instance_id: str):
        """获取或缓存实体搜索器"""
        if instance_id not in self._searcher_cache:
            searcher = get_entity_searcher(instance_id, self.graph_index_dir)
            self._searcher_cache[instance_id] = searcher
        return self._searcher_cache[instance_id]
    
    def map_blocks_to_entities(
        self,
        blocks: List[Dict[str, Any]],
        instance_id: str,
        top_k_modules: int = 10,
        top_k_entities: int = 50,
    ) -> Tuple[List[str], List[str]]:
        """
        使用Graph索引+span_ids映射代码块到实体
        
        Args:
            blocks: 代码块列表，每个包含 'file_path' 和 'span_ids'
            instance_id: 实例ID
            top_k_modules: 返回的最大模块数
            top_k_entities: 返回的最大实体数
        
        Returns:
            (found_modules, found_entities): 模块ID列表和实体ID列表
        """
        repo_name = instance_id_to_repo_name(instance_id)
        searcher = self._get_searcher(instance_id)
        
        if searcher is None:
            return [], []
        
        found_modules = []
        found_entities = []
        
        for block in blocks:
            file_path = block.get("file_path")
            if not file_path:
                continue
            
            file_path = clean_file_path(file_path, repo_name)
            
            # 从代码块的span_ids提取实体
            span_ids = block.get("span_ids", [])
            for span_id in span_ids:
                entity_id = f"{file_path}:{span_id}"
                if not searcher.has_node(entity_id):
                    continue
                
                node_data = searcher.get_node_data([entity_id])[0]
                if node_data["type"] == NODE_TYPE_FUNCTION:
                    dedupe_append(found_entities, entity_id, top_k_entities)
                    dedupe_append(found_modules, _module_id(entity_id), top_k_modules)
                elif node_data["type"] == NODE_TYPE_CLASS:
                    dedupe_append(found_modules, entity_id, top_k_modules)
            
            # 检查是否已达到所需数量
            if (
                len(found_modules) >= top_k_modules
                and len(found_entities) >= top_k_entities
            ):
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
        # 对于GraphBasedMapper，我们暂时返回空列表
        # 因为Graph索引主要关注实体而不是具体的行号
        # TODO: 可以从span_ids信息中推导出更精确的行号
        return []
