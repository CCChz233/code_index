"""
映射器基类定义

提供统一的接口，支持不同的映射方式：
- Graph索引+span_ids映射
- AST解析映射
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BlockMapper(ABC):
    """代码块到实体映射器基类"""
    
    @abstractmethod
    def map_blocks_to_entities(
        self,
        blocks: List[Dict[str, Any]],
        instance_id: str,
        top_k_modules: int = 10,
        top_k_entities: int = 50,
    ) -> Tuple[List[str], List[str]]:
        """
        将代码块映射到函数/模块
        
        Args:
            blocks: 代码块列表，每个包含 'file_path', 'start_line', 'end_line' 等
            instance_id: 实例ID（用于确定仓库）
            top_k_modules: 返回的最大模块数
            top_k_entities: 返回的最大实体数
        
        Returns:
            (found_modules, found_entities): 模块ID列表和实体ID列表
        """
        pass

