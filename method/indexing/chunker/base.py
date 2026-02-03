from abc import ABC, abstractmethod
from typing import Dict

from method.indexing.core.block import ChunkResult


class BaseChunker(ABC):
    block_type: str

    @abstractmethod
    def chunk(self, repo_path: str) -> ChunkResult:
        """将仓库代码切块，返回 ChunkResult。"""
        raise NotImplementedError

    def get_metadata(self) -> Dict:
        return {
            "name": self.__class__.__name__,
            "block_type": self.block_type,
        }
