"""
方法评测框架基类定义

提供统一的定位结果格式和方法基类，便于不同算法（检索类、生成类、智能体类）
输出标准化的 loc_outputs.jsonl，并用 evaluation/eval_metric.py 进行评估。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class LocResult:
    """
    标准定位结果

    与 evaluation/eval_metric.py 兼容的输出格式。

    Attributes:
        instance_id: 实例唯一标识（如 'repo__project-123'）
        found_files: 定位到的文件路径列表（如 ['src/utils.py', 'src/main.py']）
        found_modules: 定位到的模块列表（如 ['src/utils.py:MyClass']）
        found_entities: 定位到的实体列表（如 ['src/utils.py:MyClass.method']）
        raw_output_loc: 原始输出（可选，用于调试或记录模型原始响应）
        metadata: 额外元数据（可选）
    """
    instance_id: str
    found_files: List[str] = field(default_factory=list)
    found_modules: List[str] = field(default_factory=list)
    found_entities: List[str] = field(default_factory=list)
    raw_output_loc: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典格式，用于 JSONL 输出"""
        result = {
            "instance_id": self.instance_id,
            "found_files": self.found_files,
            "found_modules": self.found_modules,
            "found_entities": self.found_entities,
            "raw_output_loc": self.raw_output_loc,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'LocResult':
        """从字典创建 LocResult 实例"""
        return cls(
            instance_id=data["instance_id"],
            found_files=data.get("found_files", []),
            found_modules=data.get("found_modules", []),
            found_entities=data.get("found_entities", []),
            raw_output_loc=data.get("raw_output_loc", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def empty(cls, instance_id: str) -> 'LocResult':
        """创建空结果（用于定位失败的情况）"""
        return cls(instance_id=instance_id)

    def is_empty(self) -> bool:
        """检查是否为空结果"""
        return (
            not self.found_files
            and not self.found_modules
            and not self.found_entities
        )


class BaseMethod(ABC):
    """
    方法基类（可选继承）

    定义统一的接口，方便不同类型的算法实现：
    - 检索类（BM25、Dense Retrieval）
    - 生成类（RLCoder 等补全模型）
    - 智能体类（LocAgent 等多轮交互）

    Usage:
        class MyMethod(BaseMethod):
            def __init__(self, config):
                self.config = config

            def localize(self, instance: dict) -> LocResult:
                # 实现定位逻辑
                return LocResult(
                    instance_id=instance['instance_id'],
                    found_files=['src/main.py'],
                    found_modules=['src/main.py:MyClass'],
                    found_entities=['src/main.py:MyClass.my_method'],
                )
    """

    @property
    def name(self) -> str:
        """方法名称，用于日志和输出"""
        return self.__class__.__name__

    @abstractmethod
    def localize(self, instance: dict) -> LocResult:
        """
        单实例定位

        Args:
            instance: 数据集实例，包含：
                - instance_id: 唯一标识
                - problem_statement / issue / description: 问题描述
                - repo: 仓库名
                - base_commit: 基准提交
                - 其他数据集特定字段

        Returns:
            LocResult: 定位结果
        """
        pass

    def setup(self, **kwargs) -> None:
        """
        初始化设置（可选实现）

        用于加载索引、初始化模型等一次性操作。
        """
        pass

    def teardown(self) -> None:
        """
        清理资源（可选实现）

        用于释放模型、关闭连接等。
        """
        pass

    def __enter__(self):
        """支持 context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时清理资源"""
        self.teardown()
        return False


def validate_loc_result(result: LocResult) -> bool:
    """
    验证定位结果格式是否正确

    Args:
        result: 定位结果

    Returns:
        bool: 是否有效
    """
    if not result.instance_id:
        return False
    if not isinstance(result.found_files, list):
        return False
    if not isinstance(result.found_modules, list):
        return False
    if not isinstance(result.found_entities, list):
        return False
    return True
