"""
方法评测框架共享工具

提供数据加载、结果保存、索引加载等通用功能。
"""

import json
import os
import pickle
import re
import logging
from typing import Optional, List, Dict, Any, Iterator, Set

from datasets import load_dataset

from method.base import LocResult

logger = logging.getLogger(__name__)


# ============================================================================
# 数据集加载
# ============================================================================

def load_dataset_instances(
    dataset_path: str = None,
    dataset: str = None,
    split: str = "test",
    limit: int = None,
) -> List[dict]:
    """
    统一数据加载
    
    Args:
        dataset_path: 本地 JSONL 文件路径（离线模式）
        dataset: HuggingFace 数据集名称（在线模式）
        split: 数据集分片（默认 'test'）
        limit: 限制加载数量（调试用）
    
    Returns:
        List[dict]: 实例列表
    
    Usage:
        # 离线模式
        instances = load_dataset_instances(dataset_path='data/Loc-Bench_V1.jsonl')
        
        # 在线模式
        instances = load_dataset_instances(dataset='czlll/Loc-Bench_V1', split='test')
    """
    if dataset_path:
        instances = load_jsonl(dataset_path)
    elif dataset:
        data = load_dataset(dataset, split=split)
        instances = list(data)
    else:
        raise ValueError("Must provide either dataset_path or dataset")
    
    if limit:
        instances = instances[:limit]
        logger.info(f"Limiting to first {limit} instances.")
    
    return instances


def iter_dataset_instances(
    dataset_path: str = None,
    dataset: str = None,
    split: str = "test",
    limit: int = None,
) -> Iterator[dict]:
    """
    迭代数据集实例（内存友好）
    
    与 load_dataset_instances 相同参数，但返回迭代器。
    """
    if dataset_path:
        count = 0
        with open(dataset_path, 'r') as f:
            for line in f:
                if limit and count >= limit:
                    break
                yield json.loads(line)
                count += 1
    elif dataset:
        data = load_dataset(dataset, split=split)
        for i, instance in enumerate(data):
            if limit and i >= limit:
                break
            yield instance
    else:
        raise ValueError("Must provide either dataset_path or dataset")


def get_problem_text(instance: dict) -> str:
    """
    从实例中提取问题描述
    
    支持多种字段名：problem_statement, issue, description, prompt, text
    """
    for key in ("problem_statement", "issue", "description", "prompt", "text"):
        value = instance.get(key)
        if value:
            return value
    return ""


# ============================================================================
# JSONL 文件操作
# ============================================================================

def load_jsonl(filepath: str) -> List[dict]:
    """加载 JSONL 文件"""
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


def append_to_jsonl(data: dict, filepath: str) -> None:
    """追加写入 JSONL 文件"""
    with open(filepath, "a") as file:
        file.write(json.dumps(data) + "\n")


def write_to_jsonl(data: List[dict], filepath: str) -> None:
    """覆盖写入 JSONL 文件"""
    with open(filepath, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def clear_file(filepath: str) -> None:
    """清空文件"""
    with open(filepath, 'w') as f:
        f.write("")


# ============================================================================
# 实例 ID 和仓库名转换
# ============================================================================

def instance_id_to_repo_name(instance_id: str) -> str:
    """
    将 instance_id 转换为 repo_name
    
    例如：'UXARRAY__uxarray-1117' -> 'UXARRAY_uxarray'
    """
    # 移除 issue 编号后缀（如 '-1117'）
    repo_part = re.sub(r'-\d+$', '', instance_id)
    # 将双下划线替换为单下划线
    return repo_part.replace('__', '_')


# ============================================================================
# 索引加载
# ============================================================================

def get_default_index_dirs(dataset_name: str = "Loc-Bench_V1") -> tuple:
    """
    获取默认索引目录
    
    Returns:
        (graph_index_dir, bm25_index_dir)
    """
    from dependency_graph.build_graph import VERSION
    
    graph_index_dir = os.environ.get(
        "GRAPH_INDEX_DIR", 
        f"index_data/{dataset_name}/graph_index_{VERSION}"
    )
    bm25_index_dir = os.environ.get(
        "BM25_INDEX_DIR",
        f"index_data/{dataset_name}/BM25_index"
    )
    return graph_index_dir, bm25_index_dir


def load_graph_index(instance_id: str, graph_index_dir: str):
    """
    加载图索引
    
    Args:
        instance_id: 实例 ID
        graph_index_dir: 图索引目录
    
    Returns:
        nx.MultiDiGraph: 依赖图
    """
    repo_name = instance_id_to_repo_name(instance_id)
    graph_file = os.path.join(graph_index_dir, f"{repo_name}.pkl")
    
    if not os.path.exists(graph_file):
        logger.warning(f"Graph index not found: {graph_file}")
        return None
    
    with open(graph_file, 'rb') as f:
        return pickle.load(f)


def load_bm25_retriever(instance_id: str, bm25_index_dir: str):
    """
    加载 BM25 检索器
    
    Args:
        instance_id: 实例 ID
        bm25_index_dir: BM25 索引目录
    
    Returns:
        BM25Retriever: 检索器实例
    """
    from plugins.location_tools.retriever.bm25_retriever import (
        build_retriever_from_persist_dir,
    )
    
    repo_name = instance_id_to_repo_name(instance_id)
    persist_dir = os.path.join(bm25_index_dir, repo_name)
    
    if not os.path.exists(os.path.join(persist_dir, "corpus.jsonl")):
        logger.warning(f"BM25 index not found: {persist_dir}")
        return None
    
    return build_retriever_from_persist_dir(persist_dir)


def get_entity_searcher(instance_id: str, graph_index_dir: str):
    """
    获取实体搜索器
    
    Args:
        instance_id: 实例 ID
        graph_index_dir: 图索引目录
    
    Returns:
        RepoEntitySearcher: 实体搜索器
    """
    from dependency_graph import RepoEntitySearcher
    
    graph = load_graph_index(instance_id, graph_index_dir)
    if graph is None:
        return None
    
    return RepoEntitySearcher(graph)


# ============================================================================
# 结果处理
# ============================================================================

def load_processed_ids(output_file: str) -> Set[str]:
    """
    加载已处理的实例 ID（用于断点续跑）
    
    Args:
        output_file: 输出文件路径
    
    Returns:
        Set[str]: 已处理的实例 ID 集合
    """
    processed = set()
    if os.path.exists(output_file):
        for row in load_jsonl(output_file):
            instance_id = row.get("instance_id")
            if instance_id:
                processed.add(instance_id)
    return processed


def save_result(result: LocResult, output_file: str) -> None:
    """
    保存单个定位结果
    
    Args:
        result: 定位结果
        output_file: 输出文件路径
    """
    append_to_jsonl(result.to_dict(), output_file)


def dedupe_append(target: List[str], item: str, limit: int) -> None:
    """
    去重追加到列表
    
    Args:
        target: 目标列表
        item: 要追加的项
        limit: 列表最大长度
    """
    if item not in target and len(target) < limit:
        target.append(item)


def clean_file_path(file_path: str, repo_name: str) -> str:
    """
    清理文件路径，移除绝对路径前缀
    
    Args:
        file_path: 原始文件路径
        repo_name: 仓库名
    
    Returns:
        str: 清理后的相对路径
    """
    # 处理类似 'workspace/LocAgent//uxarray/grid/grid.py' 的路径
    repo_pattern = repo_name.replace('_', '[_/]')
    match = re.search(rf'{repo_pattern}/(.+)$', file_path, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # 备选：获取双斜杠后的路径
    if '//' in file_path:
        return file_path.split('//')[-1]
    
    return file_path


# ============================================================================
# CLI 参数解析辅助
# ============================================================================

def add_common_args(parser) -> None:
    """
    添加通用 CLI 参数
    
    Args:
        parser: argparse.ArgumentParser 实例
    """
    # 数据集参数
    parser.add_argument("--dataset", type=str, default="czlll/Loc-Bench_V1",
                        help="HuggingFace 数据集名称")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="本地 JSONL 数据集路径（离线模式）")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集分片")
    parser.add_argument("--eval_n_limit", type=int, default=0,
                        help="限制评测数量（0 表示不限制）")
    
    # 输出参数
    parser.add_argument("--output_folder", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl",
                        help="输出文件名")
    
    # 索引参数
    parser.add_argument("--graph_index_dir", type=str, default="",
                        help="图索引目录")
    parser.add_argument("--bm25_index_dir", type=str, default="",
                        help="BM25 索引目录")
    
    # Top-k 参数
    parser.add_argument("--top_k_files", type=int, default=10,
                        help="返回的文件数量")
    parser.add_argument("--top_k_modules", type=int, default=10,
                        help="返回的模块数量")
    parser.add_argument("--top_k_entities", type=int, default=10,
                        help="返回的实体数量")


def resolve_index_dirs(args, dataset_name: str = None) -> tuple:
    """
    解析索引目录参数
    
    Args:
        args: 命令行参数
        dataset_name: 数据集名称（可选）
    
    Returns:
        (graph_index_dir, bm25_index_dir)
    """
    if dataset_name is None:
        if args.dataset_path:
            dataset_name = "Loc-Bench_V1"
        else:
            dataset_name = args.dataset.split("/")[-1]
    
    graph_index_dir, bm25_index_dir = get_default_index_dirs(dataset_name)
    
    if args.graph_index_dir:
        graph_index_dir = args.graph_index_dir
    if args.bm25_index_dir:
        bm25_index_dir = args.bm25_index_dir
    
    return graph_index_dir, bm25_index_dir

# ============================================================================
# 注意：代码块到实体的映射功能已迁移到 method.mapping 模块
# - Graph索引+span_ids映射: method.mapping.GraphBasedMapper
# - AST解析映射: method.mapping.ASTBasedMapper
# ============================================================================

