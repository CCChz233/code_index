import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from method.utils import instance_id_to_repo_name as utils_instance_id_to_repo_name, clean_file_path


def instance_id_to_repo_name(instance_id: str) -> str:
    """将 instance_id 转换为 repo_name（去掉 issue 编号后缀）"""
    return utils_instance_id_to_repo_name(instance_id)


def get_problem_text(instance: dict) -> str:
    """从实例中提取问题描述"""
    for key in ("problem_statement", "issue", "description", "prompt", "text"):
        val = instance.get(key)
        if val:
            return val
    return ""


def rank_files(
    block_scores: List[Tuple[int, float]],
    metadata: List[dict],
    top_k_files: int,
    repo_name: str,
) -> List[str]:
    """根据代码块分数聚合到文件级别"""
    file_scores: Dict[str, float] = {}
    for block_idx, score in block_scores:
        block_meta = metadata[block_idx]
        file_path = block_meta["file_path"]
        cleaned_path = clean_file_path(file_path, repo_name)
        file_scores[cleaned_path] = max(file_scores.get(cleaned_path, 0.0), float(score))

    ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_k_files]]


def load_index(repo_name: str, index_dir: str) -> Tuple[torch.Tensor, List[dict]]:
    """加载预建的索引"""
    index_path = Path(index_dir) / repo_name / "embeddings.pt"
    metadata_path = Path(index_dir) / repo_name / "metadata.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        embeddings = torch.load(index_path, map_location="cpu", weights_only=True)
    except TypeError:
        embeddings = torch.load(index_path, map_location="cpu")

    metadata: List[dict] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    return embeddings, metadata


def load_jsonl(path: str) -> List[dict]:
    data: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
