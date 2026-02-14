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
    file_score_agg: str = "sum",
) -> List[str]:
    """根据代码块分数聚合到文件级别。"""
    if file_score_agg not in {"sum", "max"}:
        raise ValueError(f"Unknown file_score_agg: {file_score_agg}")

    file_scores: Dict[str, float] = {}
    for block_idx, score in block_scores:
        block_meta = metadata[block_idx]
        file_path = block_meta["file_path"]
        cleaned_path = clean_file_path(file_path, repo_name)
        if file_score_agg == "max":
            file_scores[cleaned_path] = max(file_scores.get(cleaned_path, 0.0), float(score))
        else:
            file_scores[cleaned_path] = file_scores.get(cleaned_path, 0.0) + float(score)

    ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_k_files]]


def load_index(repo_name: str, index_dir: str, strategy: str = "") -> Tuple[torch.Tensor, List[dict]]:
    """加载预建的索引

    构建脚本将索引保存在 {index_dir}/dense_index_{strategy}/{repo_name}/ 下，
    因此加载时需要匹配相同的目录层级。

    查找顺序:
      1. {index_dir}/dense_index_{strategy}/{repo_name}/  （如果指定了 strategy）
      2. {index_dir}/{repo_name}/                          （向后兼容 / 用户已手动拼好路径）
      3. {index_dir}/dense_index_*/{repo_name}/            （自动发现）
    """
    candidates: list[Path] = []

    # 优先: 带 strategy 的精确路径
    if strategy:
        candidates.append(Path(index_dir) / f"dense_index_{strategy}" / repo_name)

    # 向后兼容: 直接路径
    candidates.append(Path(index_dir) / repo_name)

    # 自动发现: 扫描 dense_index_* 子目录
    base = Path(index_dir)
    if base.is_dir():
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and sub.name.startswith("dense_index_"):
                candidates.append(sub / repo_name)

    # 去重并保留顺序
    seen = set()
    unique_candidates: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    index_path = None
    metadata_path = None
    for candidate in unique_candidates:
        _idx = candidate / "embeddings.pt"
        _meta = candidate / "metadata.jsonl"
        if _idx.exists() and _meta.exists():
            index_path = _idx
            metadata_path = _meta
            break

    if index_path is None or metadata_path is None:
        tried = [str(c) for c in unique_candidates]
        raise FileNotFoundError(
            f"Index not found for repo '{repo_name}'. Tried: {tried}"
        )

    try:
        embeddings = torch.load(index_path, map_location="cpu", weights_only=True)
    except TypeError:
        embeddings = torch.load(index_path, map_location="cpu")

    metadata: List[dict] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Index corrupted for repo '{repo_name}': "
            f"embeddings has {embeddings.shape[0]} rows but metadata has {len(metadata)} entries. "
            f"Please rebuild the index."
        )

    return embeddings, metadata


def load_jsonl(path: str) -> List[dict]:
    data: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
