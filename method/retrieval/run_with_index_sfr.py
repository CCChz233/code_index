#!/usr/bin/env python
"""
SFR 专用索引评估工具
使用 SentenceTransformer 后端
"""

import argparse
import json
import logging
import os
import os.path as osp
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from method.mapping import ASTBasedMapper, GraphBasedMapper
from method.retrieval.common import (
    instance_id_to_repo_name,
    get_problem_text,
    rank_files,
    load_index,
    load_jsonl,
)


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,  # 明确类型
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """SFR 专用的文本 embedding 函数"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # SFR 使用 SentenceTransformer 的编码方式
        with torch.no_grad():
            embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=False,
                normalize_embeddings=True,
            )
            all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)


from method.utils import clean_file_path


def load_model(
    model_name: str,
    device: torch.device,
    trust_remote_code: bool = False,
) -> Tuple[SentenceTransformer, Any]:
    """加载 SFR 模型（固定使用 SentenceTransformer）"""
    print(f"Loading SFR model from {model_name}")

    if not os.path.exists(model_name):
        print(f"❌ Error: Model path does not exist: {model_name}")
        raise FileNotFoundError(f"Model path not found: {model_name}")

    try:
        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code
        )
        print(f"✅ Model loaded successfully")
        tokenizer = model.tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="SFR 专用索引评估工具"
    )

    # 数据参数
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset JSONL file")
    parser.add_argument("--index_dir", type=str, required=True,
                       help="Directory containing pre-built indexes")
    parser.add_argument("--output_folder", type=str, required=True,
                       help="Output directory for results")

    # 模型参数
    parser.add_argument("--model_name", type=str, required=True,
                       help="SFR model path")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow execution of model repository code")

    # 检索参数
    parser.add_argument("--top_k_blocks", type=int, default=50,
                       help="Number of top blocks to retrieve")
    parser.add_argument("--top_k_files", type=int, default=10,
                       help="Number of top files to retrieve")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for embedding")

    # 映射参数
    parser.add_argument("--mapper_type", type=str, default="ast",
                       choices=["ast", "graph"],
                       help="Type of mapper to use")
    parser.add_argument("--repos_root", type=str,
                       default="/workspace/locbench/repos/locbench_repos",
                       help="Root directory of repositories")

    # 其他参数
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU usage")

    args = parser.parse_args()

    # 设置设备
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu_id}')

    # 创建输出目录
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    try:
        model, tokenizer = load_model(args.model_name, device, args.trust_remote_code)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 加载数据集
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_jsonl(args.dataset_path)

    print(f"Loaded {len(dataset)} instances")

    # 初始化映射器
    if args.mapper_type == "ast":
        mapper = ASTBasedMapper(args.repos_root)
    else:
        mapper = GraphBasedMapper(args.repos_root)

    results = []

    for instance in tqdm(dataset, desc="Processing instances"):
        instance_id = instance.get("instance_id", "")
        repo_name = instance_id_to_repo_name(instance_id)

        try:
            # 加载索引
            embeddings, metadata = load_index(repo_name, args.index_dir)

            # 获取问题文本
            query_text = get_problem_text(instance)
            if not query_text:
                print(f"Warning: No query text found for instance {instance_id}")
                continue

            # 编码查询
            query_embedding = embed_texts([query_text], model, tokenizer, args.max_length, 1, device)[0]

            # 计算相似度
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                embeddings,
                dim=1
            )

            # 获取top-k块（确保不超过可用块数）
            k = min(args.top_k_blocks, len(similarities))
            top_k_values, top_k_indices = torch.topk(similarities, k)

            # 按文件分组
            file_scores = {}
            for idx, score in zip(top_k_indices.tolist(), top_k_values.tolist()):
                block_meta = metadata[idx]
                file_path = block_meta["file_path"]
                if file_path not in file_scores:
                    file_scores[file_path] = []
                file_scores[file_path].append((score, block_meta))

            # 为每个文件计算最高分数
            file_results = []
            for file_path, block_list in file_scores.items():
                max_score = max(score for score, _ in block_list)
                file_results.append((max_score, file_path))

            # 排序并获取top-k文件
            file_results.sort(reverse=True)
            top_files = file_results[:args.top_k_files]

            # 映射到行号
            mapped_results = []
            for score, file_path in top_files:
                try:
                    # 获取该文件的所有相关代码块
                    relevant_blocks = [block_meta for _, block_meta in file_scores[file_path]]
                    line_numbers = mapper.map_to_line_numbers(instance_id, file_path, relevant_blocks)
                    mapped_results.append({
                        "file": file_path,
                        "score": score,
                        "line_numbers": line_numbers
                    })
                except Exception as e:
                    print(f"Warning: Failed to map {file_path} for {instance_id}: {e}")
                    mapped_results.append({
                        "file": file_path,
                        "score": score,
                        "line_numbers": []
                    })

            results.append({
                "instance_id": instance_id,
                "top_files": mapped_results
            })

        except Exception as e:
            print(f"Error processing instance {instance_id}: {e}")
            results.append({
                "instance_id": instance_id,
                "error": str(e),
                "top_files": []
            })

    # 计算统计信息
    total_instances = len(results)
    successful_instances = sum(1 for r in results if r.get('top_files') and len(r['top_files']) > 0)
    failed_instances = sum(1 for r in results if 'error' in r)

    if successful_instances > 0:
        avg_files = sum(len(r.get('top_files', [])) for r in results) / successful_instances
    else:
        avg_files = 0

    # 保存结果
    output_path = output_dir / "results.jsonl"
    with output_path.open('w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Results saved to {output_path}")
    print(f"📊 Retrieval Summary:")
    print(f"   Total instances: {total_instances}")
    print(f"   Successful: {successful_instances} ({successful_instances/total_instances:.1%})")
    print(f"   Failed: {failed_instances}")
    print(f"   Average files per instance: {avg_files:.1f}")

    if successful_instances > 0:
        print(f"\n✅ Retrieval completed successfully!")
    else:
        print(f"\n❌ No successful retrievals!")


if __name__ == "__main__":
    main()
