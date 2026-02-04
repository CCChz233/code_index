#!/usr/bin/env python
"""
CodeRankEmbed 专用索引评估工具
使用 transformers AutoModel 后端
"""

import argparse
import json
import logging
import os
import os.path as osp
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
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
    model: AutoModel,  # 明确类型
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """CodeRankEmbed 专用的文本 embedding 函数"""
    from torch.utils.data import Dataset, DataLoader

    class TextDataset(Dataset):
        def __init__(self, items: List[str]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx: int):
            encoded = tokenizer(
                self.items[idx],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    ds = TextDataset(texts)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for input_ids, attn_mask in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            # CodeRankEmbed 使用 AutoModel
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            token_embeddings = outputs[0]
            sent_emb = token_embeddings[:, 0]  # [CLS] token
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outs.append(sent_emb.cpu())
    return torch.cat(outs, dim=0)


from method.utils import clean_file_path


def load_model(
    model_name: str,
    device: torch.device,
    trust_remote_code: bool = False,
) -> Tuple[AutoModel, AutoTokenizer]:
    """加载 CodeRankEmbed 模型（固定使用 AutoModel）"""
    print(f"Loading CodeRankEmbed model from {model_name}")

    if not os.path.exists(model_name):
        print(f"❌ Error: Model path does not exist: {model_name}")
        raise FileNotFoundError(f"Model path not found: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        ).to(device)
        model.eval()
        print(f"✅ Model loaded successfully")
        return model, tokenizer
    except ValueError as e:
        if "trust_remote_code" in str(e).lower():
            print(f"❌ Error: Model requires trust_remote_code=True")
            raise
        else:
            raise


def main():
    parser = argparse.ArgumentParser(
        description="CodeRankEmbed 专用索引评估工具"
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
                       help="CodeRankEmbed model path")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Allow execution of model repository code")

    # 检索参数
    parser.add_argument("--top_k_blocks", type=int, default=50,
                       help="Number of top blocks to retrieve")
    parser.add_argument("--top_k_files", type=int, default=20,
                       help="Number of top files to retrieve")
    parser.add_argument("--top_k_modules", type=int, default=20,
                       help="Number of top modules to retrieve")
    parser.add_argument("--top_k_entities", type=int, default=50,
                       help="Number of top entities to retrieve")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8,
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

    # 缓存索引（避免重复加载）
    index_cache: Dict[str, Tuple[torch.Tensor, List[dict]]] = {}

    def get_cached_index(repo_name: str):
        if repo_name not in index_cache:
            embeddings, metadata = load_index(repo_name, args.index_dir)
            # 索引保留在 CPU，避免 GPU 内存不足
            index_cache[repo_name] = (embeddings, metadata)
        return index_cache[repo_name]

    # 处理统计
    index_found = 0
    index_missing = 0
    missing_repos = []

    for instance in tqdm(dataset, desc="Processing instances"):
        instance_id = instance.get("instance_id", "")
        repo_name = instance_id_to_repo_name(instance_id)

        try:
            # 加载索引（使用缓存）
            embeddings, metadata = get_cached_index(repo_name)

            if embeddings is None or metadata is None:
                # 索引不存在，返回空结果
                index_missing += 1
                missing_repos.append(repo_name)
                results.append({
                    "instance_id": instance_id,
                    "found_files": [],
                    "found_modules": [],
                    "found_entities": [],
                    "raw_output_loc": []
                })
                continue

            index_found += 1

            # 获取问题文本
            query_text = get_problem_text(instance)
            if not query_text:
                print(f"Warning: No query text found for instance {instance_id}")
                continue

            # 编码查询
            query_embedding = embed_texts([query_text], model, tokenizer, args.max_length, 1, device)[0]

            # 计算相似度（临时移到 GPU）
            query_emb_gpu = query_embedding.to(device)
            embeddings_gpu = embeddings.to(device)  # 临时移到 GPU
            similarities = torch.matmul(query_emb_gpu.unsqueeze(0), embeddings_gpu.t()).squeeze(0)  # (num_blocks,)
            similarities = similarities.cpu()  # 移回 CPU 以便后续处理

            # 获取top-k块（确保不超过可用块数）
            k = min(args.top_k_blocks, len(similarities))
            if k == 0:
                found_files = []
                found_modules = []
                found_entities = []
            else:
                top_k_values, top_k_indices = torch.topk(similarities, k)
                block_scores = list(zip(top_k_indices.tolist(), top_k_values.tolist()))

                # 获取文件级别结果
                found_files = rank_files(block_scores, metadata, args.top_k_files, repo_name)

                # 映射代码块到函数/模块
                # 清理 top_blocks 中的 file_path，使其与 GT 格式一致
                top_blocks = []
                for idx, _ in block_scores:
                    block = metadata[idx].copy()  # 复制以避免修改原始 metadata
                    original_path = block.get('file_path', '')
                    if original_path:
                        block['file_path'] = clean_file_path(original_path, repo_name)
                    top_blocks.append(block)

                found_modules, found_entities = mapper.map_blocks_to_entities(
                    blocks=top_blocks,
                    instance_id=instance_id,
                    top_k_modules=args.top_k_modules,
                    top_k_entities=args.top_k_entities,
                )

            # 保存结果
            results.append({
                "instance_id": instance_id,
                "found_files": found_files,
                "found_modules": found_modules,
                "found_entities": found_entities,
                "raw_output_loc": []
            })

        except Exception as e:
            print(f"Error processing instance {instance_id}: {e}")
            results.append({
                "instance_id": instance_id,
                "error": str(e),
                "found_files": [],
                "found_modules": [],
                "found_entities": [],
            })

    # 计算统计信息（结果统一用 found_files，与 loc_outputs.jsonl 一致）
    total_instances = len(results)
    successful_instances = sum(1 for r in results if r.get('found_files') and len(r['found_files']) > 0)
    failed_instances = sum(1 for r in results if 'error' in r)

    if successful_instances > 0:
        avg_files = sum(len(r.get('found_files', [])) for r in results) / successful_instances
    else:
        avg_files = 0

    # 保存结果
    output_path = output_dir / "loc_outputs.jsonl"
    with output_path.open('w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Results saved to {output_path}")
    print(f"📊 Retrieval Summary:")
    print(f"   Total instances: {total_instances}")
    print(f"   Successful: {successful_instances} ({successful_instances/total_instances:.1%})")
    print(f"   Failed: {failed_instances}")
    print(f"   Average files per instance: {avg_files:.1f}")

    # 输出索引查找统计
    print(f"\nIndex Statistics:")
    print(f"  Found: {index_found}/{len(dataset)}")
    print(f"  Missing: {index_missing}/{len(dataset)}")
    if missing_repos:
        unique_missing = list(set(missing_repos))[:10]
        print(f"  Missing repos (sample): {unique_missing}")

    if successful_instances > 0:
        print(f"\n✅ Retrieval completed successfully!")
    else:
        print(f"\n❌ No successful retrievals!")


if __name__ == "__main__":
    main()
