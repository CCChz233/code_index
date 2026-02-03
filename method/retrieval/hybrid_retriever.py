#!/usr/bin/env python
"""Hybrid retrieval: dense + sparse fusion."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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
from method.retrieval.sparse_retriever import load_sparse_index, bm25_scores
from method.retrieval.fusion.rrf import fuse_rrf
from method.retrieval.fusion.weighted import fuse_weighted
from method.utils import clean_file_path


def dense_block_scores(
    query_text: str,
    embeddings: torch.Tensor,
    model,
    tokenizer,
    max_length: int,
    device: torch.device,
    top_k: int,
) -> List[Tuple[int, float]]:
    encoded = tokenizer(
        query_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        token_embeddings = outputs[0]
        query_emb = token_embeddings[:, 0]
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)

    query_emb = query_emb.squeeze(0)
    sims = torch.matmul(query_emb.unsqueeze(0), embeddings.to(device).t()).squeeze(0)
    sims = sims.cpu()

    k = min(top_k, sims.numel())
    if k == 0:
        return []
    top_vals, top_idx = torch.topk(sims, k)
    return list(zip(top_idx.tolist(), top_vals.tolist()))


def main():
    parser = argparse.ArgumentParser(description="Hybrid retrieval (dense + sparse)")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dense_index_dir", type=str, required=True)
    parser.add_argument("--sparse_index_dir", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--top_k_blocks", type=int, default=50)
    parser.add_argument("--top_k_files", type=int, default=20)
    parser.add_argument("--top_k_modules", type=int, default=20)
    parser.add_argument("--top_k_entities", type=int, default=50)

    parser.add_argument("--fusion", type=str, default="rrf", choices=["rrf", "weighted"])
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--dense_weight", type=float, default=0.5)
    parser.add_argument("--sparse_weight", type=float, default=0.5)

    parser.add_argument("--mapper_type", type=str, default="ast", choices=["ast", "graph"])
    parser.add_argument("--repos_root", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.force_cpu else f"cuda:{args.gpu_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code).to(device)
    model.eval()

    if args.mapper_type == "ast":
        mapper = ASTBasedMapper(args.repos_root)
    else:
        mapper = GraphBasedMapper(args.repos_root)

    dense_cache: Dict[str, Tuple] = {}
    sparse_cache: Dict[str, Tuple] = {}

    def get_dense(repo_name: str):
        if repo_name not in dense_cache:
            dense_cache[repo_name] = load_index(repo_name, args.dense_index_dir)
        return dense_cache[repo_name]

    def get_sparse(repo_name: str):
        if repo_name not in sparse_cache:
            sparse_cache[repo_name] = load_sparse_index(repo_name, args.sparse_index_dir)
        return sparse_cache[repo_name]

    dataset = load_jsonl(args.dataset_path)
    results = []

    for instance in tqdm(dataset, desc="Processing instances"):
        instance_id = instance.get("instance_id", "")
        repo_name = instance_id_to_repo_name(instance_id)

        try:
            embeddings, metadata = get_dense(repo_name)
            tf_matrix, vocab, idf, doc_len, avgdl, k1, b, _ = get_sparse(repo_name)

            query_text = get_problem_text(instance)
            if not query_text:
                continue

            dense_scores = dense_block_scores(
                query_text,
                embeddings,
                model,
                tokenizer,
                args.max_length,
                device,
                args.top_k_blocks,
            )

            sparse_raw = bm25_scores(query_text, tf_matrix, vocab, idf, doc_len, avgdl, k1, b)
            k = min(args.top_k_blocks, sparse_raw.shape[0])
            if k == 0:
                sparse_scores = []
            else:
                top_idx = np.argpartition(-sparse_raw, k - 1)[:k]
                top_scores = sparse_raw[top_idx]
                order = np.argsort(-top_scores)
                sparse_scores = list(zip(top_idx[order].tolist(), top_scores[order].tolist()))

            if args.fusion == "rrf":
                fused = fuse_rrf([dense_scores, sparse_scores], k=args.rrf_k)
            else:
                fused = fuse_weighted([dense_scores, sparse_scores], weights=[args.dense_weight, args.sparse_weight])

            fused = fused[: args.top_k_blocks]

            found_files = rank_files(fused, metadata, args.top_k_files, repo_name)

            top_blocks = []
            for idx, _ in fused:
                block = metadata[idx].copy()
                original_path = block.get("file_path", "")
                if original_path:
                    block["file_path"] = clean_file_path(original_path, repo_name)
                top_blocks.append(block)

            found_modules, found_entities = mapper.map_blocks_to_entities(
                blocks=top_blocks,
                instance_id=instance_id,
                top_k_modules=args.top_k_modules,
                top_k_entities=args.top_k_entities,
            )

            results.append({
                "instance_id": instance_id,
                "found_files": found_files,
                "found_modules": found_modules,
                "found_entities": found_entities,
                "raw_output_loc": [],
            })

        except Exception as e:
            results.append({
                "instance_id": instance_id,
                "error": str(e),
                "found_files": [],
                "found_modules": [],
                "found_entities": [],
                "raw_output_loc": [],
            })

    output_path = output_dir / "loc_outputs.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
