#!/usr/bin/env python
"""Sparse BM25 retrieval."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.sparse import load_npz
from tqdm import tqdm

from method.mapping import ASTBasedMapper, GraphBasedMapper
from method.retrieval.common import (
    instance_id_to_repo_name,
    get_problem_text,
    rank_files,
    load_jsonl,
)
from method.utils import clean_file_path
from method.indexing.encoder.sparse.bm25 import tokenize


def load_sparse_index(repo_name: str, index_dir: str):
    index_path = Path(index_dir) / repo_name / "index.npz"
    vocab_path = Path(index_dir) / repo_name / "vocab.json"
    meta_path = Path(index_dir) / repo_name / "metadata.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    tf_matrix = load_npz(index_path).tocsc()
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_payload = json.load(f)

    vocab = vocab_payload["vocab"]
    idf = np.array(vocab_payload["idf"], dtype=np.float32)
    doc_len = np.array(vocab_payload["doc_len"], dtype=np.float32)
    avgdl = float(vocab_payload.get("avgdl", 0.0))
    k1 = float(vocab_payload.get("k1", 1.5))
    b = float(vocab_payload.get("b", 0.75))

    metadata: List[dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    return tf_matrix, vocab, idf, doc_len, avgdl, k1, b, metadata


def bm25_scores(
    query: str,
    tf_matrix,
    vocab: Dict[str, int],
    idf: np.ndarray,
    doc_len: np.ndarray,
    avgdl: float,
    k1: float,
    b: float,
) -> np.ndarray:
    tokens = tokenize(query)
    if not tokens:
        return np.zeros(tf_matrix.shape[0], dtype=np.float32)

    scores = np.zeros(tf_matrix.shape[0], dtype=np.float32)
    doc_len = doc_len.astype(np.float32)

    # query term frequency
    qtf: Dict[int, int] = {}
    for t in tokens:
        idx = vocab.get(t)
        if idx is None:
            continue
        qtf[idx] = qtf.get(idx, 0) + 1

    for term_id, qf in qtf.items():
        col = tf_matrix.getcol(term_id)
        if col.nnz == 0:
            continue
        tf_vals = col.data.astype(np.float32)
        doc_idx = col.indices
        denom = tf_vals + k1 * (1.0 - b + b * (doc_len[doc_idx] / max(avgdl, 1e-6)))
        inc = idf[term_id] * (tf_vals * (k1 + 1.0)) / denom
        if qf > 1:
            inc = inc * qf
        scores[doc_idx] += inc

    return scores


def main():
    parser = argparse.ArgumentParser(description="Sparse BM25 retrieval")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--top_k_blocks", type=int, default=50)
    parser.add_argument("--top_k_files", type=int, default=20)
    parser.add_argument("--top_k_modules", type=int, default=20)
    parser.add_argument("--top_k_entities", type=int, default=50)
    parser.add_argument("--file_score_agg", type=str, default="sum", choices=["sum", "max"])
    parser.add_argument("--mapper_type", type=str, default="ast", choices=["ast", "graph"])
    parser.add_argument("--repos_root", type=str, required=True)
    parser.add_argument("--graph_index_dir", type=str, default="")

    args = parser.parse_args()

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl(args.dataset_path)

    if args.mapper_type == "ast":
        mapper = ASTBasedMapper(args.repos_root)
    else:
        if not args.graph_index_dir:
            raise ValueError("mapper_type=graph requires --graph_index_dir")
        mapper = GraphBasedMapper(args.graph_index_dir)

    index_cache: Dict[str, Tuple] = {}

    def get_cached(repo_name: str):
        if repo_name not in index_cache:
            index_cache[repo_name] = load_sparse_index(repo_name, args.index_dir)
        return index_cache[repo_name]

    results = []

    for instance in tqdm(dataset, desc="Processing instances"):
        instance_id = instance.get("instance_id", "")
        repo_name = instance_id_to_repo_name(instance_id)

        try:
            tf_matrix, vocab, idf, doc_len, avgdl, k1, b, metadata = get_cached(repo_name)

            query_text = get_problem_text(instance)
            if not query_text:
                continue

            scores = bm25_scores(query_text, tf_matrix, vocab, idf, doc_len, avgdl, k1, b)

            k = min(args.top_k_blocks, scores.shape[0])
            if k == 0:
                found_files = []
                found_modules = []
                found_entities = []
            else:
                top_idx = np.argpartition(-scores, k - 1)[:k]
                top_scores = scores[top_idx]
                order = np.argsort(-top_scores)
                top_k_indices = top_idx[order]
                block_scores = list(zip(top_k_indices.tolist(), top_scores[order].tolist()))

                found_files = rank_files(
                    block_scores,
                    metadata,
                    args.top_k_files,
                    repo_name,
                    file_score_agg=args.file_score_agg,
                )

                top_blocks = []
                for idx, _ in block_scores:
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
