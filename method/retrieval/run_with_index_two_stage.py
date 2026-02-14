#!/usr/bin/env python
"""
Two-stage dense retrieval on top of prebuilt block index.

Stage 1:
  Global top-N block retrieval.
Stage 2:
  Restrict candidates to files observed in Stage 1, then retrieve final top-K blocks.
Fallback:
  Fail-open to Stage 1 when Stage 2 is empty/too small/failed.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from method.core.embedding import POOLING_CHOICES, extract_last_hidden_state, pool_hidden_states
from method.mapping import ASTBasedMapper, GraphBasedMapper
from method.retrieval.common import (
    get_problem_text,
    instance_id_to_repo_name,
    load_index,
    load_jsonl,
    rank_files,
)
from method.utils import clean_file_path


def embed_texts(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    pooling: str = "first_non_pad",
) -> torch.Tensor:
    """CodeRankEmbed-oriented text embedding helper."""
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            token_embeddings = extract_last_hidden_state(outputs)
            sent_emb = pool_hidden_states(token_embeddings, attn_mask, pooling=pooling)
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outs.append(sent_emb.cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty((0, 0))


def load_model(
    model_name: str,
    device: torch.device,
    trust_remote_code: bool = False,
) -> Tuple[AutoModel, AutoTokenizer]:
    """Load dense bi-encoder model (AutoModel backend)."""
    print(f"Loading CodeRankEmbed model from {model_name}")

    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Model path not found: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    print("Model loaded successfully")
    return model, tokenizer


def topk_block_scores(
    query_embedding: torch.Tensor,
    embeddings_gpu: torch.Tensor,
    top_k: int,
    candidate_indices: Optional[torch.Tensor] = None,
) -> List[Tuple[int, float]]:
    """Compute top-k block scores globally or within candidate indices."""
    if top_k <= 0:
        return []

    with torch.no_grad():
        query_emb_gpu = query_embedding.to(embeddings_gpu.device)

        if candidate_indices is None:
            sims = torch.matmul(query_emb_gpu.unsqueeze(0), embeddings_gpu.t()).squeeze(0)
            k = min(top_k, sims.numel())
            if k == 0:
                return []
            top_vals, top_idx = torch.topk(sims, k)
            return list(zip(top_idx.cpu().tolist(), top_vals.cpu().tolist()))

        if candidate_indices.numel() == 0:
            return []

        candidate_gpu = candidate_indices.to(embeddings_gpu.device)
        candidate_embeddings = embeddings_gpu.index_select(0, candidate_gpu)
        sims = torch.matmul(query_emb_gpu.unsqueeze(0), candidate_embeddings.t()).squeeze(0)
        k = min(top_k, sims.numel())
        if k == 0:
            return []

        top_vals, local_idx = torch.topk(sims, k)
        global_idx = candidate_gpu.index_select(0, local_idx).cpu().tolist()
        return list(zip(global_idx, top_vals.cpu().tolist()))


def extract_seed_files_from_blocks(
    block_scores: List[Tuple[int, float]],
    metadata: List[dict],
    repo_name: str,
    max_files: int = 0,
) -> List[str]:
    """Extract unique file paths in Stage-1 ranking order."""
    files: List[str] = []
    seen = set()

    for idx, _ in block_scores:
        raw_file_path = metadata[idx].get("file_path", "")
        if not raw_file_path:
            continue
        file_path = clean_file_path(raw_file_path, repo_name)
        if file_path in seen:
            continue
        seen.add(file_path)
        files.append(file_path)
        if max_files > 0 and len(files) >= max_files:
            break

    return files


def build_candidate_indices_from_files(
    metadata: List[dict],
    seed_files: List[str],
    repo_name: str,
) -> torch.Tensor:
    """Map seed files to candidate block indices."""
    seed_set = set(seed_files)
    if not seed_set:
        return torch.empty((0,), dtype=torch.long)

    idx_list = [
        idx
        for idx, item in enumerate(metadata)
        if clean_file_path(item.get("file_path", ""), repo_name) in seed_set
    ]
    if not idx_list:
        return torch.empty((0,), dtype=torch.long)
    return torch.tensor(idx_list, dtype=torch.long)


def finalize_block_scores_with_fail_open(
    stage1_scores: List[Tuple[int, float]],
    stage2_scores: List[Tuple[int, float]],
    final_top_k: int,
    min_stage2_accept: int,
) -> Tuple[List[Tuple[int, float]], str]:
    """
    Decide final block ranking.

    Returns:
      - final block scores
      - reason tag: stage2 / stage2+pad / fallback_stage1
    """
    stage1_top = stage1_scores[:final_top_k]
    if not stage2_scores:
        return stage1_top, "fallback_stage1"

    accept_threshold = min(max(1, min_stage2_accept), max(1, final_top_k))
    if len(stage2_scores) < accept_threshold:
        return stage1_top, "fallback_stage1"

    # Keep Stage-2 head; if candidate pool is small, pad with Stage-1 tail.
    if len(stage2_scores) >= final_top_k:
        return stage2_scores[:final_top_k], "stage2"

    merged = stage2_scores[:]
    seen = {idx for idx, _ in merged}
    for idx, score in stage1_top:
        if idx in seen:
            continue
        merged.append((idx, score))
        seen.add(idx)
        if len(merged) >= final_top_k:
            break
    return merged, "stage2+pad"


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k_blocks < 1:
        raise ValueError("--top_k_blocks must be >= 1.")
    if args.stage1_top_k_blocks < 1:
        raise ValueError("--stage1_top_k_blocks must be >= 1.")
    if args.stage1_max_files < 0:
        raise ValueError("--stage1_max_files must be >= 0.")
    if args.stage2_min_accept_blocks < 1:
        raise ValueError("--stage2_min_accept_blocks must be >= 1.")
    if args.top_k_files < 1:
        raise ValueError("--top_k_files must be >= 1.")
    if args.top_k_modules < 1:
        raise ValueError("--top_k_modules must be >= 1.")
    if args.top_k_entities < 1:
        raise ValueError("--top_k_entities must be >= 1.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage CodeRankEmbed retrieval with file-constrained second pass",
    )

    # Data
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSONL file")
    parser.add_argument("--index_dir", type=str, required=True, help="Directory containing pre-built indexes")
    parser.add_argument("--output_folder", type=str, required=True, help="Output directory for results")

    # Model
    parser.add_argument("--model_name", type=str, required=True, help="CodeRankEmbed model path")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow execution of model repository code")

    # Index strategy
    parser.add_argument(
        "--strategy",
        type=str,
        default="",
        help=(
            "Chunking strategy used during indexing (e.g. llamaindex_code). "
            "Used to locate the dense_index_{strategy}/ subdirectory."
        ),
    )

    # Retrieval
    parser.add_argument("--stage1_top_k_blocks", type=int, default=100, help="Global Stage-1 top blocks")
    parser.add_argument("--top_k_blocks", type=int, default=20, help="Final top blocks after Stage-2")
    parser.add_argument(
        "--stage1_max_files",
        type=int,
        default=0,
        help="Optional cap for Stage-1 seed files; 0 means no cap",
    )
    parser.add_argument(
        "--stage2_min_accept_blocks",
        type=int,
        default=5,
        help="Fail-open to Stage-1 if Stage-2 returns fewer than this many blocks",
    )
    parser.add_argument("--top_k_files", type=int, default=20, help="Number of top files to retrieve")
    parser.add_argument("--top_k_modules", type=int, default=20, help="Number of top modules to retrieve")
    parser.add_argument("--top_k_entities", type=int, default=50, help="Number of top entities to retrieve")
    parser.add_argument(
        "--file_score_agg",
        type=str,
        default="sum",
        choices=["sum", "max"],
        help="How to aggregate block scores to file scores.",
    )
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument(
        "--pooling",
        type=str,
        default="first_non_pad",
        choices=list(POOLING_CHOICES),
        help="Pooling strategy for dense query embeddings.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding")

    # Mapping
    parser.add_argument("--mapper_type", type=str, default="ast", choices=["ast", "graph"], help="Mapper type")
    parser.add_argument(
        "--repos_root",
        type=str,
        default="/workspace/locbench/repos/locbench_repos",
        help="Root directory of repositories",
    )
    parser.add_argument(
        "--graph_index_dir",
        type=str,
        default="",
        help="Graph index directory (required when mapper_type=graph).",
    )

    # Runtime
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()
    validate_args(args)

    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, tokenizer = load_model(args.model_name, device, args.trust_remote_code)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_jsonl(args.dataset_path)
    print(f"Loaded {len(dataset)} instances")
    print(
        "Two-stage config: "
        f"stage1_top_k_blocks={args.stage1_top_k_blocks}, "
        f"top_k_blocks={args.top_k_blocks}, "
        f"stage2_min_accept_blocks={args.stage2_min_accept_blocks}, "
        f"stage1_max_files={args.stage1_max_files}"
    )

    if args.mapper_type == "ast":
        mapper = ASTBasedMapper(args.repos_root)
    else:
        if not args.graph_index_dir:
            raise ValueError("mapper_type=graph requires --graph_index_dir")
        mapper = GraphBasedMapper(args.graph_index_dir)

    results = []

    # Cache per repo, keep embeddings on CPU to control GPU memory.
    index_cache: Dict[str, Tuple[Optional[torch.Tensor], Optional[List[dict]]]] = {}

    def get_cached_index(repo_name: str) -> Tuple[Optional[torch.Tensor], Optional[List[dict]]]:
        if repo_name not in index_cache:
            try:
                embeddings, metadata = load_index(repo_name, args.index_dir, strategy=args.strategy)
            except FileNotFoundError:
                index_cache[repo_name] = (None, None)
            else:
                index_cache[repo_name] = (embeddings, metadata)
        return index_cache[repo_name]

    index_found = 0
    index_missing = 0
    missing_repos: List[str] = []

    stage1_fallback_count = 0
    stage2_used_count = 0
    stage2_padded_count = 0
    stage2_exception_count = 0
    stage2_empty_candidate_count = 0
    stage2_too_small_count = 0

    for instance in tqdm(dataset, desc="Processing instances"):
        instance_id = instance.get("instance_id", "")
        repo_name = instance_id_to_repo_name(instance_id)

        try:
            embeddings, metadata = get_cached_index(repo_name)
            if embeddings is None or metadata is None:
                index_missing += 1
                missing_repos.append(repo_name)
                results.append(
                    {
                        "instance_id": instance_id,
                        "found_files": [],
                        "found_modules": [],
                        "found_entities": [],
                        "raw_output_loc": [],
                    }
                )
                continue

            index_found += 1

            query_text = get_problem_text(instance)
            if not query_text:
                print(f"Warning: No query text found for instance {instance_id}")
                results.append(
                    {
                        "instance_id": instance_id,
                        "found_files": [],
                        "found_modules": [],
                        "found_entities": [],
                        "raw_output_loc": [],
                    }
                )
                continue

            query_embedding = embed_texts(
                [query_text],
                model,
                tokenizer,
                args.max_length,
                1,
                device,
                pooling=args.pooling,
            )[0]

            embeddings_gpu = embeddings.to(device)

            # Stage 1: global retrieval
            stage1_scores = topk_block_scores(
                query_embedding,
                embeddings_gpu,
                args.stage1_top_k_blocks,
            )
            if not stage1_scores:
                results.append(
                    {
                        "instance_id": instance_id,
                        "found_files": [],
                        "found_modules": [],
                        "found_entities": [],
                        "raw_output_loc": [],
                    }
                )
                continue

            seed_files = extract_seed_files_from_blocks(
                stage1_scores,
                metadata,
                repo_name,
                max_files=args.stage1_max_files,
            )
            candidate_indices = build_candidate_indices_from_files(metadata, seed_files, repo_name)

            # Stage 2: file-constrained retrieval
            stage2_scores: List[Tuple[int, float]] = []
            stage2_failed = False
            if candidate_indices.numel() == 0:
                stage2_empty_candidate_count += 1
            else:
                try:
                    stage2_scores = topk_block_scores(
                        query_embedding,
                        embeddings_gpu,
                        args.top_k_blocks,
                        candidate_indices=candidate_indices,
                    )
                except Exception as e:
                    stage2_failed = True
                    stage2_exception_count += 1
                    print(f"[WARN] Stage-2 retrieval failed for {instance_id}: {e}")

            final_scores, select_reason = finalize_block_scores_with_fail_open(
                stage1_scores=stage1_scores,
                stage2_scores=stage2_scores,
                final_top_k=args.top_k_blocks,
                min_stage2_accept=args.stage2_min_accept_blocks,
            )

            if select_reason == "fallback_stage1":
                stage1_fallback_count += 1
                accept_threshold = min(max(1, args.stage2_min_accept_blocks), max(1, args.top_k_blocks))
                if not stage2_failed and stage2_scores and len(stage2_scores) < accept_threshold:
                    stage2_too_small_count += 1
            elif select_reason == "stage2+pad":
                stage2_used_count += 1
                stage2_padded_count += 1
            else:
                stage2_used_count += 1

            found_files = rank_files(
                final_scores,
                metadata,
                args.top_k_files,
                repo_name,
                file_score_agg=args.file_score_agg,
            )

            top_blocks = []
            for idx, _ in final_scores:
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

            results.append(
                {
                    "instance_id": instance_id,
                    "found_files": found_files,
                    "found_modules": found_modules,
                    "found_entities": found_entities,
                    "raw_output_loc": [],
                }
            )

        except Exception as e:
            print(f"Error processing instance {instance_id}: {e}")
            results.append(
                {
                    "instance_id": instance_id,
                    "error": str(e),
                    "found_files": [],
                    "found_modules": [],
                    "found_entities": [],
                }
            )

    total_instances = len(results)
    successful_instances = sum(1 for r in results if r.get("found_files"))
    failed_instances = sum(1 for r in results if "error" in r)

    if successful_instances > 0:
        avg_files = sum(len(r.get("found_files", [])) for r in results) / successful_instances
    else:
        avg_files = 0.0

    output_path = output_dir / "loc_outputs.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")
    print("Retrieval Summary:")
    print(f"  Total instances: {total_instances}")
    success_rate = (successful_instances / total_instances) if total_instances > 0 else 0.0
    print(f"  Successful: {successful_instances} ({success_rate:.1%})")
    print(f"  Failed: {failed_instances}")
    print(f"  Average files per successful instance: {avg_files:.1f}")

    print("\nIndex Statistics:")
    print(f"  Found: {index_found}/{len(dataset)}")
    print(f"  Missing: {index_missing}/{len(dataset)}")
    if missing_repos:
        unique_missing = list(set(missing_repos))[:10]
        print(f"  Missing repos (sample): {unique_missing}")

    print("\nTwo-stage Selection Statistics:")
    print(f"  Stage-2 used: {stage2_used_count}")
    print(f"  Stage-2 used with Stage-1 padding: {stage2_padded_count}")
    print(f"  Fallback to Stage-1: {stage1_fallback_count}")
    print(f"  Stage-2 empty candidates: {stage2_empty_candidate_count}")
    print(f"  Stage-2 too small (< stage2_min_accept_blocks): {stage2_too_small_count}")
    print(f"  Stage-2 exceptions: {stage2_exception_count}")


if __name__ == "__main__":
    main()
