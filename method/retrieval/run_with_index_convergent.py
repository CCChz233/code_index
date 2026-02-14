#!/usr/bin/env python
"""
Dense retrieval with optional multi-step convergence and optional reranking.
Single entrypoint for baseline / PRF / global_local / rerank combinations.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
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
from method.retrieval.fusion.rrf import fuse_rrf
from method.utils import clean_file_path

logger = logging.getLogger(__name__)


def embed_texts(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    pooling: str = "first_non_pad",
) -> torch.Tensor:
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
            # Warn if any text was truncated
            for j, text in enumerate(batch):
                token_count = encoded["attention_mask"][j].sum().item()
                if token_count >= max_length:
                    orig_tokens = len(tokenizer.encode(text, add_special_tokens=True))
                    if orig_tokens > max_length:
                        logger.warning(
                            "Query text truncated from %d to %d tokens (%.0f%% lost)",
                            orig_tokens, max_length,
                            (orig_tokens - max_length) / orig_tokens * 100,
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
    print(f"Loading dense model from {model_name}")

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
    print("Dense model loaded successfully")
    return model, tokenizer


class CrossEncoderReranker:
    """Cross-Encoder reranker that scores (query, snippet) pairs."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        *,
        batch_size: int,
        max_length: int,
        context_lines: int,
        snippet_max_lines: int,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.context_lines = context_lines
        self.snippet_max_lines = snippet_max_lines
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self.model.eval()
        self._file_cache: Dict[str, List[str]] = {}
        self._file_cache_max_size: int = 2000

    def _resolve_file_path(self, repos_root: str, repo_name: str, rel_file_path: str) -> Optional[Path]:
        repo_root = Path(repos_root) / repo_name
        repo_root_resolved = repo_root.resolve()
        candidate = (repo_root / rel_file_path).resolve()
        try:
            candidate.relative_to(repo_root_resolved)
        except ValueError:
            return None
        return candidate

    def _read_file_lines(self, file_path: Path) -> Optional[List[str]]:
        key = str(file_path)
        if key in self._file_cache:
            return self._file_cache[key]

        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            return None

        # Evict oldest entries when cache is full
        if len(self._file_cache) >= self._file_cache_max_size:
            evict_keys = list(self._file_cache.keys())[: self._file_cache_max_size // 4]
            for k in evict_keys:
                del self._file_cache[k]

        self._file_cache[key] = lines
        return lines

    def _build_snippet(self, repos_root: str, repo_name: str, meta: dict) -> Optional[str]:
        rel_file_path = meta.get("file_path", "")
        if not rel_file_path:
            return None

        abs_path = self._resolve_file_path(repos_root, repo_name, rel_file_path)
        if abs_path is None:
            return None

        lines = self._read_file_lines(abs_path)
        if lines is None or len(lines) == 0:
            return None

        try:
            start_line = int(meta.get("start_line", 0))
            end_line = int(meta.get("end_line", start_line))
        except (TypeError, ValueError):
            return None

        if end_line < start_line:
            end_line = start_line

        start_line = max(0, min(start_line, len(lines) - 1))
        end_line = max(start_line, min(end_line, len(lines) - 1))

        snippet_start = max(0, start_line - self.context_lines)
        snippet_end = min(len(lines) - 1, end_line + self.context_lines)

        snippet_lines = lines[snippet_start : snippet_end + 1]
        if self.snippet_max_lines > 0 and len(snippet_lines) > self.snippet_max_lines:
            snippet_lines = snippet_lines[: self.snippet_max_lines]
            snippet_end = snippet_start + len(snippet_lines) - 1

        code = "\n".join(snippet_lines).strip()
        if not code:
            return None

        return (
            f"File: {rel_file_path}\n"
            f"Lines: {snippet_start + 1}-{snippet_end + 1}\n"
            f"Code:\n{code}"
        )

    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                queries = [q for q, _ in batch]
                snippets = [s for _, s in batch]
                encoded = self.tokenizer(
                    queries,
                    snippets,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                logits = outputs.logits
                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)
                if logits.size(-1) == 1:
                    batch_scores = torch.sigmoid(logits.squeeze(-1))
                else:
                    batch_scores = torch.softmax(logits, dim=-1)[:, -1]
                scores.extend(batch_scores.detach().cpu().tolist())
        return scores

    def rerank_head(
        self,
        query_text: str,
        repo_name: str,
        head_candidates: List[Tuple[int, float]],
        metadata: List[dict],
        repos_root: str,
    ) -> List[Tuple[int, float]]:
        if not head_candidates:
            return []

        pair_payload: List[Tuple[int, float, Tuple[str, str]]] = []
        fallback: List[Tuple[int, float]] = []

        for idx, dense_score in head_candidates:
            snippet = self._build_snippet(repos_root, repo_name, metadata[idx])
            if snippet is None:
                fallback.append((idx, float(dense_score)))
                continue
            pair_payload.append((idx, float(dense_score), (query_text, snippet)))

        if not pair_payload:
            return head_candidates

        pair_inputs = [item[2] for item in pair_payload]
        rerank_scores = self._score_pairs(pair_inputs)

        reranked = sorted(
            [(pair_payload[i][0], float(rerank_scores[i])) for i in range(len(pair_payload))],
            key=lambda x: x[1],
            reverse=True,
        )
        return reranked + fallback


def topk_block_scores(
    query_embedding: torch.Tensor,
    embeddings_gpu: torch.Tensor,
    top_k: int,
    candidate_indices: Optional[torch.Tensor] = None,
) -> List[Tuple[int, float]]:
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


def rank_raw_files(
    block_scores: List[Tuple[int, float]],
    metadata: List[dict],
    top_k_files: int,
    repo_name: str,
    *,
    file_score_agg: str,
) -> List[str]:
    if file_score_agg not in {"sum", "max"}:
        raise ValueError(f"Unknown file_score_agg: {file_score_agg}")

    file_scores: Dict[str, float] = {}
    for block_idx, score in block_scores:
        file_path = metadata[block_idx].get("file_path", "")
        if not file_path:
            continue
        file_path = clean_file_path(file_path, repo_name)
        if file_score_agg == "max":
            file_scores[file_path] = max(file_scores.get(file_path, 0.0), float(score))
        else:
            file_scores[file_path] = file_scores.get(file_path, 0.0) + float(score)

    ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_k_files]]


def build_candidate_indices_from_files(
    metadata: List[dict],
    seed_files: List[str],
    repo_name: str,
) -> torch.Tensor:
    seed_set = set(seed_files)
    if not seed_set:
        return torch.empty((0,), dtype=torch.long)
    idx_list = [
        idx for idx, item in enumerate(metadata)
        if clean_file_path(item.get("file_path", ""), repo_name) in seed_set
    ]
    if not idx_list:
        return torch.empty((0,), dtype=torch.long)
    return torch.tensor(idx_list, dtype=torch.long)


def select_feedback_blocks(
    block_scores: List[Tuple[int, float]],
    metadata: List[dict],
    feedback_top_m: int,
    feedback_file_cap: int,
) -> List[Tuple[int, float]]:
    if feedback_top_m <= 0:
        return []

    selected: List[Tuple[int, float]] = []
    selected_ids = set()
    file_counter: Dict[str, int] = {}

    for idx, score in block_scores:
        file_path = metadata[idx].get("file_path", "")
        if feedback_file_cap > 0 and file_counter.get(file_path, 0) >= feedback_file_cap:
            continue
        selected.append((idx, float(score)))
        selected_ids.add(idx)
        file_counter[file_path] = file_counter.get(file_path, 0) + 1
        if len(selected) >= feedback_top_m:
            break

    return selected


def update_query_embedding(
    q_cur: torch.Tensor,
    q0: torch.Tensor,
    feedback_blocks: List[Tuple[int, float]],
    embeddings_cpu: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    temperature: float,
) -> torch.Tensor:
    if not feedback_blocks:
        return q_cur

    idx = torch.tensor([i for i, _ in feedback_blocks], dtype=torch.long)
    score = torch.tensor([s for _, s in feedback_blocks], dtype=torch.float32)

    feedback_emb = embeddings_cpu.index_select(0, idx)
    temp = max(temperature, 1e-6)
    weights = torch.softmax(score / temp, dim=0)
    centroid = (feedback_emb * weights.unsqueeze(1)).sum(dim=0)
    centroid = torch.nn.functional.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)

    base_weight = max(0.0, 1.0 - alpha - beta)
    # q0 anchors with base_weight, centroid provides feedback, q_cur carries
    # limited momentum via beta — prevents multi-round drift from q0.
    q_next = base_weight * q0 + alpha * centroid + beta * q_cur
    q_next = torch.nn.functional.normalize(q_next.unsqueeze(0), p=2, dim=1).squeeze(0)
    return q_next


def jaccard_at_k(cur_top_idx: List[int], prev_top_idx: List[int], k: int) -> float:
    if k <= 0:
        return 1.0
    cur = set(cur_top_idx[:k])
    prev = set(prev_top_idx[:k])
    if not cur and not prev:
        return 1.0
    union = cur | prev
    if not union:
        return 1.0
    return len(cur & prev) / len(union)


def iterative_block_retrieval(
    query_embedding: torch.Tensor,
    embeddings_cpu: torch.Tensor,
    embeddings_gpu: torch.Tensor,
    metadata: List[dict],
    args: argparse.Namespace,
    retrieve_top_k: int,
    repo_name: str = "",
) -> Tuple[List[Tuple[int, float]], int]:
    if retrieve_top_k <= 0:
        return [], 0

    per_round_top_k = max(retrieve_top_k, args.top_k_blocks_expand)

    if args.convergence_mode == "off":
        block_scores = topk_block_scores(query_embedding, embeddings_gpu, retrieve_top_k)
        return block_scores, (1 if block_scores else 0)

    q0 = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
    q = q0.clone()

    round_rankings: List[List[Tuple[int, float]]] = []
    prev_top_idx: Optional[List[int]] = None
    prev_mean_score: Optional[float] = None
    stable_rounds = 0
    candidate_indices: Optional[torch.Tensor] = None

    for step in range(args.max_steps):
        if args.convergence_mode == "global_local" and step > 0:
            current_candidates = candidate_indices
        else:
            current_candidates = None

        block_scores = topk_block_scores(
            q,
            embeddings_gpu,
            per_round_top_k,
            candidate_indices=current_candidates,
        )
        if not block_scores:
            break

        round_rankings.append(block_scores)

        top_idx = [i for i, _ in block_scores]
        top_scores = [float(s) for _, s in block_scores]
        gain_k = min(args.converge_jaccard_k, len(top_scores))
        cur_mean_score = (sum(top_scores[:gain_k]) / gain_k) if gain_k > 0 else 0.0

        if prev_top_idx is not None and prev_mean_score is not None:
            jaccard = jaccard_at_k(top_idx, prev_top_idx, args.converge_jaccard_k)
            gain = cur_mean_score - prev_mean_score
            if jaccard >= args.converge_jaccard_threshold and gain <= args.converge_min_improve:
                stable_rounds += 1
            else:
                stable_rounds = 0
            if stable_rounds >= args.patience:
                break

        prev_top_idx = top_idx
        prev_mean_score = cur_mean_score

        if step >= args.max_steps - 1:
            break

        if args.convergence_mode == "global_local" and step == 0:
            seed_files = rank_raw_files(
                block_scores,
                metadata,
                args.top_k_seed_files,
                repo_name,
                file_score_agg=args.file_score_agg,
            )
            candidate_indices = build_candidate_indices_from_files(metadata, seed_files, repo_name)
            if candidate_indices.numel() == 0:
                candidate_indices = None

        feedback_blocks = select_feedback_blocks(
            block_scores,
            metadata,
            args.feedback_top_m,
            args.feedback_file_cap,
        )
        if not feedback_blocks:
            break

        q_next = update_query_embedding(
            q,
            q0,
            feedback_blocks,
            embeddings_cpu,
            alpha=args.query_update_alpha,
            beta=args.query_anchor_beta,
            temperature=args.feedback_temp,
        )
        cos_to_q0 = torch.nn.functional.cosine_similarity(
            q_next.unsqueeze(0),
            q0.unsqueeze(0),
        ).item()
        if cos_to_q0 < args.min_cos_to_q0:
            break
        q = q_next

    if not round_rankings:
        return [], 0

    if args.round_fusion == "last" or len(round_rankings) == 1:
        return round_rankings[-1][:retrieve_top_k], len(round_rankings)

    fused = fuse_rrf(round_rankings, k=args.rrf_k)
    return fused[:retrieve_top_k], len(round_rankings)


def maybe_rerank(
    block_scores: List[Tuple[int, float]],
    query_text: str,
    repo_name: str,
    metadata: List[dict],
    reranker: Optional[CrossEncoderReranker],
    args: argparse.Namespace,
) -> Tuple[List[Tuple[int, float]], bool, bool]:
    if reranker is None or not args.enable_rerank:
        return block_scores, False, False

    if not block_scores or args.rerank_top_k <= 0:
        return block_scores, False, False

    top_n = min(args.rerank_top_k, len(block_scores))
    head = block_scores[:top_n]
    tail = block_scores[top_n:]

    try:
        reranked_head = reranker.rerank_head(
            query_text=query_text,
            repo_name=repo_name,
            head_candidates=head,
            metadata=metadata,
            repos_root=args.repos_root,
        )
        # Assign tail items monotonically decreasing scores below the min
        # reranked head score so that score semantics stay consistent when
        # downstream aggregation (e.g. sum/max) mixes head and tail.
        if reranked_head:
            min_rerank_score = min(s for _, s in reranked_head)
        else:
            min_rerank_score = 0.0
        adjusted_tail = [
            (idx, min_rerank_score - (i + 1) * 1e-6)
            for i, (idx, _) in enumerate(tail)
        ]
        return reranked_head + adjusted_tail, True, False
    except Exception as e:
        if args.rerank_fail_open:
            print(f"[WARN] Rerank failed for {repo_name}: {e}. Fallback to stage-1 ranking.")
            return block_scores, False, True
        raise


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k_blocks < 1:
        raise ValueError("--top_k_blocks must be >= 1.")
    if args.top_k_files < 1:
        raise ValueError("--top_k_files must be >= 1.")
    if args.top_k_modules < 1:
        raise ValueError("--top_k_modules must be >= 1.")
    if args.top_k_entities < 1:
        raise ValueError("--top_k_entities must be >= 1.")

    if args.enable_rerank:
        if args.rerank_top_k < 1:
            raise ValueError("--rerank_top_k must be >= 1 when --enable_rerank is set.")
        if args.rerank_batch_size < 1:
            raise ValueError("--rerank_batch_size must be >= 1.")
        if args.rerank_max_length < 8:
            raise ValueError("--rerank_max_length must be >= 8.")
        if args.rerank_context_lines < 0:
            raise ValueError("--rerank_context_lines must be >= 0.")
        if args.rerank_snippet_max_lines < 1:
            raise ValueError("--rerank_snippet_max_lines must be >= 1.")

    if args.convergence_mode == "off":
        return

    if args.max_steps < 1:
        raise ValueError("--max_steps must be >= 1 when convergence is enabled.")
    if args.top_k_blocks_expand < 1:
        raise ValueError("--top_k_blocks_expand must be >= 1 when convergence is enabled.")
    if args.feedback_top_m < 1:
        raise ValueError("--feedback_top_m must be >= 1 when convergence is enabled.")
    if args.feedback_file_cap < 0:
        raise ValueError("--feedback_file_cap must be >= 0.")
    if args.patience < 1:
        raise ValueError("--patience must be >= 1.")
    if args.min_cos_to_q0 < -1.0 or args.min_cos_to_q0 > 1.0:
        raise ValueError("--min_cos_to_q0 must be in [-1, 1].")
    if args.query_update_alpha < 0.0 or args.query_anchor_beta < 0.0:
        raise ValueError("--query_update_alpha and --query_anchor_beta must be >= 0.")
    if args.query_update_alpha + args.query_anchor_beta > 1.0:
        raise ValueError("query_update_alpha + query_anchor_beta must be <= 1.0.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dense retrieval with optional convergence and optional reranking",
    )

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSONL file")
    parser.add_argument("--index_dir", type=str, required=True, help="Directory containing pre-built indexes")
    parser.add_argument("--output_folder", type=str, required=True, help="Output directory for results")

    parser.add_argument("--model_name", type=str, required=True, help="Dense bi-encoder model path")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow execution of model repository code")

    parser.add_argument("--top_k_blocks", type=int, default=50, help="Number of final top blocks")
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

    parser.add_argument(
        "--convergence_mode",
        type=str,
        default="off",
        choices=["off", "prf", "global_local"],
        help="Convergence mode for stage-1 retrieval.",
    )
    parser.add_argument("--max_steps", type=int, default=3, help="Max retrieval rounds when convergence is enabled")
    parser.add_argument(
        "--top_k_blocks_expand",
        type=int,
        default=100,
        help="Per-round block cutoff before round fusion",
    )
    parser.add_argument("--feedback_top_m", type=int, default=8, help="Number of feedback blocks per round")
    parser.add_argument(
        "--feedback_file_cap",
        type=int,
        default=2,
        help="Max feedback blocks from a single file; 0 means no cap",
    )
    parser.add_argument("--query_update_alpha", type=float, default=0.35, help="Feedback update weight")
    parser.add_argument("--query_anchor_beta", type=float, default=0.15, help="Anchor-to-q0 weight")
    parser.add_argument("--feedback_temp", type=float, default=0.05, help="Softmax temperature for feedback")
    parser.add_argument(
        "--round_fusion",
        type=str,
        default="rrf",
        choices=["last", "rrf"],
        help="How to merge multi-round rankings",
    )
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF k parameter")
    parser.add_argument("--converge_jaccard_k", type=int, default=50, help="Top-k for Jaccard convergence check")
    parser.add_argument(
        "--converge_jaccard_threshold",
        type=float,
        default=0.90,
        help="Jaccard threshold for stable convergence",
    )
    parser.add_argument(
        "--converge_min_improve",
        type=float,
        default=0.002,
        help="Min mean-score improvement threshold for convergence",
    )
    parser.add_argument("--patience", type=int, default=1, help="Stable rounds required before early stop")
    parser.add_argument("--min_cos_to_q0", type=float, default=0.75, help="Minimum cosine(query_t, q0)")
    parser.add_argument("--top_k_seed_files", type=int, default=20, help="Seed files for global_local mode")

    parser.add_argument("--enable_rerank", action="store_true", help="Enable Cross-Encoder reranking")
    parser.add_argument(
        "--rerank_model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-Encoder model name/path",
    )
    parser.add_argument("--rerank_top_k", type=int, default=50, help="Rerank top-N stage-1 candidates")
    parser.add_argument("--rerank_batch_size", type=int, default=8, help="Batch size for rerank inference")
    parser.add_argument("--rerank_max_length", type=int, default=512, help="Max pair length for reranker")
    parser.add_argument(
        "--rerank_context_lines",
        type=int,
        default=0,
        help="Extra context lines around block when building rerank snippet",
    )
    parser.add_argument(
        "--rerank_snippet_max_lines",
        type=int,
        default=120,
        help="Maximum snippet lines for rerank input",
    )
    parser.add_argument("--rerank_trust_remote_code", action="store_true", help="Trust remote code for reranker")
    parser.add_argument(
        "--rerank_fail_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to stage-1 ordering if rerank fails",
    )

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

    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()
    validate_args(args)

    retrieve_top_k = max(args.top_k_blocks, args.rerank_top_k if args.enable_rerank else 0)
    if args.convergence_mode != "off" and args.top_k_blocks_expand < retrieve_top_k:
        args.top_k_blocks_expand = retrieve_top_k

    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model, tokenizer = load_model(args.model_name, device, args.trust_remote_code)
    except Exception as e:
        print(f"Failed to load dense model: {e}")
        return

    reranker: Optional[CrossEncoderReranker] = None
    if args.enable_rerank:
        try:
            print(f"Loading reranker model from {args.rerank_model}")
            reranker = CrossEncoderReranker(
                model_name=args.rerank_model,
                device=device,
                batch_size=args.rerank_batch_size,
                max_length=args.rerank_max_length,
                context_lines=args.rerank_context_lines,
                snippet_max_lines=args.rerank_snippet_max_lines,
                trust_remote_code=args.rerank_trust_remote_code,
            )
            print("Reranker loaded successfully")
        except Exception as e:
            if args.rerank_fail_open:
                print(f"[WARN] Failed to load reranker: {e}. Continue without rerank.")
                reranker = None
            else:
                print(f"Failed to load reranker: {e}")
                return

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_jsonl(args.dataset_path)
    print(f"Loaded {len(dataset)} instances")
    print(f"Convergence mode: {args.convergence_mode}")
    print(f"Rerank enabled: {args.enable_rerank and reranker is not None}")

    if args.mapper_type == "ast":
        mapper = ASTBasedMapper(args.repos_root)
    else:
        if not args.graph_index_dir:
            raise ValueError("mapper_type=graph requires --graph_index_dir")
        mapper = GraphBasedMapper(args.graph_index_dir)

    results = []
    index_cache: Dict[str, Tuple[Optional[torch.Tensor], Optional[List[dict]]]] = {}

    def get_cached_index(repo_name: str) -> Tuple[Optional[torch.Tensor], Optional[List[dict]]]:
        if repo_name not in index_cache:
            try:
                embeddings, metadata = load_index(repo_name, args.index_dir)
            except FileNotFoundError:
                index_cache[repo_name] = (None, None)
            else:
                index_cache[repo_name] = (embeddings, metadata)
        return index_cache[repo_name]

    index_found = 0
    index_missing = 0
    missing_repos = []
    convergence_rounds_total = 0
    convergence_instances = 0
    rerank_attempted = 0
    rerank_applied = 0
    rerank_failed = 0

    # Cache the current repo's GPU tensor to avoid repeated .to(device)
    _gpu_cache_repo: Optional[str] = None
    _gpu_cache_tensor: Optional[torch.Tensor] = None

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

            # Reuse cached GPU tensor for the same repo
            if repo_name != _gpu_cache_repo:
                _gpu_cache_tensor = embeddings.to(device)
                _gpu_cache_repo = repo_name
            embeddings_gpu = _gpu_cache_tensor

            block_scores, rounds_used = iterative_block_retrieval(
                query_embedding=query_embedding,
                embeddings_cpu=embeddings,
                embeddings_gpu=embeddings_gpu,
                metadata=metadata,
                args=args,
                retrieve_top_k=retrieve_top_k,
                repo_name=repo_name,
            )
            if rounds_used > 0:
                convergence_rounds_total += rounds_used
                convergence_instances += 1

            if args.enable_rerank and reranker is not None:
                rerank_attempted += 1
            block_scores, applied, failed = maybe_rerank(
                block_scores=block_scores,
                query_text=query_text,
                repo_name=repo_name,
                metadata=metadata,
                reranker=reranker,
                args=args,
            )
            if applied:
                rerank_applied += 1
            if failed:
                rerank_failed += 1

            block_scores = block_scores[: args.top_k_blocks]

            if not block_scores:
                found_files = []
                found_modules = []
                found_entities = []
            else:
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
    successful_instances = sum(1 for r in results if r.get("found_files") and len(r["found_files"]) > 0)
    failed_instances = sum(1 for r in results if "error" in r)
    avg_files = (
        sum(len(r.get("found_files", [])) for r in results) / successful_instances
        if successful_instances > 0
        else 0.0
    )

    output_path = output_dir / "loc_outputs.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Results saved to {output_path}")
    print("Retrieval Summary:")
    print(f"  Total instances: {total_instances}")
    print(f"  Successful: {successful_instances} ({successful_instances / total_instances:.1%})")
    print(f"  Failed: {failed_instances}")
    print(f"  Average files per successful instance: {avg_files:.1f}")

    print("\nIndex Statistics:")
    print(f"  Found: {index_found}/{len(dataset)}")
    print(f"  Missing: {index_missing}/{len(dataset)}")
    if missing_repos:
        unique_missing = list(set(missing_repos))[:10]
        print(f"  Missing repos (sample): {unique_missing}")

    if convergence_instances > 0:
        avg_rounds = convergence_rounds_total / convergence_instances
        print("\nConvergence Statistics:")
        print(f"  Instances with retrieval rounds: {convergence_instances}")
        print(f"  Average rounds used: {avg_rounds:.2f}")

    if args.enable_rerank:
        print("\nRerank Statistics:")
        print(f"  Attempted instances: {rerank_attempted}")
        print(f"  Applied instances: {rerank_applied}")
        print(f"  Failed instances: {rerank_failed}")


if __name__ == "__main__":
    main()
