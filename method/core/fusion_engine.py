from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from method.core.embedding import extract_last_hidden_state, pool_hidden_states
from method.retrieval.common import load_index
from method.retrieval.sparse_retriever import load_sparse_index, bm25_scores
from method.retrieval.summary_retriever import load_summary_index
from method.utils import clean_file_path


@dataclass
class BestProvenance:
    source: str
    original_id: str
    original_score: float
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    snippet: Optional[str] = None


@dataclass
class RetrievalItem:
    item_id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    source: str
    granularity: str
    items: List[RetrievalItem]


@dataclass
class AlignedResult:
    source: str
    items: List[RetrievalItem]
    provenance: Dict[str, BestProvenance]


class ModelCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, bool, str], Tuple[AutoTokenizer, AutoModel]] = {}

    def get(
        self,
        model_name: str,
        trust_remote_code: bool,
        device: torch.device,
    ) -> Tuple[AutoTokenizer, AutoModel]:
        key = (model_name, trust_remote_code, str(device))
        if key in self._cache:
            return self._cache[key]
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = model.to(device)
        model.eval()
        self._cache[key] = (tokenizer, model)
        return tokenizer, model


def _device(force_cpu: bool, gpu_id: int) -> torch.device:
    return torch.device("cpu" if force_cpu else f"cuda:{gpu_id}")


def _embed_query(
    text: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device,
    pooling: str = "first_non_pad",
) -> torch.Tensor:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        token_embeddings = extract_last_hidden_state(outputs)
        query_emb = pool_hidden_states(token_embeddings, attn_mask, pooling=pooling)
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    return query_emb.squeeze(0)


def _sum_decay(scores: List[float], decay: float, top_n: int) -> float:
    total = 0.0
    for i, s in enumerate(scores[:top_n]):
        total += s * (decay ** i)
    return total


def _normalize_scores(scores: List[float], method: str) -> List[float]:
    if not scores:
        return []
    if method == "rank":
        # caller uses ranks; return scores unchanged
        return scores
    if method == "minmax":
        lo = min(scores)
        hi = max(scores)
        if hi - lo <= 1e-8:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]
    if method == "zscore":
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std <= 1e-8:
            return [0.0 for _ in scores]
        return [(s - mean) / std for s in scores]
    if method == "softmax":
        arr = np.array(scores, dtype=np.float32)
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        denom = float(np.sum(exp)) if exp.size else 1.0
        return (exp / denom).tolist()
    if method == "sigmoid":
        return [1.0 / (1.0 + math.exp(-float(s))) for s in scores]
    raise ValueError(f"Unknown normalize method: {method}")


def fuse_rrf(
    ranked_lists: Iterable[List[RetrievalItem]],
    *,
    k: int = 60,
) -> List[RetrievalItem]:
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item.item_id] = scores.get(item.item_id, 0.0) + 1.0 / (k + rank)
    return [
        RetrievalItem(item_id=item_id, score=score, payload={"file_path": item_id})
        for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


def fuse_weighted(
    lists: Iterable[List[RetrievalItem]],
    *,
    weights: Iterable[float],
) -> List[RetrievalItem]:
    scores: Dict[str, float] = {}
    for items, w in zip(lists, weights):
        for item in items:
            scores[item.item_id] = scores.get(item.item_id, 0.0) + float(w) * float(item.score)
    return [
        RetrievalItem(item_id=item_id, score=score, payload={"file_path": item_id})
        for item_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


class BaseSource:
    source_type: str = ""

    def __init__(self, name: str, config: Dict[str, Any], model_cache: ModelCache) -> None:
        self.name = name
        self.config = config
        self.model_cache = model_cache

    def retrieve(self, query_text: str, repo_name: str) -> RetrievalResult:
        raise NotImplementedError


class DenseSource(BaseSource):
    source_type = "dense"

    def __init__(self, name: str, config: Dict[str, Any], model_cache: ModelCache) -> None:
        super().__init__(name, config, model_cache)
        self.index_dir = config["index_dir"]
        self.model_name = config["model_name"]
        self.trust_remote_code = bool(config.get("trust_remote_code", False))
        self.max_length = int(config.get("max_length", 1024))
        self.pooling = str(config.get("pooling", "first_non_pad"))
        self.top_k = int(config.get("top_k", 50))
        self.gpu_id = int(config.get("gpu_id", 0))
        self.force_cpu = bool(config.get("force_cpu", False))
        self._cache: Dict[str, Tuple[torch.Tensor, List[dict]]] = {}

    def _load(self, repo_name: str) -> Tuple[torch.Tensor, List[dict]]:
        if repo_name not in self._cache:
            self._cache[repo_name] = load_index(repo_name, self.index_dir)
        return self._cache[repo_name]

    def retrieve(self, query_text: str, repo_name: str) -> RetrievalResult:
        embeddings, metadata = self._load(repo_name)
        device = _device(self.force_cpu, self.gpu_id)
        tokenizer, model = self.model_cache.get(self.model_name, self.trust_remote_code, device)

        query_emb = _embed_query(
            query_text,
            model,
            tokenizer,
            self.max_length,
            device,
            pooling=self.pooling,
        )
        sims = torch.matmul(query_emb.unsqueeze(0), embeddings.to(device).t()).squeeze(0).cpu()
        k = min(self.top_k, sims.numel())
        if k <= 0:
            return RetrievalResult(source=self.name, granularity="block", items=[])
        top_vals, top_idx = torch.topk(sims, k)
        items: List[RetrievalItem] = []
        for idx, score in zip(top_idx.tolist(), top_vals.tolist()):
            meta = metadata[idx]
            item_id = f"{meta.get('file_path')}:{meta.get('start_line')}:{meta.get('end_line')}"
            items.append(
                RetrievalItem(
                    item_id=item_id,
                    score=float(score),
                    payload={
                        "block_id": idx,
                        "file_path": meta.get("file_path"),
                        "start_line": meta.get("start_line"),
                        "end_line": meta.get("end_line"),
                        "block_type": meta.get("block_type"),
                    },
                )
            )
        return RetrievalResult(source=self.name, granularity="block", items=items)


class SparseSource(BaseSource):
    source_type = "sparse"

    def __init__(self, name: str, config: Dict[str, Any], model_cache: ModelCache) -> None:
        super().__init__(name, config, model_cache)
        self.index_dir = config["index_dir"]
        self.top_k = int(config.get("top_k", 50))
        self._cache: Dict[str, Tuple] = {}

    def _load(self, repo_name: str):
        if repo_name not in self._cache:
            self._cache[repo_name] = load_sparse_index(repo_name, self.index_dir)
        return self._cache[repo_name]

    def retrieve(self, query_text: str, repo_name: str) -> RetrievalResult:
        tf_matrix, vocab, idf, doc_len, avgdl, k1, b, metadata = self._load(repo_name)
        sparse_raw = bm25_scores(query_text, tf_matrix, vocab, idf, doc_len, avgdl, k1, b)
        k = min(self.top_k, sparse_raw.shape[0])
        if k <= 0:
            return RetrievalResult(source=self.name, granularity="block", items=[])
        top_idx = np.argpartition(-sparse_raw, k - 1)[:k]
        top_scores = sparse_raw[top_idx]
        order = np.argsort(-top_scores)
        items: List[RetrievalItem] = []
        for idx, score in zip(top_idx[order].tolist(), top_scores[order].tolist()):
            meta = metadata[idx]
            item_id = f"{meta.get('file_path')}:{meta.get('start_line')}:{meta.get('end_line')}"
            items.append(
                RetrievalItem(
                    item_id=item_id,
                    score=float(score),
                    payload={
                        "block_id": idx,
                        "file_path": meta.get("file_path"),
                        "start_line": meta.get("start_line"),
                        "end_line": meta.get("end_line"),
                        "block_type": meta.get("block_type"),
                    },
                )
            )
        return RetrievalResult(source=self.name, granularity="block", items=items)


class SummarySource(BaseSource):
    source_type = "summary"

    def __init__(self, name: str, config: Dict[str, Any], model_cache: ModelCache) -> None:
        super().__init__(name, config, model_cache)
        self.index_dir = config["index_dir"]
        self.model_name = config["model_name"]
        self.trust_remote_code = bool(config.get("trust_remote_code", False))
        self.max_length = int(config.get("max_length", 512))
        self.pooling = str(config.get("pooling", "first_non_pad"))
        self.top_k = int(config.get("top_k", 80))
        self.gpu_id = int(config.get("gpu_id", 0))
        self.force_cpu = bool(config.get("force_cpu", False))
        self._cache: Dict[str, Tuple[torch.Tensor, List[str], Dict[str, dict]]] = {}

    def _load(self, repo_name: str) -> Tuple[torch.Tensor, List[str], Dict[str, dict]]:
        if repo_name not in self._cache:
            self._cache[repo_name] = load_summary_index(repo_name, self.index_dir)
        return self._cache[repo_name]

    def retrieve(self, query_text: str, repo_name: str) -> RetrievalResult:
        embeddings, index_map, docs = self._load(repo_name)
        device = _device(self.force_cpu, self.gpu_id)
        tokenizer, model = self.model_cache.get(self.model_name, self.trust_remote_code, device)

        query_emb = _embed_query(
            query_text,
            model,
            tokenizer,
            self.max_length,
            device,
            pooling=self.pooling,
        )
        scores = torch.matmul(embeddings.to(device), query_emb).cpu().tolist()
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]

        items: List[RetrievalItem] = []
        for idx in top_indices:
            if idx >= len(index_map):
                continue
            doc_id = index_map[idx]
            doc = docs.get(doc_id)
            if not doc:
                continue
            items.append(
                RetrievalItem(
                    item_id=str(doc_id),
                    score=float(scores[idx]),
                    payload={
                        "doc_id": doc_id,
                        "doc_type": doc.get("type"),
                        "file_path": doc.get("file_path"),
                        "module_path": doc.get("module_path"),
                        "start_line": doc.get("start_line"),
                        "end_line": doc.get("end_line"),
                    },
                )
            )
        return RetrievalResult(source=self.name, granularity="doc", items=items)


SOURCE_REGISTRY = {
    DenseSource.source_type: DenseSource,
    SparseSource.source_type: SparseSource,
    SummarySource.source_type: SummarySource,
}


class FusionEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.target_granularity = config.get("target_granularity", "file")

        align_cfg = config.get("aligner", {})
        self.aligner_type = align_cfg.get("type", "sum_decay")
        self.aligner_top_n = int(align_cfg.get("top_n", 5))
        self.aligner_decay = float(align_cfg.get("decay", 0.5))

        fusion_cfg = config.get("fusion", {})
        self.fusion_type = fusion_cfg.get("type", "rrf")
        self.rrf_k = int(fusion_cfg.get("rrf_k", 60))

        self.top_k_files = int(config.get("top_k", {}).get("files", 20))
        self.top_k_blocks = int(config.get("top_k", {}).get("blocks", 50))

        self.model_cache = ModelCache()
        self.sources: List[BaseSource] = []
        for src in config.get("sources", []):
            src_type = src.get("type")
            name = src.get("name", src_type)
            if src_type not in SOURCE_REGISTRY:
                raise ValueError(f"Unknown source type: {src_type}")
            self.sources.append(SOURCE_REGISTRY[src_type](name, src, self.model_cache))
        if not self.sources:
            raise ValueError("No sources configured for fusion engine.")
        if self.aligner_type != "sum_decay":
            raise ValueError(f"Unsupported aligner type in v1.0: {self.aligner_type}")

    def _align_result(self, result: RetrievalResult, repo_name: str) -> AlignedResult:
        if self.target_granularity != "file":
            raise NotImplementedError("Only file-level alignment is supported in v1.0")

        grouped: Dict[str, List[RetrievalItem]] = {}
        for item in result.items:
            doc_type = item.payload.get("doc_type")
            if result.granularity == "doc" and doc_type == "module":
                # 模块级文档不参与 file 级对齐
                continue
            file_path = item.payload.get("file_path")
            if not file_path:
                continue
            cleaned = clean_file_path(str(file_path), repo_name)
            grouped.setdefault(cleaned, []).append(item)

        aligned_items: List[RetrievalItem] = []
        provenance: Dict[str, BestProvenance] = {}

        for file_path, items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
            scores = [float(i.score) for i in items_sorted]
            agg_score = _sum_decay(scores, self.aligner_decay, self.aligner_top_n)

            aligned_items.append(RetrievalItem(item_id=file_path, score=agg_score, payload={"file_path": file_path}))

            best_item = items_sorted[0]
            provenance[file_path] = BestProvenance(
                source=result.source,
                original_id=str(best_item.payload.get("doc_id") or best_item.item_id),
                original_score=float(best_item.score),
                file_path=file_path,
                start_line=best_item.payload.get("start_line"),
                end_line=best_item.payload.get("end_line"),
            )

        return AlignedResult(source=result.source, items=aligned_items, provenance=provenance)

    def _choose_best_provenance(
        self,
        aligned_results: List[AlignedResult],
        file_id: str,
    ) -> Optional[BestProvenance]:
        best: Optional[BestProvenance] = None
        best_score = -float("inf")
        for aligned in aligned_results:
            for item in aligned.items:
                if item.item_id != file_id:
                    continue
                score = float(item.score)
                if score > best_score:
                    best_score = score
                    best = aligned.provenance.get(file_id)
        return best

    def retrieve(self, query_text: str, repo_name: str) -> Tuple[List[RetrievalItem], Dict[str, BestProvenance], List[Dict[str, Any]]]:
        source_results: List[RetrievalResult] = []
        for source in self.sources:
            source_results.append(source.retrieve(query_text, repo_name))

        # Build mapping blocks (best-effort)
        mapping_blocks = self._build_mapping_blocks(source_results, repo_name, self.top_k_blocks)

        aligned_results = [self._align_result(r, repo_name) for r in source_results]

        ranked_lists = []
        for aligned in aligned_results:
            ranked = sorted(aligned.items, key=lambda x: x.score, reverse=True)
            ranked_lists.append(ranked)

        if self.fusion_type == "rrf":
            fused = fuse_rrf(ranked_lists, k=self.rrf_k)
        elif self.fusion_type == "weighted_sum":
            weights = [float(s.config.get("weight", 1.0)) for s in self.sources]
            normalized_lists: List[List[RetrievalItem]] = []
            for aligned, source in zip(aligned_results, self.sources):
                method = source.config.get("normalize", "sigmoid")
                scores = [i.score for i in aligned.items]
                norm_scores = _normalize_scores(scores, method)
                normalized_lists.append([
                    RetrievalItem(item_id=i.item_id, score=ns, payload=i.payload)
                    for i, ns in zip(aligned.items, norm_scores)
                ])
            fused = fuse_weighted(normalized_lists, weights=weights)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        fused = fused[: self.top_k_files]

        best_prov: Dict[str, BestProvenance] = {}
        for item in fused:
            prov = self._choose_best_provenance(aligned_results, item.item_id)
            if prov is not None:
                best_prov[item.item_id] = prov

        return fused, best_prov, mapping_blocks

    def _build_mapping_blocks(
        self,
        results: List[RetrievalResult],
        repo_name: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        seen = set()
        for result in results:
            for item in result.items[:limit]:
                file_path = item.payload.get("file_path")
                start_line = item.payload.get("start_line")
                end_line = item.payload.get("end_line")
                if file_path is None or start_line is None or end_line is None:
                    continue
                cleaned = clean_file_path(str(file_path), repo_name)
                key = (cleaned, int(start_line), int(end_line))
                if key in seen:
                    continue
                seen.add(key)
                blocks.append(
                    {"file_path": cleaned, "start_line": int(start_line), "end_line": int(end_line)}
                )
        return blocks
