"""
Decoupled Summary Generation + Hybrid Storage (MVP skeleton).

Generator (heavy): scan -> AST -> LLM -> SQLite (nodes/edges)
Indexer   (fast):  read SQLite -> embeddings/BM25/graph -> stores
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from method.indexing import summary_index as summary_impl
from method.indexing.summary_index import (
    _apply_llm_timeout,
    _collect_function_docs,
    _aggregate_class_docs,
    _aggregate_file_docs,
    _aggregate_module_docs,
    _save_summary_jsonl,
    save_nodes_as_hierarchical_json,
    _load_prompt,
    _load_graph,
    _compute_graph_stats,
    DEFAULT_SUMMARY_PROMPT,
    DEFAULT_CLASS_PROMPT,
    DEFAULT_FILE_PROMPT,
)
from method.indexing.utils.file import set_file_scan_options


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_stable_id(relative_file_path: str, qualified_name: str) -> str:
    """AST strategy: {relative_file_path}::{qualified_name}"""
    return f"{relative_file_path}::{qualified_name}"


def make_chunk_id(relative_file_path: str, block_content: str) -> str:
    """Chunk strategy: {relative_file_path}::{md5(block_content)[:8]}"""
    digest = hashlib.md5(block_content.encode("utf-8")).hexdigest()[:8]
    return f"{relative_file_path}::{digest}"


def doc_to_stable_id(doc: Dict[str, Any]) -> str:
    file_path = doc.get("file_path") or ""
    doc_type = doc.get("type") or "function"

    qualified = doc.get("qualified_name") or doc.get("id") or ""
    name_parts: List[str] = []
    if "::" in qualified:
        parts = qualified.split("::")
        if len(parts) >= 2:
            name_parts = parts[1:]
    elif qualified:
        name_parts = [qualified]

    if doc_type == "file":
        return f"{file_path}::file"
    if doc_type == "module":
        module_path = doc.get("module_path") or file_path or "module"
        return f"{module_path}::module"

    qualified_dot = ".".join(name_parts) if name_parts else "unknown"
    return f"{file_path}::{qualified_dot}"


def stable_id_to_doc_id(stable_id: str, node_type: str) -> str:
    if "::" not in stable_id:
        return stable_id
    file_path, name = stable_id.split("::", 1)
    if node_type == "file":
        return file_path
    if node_type == "module":
        return file_path
    if "." in name:
        name = "::".join(name.split("."))
    return f"{file_path}::{name}"


class SummaryPayload(BaseModel):
    summary: str = ""
    business_intent: str = ""
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeModel(BaseModel):
    stable_id: str
    file_path: str
    type: Literal["function", "class", "file", "module", "chunk"]
    summary: SummaryPayload
    content_hash: str
    summary_hash: str
    updated_at: str = Field(default_factory=_utc_now_iso)


class EdgeModel(BaseModel):
    src_id: str
    dst_id: str
    edge_type: str
    updated_at: str = Field(default_factory=_utc_now_iso)


class GeneratorConfig(BaseModel):
    summary_levels: List[str] = Field(default_factory=lambda: ["function", "class", "file", "module"])
    graph_index_dir: Optional[str] = None

    summary_llm_provider: str = "openai"
    summary_llm_model: str = "gpt-3.5-turbo"
    summary_llm_api_key: Optional[str] = None
    summary_llm_api_base: Optional[str] = None
    summary_llm_max_new_tokens: int = 512
    summary_llm_temperature: float = 0.1
    llm_timeout: Optional[float] = 300
    llm_max_retries: int = 3
    llm_backoff: float = 1.0

    summary_prompt: Optional[str] = None
    summary_prompt_file: Optional[str] = None
    summary_language: str = "English"

    summary_cache_dir: Optional[str] = None
    summary_cache_policy: str = "read_write"
    summary_cache_miss: str = "empty"
    raw_log_path: Optional[str] = None

    filter_min_lines: int = 3
    filter_min_complexity: int = 2
    filter_skip_patterns: List[str] = Field(default_factory=lambda: ["^get_", "^set_", "^__.*__$"])

    top_k_callers: int = 5
    top_k_callees: int = 5
    context_similarity_threshold: float = 0.8

    lang_allowlist: str = ".py"
    skip_patterns: str = ""
    max_file_size_mb: Optional[float] = None


class IndexerConfig(BaseModel):
    embed_model_name: str
    trust_remote_code: bool = False
    batch_size: int = 8
    max_length: int = 512
    device: str = "cpu"
    output_dir: Optional[str] = None


class StorageManager:
    """
    Manages SQLite (gold), Chroma (dense), BM25 (sparse), NetworkX (graph).
    MVP: SQLite is required; others are optional based on availability.
    """

    def __init__(
        self,
        sqlite_path: str,
        chroma_dir: Optional[str] = None,
        bm25_dir: Optional[str] = None,
        graph_path: Optional[str] = None,
    ) -> None:
        self.sqlite_path = sqlite_path
        self.chroma_dir = chroma_dir
        self.bm25_dir = bm25_dir
        self.graph_path = graph_path
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        self._conn = sqlite3.connect(sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                stable_id TEXT PRIMARY KEY,
                file_path TEXT,
                type TEXT,
                summary_json TEXT,
                content_hash TEXT,
                summary_hash TEXT,
                updated_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                src_id TEXT,
                dst_id TEXT,
                edge_type TEXT,
                updated_at TEXT,
                UNIQUE(src_id, dst_id, edge_type)
            )
            """
        )
        self._conn.commit()

    def upsert_nodes(self, nodes: Sequence[NodeModel]) -> None:
        rows = [
            (
                n.stable_id,
                n.file_path,
                n.type,
                json.dumps(n.summary.dict(), ensure_ascii=False),
                n.content_hash,
                n.summary_hash,
                n.updated_at,
            )
            for n in nodes
        ]
        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT INTO nodes (stable_id, file_path, type, summary_json, content_hash, summary_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stable_id) DO UPDATE SET
              file_path=excluded.file_path,
              type=excluded.type,
              summary_json=excluded.summary_json,
              content_hash=excluded.content_hash,
              summary_hash=excluded.summary_hash,
              updated_at=excluded.updated_at
            """,
            rows,
        )
        self._conn.commit()

    def upsert_edges(self, edges: Sequence[EdgeModel]) -> None:
        rows = [(e.src_id, e.dst_id, e.edge_type, e.updated_at) for e in edges]
        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT INTO edges (src_id, dst_id, edge_type, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, edge_type) DO UPDATE SET
              updated_at=excluded.updated_at
            """,
            rows,
        )
        self._conn.commit()

    def load_nodes(self, updated_since: Optional[str] = None) -> List[NodeModel]:
        cur = self._conn.cursor()
        if updated_since:
            cur.execute("SELECT * FROM nodes WHERE updated_at > ?", (updated_since,))
        else:
            cur.execute("SELECT * FROM nodes")
        rows = cur.fetchall()
        nodes: List[NodeModel] = []
        for r in rows:
            payload = json.loads(r["summary_json"]) if r["summary_json"] else {}
            nodes.append(
                NodeModel(
                    stable_id=r["stable_id"],
                    file_path=r["file_path"],
                    type=r["type"],
                    summary=SummaryPayload(**payload),
                    content_hash=r["content_hash"],
                    summary_hash=r["summary_hash"],
                    updated_at=r["updated_at"],
                )
            )
        return nodes

    def load_edges(self, updated_since: Optional[str] = None) -> List[EdgeModel]:
        cur = self._conn.cursor()
        if updated_since:
            cur.execute("SELECT * FROM edges WHERE updated_at > ?", (updated_since,))
        else:
            cur.execute("SELECT * FROM edges")
        rows = cur.fetchall()
        return [
            EdgeModel(
                src_id=r["src_id"],
                dst_id=r["dst_id"],
                edge_type=r["edge_type"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    # ---- Dense vectors (Chroma) ----
    def upsert_vectors(self, ids: Sequence[str], embeddings: Sequence[Sequence[float]], metadatas=None) -> None:
        if not self.chroma_dir:
            return
        try:
            import chromadb
        except Exception as exc:  # pragma: no cover
            raise ImportError("chromadb is required for vector storage") from exc
        client = chromadb.PersistentClient(path=self.chroma_dir)
        col = client.get_or_create_collection("summary_nodes")
        col.upsert(ids=list(ids), embeddings=list(embeddings), metadatas=metadatas)

    # ---- BM25 ----
    def save_bm25(self, tokenized_docs: List[List[str]], ids: List[str]) -> None:
        if not self.bm25_dir:
            return
        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:  # pragma: no cover
            raise ImportError("rank_bm25 is required for BM25 storage") from exc
        bm25 = BM25Okapi(tokenized_docs)
        os.makedirs(self.bm25_dir, exist_ok=True)
        with open(os.path.join(self.bm25_dir, "bm25.pkl"), "wb") as f:
            pickle.dump({"bm25": bm25, "ids": ids}, f)

    # ---- Graph ----
    def save_graph(self, edges: Sequence[EdgeModel]) -> None:
        if not self.graph_path:
            return
        try:
            import networkx as nx
        except Exception as exc:  # pragma: no cover
            raise ImportError("networkx is required for graph storage") from exc
        g = nx.DiGraph()
        for e in edges:
            g.add_edge(e.src_id, e.dst_id, type=e.edge_type)
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        nx.write_gpickle(g, self.graph_path)


class Generator:
    """
    Heavy stage: scan -> AST -> edges -> LLM -> SQLite.
    Only writes nodes/edges to SQLite (gold).
    """

    def __init__(self, storage: StorageManager, config: GeneratorConfig) -> None:
        self.storage = storage
        self.config = config

    def run(self, repo_path: str) -> None:
        # NOTE: skeleton only. Implement actual AST parse + LLM later.
        nodes, edges = self._generate_nodes_edges(repo_path)
        self.storage.upsert_nodes(nodes)
        self.storage.upsert_edges(edges)

    def _generate_nodes_edges(self, repo_path: str) -> Tuple[List[NodeModel], List[EdgeModel]]:
        cfg = self.config
        repo_root = Path(repo_path)
        repo_name = repo_root.name

        set_file_scan_options(
            suffixes={p if p.startswith(".") else f".{p}" for p in cfg.lang_allowlist.split(",") if p.strip()},
            skip_patterns=[p.strip() for p in cfg.skip_patterns.split(",") if p.strip()] or None,
            max_file_size_mb=cfg.max_file_size_mb,
        )

        summary_prompt_template = _load_prompt(
            cfg.summary_prompt, cfg.summary_prompt_file, cfg.summary_language, default_template=DEFAULT_SUMMARY_PROMPT
        )
        class_prompt_template = _load_prompt(
            cfg.summary_prompt, cfg.summary_prompt_file, cfg.summary_language, default_template=DEFAULT_CLASS_PROMPT
        )
        file_prompt_template = _load_prompt(
            cfg.summary_prompt, cfg.summary_prompt_file, cfg.summary_language, default_template=DEFAULT_FILE_PROMPT
        )

        llm = summary_impl.get_llm(
            provider=cfg.summary_llm_provider,
            model_name=cfg.summary_llm_model,
            api_key=cfg.summary_llm_api_key,
            api_base=cfg.summary_llm_api_base,
            max_new_tokens=cfg.summary_llm_max_new_tokens,
            temperature=cfg.summary_llm_temperature,
        )
        _apply_llm_timeout(llm, cfg.llm_timeout)
        raw_log_hook = summary_impl._make_raw_log_hook(cfg.raw_log_path)

        # Build existing_docs mapping from SQLite for reuse
        existing_docs: Dict[str, Dict[str, Any]] = {}
        for node in self.storage.load_nodes():
            doc_id = stable_id_to_doc_id(node.stable_id, node.type)
            payload = node.summary.dict()
            existing_docs[doc_id] = {
                "summary": payload.get("summary", ""),
                "business_intent": payload.get("business_intent", ""),
                "keywords": payload.get("keywords", []),
                "content_hash": node.content_hash,
                "summary_hash": node.summary_hash,
                "context": (payload.get("metadata") or {}).get("context", {}),
            }

        graph = _load_graph(cfg.graph_index_dir, repo_name)
        graph_stats = _compute_graph_stats(graph)

        function_docs, _ = _collect_function_docs(
            repo_path=repo_path,
            graph=graph,
            graph_stats=graph_stats,
            llm=llm,
            summary_prompt=summary_prompt_template,
            summary_language=cfg.summary_language,
            summary_cache_dir=cfg.summary_cache_dir,
            summary_cache_policy=cfg.summary_cache_policy,
            summary_cache_miss=cfg.summary_cache_miss,
            min_lines=cfg.filter_min_lines,
            min_complexity=cfg.filter_min_complexity,
            skip_patterns=cfg.filter_skip_patterns,
            top_k_callers=cfg.top_k_callers,
            top_k_callees=cfg.top_k_callees,
            context_similarity_threshold=cfg.context_similarity_threshold,
            existing_docs=existing_docs,
            show_progress=False,
            progress_desc=f"{repo_name} functions",
            llm_max_retries=cfg.llm_max_retries,
            llm_backoff=cfg.llm_backoff,
            metrics_hook=None,
            summary_llm_model=cfg.summary_llm_model,
            raw_log_hook=raw_log_hook,
        )

        all_docs: List[Dict[str, Any]] = list(function_docs)
        class_docs: List[Dict[str, Any]] = []
        file_docs: List[Dict[str, Any]] = []
        module_docs: List[Dict[str, Any]] = []

        if "class" in cfg.summary_levels:
            class_docs = _aggregate_class_docs(
                repo_path=repo_path,
                function_docs=function_docs,
                llm=llm,
                summary_prompt=class_prompt_template,
                summary_language=cfg.summary_language,
                summary_cache_dir=cfg.summary_cache_dir,
                summary_cache_policy=cfg.summary_cache_policy,
                summary_cache_miss=cfg.summary_cache_miss,
                existing_docs=existing_docs,
                llm_max_retries=cfg.llm_max_retries,
                llm_backoff=cfg.llm_backoff,
                metrics_hook=None,
                summary_llm_model=cfg.summary_llm_model,
                raw_log_hook=raw_log_hook,
            )
            all_docs.extend(class_docs)

        if "file" in cfg.summary_levels:
            file_docs = _aggregate_file_docs(
                repo_path=repo_path,
                function_docs=function_docs,
                class_docs=class_docs,
                llm=llm,
                summary_prompt=file_prompt_template,
                summary_language=cfg.summary_language,
                summary_cache_dir=cfg.summary_cache_dir,
                summary_cache_policy=cfg.summary_cache_policy,
                summary_cache_miss=cfg.summary_cache_miss,
                existing_docs=existing_docs,
                llm_max_retries=cfg.llm_max_retries,
                llm_backoff=cfg.llm_backoff,
                metrics_hook=None,
                summary_llm_model=cfg.summary_llm_model,
                raw_log_hook=raw_log_hook,
            )
            all_docs.extend(file_docs)

        if "module" in cfg.summary_levels and file_docs:
            module_docs = _aggregate_module_docs(
                repo_path=repo_path,
                file_docs=file_docs,
                llm=llm,
                summary_prompt=file_prompt_template,
                summary_cache_dir=cfg.summary_cache_dir,
                summary_cache_policy=cfg.summary_cache_policy,
                summary_cache_miss=cfg.summary_cache_miss,
                existing_docs=existing_docs,
                llm_max_retries=cfg.llm_max_retries,
                llm_backoff=cfg.llm_backoff,
                metrics_hook=None,
                summary_llm_model=cfg.summary_llm_model,
                raw_log_hook=raw_log_hook,
            )
            all_docs.extend(module_docs)

        nodes: List[NodeModel] = []
        for doc in all_docs:
            stable_id = doc_to_stable_id(doc)
            payload = SummaryPayload(
                summary=doc.get("summary", ""),
                business_intent=doc.get("business_intent", ""),
                keywords=doc.get("keywords", []),
                metadata={
                    "context": doc.get("context", {}),
                    "qualified_name": doc.get("qualified_name"),
                    "start_line": doc.get("start_line"),
                    "end_line": doc.get("end_line"),
                    "module_path": doc.get("module_path"),
                    "type": doc.get("type"),
                    "is_placeholder": doc.get("is_placeholder", False),
                },
            )
            nodes.append(
                NodeModel(
                    stable_id=stable_id,
                    file_path=doc.get("file_path", ""),
                    type=doc.get("type", "function"),
                    summary=payload,
                    content_hash=doc.get("content_hash", ""),
                    summary_hash=doc.get("summary_hash", ""),
                )
            )

        edges: List[EdgeModel] = []
        if graph is not None:
            from dependency_graph.build_graph import EDGE_TYPE_INVOKES

            for u, v, data in graph.edges(data=True):
                if data.get("type") != EDGE_TYPE_INVOKES:
                    continue
                if ":" not in u or ":" not in v:
                    continue
                src_file, src_name = u.split(":", 1)
                dst_file, dst_name = v.split(":", 1)
                src_id = make_stable_id(src_file, src_name)
                dst_id = make_stable_id(dst_file, dst_name)
                edges.append(EdgeModel(src_id=src_id, dst_id=dst_id, edge_type=EDGE_TYPE_INVOKES))

        return nodes, edges


class Indexer:
    """
    Fast stage: read SQLite -> embeddings/BM25/graph.
    """

    def __init__(
        self,
        storage: StorageManager,
        embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def run(self, updated_since: Optional[str] = None) -> None:
        nodes = self.storage.load_nodes(updated_since=updated_since)
        edges = self.storage.load_edges(updated_since=updated_since)

        dense_nodes = [n for n in nodes if not (n.summary.metadata or {}).get("is_placeholder")]

        # Dense vectors (Chroma + optional flat files)
        embeddings = None
        if self.embedder:
            texts = [n.summary.summary for n in dense_nodes]
            if texts:
                embeddings = self.embedder(texts)
                metadatas = [{"file_path": n.file_path, "type": n.type} for n in dense_nodes]
                self.storage.upsert_vectors([n.stable_id for n in dense_nodes], embeddings, metadatas=metadatas)

        # BM25
        if self.tokenizer:
            tokenized = [self.tokenizer(n.summary.summary) for n in nodes]
            self.storage.save_bm25(tokenized, [n.stable_id for n in nodes])

        # Graph
        self.storage.save_graph(edges)

        # Optional: flat summary + dense outputs (compat with old retriever)
        if self.output_dir:
            out_root = Path(self.output_dir)
            out_root.mkdir(parents=True, exist_ok=True)

            docs = []
            for n in nodes:
                meta = n.summary.metadata or {}
                doc_id = stable_id_to_doc_id(n.stable_id, n.type)
                docs.append(
                    {
                        "id": doc_id,
                        "type": n.type,
                        "file_path": n.file_path,
                        "module_path": meta.get("module_path"),
                        "qualified_name": meta.get("qualified_name"),
                        "summary": n.summary.summary,
                        "business_intent": n.summary.business_intent,
                        "keywords": n.summary.keywords,
                        "context": meta.get("context", {}),
                        "content_hash": n.content_hash,
                        "summary_hash": n.summary_hash,
                        "hash": n.summary_hash,
                        "language": meta.get("language", "python"),
                        "last_updated": n.updated_at,
                        "is_placeholder": meta.get("is_placeholder", False),
                        "start_line": meta.get("start_line"),
                        "end_line": meta.get("end_line"),
                    }
                )

            _save_summary_jsonl(out_root / "summary.jsonl", docs)
            save_nodes_as_hierarchical_json(docs, out_root / "summary_ast")

            try:
                import torch

                dense_dir = out_root / "dense"
                dense_dir.mkdir(parents=True, exist_ok=True)
                if embeddings is None:
                    tensor = torch.empty((0, 0))
                    index_map = []
                else:
                    tensor = torch.tensor(embeddings)
                    index_map = [stable_id_to_doc_id(n.stable_id, n.type) for n in dense_nodes]
                torch.save(tensor, dense_dir / "embeddings.pt")
                with (dense_dir / "index_map.json").open("w", encoding="utf-8") as f:
                    json.dump(index_map, f, ensure_ascii=False)
            except Exception:
                pass
