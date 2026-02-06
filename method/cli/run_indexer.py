#!/usr/bin/env python
import argparse

import torch
from transformers import AutoModel, AutoTokenizer

from method.indexing.decoupled_pipeline import Indexer, IndexerConfig, StorageManager
from method.indexing.encoder.sparse.bm25 import tokenize
from method.indexing.summary_index import _embed_texts


def build_embedder(cfg: IndexerConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.embed_model_name, trust_remote_code=cfg.trust_remote_code)
    model = AutoModel.from_pretrained(cfg.embed_model_name, trust_remote_code=cfg.trust_remote_code)
    device = torch.device(cfg.device)
    model.to(device)

    def _embed(texts):
        embeddings = _embed_texts(
            texts,
            model=model,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            device=device,
        )
        return embeddings.tolist()

    return _embed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build indexes from SQLite (decoupled).")
    parser.add_argument("--sqlite_path", required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--chroma_dir", type=str, default="")
    parser.add_argument("--bm25_dir", type=str, default="")
    parser.add_argument("--graph_path", type=str, default="")
    parser.add_argument("--embed_model_name", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--updated_since", type=str, default=None)
    args = parser.parse_args()

    device = "cpu" if args.force_cpu else f"cuda:{args.gpu_id}"
    cfg = IndexerConfig(
        embed_model_name=args.embed_model_name,
        trust_remote_code=args.trust_remote_code,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    storage = StorageManager(
        sqlite_path=args.sqlite_path,
        chroma_dir=args.chroma_dir or None,
        bm25_dir=args.bm25_dir or None,
        graph_path=args.graph_path or None,
    )

    embedder = build_embedder(cfg) if args.embed_model_name else None
    indexer = Indexer(
        storage=storage,
        embedder=embedder,
        tokenizer=tokenize,
        output_dir=args.output_dir or None,
    )
    indexer.run(updated_since=args.updated_since)


if __name__ == "__main__":
    main()
