#!/usr/bin/env python
"""
Summary Index retriever (dense-only with summary.jsonl metadata).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from method.mapping import ASTBasedMapper, GraphBasedMapper
from method.retrieval.common import (
    instance_id_to_repo_name,
    get_problem_text,
    load_jsonl,
)


def load_summary_index(repo_name: str, index_dir: str) -> Tuple[torch.Tensor, List[str], Dict[str, dict]]:
    base = Path(index_dir) / repo_name
    summary_path = base / "summary.jsonl"
    dense_dir = base / "dense"
    embeddings_path = dense_dir / "embeddings.pt"
    index_map_path = dense_dir / "index_map.json"

    if not embeddings_path.exists() or not index_map_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Summary index missing for repo {repo_name}")

    try:
        embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    except TypeError:
        embeddings = torch.load(embeddings_path, map_location="cpu")

    with index_map_path.open("r", encoding="utf-8") as f:
        index_map = json.load(f)
    if not isinstance(index_map, list):
        raise ValueError("index_map.json must be a list of doc_id")

    docs: Dict[str, dict] = {}
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("id")
            if doc_id:
                docs[doc_id] = doc

    return embeddings, index_map, docs


def embed_texts(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
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
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            token_embeddings = outputs[0]
            sent_emb = token_embeddings[:, 0]
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outs.append(sent_emb.cpu())
    return torch.cat(outs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summary index retriever")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--top_k_docs", type=int, default=50)
    parser.add_argument("--top_k_files", type=int, default=20)
    parser.add_argument("--top_k_modules", type=int, default=20)
    parser.add_argument("--top_k_entities", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mapper_type", type=str, default="ast", choices=["ast", "graph"])
    parser.add_argument("--repos_root", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.force_cpu else f"cuda:{args.gpu_id}")

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code).to(device)
    model.eval()

    dataset = load_jsonl(args.dataset_path)

    mapper = ASTBasedMapper(args.repos_root) if args.mapper_type == "ast" else GraphBasedMapper(args.repos_root)

    index_cache: Dict[str, Tuple[torch.Tensor, List[str], Dict[str, dict]]] = {}

    def get_index(repo_name: str):
        if repo_name not in index_cache:
            index_cache[repo_name] = load_summary_index(repo_name, args.index_dir)
        return index_cache[repo_name]

    results = []
    for instance in tqdm(dataset, desc="Processing instances"):
        instance_id = instance.get("instance_id", "")
        repo_name = instance_id_to_repo_name(instance_id)

        try:
            embeddings, index_map, docs = get_index(repo_name)
        except Exception:
            results.append({
                "instance_id": instance_id,
                "found_files": [],
                "found_modules": [],
                "found_entities": [],
                "raw_output_loc": [],
            })
            continue

        query_text = get_problem_text(instance)
        if not query_text:
            results.append({
                "instance_id": instance_id,
                "found_files": [],
                "found_modules": [],
                "found_entities": [],
                "raw_output_loc": [],
            })
            continue

        query_embedding = embed_texts([query_text], model, tokenizer, args.max_length, 1, device)[0]
        query_emb_gpu = query_embedding.to(device)
        embeddings_gpu = embeddings.to(device)
        scores = torch.matmul(embeddings_gpu, query_emb_gpu)
        scores = scores.cpu().tolist()

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.top_k_docs]
        top_docs = []
        for idx in top_indices:
            if idx >= len(index_map):
                continue
            doc_id = index_map[idx]
            doc = docs.get(doc_id)
            if not doc:
                continue
            top_docs.append((doc, float(scores[idx])))

        # File ranking by max score (avoids bias toward large files with many docs)
        file_scores: Dict[str, float] = {}
        for doc, score in top_docs:
            file_path = doc.get("file_path")
            if not file_path:
                continue
            file_scores[file_path] = max(file_scores.get(file_path, 0.0), score)
        found_files = [f for f, _ in sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[: args.top_k_files]]

        # Module docs by score
        module_scores: Dict[str, float] = {}
        for doc, score in top_docs:
            if doc.get("type") != "module":
                continue
            module_path = doc.get("module_path") or doc.get("file_path")
            if not module_path:
                continue
            module_scores[module_path] = max(module_scores.get(module_path, 0.0), score)

        # Map function/class docs to entities/modules
        blocks = []
        for doc, _score in top_docs:
            if doc.get("type") not in {"function", "class"}:
                continue
            if doc.get("start_line") is None or doc.get("end_line") is None:
                continue
            blocks.append(
                {
                    "file_path": doc.get("file_path"),
                    "start_line": doc.get("start_line"),
                    "end_line": doc.get("end_line"),
                }
            )
        found_modules, found_entities = mapper.map_blocks_to_entities(
            blocks,
            instance_id,
            top_k_modules=args.top_k_modules,
            top_k_entities=args.top_k_entities,
        )

        for module_path, _score in sorted(module_scores.items(), key=lambda x: x[1], reverse=True):
            if module_path in found_modules:
                continue
            if len(found_modules) >= args.top_k_modules:
                break
            found_modules.append(module_path)

        results.append({
            "instance_id": instance_id,
            "found_files": found_files[: args.top_k_files],
            "found_modules": found_modules[: args.top_k_modules],
            "found_entities": found_entities[: args.top_k_entities],
            "raw_output_loc": [
                {"doc_id": doc.get("id"), "type": doc.get("type"), "score": score}
                for doc, score in top_docs
            ],
        })

    output_path = output_dir / "summary_results.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
