#!/usr/bin/env python
"""YAML-driven fusion search CLI (dense + sparse + summary)."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
    HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None
    HAS_YAML = False

from tqdm import tqdm

from method.core.fusion_engine import FusionEngine, BestProvenance
from method.mapping import ASTBasedMapper, GraphBasedMapper
from method.retrieval.common import instance_id_to_repo_name, get_problem_text, load_jsonl
from method.utils import load_dataset_instances


def _load_config(path: str) -> Dict[str, Any]:
    if not HAS_YAML:
        raise ImportError("PyYAML is required for --config_path. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must be a YAML mapping at top level.")
    return data


def _best_prov_to_dict(prov: BestProvenance) -> Dict[str, Any]:
    data = asdict(prov)
    return {k: v for k, v in data.items() if v is not None}


def _resolve_dataset(config: Dict[str, Any]) -> List[dict]:
    dataset_path = config.get("dataset_path", "")
    if dataset_path:
        return load_jsonl(dataset_path)
    dataset = config.get("dataset", "")
    if not dataset:
        raise ValueError("Config must include dataset_path or dataset.")
    split = config.get("split", "test")
    limit = config.get("limit")
    return load_dataset_instances(dataset_path="", dataset=dataset, split=split, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion search (YAML config)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to fusion YAML config")
    args = parser.parse_args()

    config = _load_config(args.config_path)

    repos_root = config.get("repos_root")
    output_folder = config.get("output_folder")
    if not repos_root:
        raise ValueError("Config missing repos_root")
    if not output_folder:
        raise ValueError("Config missing output_folder")

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapper_type = config.get("mapper_type", "ast")
    if mapper_type == "ast":
        mapper = ASTBasedMapper(repos_root)
    else:
        mapper = GraphBasedMapper(repos_root)

    engine = FusionEngine(config)

    dataset = _resolve_dataset(config)

    top_k_modules = int(config.get("top_k", {}).get("modules", 20))
    top_k_entities = int(config.get("top_k", {}).get("entities", 50))

    results = []
    for instance in tqdm(dataset, desc="Processing instances"):
        instance_id = instance.get("instance_id", "")
        repo_name = instance_id_to_repo_name(instance_id)
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

        try:
            fused, best_prov, mapping_blocks = engine.retrieve(query_text, repo_name)
            found_files = [item.item_id for item in fused]

            found_modules, found_entities = mapper.map_blocks_to_entities(
                blocks=mapping_blocks,
                instance_id=instance_id,
                top_k_modules=top_k_modules,
                top_k_entities=top_k_entities,
            )

            raw_output_loc = []
            for item in fused:
                entry = {
                    "file_path": item.item_id,
                    "score": item.score,
                }
                prov = best_prov.get(item.item_id)
                if prov:
                    entry["best_provenance"] = _best_prov_to_dict(prov)
                raw_output_loc.append(entry)

            results.append({
                "instance_id": instance_id,
                "found_files": found_files,
                "found_modules": found_modules[:top_k_modules],
                "found_entities": found_entities[:top_k_entities],
                "raw_output_loc": raw_output_loc,
            })
        except Exception as exc:
            results.append({
                "instance_id": instance_id,
                "error": str(exc),
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
