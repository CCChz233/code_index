import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from method.core.contract import Block


_DEFAULT_EPIC_CFG = {
    "min_chunk_size": 512,
    "chunk_size": 1024,
    "max_chunk_size": 2048,
    "hard_token_limit": 4096,
    "max_chunks": 512,
}


def _epic_config_dict(epic_config: Optional[Any]) -> Dict[str, Any]:
    if epic_config is None:
        return dict(_DEFAULT_EPIC_CFG)
    if isinstance(epic_config, dict):
        data = dict(_DEFAULT_EPIC_CFG)
        data.update(epic_config)
        return data
    # try attribute access
    data = {}
    for key, default in _DEFAULT_EPIC_CFG.items():
        data[key] = getattr(epic_config, key, default)
    return data


def save_index(
    output_dir: Path,
    embeddings: torch.Tensor,
    blocks: List[Block],
    strategy: str,
    span_ids_map: Optional[Dict[int, List[str]]] = None,
    epic_config: Optional[Any] = None,
    function_metadata: Optional[Dict[int, Dict[str, Optional[str]]]] = None,
    summary_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_dir / "embeddings.pt")

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for idx, b in enumerate(blocks):
            metadata: Dict[str, Any] = {
                "block_id": idx,
                "file_path": b.file_path,
                "start_line": b.start_line,
                "end_line": b.end_line,
                "block_type": b.block_type,
                "strategy": strategy,
            }

            if span_ids_map and idx in span_ids_map:
                metadata["span_ids"] = span_ids_map[idx]

            if strategy == "epic":
                cfg = _epic_config_dict(epic_config)
                metadata.update(
                    {
                        "epic_splitter_version": "1.0",
                        "chunking_config": cfg,
                        "context_enhanced": True,
                        "chunk_tokens": len(b.content.split()) if b.content else 0,
                        "created_at": datetime.now().isoformat(),
                    }
                )

            if strategy in {"function_level", "ir_function"} and function_metadata and idx in function_metadata:
                metadata.update({
                    "qualified_name": function_metadata[idx].get("qualified_name"),
                    "class_name": function_metadata[idx].get("class_name"),
                    "function_name": function_metadata[idx].get("function_name"),
                    "created_at": datetime.now().isoformat(),
                })
                metadata["function_level_version" if strategy == "function_level" else "ir_function_version"] = "1.0"

            if summary_metadata and idx in summary_metadata:
                for key, value in summary_metadata[idx].items():
                    if value is not None:
                        metadata[key] = value

            f.write(json.dumps(metadata) + "\n")
