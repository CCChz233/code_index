import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from method.core.contract import Block


def save_bm25_index(
    output_dir: Path,
    tf_matrix: csr_matrix,
    vocab: Dict[str, int],
    idf: np.ndarray,
    doc_len: np.ndarray,
    avgdl: float,
    *,
    strategy: str,
    blocks: List[Block],
    k1: float = 1.5,
    b: float = 0.75,
    span_ids_map: Optional[Dict[int, List[str]]] = None,
    function_metadata: Optional[Dict[int, Dict[str, Optional[str]]]] = None,
    summary_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(output_dir / "index.npz", tf_matrix)

    vocab_payload = {
        "vocab": vocab,
        "idf": idf.tolist(),
        "doc_len": doc_len.tolist(),
        "avgdl": avgdl,
        "k1": k1,
        "b": b,
    }
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab_payload, f)

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
