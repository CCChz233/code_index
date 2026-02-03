from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Block:
    file_path: str
    start_line: int
    end_line: int
    content: str
    block_type: str
    context_text: Optional[str] = None
    function_text: Optional[str] = None
    summary_text: Optional[str] = None
    original_content: Optional[str] = None

    def __init__(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        content: str = "",
        block_type: str = "",
        context_text: Optional[str] = None,
        function_text: Optional[str] = None,
        summary_text: Optional[str] = None,
        original_content: Optional[str] = None,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        if start_line is None and start is not None:
            start_line = start
        if end_line is None and end is not None:
            end_line = end
        if start_line is None or end_line is None:
            raise ValueError("Block requires start_line/end_line (or start/end).")
        self.file_path = file_path
        self.start_line = int(start_line)
        self.end_line = int(end_line)
        self.content = content
        self.block_type = block_type
        self.context_text = context_text
        self.function_text = function_text
        self.summary_text = summary_text
        self.original_content = original_content

    @property
    def start(self) -> int:
        return self.start_line

    @property
    def end(self) -> int:
        return self.end_line


@dataclass
class ChunkResult:
    blocks: List[Block]
    metadata: Dict[str, Dict[str, Any]]
    aux: Dict[str, Dict[str, Any]]


@dataclass
class IndexArtifact:
    index_type: str
    strategy: str
    schema_version: str
    repo_name: str
    block_count: int
    build_time: str
    config_hash: str
