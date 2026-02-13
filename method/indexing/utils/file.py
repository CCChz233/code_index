import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

CODE_SUFFIXES = {".py"}
_DEFAULT_SKIP_PATTERNS = [
    "**/.git/**",
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/target/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
]
_DEFAULT_MAX_FILE_SIZE_MB: Optional[float] = None

# Mutable runtime overrides (set via CLI)
_RUNTIME_SUFFIXES: Optional[Set[str]] = None
_RUNTIME_SKIP_PATTERNS: Optional[List[str]] = None
_RUNTIME_MAX_FILE_SIZE_MB: Optional[float] = None


def set_file_scan_options(
    *,
    suffixes: Optional[Iterable[str]] = None,
    skip_patterns: Optional[Iterable[str]] = None,
    max_file_size_mb: Optional[float] = None,
) -> None:
    global _RUNTIME_SUFFIXES, _RUNTIME_SKIP_PATTERNS, _RUNTIME_MAX_FILE_SIZE_MB
    _RUNTIME_SUFFIXES = {s.lower() for s in suffixes} if suffixes else None
    _RUNTIME_SKIP_PATTERNS = [p for p in skip_patterns] if skip_patterns else None
    _RUNTIME_MAX_FILE_SIZE_MB = max_file_size_mb


def _resolve_suffixes(suffixes: Optional[Set[str]]) -> Set[str]:
    if suffixes is not None:
        return {s.lower() for s in suffixes}
    if _RUNTIME_SUFFIXES is not None:
        return _RUNTIME_SUFFIXES
    return CODE_SUFFIXES


def _resolve_skip_patterns(skip_patterns: Optional[Iterable[str]]) -> List[str]:
    if skip_patterns is not None:
        return [p.replace("\\", "/") for p in skip_patterns]
    if _RUNTIME_SKIP_PATTERNS is not None:
        return [p.replace("\\", "/") for p in _RUNTIME_SKIP_PATTERNS]
    return [p.replace("\\", "/") for p in _DEFAULT_SKIP_PATTERNS]


def _resolve_max_file_size_mb(max_file_size_mb: Optional[float]) -> Optional[float]:
    if max_file_size_mb is not None:
        return max_file_size_mb
    return _RUNTIME_MAX_FILE_SIZE_MB if _RUNTIME_MAX_FILE_SIZE_MB is not None else _DEFAULT_MAX_FILE_SIZE_MB


def get_file_scan_options(
    *,
    suffixes: Optional[Set[str]] = None,
    skip_patterns: Optional[Iterable[str]] = None,
    max_file_size_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Return effective file scan options after applying runtime overrides."""
    return {
        "suffixes": sorted(_resolve_suffixes(suffixes)),
        "skip_patterns": _resolve_skip_patterns(skip_patterns),
        "max_file_size_mb": _resolve_max_file_size_mb(max_file_size_mb),
    }


def _is_skipped(path: Path, repo_root: Path, patterns: List[str]) -> bool:
    try:
        rel = str(path.relative_to(repo_root)).replace("\\", "/")
    except Exception:
        rel = str(path).replace("\\", "/")
    rel_with_slash = "/" + rel
    for pat in patterns:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel_with_slash, pat):
            return True
    return False


def iter_files(
    repo_root: Path,
    *,
    suffixes: Optional[Set[str]] = None,
    skip_patterns: Optional[Iterable[str]] = None,
    max_file_size_mb: Optional[float] = None,
) -> List[Path]:
    target_suffixes = _resolve_suffixes(suffixes)
    patterns = _resolve_skip_patterns(skip_patterns)
    max_mb = _resolve_max_file_size_mb(max_file_size_mb)

    results: List[Path] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in target_suffixes:
            continue
        if patterns and _is_skipped(p, repo_root, patterns):
            continue
        if max_mb is not None:
            try:
                if p.stat().st_size > max_mb * 1024 * 1024:
                    continue
            except OSError:
                continue
        results.append(p)
    return results
