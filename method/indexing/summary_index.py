import argparse
import ast
import hashlib
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from dependency_graph import RepoDependencySearcher
from dependency_graph.build_graph import EDGE_TYPE_INVOKES, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION
from method.indexing.chunker.function import collect_function_blocks
from method.indexing.core.llm_factory import get_llm
from method.indexing.encoder.sparse import build_bm25_matrix
from method.indexing.utils.file import set_file_scan_options

try:
    from scipy.sparse import save_npz
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for sparse indexing") from exc


DEFAULT_SUMMARY_PROMPT = (
    "You are a senior code analyst. Given the context below, output a JSON object with keys: "
    "`summary`, `business_intent`, `keywords` (list).\n"
    "Constraints:\n"
    "- Do NOT mention any specific caller/callee names.\n"
    "- Focus on business intent and behavior.\n"
    "- Use {language}.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "JSON:\n"
)

DEFAULT_CLASS_PROMPT = (
    "You are a senior code analyst. Given class-level metadata and method summaries, output a JSON object "
    "with keys: `summary`, `business_intent`, `keywords` (list).\n"
    "Constraints:\n"
    "- Do NOT list method names verbatim.\n"
    "- Focus on the class responsibility and state/behavior.\n"
    "- Use {language}.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "JSON:\n"
)

DEFAULT_FILE_PROMPT = (
    "You are a senior code analyst. Given file-level metadata and function summaries, output a JSON object "
    "with keys: `summary`, `business_intent`, `keywords` (list).\n"
    "Constraints:\n"
    "- Do NOT list function names verbatim.\n"
    "- Focus on the file/module responsibility.\n"
    "- Use {language}.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "JSON:\n"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _emit_metric(
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]],
    event: str,
    *,
    duration_ms: Optional[float] = None,
    counts: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    if metrics_hook is None:
        return
    metric: Dict[str, Any] = {"ts": _utc_now_iso(), "event": event}
    if duration_ms is not None:
        metric["duration_ms"] = int(duration_ms)
    if counts:
        metric["counts"] = counts
    if payload:
        metric["payload"] = payload
    try:
        metrics_hook(metric)
    except Exception:
        return


def _get_usage_value(usage: Any, key: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return None
    if hasattr(usage, key):
        try:
            return int(getattr(usage, key))
        except Exception:
            return None
    return None


def _extract_usage_from_raw(raw: Any) -> Optional[Any]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "usage" in raw:
            return raw.get("usage")
        if "data" in raw and isinstance(raw.get("data"), dict):
            data_usage = raw["data"].get("usage")
            if data_usage:
                return data_usage
    if hasattr(raw, "usage"):
        return getattr(raw, "usage")
    return None


def _extract_usage(response: Any) -> Optional[Any]:
    if response is None:
        return None
    if isinstance(response, dict):
        if "usage" in response:
            return response.get("usage")
        if "data" in response and isinstance(response.get("data"), dict):
            return response["data"].get("usage")
    for attr in ("raw", "raw_response", "raw_output", "additional_kwargs"):
        if hasattr(response, attr):
            usage = _extract_usage_from_raw(getattr(response, attr))
            if usage is not None:
                return usage
    if hasattr(response, "usage"):
        return getattr(response, "usage")
    return None


def _extract_model_name(response: Any, fallback: Optional[str]) -> Optional[str]:
    for attr in ("model", "model_name"):
        if hasattr(response, attr):
            value = getattr(response, attr)
            if value:
                return str(value)
    if isinstance(response, dict):
        if response.get("model"):
            return str(response.get("model"))
    for attr in ("raw", "raw_response", "raw_output"):
        if hasattr(response, attr):
            raw = getattr(response, attr)
            if isinstance(raw, dict) and raw.get("model"):
                return str(raw.get("model"))
            if hasattr(raw, "model") and getattr(raw, "model", None):
                return str(getattr(raw, "model"))
    return fallback


def _normalize_usage(usage: Any, model_name: Optional[str]) -> Dict[str, Any]:
    prompt_tokens = _get_usage_value(usage, "prompt_tokens")
    completion_tokens = _get_usage_value(usage, "completion_tokens")
    total_tokens = _get_usage_value(usage, "total_tokens")
    if total_tokens is None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "model": model_name or "",
    }


def _classify_llm_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "context" in message and ("length" in message or "token" in message):
        return "CONTEXT_LENGTH_ERROR"
    if "maximum context" in message or "max context" in message:
        return "CONTEXT_LENGTH_ERROR"
    if "timeout" in message or "timed out" in message:
        return "LLM_API_ERROR"
    if any(code in message for code in ("429", "500", "502", "503", "504")):
        return "LLM_API_ERROR"
    if "rate limit" in message or "server error" in message:
        return "LLM_API_ERROR"
    return "SYSTEM_ERROR"


def _sha1(text: str, *, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _normalize_code(code: str) -> str:
    import io
    import tokenize

    try:
        tokens = []
        reader = io.StringIO(code).readline
        for tok in tokenize.generate_tokens(reader):
            if tok.type == tokenize.COMMENT:
                continue
            tokens.append(tok)
        code = tokenize.untokenize(tokens)
    except Exception:
        pass

    lines = []
    for line in (code or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"\s+", " ", stripped)
        lines.append(stripped)
    return "\n".join(lines)


def _normalize_context(called_by: Sequence[str], calls: Sequence[str]) -> str:
    called_by_sorted = sorted(set(called_by))
    calls_sorted = sorted(set(calls))
    return f"called_by={'|'.join(called_by_sorted)}\ncalls={'|'.join(calls_sorted)}"


def _jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _is_test_node(node_id: str) -> bool:
    file_path = node_id.split(":")[0].lower()
    parts = re.split(r" |_|\/", file_path)
    return any(part.startswith("test") for part in parts)


def _load_prompt(
    prompt: Optional[str],
    prompt_file: Optional[str],
    language: str,
    *,
    default_template: str,
) -> str:
    template = prompt
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception as exc:
            logging.warning("Failed to read summary prompt file %s: %s", prompt_file, exc)

    if not template:
        template = default_template

    if "{language}" in template:
        template = template.replace("{language}", language or "English")

    if "{context_str}" not in template:
        template = template.rstrip() + "\n\nContext:\n{context_str}\n\nJSON:\n"
    return template


def _call_llm(
    llm,
    prompt: str,
    *,
    max_retries: int,
    base_backoff: float,
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    metrics_payload: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            _emit_metric(metrics_hook, "llm_call", payload=dict(metrics_payload or {}))
            start_ts = time.time()
            response = llm.complete(prompt)
            duration_ms = (time.time() - start_ts) * 1000
            usage = _normalize_usage(_extract_usage(response), _extract_model_name(response, model_name))
            if hasattr(response, "text") and response.text:
                text = str(response.text).strip()
            elif hasattr(response, "message") and getattr(response.message, "content", None):
                text = str(response.message.content).strip()
            else:
                text = str(response).strip()

            payload = dict(metrics_payload or {})
            payload.update({"latency_ms": int(duration_ms), "usage": usage})
            _emit_metric(metrics_hook, "llm_finish", duration_ms=duration_ms, payload=payload)
            return text
        except Exception as exc:
            last_exc = exc
            payload = dict(metrics_payload or {})
            payload.update(
                {
                    "error_type": _classify_llm_error(exc),
                    "error_msg": str(exc),
                    "attempt": attempt + 1,
                }
            )
            if attempt >= max_retries:
                _emit_metric(metrics_hook, "error", payload=payload)
                break
            _emit_metric(metrics_hook, "llm_retry", payload=payload)
            sleep_s = base_backoff * (2**attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM completion failed after retries: {last_exc}") from last_exc


def _render_prompt(template: str, context_str: str) -> str:
    return template.replace("{context_str}", context_str)


def _apply_llm_timeout(llm, timeout: Optional[float]) -> None:
    if timeout is None:
        return
    for attr in ("timeout", "request_timeout"):
        if hasattr(llm, attr):
            try:
                setattr(llm, attr, timeout)
                return
            except Exception:
                continue
    if hasattr(llm, "llm") and getattr(llm, "llm", None) is not None:
        inner = getattr(llm, "llm")
        for attr in ("timeout", "request_timeout"):
            if hasattr(inner, attr):
                try:
                    setattr(inner, attr, timeout)
                    return
                except Exception:
                    continue


def _parse_summary_json(
    text: str,
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    metrics_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def _coerce_keywords(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            parts = re.split(r"[,\n;]", value)
            return [p.strip() for p in parts if p.strip()]
        return [str(value).strip()]

    payload = None
    try:
        payload = json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                payload = json.loads(text[start : end + 1])
        except Exception:
            payload = None

    if not isinstance(payload, dict):
        error_payload = dict(metrics_payload or {})
        error_payload.update(
            {
                "error_type": "PARSING_ERROR",
                "error_msg": "Failed to parse summary JSON",
            }
        )
        _emit_metric(metrics_hook, "error", payload=error_payload)
        summary_text = text.strip()
        return {
            "summary": summary_text,
            "business_intent": summary_text,
            "keywords": [],
        }

    summary = str(payload.get("summary") or "").strip()
    business_intent = str(payload.get("business_intent") or summary).strip()
    keywords = _coerce_keywords(payload.get("keywords"))
    return {
        "summary": summary,
        "business_intent": business_intent,
        "keywords": keywords,
    }


def _extract_function_code(content: str) -> str:
    if content.startswith("# ") and "\n\n" in content:
        _, code = content.split("\n\n", 1)
        return code
    return content


def _compute_cyclomatic_complexity(code: str) -> int:
    try:
        tree = ast.parse(code)
    except Exception:
        return 0
    count = 1
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.ExceptHandler,
                ast.With,
                ast.Assert,
            ),
        ):
            count += 1
    return count


def _build_placeholder_summary(function_name: str) -> Dict[str, Any]:
    lower = function_name.lower()
    if lower.startswith("get_"):
        target = function_name[4:] or "value"
        summary = f"Standard getter for {target}."
    elif lower.startswith("set_"):
        target = function_name[4:] or "value"
        summary = f"Standard setter for {target}."
    elif lower.startswith("__") and lower.endswith("__"):
        summary = "Standard magic method implementation."
    else:
        summary = f"Simple helper {function_name}."
    return {
        "summary": summary,
        "business_intent": summary,
        "keywords": [],
    }


def _function_name_matches_skip(name: str, patterns: Sequence[str]) -> bool:
    for pat in patterns:
        if re.match(pat, name):
            return True
    return False


def _qualified_to_graph_id(qualified_name: str) -> Optional[str]:
    if "::" not in qualified_name:
        return None
    parts = qualified_name.split("::")
    if len(parts) < 2:
        return None
    rel_path = parts[0]
    symbol = ".".join(parts[1:])
    return f"{rel_path}:{symbol}"


def _module_path_for_file(file_path: str) -> str:
    parent = str(Path(file_path).parent)
    return parent if parent and parent != "." else "."


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    except Exception:
        return None


def _extract_file_metadata(source: str) -> Tuple[str, List[str], List[str]]:
    docstring = ""
    imports: List[str] = []
    globals_list: List[str] = []

    try:
        tree = ast.parse(source)
    except Exception:
        return docstring, imports, globals_list

    try:
        doc = ast.get_docstring(tree)
        if doc:
            docstring = doc.strip()
    except Exception:
        docstring = ""

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    imports.append(f"{base}.*" if base else "*")
                else:
                    name = f"{base}.{alias.name}" if base else alias.name
                    imports.append(name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if name.isupper() and not name.startswith("_"):
                        globals_list.append(name)

    return docstring, sorted(set(imports)), sorted(set(globals_list))


def _extract_class_info(source: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    info: Dict[str, Dict[str, Any]] = {}
    try:
        tree = ast.parse(source)
    except Exception:
        return info

    def visit(node: ast.AST, stack: List[str]) -> None:
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            full_name = ".".join(stack + [class_name]) if stack else class_name
            doc = ast.get_docstring(node) or ""
            start_line = node.lineno - 1 if hasattr(node, "lineno") else None
            end_line = node.end_lineno - 1 if hasattr(node, "end_lineno") else None
            method_names: List[str] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_names.append(child.name)
            info[full_name] = {
                "docstring": doc.strip(),
                "start_line": start_line,
                "end_line": end_line,
                "methods": method_names,
            }
            for child in node.body:
                visit(child, stack + [class_name])
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, stack)

    visit(tree, [])

    short_map: Dict[str, Dict[str, Any]] = {}
    for full_name, payload in info.items():
        short = full_name.split(".")[-1]
        if short not in short_map:
            short_map[short] = payload
    return info, short_map


def _load_graph(graph_index_dir: Optional[str], repo_name: str):
    if not graph_index_dir:
        return None
    graph_path = Path(graph_index_dir) / f"{repo_name}.pkl"
    if not graph_path.exists():
        return None
    try:
        import pickle

        with graph_path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logging.warning("Failed to load graph index %s: %s", graph_path, exc)
        return None


def _compute_graph_stats(graph) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    if graph is None:
        return stats

    in_degree = {node: 0 for node in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        if data.get("type") != EDGE_TYPE_INVOKES:
            continue
        in_degree[v] = in_degree.get(v, 0) + 1
    for node, deg in in_degree.items():
        stats.setdefault(node, {})["in_degree"] = float(deg)

    try:
        import networkx as nx

        inv_edges = [(u, v) for u, v, data in graph.edges(data=True) if data.get("type") == EDGE_TYPE_INVOKES]
        if inv_edges:
            dg = nx.DiGraph()
            dg.add_edges_from(inv_edges)
            pr = nx.pagerank(dg, alpha=0.85)
            for node, score in pr.items():
                stats.setdefault(node, {})["pagerank"] = float(score)
    except Exception:
        pass

    return stats


def _rank_nodes(
    candidates: Sequence[str],
    *,
    target_module: str,
    stats: Dict[str, Dict[str, float]],
    top_k: int,
) -> List[str]:
    def _score(nid: str) -> Tuple[float, float, float, str]:
        module = _module_path_for_file(nid.split(":")[0])
        same_module = 1.0 if module == target_module else 0.0
        is_test = 1.0 if _is_test_node(nid) else 0.0
        base = stats.get(nid, {}).get("pagerank")
        if base is None:
            base = stats.get(nid, {}).get("in_degree", 0.0)
        # Higher score first; penalize tests
        return (same_module, base, -is_test, nid)

    ranked = sorted(set(candidates), key=_score, reverse=True)
    return ranked[:top_k]


def _build_context_summary(called_by: Sequence[str], calls: Sequence[str]) -> str:
    callers_count = len(called_by)
    callees_count = len(calls)
    called_by_tests = any(_is_test_node(nid) for nid in called_by)
    return (
        f"callers_count: {callers_count}\n"
        f"callees_count: {callees_count}\n"
        f"called_by_tests: {called_by_tests}\n"
    )


def _load_cache_entry(cache_dir: Optional[str], cache_key: str) -> Optional[Dict[str, Any]]:
    if not cache_dir:
        return None
    path = Path(cache_dir) / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache_entry(cache_dir: Optional[str], cache_key: str, payload: Dict[str, Any]) -> None:
    if not cache_dir:
        return
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{cache_key}.json"
    tmp = target.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, target)
    except Exception as exc:
        logging.warning("Failed to write cache %s: %s", target, exc)
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _make_cache_key(doc_id: str, summary_hash: str) -> str:
    return hashlib.md5(f"{doc_id}:{summary_hash}".encode("utf-8")).hexdigest()


def _ensure_keywords(value: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in value:
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _collect_function_docs(
    *,
    repo_path: str,
    graph,
    graph_stats: Dict[str, Dict[str, float]],
    llm,
    summary_prompt: str,
    summary_language: str,
    summary_cache_dir: Optional[str],
    summary_cache_policy: str,
    summary_cache_miss: str,
    min_lines: int,
    min_complexity: int,
    skip_patterns: Sequence[str],
    top_k_callers: int,
    top_k_callees: int,
    context_similarity_threshold: float,
    existing_docs: Dict[str, Dict[str, Any]],
    show_progress: bool,
    progress_desc: str,
    llm_max_retries: int,
    llm_backoff: float,
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    summary_llm_model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    blocks, metadata = collect_function_blocks(repo_path, block_size=15)
    docs: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}

    searcher = RepoDependencySearcher(graph) if graph is not None else None

    total_blocks = len(blocks)
    placeholders = 0
    cache_hits = 0
    cache_misses = 0
    llm_calls = 0
    llm_fail = 0
    reused_existing = 0

    iterator = blocks
    if show_progress and tqdm is not None:
        iterator = tqdm(blocks, total=len(blocks), desc=progress_desc)

    for idx, block in enumerate(iterator):
        if block.block_type != "function_level":
            continue
        meta = metadata.get(idx, {})
        qualified_name = meta.get("qualified_name")
        function_name = meta.get("function_name") or ""
        if not qualified_name:
            continue

        file_path = block.file_path
        module_path = _module_path_for_file(file_path)
        doc_id = qualified_name
        metric_payload = {"stage": "function", "doc_id": doc_id}

        raw_code = _extract_function_code(block.content)
        lines = [ln for ln in raw_code.splitlines() if ln.strip()]
        line_count = len(lines)
        complexity = _compute_cyclomatic_complexity(raw_code)

        is_placeholder = (
            line_count < min_lines
            or complexity < min_complexity
            or _function_name_matches_skip(function_name, skip_patterns)
        )
        if is_placeholder:
            placeholders += 1

        graph_id = _qualified_to_graph_id(qualified_name)
        called_by: List[str] = []
        calls: List[str] = []

        if searcher is not None and graph_id and searcher.has_node(graph_id, include_test=True):
            callers, _ = searcher.get_neighbors(
                graph_id,
                direction="backward",
                ntype_filter=[NODE_TYPE_FUNCTION, NODE_TYPE_CLASS],
                etype_filter=[EDGE_TYPE_INVOKES],
                ignore_test_file=False,
            )
            callees, _ = searcher.get_neighbors(
                graph_id,
                direction="forward",
                ntype_filter=[NODE_TYPE_FUNCTION, NODE_TYPE_CLASS],
                etype_filter=[EDGE_TYPE_INVOKES],
                ignore_test_file=False,
            )
            called_by = _rank_nodes(callers, target_module=module_path, stats=graph_stats, top_k=top_k_callers)
            calls = _rank_nodes(callees, target_module=module_path, stats=graph_stats, top_k=top_k_callees)

        context_str = _build_context_summary(called_by, calls)

        normalized_source = _normalize_code(raw_code)
        content_hash = _sha1(normalized_source)
        graph_hash = _sha1(_normalize_context(called_by, calls))
        summary_hash = _sha1(summary_prompt + normalized_source)

        existing = existing_docs.get(doc_id)
        if existing:
            old_called_by = existing.get("context", {}).get("called_by", [])
            old_calls = existing.get("context", {}).get("calls", [])
            similarity = _jaccard_similarity(old_called_by + old_calls, called_by + calls)
        else:
            similarity = 0.0

        summary_payload: Dict[str, Any] = {}
        if is_placeholder:
            summary_payload = _build_placeholder_summary(function_name or "function")
        else:
            if existing and existing.get("content_hash") == content_hash and existing.get("summary_hash") == summary_hash:
                if similarity >= context_similarity_threshold:
                    summary_payload = {
                        "summary": existing.get("summary", ""),
                        "business_intent": existing.get("business_intent", existing.get("summary", "")),
                        "keywords": existing.get("keywords", []),
                    }
                    reused_existing += 1
                    _emit_metric(
                        metrics_hook,
                        "cache_hit",
                        payload={**metric_payload, "source": "existing"},
                    )
            if not summary_payload and summary_cache_policy != "disabled":
                cache_key = _make_cache_key(doc_id, summary_hash)
                cache_entry = _load_cache_entry(summary_cache_dir, cache_key)
                if cache_entry and cache_entry.get("summary_hash") == summary_hash:
                    summary_payload = {
                        "summary": cache_entry.get("summary", ""),
                        "business_intent": cache_entry.get("business_intent", cache_entry.get("summary", "")),
                        "keywords": cache_entry.get("keywords", []),
                    }
                    cache_hits += 1
                    _emit_metric(
                        metrics_hook,
                        "cache_hit",
                        payload={**metric_payload, "source": "cache"},
                    )
                else:
                    cache_misses += 1
                    _emit_metric(metrics_hook, "cache_miss", payload=dict(metric_payload))

            if not summary_payload and summary_cache_policy != "read_only":
                prompt = _render_prompt(summary_prompt, context_str)
                try:
                    llm_calls += 1
                    llm_text = _call_llm(
                        llm,
                        prompt,
                        max_retries=llm_max_retries,
                        base_backoff=llm_backoff,
                        metrics_hook=metrics_hook,
                        metrics_payload=metric_payload,
                        model_name=summary_llm_model,
                    )
                    summary_payload = _parse_summary_json(
                        llm_text,
                        metrics_hook=metrics_hook,
                        metrics_payload=metric_payload,
                    )
                except Exception as exc:
                    logging.warning("Summary generation failed for %s: %s", qualified_name, exc)
                    summary_payload = {}
                    llm_fail += 1

                if summary_payload and summary_cache_policy == "read_write":
                    cache_key = _make_cache_key(doc_id, summary_hash)
                    _save_cache_entry(
                        summary_cache_dir,
                        cache_key,
                        {
                            "summary": summary_payload.get("summary", ""),
                            "business_intent": summary_payload.get("business_intent", ""),
                            "keywords": summary_payload.get("keywords", []),
                            "summary_hash": summary_hash,
                            "cached_at": _utc_now_iso(),
                        },
                    )

            if not summary_payload:
                if summary_cache_miss == "error":
                    raise RuntimeError(f"Summary cache miss for {qualified_name}")
                if summary_cache_miss == "skip":
                    continue
                summary_payload = {"summary": "", "business_intent": "", "keywords": []}

        summary = summary_payload.get("summary", "")
        business_intent = summary_payload.get("business_intent", summary)
        keywords = _ensure_keywords(summary_payload.get("keywords", []))

        doc = {
            "id": doc_id,
            "type": "function",
            "file_path": file_path,
            "module_path": module_path,
            "qualified_name": qualified_name,
            "summary": summary,
            "business_intent": business_intent,
            "keywords": keywords,
            "dependencies": calls,
            "context": {"called_by": called_by, "calls": calls},
            "content_hash": content_hash,
            "graph_hash": graph_hash,
            "summary_hash": summary_hash,
            "hash": summary_hash,
            "language": "python",
            "last_updated": _utc_now_iso(),
            "is_placeholder": bool(is_placeholder),
            "start_line": block.start_line,
            "end_line": block.end_line,
        }
        docs.append(doc)
        by_id[doc_id] = doc

    logging.info(
        "Function summaries: total=%d placeholders=%d cache_hit=%d cache_miss=%d "
        "llm_calls=%d llm_fail=%d reused_existing=%d",
        total_blocks,
        placeholders,
        cache_hits,
        cache_misses,
        llm_calls,
        llm_fail,
        reused_existing,
    )

    return docs, by_id


def _aggregate_file_docs(
    *,
    repo_path: str,
    function_docs: List[Dict[str, Any]],
    class_docs: List[Dict[str, Any]],
    llm,
    summary_prompt: str,
    summary_language: str,
    summary_cache_dir: Optional[str],
    summary_cache_policy: str,
    summary_cache_miss: str,
    existing_docs: Dict[str, Dict[str, Any]],
    llm_max_retries: int,
    llm_backoff: float,
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    summary_llm_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    repo_root = Path(repo_path)
    docs_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for doc in function_docs:
        if doc.get("type") != "function":
            continue
        docs_by_file.setdefault(doc["file_path"], []).append(doc)

    classes_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for doc in class_docs:
        if doc.get("type") != "class":
            continue
        classes_by_file.setdefault(doc["file_path"], []).append(doc)

    file_docs: List[Dict[str, Any]] = []
    for file_path, funcs in docs_by_file.items():
        abs_path = repo_root / file_path
        source = _read_text(abs_path) or ""
        docstring, imports, globals_list = _extract_file_metadata(source)
        funcs_sorted = sorted(funcs, key=lambda d: d.get("qualified_name", ""))
        classes_sorted = sorted(classes_by_file.get(file_path, []), key=lambda d: d.get("qualified_name", ""))

        standalone_funcs = []
        for d in funcs_sorted:
            parts = (d.get("qualified_name") or "").split("::")
            if len(parts) >= 3:
                continue
            standalone_funcs.append(d)

        func_summaries = "\n".join(
            f"- {d.get('qualified_name','')}: {d.get('summary','')}" for d in standalone_funcs
        )
        class_summaries = "\n".join(
            f"- {d.get('qualified_name','')}: {d.get('summary','')}" for d in classes_sorted
        )
        context_str = (
            f"file_path: {file_path}\n"
            f"docstring: {docstring}\n"
            f"imports: {', '.join(imports)}\n"
            f"globals: {', '.join(globals_list)}\n"
            f"class_summaries:\n{class_summaries}\n"
            f"standalone_function_summaries:\n{func_summaries}\n"
        )

        normalized_source = _normalize_code(context_str)
        content_hash = _sha1(normalized_source)
        summary_hash = _sha1(summary_prompt + normalized_source)

        doc_id = file_path
        module_path = _module_path_for_file(file_path)
        existing = existing_docs.get(doc_id)
        metric_payload = {"stage": "file", "doc_id": doc_id}

        summary_payload: Dict[str, Any] = {}
        if existing and existing.get("content_hash") == content_hash and existing.get("summary_hash") == summary_hash:
            summary_payload = {
                "summary": existing.get("summary", ""),
                "business_intent": existing.get("business_intent", existing.get("summary", "")),
                "keywords": existing.get("keywords", []),
            }
            _emit_metric(
                metrics_hook,
                "cache_hit",
                payload={**metric_payload, "source": "existing"},
            )

        if not summary_payload and summary_cache_policy != "disabled":
            cache_key = _make_cache_key(doc_id, summary_hash)
            cache_entry = _load_cache_entry(summary_cache_dir, cache_key)
            if cache_entry and cache_entry.get("summary_hash") == summary_hash:
                summary_payload = {
                    "summary": cache_entry.get("summary", ""),
                    "business_intent": cache_entry.get("business_intent", cache_entry.get("summary", "")),
                    "keywords": cache_entry.get("keywords", []),
                }
                _emit_metric(
                    metrics_hook,
                    "cache_hit",
                    payload={**metric_payload, "source": "cache"},
                )
            else:
                _emit_metric(metrics_hook, "cache_miss", payload=dict(metric_payload))

        if not summary_payload and summary_cache_policy != "read_only":
            prompt = _render_prompt(summary_prompt, context_str)
            try:
                llm_text = _call_llm(
                    llm,
                    prompt,
                    max_retries=llm_max_retries,
                    base_backoff=llm_backoff,
                    metrics_hook=metrics_hook,
                    metrics_payload=metric_payload,
                    model_name=summary_llm_model,
                )
                summary_payload = _parse_summary_json(
                    llm_text,
                    metrics_hook=metrics_hook,
                    metrics_payload=metric_payload,
                )
            except Exception as exc:
                logging.warning("File summary generation failed for %s: %s", file_path, exc)
                summary_payload = {}

            if summary_payload and summary_cache_policy == "read_write":
                cache_key = _make_cache_key(doc_id, summary_hash)
                _save_cache_entry(
                    summary_cache_dir,
                    cache_key,
                    {
                        "summary": summary_payload.get("summary", ""),
                        "business_intent": summary_payload.get("business_intent", ""),
                        "keywords": summary_payload.get("keywords", []),
                        "summary_hash": summary_hash,
                        "cached_at": _utc_now_iso(),
                    },
                )

        if not summary_payload:
            if summary_cache_miss == "error":
                raise RuntimeError(f"Summary cache miss for {file_path}")
            if summary_cache_miss == "skip":
                continue
            summary_payload = {"summary": "", "business_intent": "", "keywords": []}

        summary = summary_payload.get("summary", "")
        business_intent = summary_payload.get("business_intent", summary)
        keywords = _ensure_keywords(summary_payload.get("keywords", []))

        file_docs.append(
            {
                "id": doc_id,
                "type": "file",
                "file_path": file_path,
                "module_path": module_path,
                "qualified_name": file_path,
                "summary": summary,
                "business_intent": business_intent,
                "keywords": keywords,
                "dependencies": imports,
                "context": {},
                "content_hash": content_hash,
                "graph_hash": "",
                "summary_hash": summary_hash,
                "hash": summary_hash,
                "language": "python",
                "last_updated": _utc_now_iso(),
                "is_placeholder": False,
            }
        )

    return file_docs


def _aggregate_module_docs(
    *,
    repo_path: str,
    file_docs: List[Dict[str, Any]],
    llm,
    summary_prompt: str,
    summary_cache_dir: Optional[str],
    summary_cache_policy: str,
    summary_cache_miss: str,
    existing_docs: Dict[str, Dict[str, Any]],
    llm_max_retries: int,
    llm_backoff: float,
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    summary_llm_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    repo_root = Path(repo_path)
    docs_by_module: Dict[str, List[Dict[str, Any]]] = {}
    for doc in file_docs:
        module_path = doc.get("module_path") or "."
        docs_by_module.setdefault(module_path, []).append(doc)

    module_docs: List[Dict[str, Any]] = []
    for module_path, files in docs_by_module.items():
        files_sorted = sorted(files, key=lambda d: d.get("file_path", ""))

        readme_text = ""
        module_dir = repo_root / module_path if module_path != "." else repo_root
        for name in ("README.md", "README.rst", "README.txt"):
            readme_path = module_dir / name
            if readme_path.exists():
                readme_text = _read_text(readme_path) or ""
                break

        file_summaries = "\n".join(
            f"- {d.get('file_path','')}: {d.get('summary','')}" for d in files_sorted
        )
        context_str = (
            f"module_path: {module_path}\n"
            f"readme: {readme_text}\n"
            f"file_summaries:\n{file_summaries}\n"
        )

        normalized_source = _normalize_code(context_str)
        content_hash = _sha1(normalized_source)
        summary_hash = _sha1(summary_prompt + normalized_source)
        doc_id = module_path
        existing = existing_docs.get(doc_id)
        metric_payload = {"stage": "module", "doc_id": doc_id}

        summary_payload: Dict[str, Any] = {}
        if existing and existing.get("content_hash") == content_hash and existing.get("summary_hash") == summary_hash:
            summary_payload = {
                "summary": existing.get("summary", ""),
                "business_intent": existing.get("business_intent", existing.get("summary", "")),
                "keywords": existing.get("keywords", []),
            }
            _emit_metric(
                metrics_hook,
                "cache_hit",
                payload={**metric_payload, "source": "existing"},
            )

        if not summary_payload and summary_cache_policy != "disabled":
            cache_key = _make_cache_key(doc_id, summary_hash)
            cache_entry = _load_cache_entry(summary_cache_dir, cache_key)
            if cache_entry and cache_entry.get("summary_hash") == summary_hash:
                summary_payload = {
                    "summary": cache_entry.get("summary", ""),
                    "business_intent": cache_entry.get("business_intent", cache_entry.get("summary", "")),
                    "keywords": cache_entry.get("keywords", []),
                }
                _emit_metric(
                    metrics_hook,
                    "cache_hit",
                    payload={**metric_payload, "source": "cache"},
                )
            else:
                _emit_metric(metrics_hook, "cache_miss", payload=dict(metric_payload))

        if not summary_payload and summary_cache_policy != "read_only":
            prompt = _render_prompt(summary_prompt, context_str)
            try:
                llm_text = _call_llm(
                    llm,
                    prompt,
                    max_retries=llm_max_retries,
                    base_backoff=llm_backoff,
                    metrics_hook=metrics_hook,
                    metrics_payload=metric_payload,
                    model_name=summary_llm_model,
                )
                summary_payload = _parse_summary_json(
                    llm_text,
                    metrics_hook=metrics_hook,
                    metrics_payload=metric_payload,
                )
            except Exception as exc:
                logging.warning("Module summary generation failed for %s: %s", module_path, exc)
                summary_payload = {}

            if summary_payload and summary_cache_policy == "read_write":
                cache_key = _make_cache_key(doc_id, summary_hash)
                _save_cache_entry(
                    summary_cache_dir,
                    cache_key,
                    {
                        "summary": summary_payload.get("summary", ""),
                        "business_intent": summary_payload.get("business_intent", ""),
                        "keywords": summary_payload.get("keywords", []),
                        "summary_hash": summary_hash,
                        "cached_at": _utc_now_iso(),
                    },
                )

        if not summary_payload:
            if summary_cache_miss == "error":
                raise RuntimeError(f"Summary cache miss for {module_path}")
            if summary_cache_miss == "skip":
                continue
            summary_payload = {"summary": "", "business_intent": "", "keywords": []}

        summary = summary_payload.get("summary", "")
        business_intent = summary_payload.get("business_intent", summary)
        keywords = _ensure_keywords(summary_payload.get("keywords", []))

        module_docs.append(
            {
                "id": doc_id,
                "type": "module",
                "file_path": module_path,
                "module_path": module_path,
                "qualified_name": module_path,
                "summary": summary,
                "business_intent": business_intent,
                "keywords": keywords,
                "dependencies": [],
                "context": {},
                "content_hash": content_hash,
                "graph_hash": "",
                "summary_hash": summary_hash,
                "hash": summary_hash,
                "language": "python",
                "last_updated": _utc_now_iso(),
                "is_placeholder": False,
            }
        )

    return module_docs


def _aggregate_class_docs(
    *,
    repo_path: str,
    function_docs: List[Dict[str, Any]],
    llm,
    summary_prompt: str,
    summary_language: str,
    summary_cache_dir: Optional[str],
    summary_cache_policy: str,
    summary_cache_miss: str,
    existing_docs: Dict[str, Dict[str, Any]],
    llm_max_retries: int,
    llm_backoff: float,
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    summary_llm_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    repo_root = Path(repo_path)
    docs_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for doc in function_docs:
        if doc.get("type") != "function":
            continue
        docs_by_file.setdefault(doc["file_path"], []).append(doc)

    class_docs: List[Dict[str, Any]] = []
    for file_path, funcs in docs_by_file.items():
        abs_path = repo_root / file_path
        source = _read_text(abs_path) or ""
        class_info, class_alias = _extract_class_info(source)

        classes: Dict[str, List[Dict[str, Any]]] = {}
        for d in funcs:
            parts = (d.get("qualified_name") or "").split("::")
            if len(parts) < 3:
                continue
            class_name = parts[1]
            classes.setdefault(class_name, []).append(d)

        for full_name in class_info.keys():
            short = full_name.split(".")[-1]
            if short not in classes:
                classes.setdefault(full_name, [])

        for class_name, methods in classes.items():
            info = class_info.get(class_name) or class_alias.get(class_name) or {}
            docstring = info.get("docstring", "")
            start_line = info.get("start_line")
            end_line = info.get("end_line")

            methods_sorted = sorted(methods, key=lambda d: d.get("qualified_name", ""))
            method_summaries = "\n".join(
                f"- {d.get('qualified_name','').split('::')[-1]}: {d.get('summary','')}"
                for d in methods_sorted
            )

            context_str = (
                f"class_name: {class_name}\n"
                f"docstring: {docstring}\n"
                f"method_summaries:\n{method_summaries}\n"
            )

            if not docstring and not method_summaries.strip():
                continue

            normalized_source = _normalize_code(context_str)
            content_hash = _sha1(normalized_source)
            summary_hash = _sha1(summary_prompt + normalized_source)
            doc_id = f"{file_path}::{class_name}"
            module_path = _module_path_for_file(file_path)
            existing = existing_docs.get(doc_id)
            metric_payload = {"stage": "class", "doc_id": doc_id}

            summary_payload: Dict[str, Any] = {}
            if existing and existing.get("content_hash") == content_hash and existing.get("summary_hash") == summary_hash:
                summary_payload = {
                    "summary": existing.get("summary", ""),
                    "business_intent": existing.get("business_intent", existing.get("summary", "")),
                    "keywords": existing.get("keywords", []),
                }
                _emit_metric(
                    metrics_hook,
                    "cache_hit",
                    payload={**metric_payload, "source": "existing"},
                )

            if not summary_payload and summary_cache_policy != "disabled":
                cache_key = _make_cache_key(doc_id, summary_hash)
                cache_entry = _load_cache_entry(summary_cache_dir, cache_key)
                if cache_entry and cache_entry.get("summary_hash") == summary_hash:
                    summary_payload = {
                        "summary": cache_entry.get("summary", ""),
                        "business_intent": cache_entry.get("business_intent", cache_entry.get("summary", "")),
                        "keywords": cache_entry.get("keywords", []),
                    }
                    _emit_metric(
                        metrics_hook,
                        "cache_hit",
                        payload={**metric_payload, "source": "cache"},
                    )
                else:
                    _emit_metric(metrics_hook, "cache_miss", payload=dict(metric_payload))

            if not summary_payload and summary_cache_policy != "read_only":
                prompt = _render_prompt(summary_prompt, context_str)
                try:
                    llm_text = _call_llm(
                        llm,
                        prompt,
                        max_retries=llm_max_retries,
                        base_backoff=llm_backoff,
                        metrics_hook=metrics_hook,
                        metrics_payload=metric_payload,
                        model_name=summary_llm_model,
                    )
                    summary_payload = _parse_summary_json(
                        llm_text,
                        metrics_hook=metrics_hook,
                        metrics_payload=metric_payload,
                    )
                except Exception as exc:
                    logging.warning("Class summary generation failed for %s: %s", doc_id, exc)
                    summary_payload = {}

                if summary_payload and summary_cache_policy == "read_write":
                    cache_key = _make_cache_key(doc_id, summary_hash)
                    _save_cache_entry(
                        summary_cache_dir,
                        cache_key,
                        {
                            "summary": summary_payload.get("summary", ""),
                            "business_intent": summary_payload.get("business_intent", ""),
                            "keywords": summary_payload.get("keywords", []),
                            "summary_hash": summary_hash,
                            "cached_at": _utc_now_iso(),
                        },
                    )

            if not summary_payload:
                if summary_cache_miss == "error":
                    raise RuntimeError(f"Summary cache miss for {doc_id}")
                if summary_cache_miss == "skip":
                    continue
                summary_payload = {"summary": "", "business_intent": "", "keywords": []}

            summary = summary_payload.get("summary", "")
            business_intent = summary_payload.get("business_intent", summary)
            keywords = _ensure_keywords(summary_payload.get("keywords", []))
            dependencies: List[str] = []
            for m in methods_sorted:
                dependencies.extend(m.get("dependencies", []) or [])
            dependencies = _ensure_keywords(dependencies)

            class_docs.append(
                {
                    "id": doc_id,
                    "type": "class",
                    "file_path": file_path,
                    "module_path": module_path,
                    "qualified_name": doc_id,
                    "summary": summary,
                    "business_intent": business_intent,
                    "keywords": keywords,
                    "dependencies": dependencies,
                    "context": {},
                    "content_hash": content_hash,
                    "graph_hash": "",
                    "summary_hash": summary_hash,
                    "hash": summary_hash,
                    "language": "python",
                    "last_updated": _utc_now_iso(),
                    "is_placeholder": False,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )

    return class_docs


def _save_summary_jsonl(path: Path, docs: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def _load_summary_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    docs: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except Exception:
                continue
            doc_id = doc.get("id")
            if doc_id:
                docs[doc_id] = doc
    return docs


def _embed_texts(
    texts: Sequence[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(out, (tuple, list)):
                token_embeddings = out[0]
            else:
                token_embeddings = out.last_hidden_state
            sent_emb = token_embeddings[:, 0]
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outputs.append(sent_emb.cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0))


def _save_bm25(output_dir: Path, texts: Sequence[str], docs: Sequence[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tf_matrix, vocab, idf, doc_len, avgdl = build_bm25_matrix(texts)
    save_npz(output_dir / "index.npz", tf_matrix)
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab": vocab,
                "idf": idf.tolist(),
                "doc_len": doc_len.tolist(),
                "avgdl": avgdl,
                "k1": 1.5,
                "b": 0.75,
            },
            f,
        )
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for idx, doc in enumerate(docs):
            payload = {
                "doc_id": doc.get("id"),
                "type": doc.get("type"),
                "file_path": doc.get("file_path"),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_summary_index(
    *,
    repo_path: str,
    output_dir: str,
    model_name: str,
    trust_remote_code: bool,
    batch_size: int,
    summary_levels: Sequence[str],
    graph_index_dir: Optional[str],
    summary_llm_provider: str,
    summary_llm_model: str,
    summary_llm_api_key: Optional[str],
    summary_llm_api_base: Optional[str],
    summary_llm_max_new_tokens: int,
    summary_llm_temperature: float,
    llm_timeout: Optional[float],
    llm_max_retries: int,
    summary_prompt: Optional[str],
    summary_prompt_file: Optional[str],
    summary_language: str,
    summary_cache_dir: Optional[str],
    summary_cache_policy: str,
    summary_cache_miss: str,
    min_lines: int,
    min_complexity: int,
    skip_patterns: Sequence[str],
    top_k_callers: int,
    top_k_callees: int,
    context_similarity_threshold: float,
    summary_embed_max_tokens: int,
    lang_allowlist: Optional[str],
    skip_file_patterns: Optional[str],
    max_file_size_mb: Optional[float],
    show_progress: bool,
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    repo_root = Path(repo_path)
    repo_name = repo_root.name
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_file_scan_options(
        suffixes={p if p.startswith(".") else f".{p}" for p in (lang_allowlist or ".py").split(",") if p.strip()},
        skip_patterns=[p.strip() for p in (skip_file_patterns or "").split(",") if p.strip()] or None,
        max_file_size_mb=max_file_size_mb,
    )

    summary_prompt_template = _load_prompt(
        summary_prompt, summary_prompt_file, summary_language, default_template=DEFAULT_SUMMARY_PROMPT
    )
    class_prompt_template = _load_prompt(
        summary_prompt, summary_prompt_file, summary_language, default_template=DEFAULT_CLASS_PROMPT
    )
    file_prompt_template = _load_prompt(
        summary_prompt, summary_prompt_file, summary_language, default_template=DEFAULT_FILE_PROMPT
    )

    llm = get_llm(
        provider=summary_llm_provider,
        model_name=summary_llm_model,
        api_key=summary_llm_api_key,
        api_base=summary_llm_api_base,
        max_new_tokens=summary_llm_max_new_tokens,
        temperature=summary_llm_temperature,
    )
    _apply_llm_timeout(llm, llm_timeout)

    existing_docs = _load_summary_jsonl(out_dir / "summary.jsonl")

    graph = _load_graph(graph_index_dir, repo_name)
    graph_stats = _compute_graph_stats(graph)
    if graph_index_dir and graph is not None:
        try:
            stats_path = Path(graph_index_dir) / "stats.json"
            with stats_path.open("w", encoding="utf-8") as f:
                json.dump(graph_stats, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logging.warning("Failed to write graph stats: %s", exc)

    function_docs, function_by_id = _collect_function_docs(
        repo_path=repo_path,
        graph=graph,
        graph_stats=graph_stats,
        llm=llm,
        summary_prompt=summary_prompt_template,
        summary_language=summary_language,
        summary_cache_dir=summary_cache_dir,
        summary_cache_policy=summary_cache_policy,
        summary_cache_miss=summary_cache_miss,
        min_lines=min_lines,
        min_complexity=min_complexity,
        skip_patterns=skip_patterns,
        top_k_callers=top_k_callers,
        top_k_callees=top_k_callees,
        context_similarity_threshold=context_similarity_threshold,
        existing_docs=existing_docs,
        show_progress=show_progress,
        progress_desc=f"{repo_name} functions",
        llm_max_retries=llm_max_retries,
        llm_backoff=1.0,
        metrics_hook=metrics_hook,
        summary_llm_model=summary_llm_model,
    )

    all_docs = list(function_docs)

    class_docs: List[Dict[str, Any]] = []
    file_docs: List[Dict[str, Any]] = []
    module_docs: List[Dict[str, Any]] = []

    if "class" in summary_levels:
        class_docs = _aggregate_class_docs(
            repo_path=repo_path,
            function_docs=function_docs,
            llm=llm,
            summary_prompt=class_prompt_template,
            summary_language=summary_language,
            summary_cache_dir=summary_cache_dir,
            summary_cache_policy=summary_cache_policy,
            summary_cache_miss=summary_cache_miss,
            existing_docs=existing_docs,
            llm_max_retries=llm_max_retries,
            llm_backoff=1.0,
            metrics_hook=metrics_hook,
            summary_llm_model=summary_llm_model,
        )
        all_docs.extend(class_docs)

    if "file" in summary_levels:
        file_docs = _aggregate_file_docs(
            repo_path=repo_path,
            function_docs=function_docs,
            class_docs=class_docs,
            llm=llm,
            summary_prompt=file_prompt_template,
            summary_language=summary_language,
            summary_cache_dir=summary_cache_dir,
            summary_cache_policy=summary_cache_policy,
            summary_cache_miss=summary_cache_miss,
            existing_docs=existing_docs,
            llm_max_retries=llm_max_retries,
            llm_backoff=1.0,
            metrics_hook=metrics_hook,
            summary_llm_model=summary_llm_model,
        )
        all_docs.extend(file_docs)

    if "module" in summary_levels and file_docs:
        module_docs = _aggregate_module_docs(
            repo_path=repo_path,
            file_docs=file_docs,
            llm=llm,
            summary_prompt=file_prompt_template,
            summary_cache_dir=summary_cache_dir,
            summary_cache_policy=summary_cache_policy,
            summary_cache_miss=summary_cache_miss,
            existing_docs=existing_docs,
            llm_max_retries=llm_max_retries,
            llm_backoff=1.0,
            metrics_hook=metrics_hook,
            summary_llm_model=summary_llm_model,
        )
        all_docs.extend(module_docs)

    _save_summary_jsonl(out_dir / "summary.jsonl", all_docs)

    dense_docs = [doc for doc in all_docs if not doc.get("is_placeholder")]
    dense_texts = [
        f"{doc.get('qualified_name','')}\n{doc.get('business_intent','')}\n{doc.get('summary','')}"
        for doc in dense_docs
    ]

    if dense_docs:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        embed_start = time.time()
        embeddings = _embed_texts(
            dense_texts,
            model=model,
            tokenizer=tokenizer,
            max_length=summary_embed_max_tokens,
            batch_size=batch_size,
            device=device,
        )
        _emit_metric(
            metrics_hook,
            "embedding_batch",
            duration_ms=(time.time() - embed_start) * 1000,
            payload={"count": len(dense_texts)},
        )
        dense_dir = out_dir / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, dense_dir / "embeddings.pt")
    else:
        dense_dir = out_dir / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.empty((0, 0)), dense_dir / "embeddings.pt")

    sparse_docs = dense_docs
    sparse_texts = [
        " ".join(
            [
                " ".join(doc.get("keywords", [])),
                doc.get("qualified_name", ""),
                doc.get("file_path", ""),
            ]
        ).strip()
        for doc in sparse_docs
    ]
    _save_bm25(out_dir / "bm25", sparse_texts, sparse_docs)

    dense_index_map = [doc.get("id") for doc in dense_docs]
    with (dense_dir / "index_map.json").open("w", encoding="utf-8") as f:
        json.dump(dense_index_map, f, ensure_ascii=False)

    manifest = {
        "schema_version": "summary_index_v2.1",
        "summary_levels": list(summary_levels),
        "base_strategy": "function_level",
        "prompt_template_hash": _sha1(summary_prompt_template),
        "summary_llm_provider": summary_llm_provider,
        "summary_llm_model": summary_llm_model,
        "summary_llm_temperature": summary_llm_temperature,
        "summary_llm_seed": None,
        "graph_version": "v2.3" if graph is not None else None,
        "hash_policy": {
            "content_hash": "sha1(normalized_source)",
            "graph_hash": "sha1(normalized_graph_context)",
            "summary_hash": "sha1(prompt + normalized_source)",
        },
        "lang_allowlist": lang_allowlist or ".py",
        "skip_patterns": skip_file_patterns or "",
        "max_file_size_mb": max_file_size_mb,
        "summary_embed_max_tokens": summary_embed_max_tokens,
        "context_similarity_threshold": context_similarity_threshold,
        "top_k_callers": top_k_callers,
        "top_k_callees": top_k_callees,
        "build_time": _utc_now_iso(),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-repo summary index (v2.1).")
    parser.add_argument("--repo_path", required=True, help="Repository root to index.")
    parser.add_argument("--output_dir", required=True, help="Where to save summary index.")

    parser.add_argument("--model_name", default="nov3630/RLRetriever", help="HF model id or local path.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow execution of model code.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding.")
    parser.add_argument("--summary_embed_max_tokens", type=int, default=512, help="Max tokens for summary embeddings.")

    parser.add_argument(
        "--summary_levels",
        type=str,
        default="function,class,file,module",
        help="Comma separated summary levels: function,class,file,module",
    )
    parser.add_argument("--graph_index_dir", type=str, default="", help="Graph index dir.")

    parser.add_argument("--summary_llm_provider", type=str, default="openai", choices=["openai", "vllm"])
    parser.add_argument("--summary_llm_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--summary_llm_api_key", type=str, default=None)
    parser.add_argument("--summary_llm_api_base", type=str, default=None)
    parser.add_argument("--summary_llm_max_new_tokens", type=int, default=512)
    parser.add_argument("--summary_llm_temperature", type=float, default=0.1)
    parser.add_argument("--llm_timeout", type=float, default=300, help="LLM request timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for LLM requests")
    parser.add_argument("--summary_prompt", type=str, default=None)
    parser.add_argument("--summary_prompt_file", type=str, default=None)
    parser.add_argument("--summary_language", type=str, default="English")
    parser.add_argument("--summary_cache_dir", type=str, default="")
    parser.add_argument(
        "--summary_cache_policy",
        type=str,
        default="read_write",
        choices=["read_only", "read_write", "disabled"],
    )
    parser.add_argument(
        "--summary_cache_miss",
        type=str,
        default="empty",
        choices=["empty", "skip", "error"],
    )

    parser.add_argument("--filter_min_lines", type=int, default=3)
    parser.add_argument("--filter_min_complexity", type=int, default=2)
    parser.add_argument(
        "--filter_skip_patterns",
        type=str,
        default="^get_,^set_,^__.*__$",
        help="Comma-separated regex patterns to skip.",
    )

    parser.add_argument("--top_k_callers", type=int, default=5)
    parser.add_argument("--top_k_callees", type=int, default=5)
    parser.add_argument("--context_similarity_threshold", type=float, default=0.8)

    parser.add_argument("--lang_allowlist", type=str, default=".py")
    parser.add_argument("--skip_patterns", type=str, default="")
    parser.add_argument("--max_file_size_mb", type=float, default=None)
    parser.add_argument("--show_progress", action="store_true", help="Show per-repo function summary progress.")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    summary_levels = [s.strip() for s in args.summary_levels.split(",") if s.strip()]
    skip_patterns = [p.strip() for p in args.filter_skip_patterns.split(",") if p.strip()]

    build_summary_index(
        repo_path=args.repo_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        trust_remote_code=args.trust_remote_code,
        batch_size=args.batch_size,
        summary_levels=summary_levels,
        graph_index_dir=args.graph_index_dir or None,
        summary_llm_provider=args.summary_llm_provider,
        summary_llm_model=args.summary_llm_model,
        summary_llm_api_key=args.summary_llm_api_key,
        summary_llm_api_base=args.summary_llm_api_base,
        summary_llm_max_new_tokens=args.summary_llm_max_new_tokens,
        summary_llm_temperature=args.summary_llm_temperature,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.max_retries,
        summary_prompt=args.summary_prompt,
        summary_prompt_file=args.summary_prompt_file,
        summary_language=args.summary_language,
        summary_cache_dir=args.summary_cache_dir or None,
        summary_cache_policy=args.summary_cache_policy,
        summary_cache_miss=args.summary_cache_miss,
        min_lines=args.filter_min_lines,
        min_complexity=args.filter_min_complexity,
        skip_patterns=skip_patterns,
        top_k_callers=args.top_k_callers,
        top_k_callees=args.top_k_callees,
        context_similarity_threshold=args.context_similarity_threshold,
        summary_embed_max_tokens=args.summary_embed_max_tokens,
        lang_allowlist=args.lang_allowlist,
        skip_file_patterns=args.skip_patterns,
        max_file_size_mb=args.max_file_size_mb,
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    main()
