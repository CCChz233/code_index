#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _shorten(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def resolve_fuzzy_path(repo_root: Path, partial_path: str) -> Optional[Path]:
    partial_path = partial_path.lstrip("/").replace("\\", "/")

    direct = repo_root / partial_path
    if direct.exists():
        return direct

    direct_json = repo_root / f"{partial_path}.json"
    if direct_json.exists():
        return direct_json

    matches = list(repo_root.rglob(f"*{partial_path}*.json"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"Found multiple matches for '{partial_path}', using first one:")
        for match in matches[:5]:
            try:
                rel = match.relative_to(repo_root)
            except Exception:
                rel = match
            print(f" - {rel}")
    return matches[0]


def inspect_ast(repo: str, file_path: str, index_root: str, max_len: int) -> None:
    repo_root = Path(index_root) / repo / "summary_ast"
    json_path = resolve_fuzzy_path(repo_root, file_path)
    json_path_str = str(json_path) if json_path else ""

    if not json_path or not os.path.exists(json_path_str):
        print(f"File not found: {json_path_str or file_path}")
        print("Check that the repo was indexed and the file path is correct.")
        return

    data = _load_json(json_path_str)

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.tree import Tree

        console = Console()
        console.print(f"Inspecting: {repo} :: {file_path}")

        summary = data.get("summary", "")
        if summary:
            console.print(Panel(summary, title=f"File: {os.path.basename(file_path)}", border_style="green"))

        tree = Tree(f"[bold]{file_path}[/bold]")

        for cls in data.get("classes", []):
            cls_branch = tree.add(f"[bold cyan]C: {cls.get('name','')}[/bold cyan]")
            cls_summary = _shorten(cls.get("summary", ""), max_len)
            if cls_summary:
                cls_branch.add(f"[dim]{cls_summary}[/dim]")

            for method in cls.get("methods", []):
                method_name = method.get("name", "")
                method_summary = _shorten(method.get("summary", ""), max_len)
                m_branch = cls_branch.add(f"[magenta]M: {method_name}[/magenta]")
                if method_summary:
                    m_branch.add(f"[dim]{method_summary}[/dim]")

        for func in data.get("functions", []):
            func_name = func.get("name", "")
            func_summary = _shorten(func.get("summary", ""), max_len)
            f_branch = tree.add(f"[bold yellow]F: {func_name}[/bold yellow]")
            if func_summary:
                f_branch.add(f"[dim]{func_summary}[/dim]")

        console.print(tree)
    except Exception:
        print(f"Inspecting: {repo} :: {file_path}")
        summary = data.get("summary", "")
        if summary:
            print(f"File summary: {summary}")

        print("Classes:")
        for cls in data.get("classes", []):
            print(f"  C: {cls.get('name','')}")
            cls_summary = _shorten(cls.get("summary", ""), max_len)
            if cls_summary:
                print(f"    {cls_summary}")
            for method in cls.get("methods", []):
                method_summary = _shorten(method.get("summary", ""), max_len)
                print(f"    M: {method.get('name','')}")
                if method_summary:
                    print(f"      {method_summary}")

        print("Functions:")
        for func in data.get("functions", []):
            func_summary = _shorten(func.get("summary", ""), max_len)
            print(f"  F: {func.get('name','')}")
            if func_summary:
                print(f"    {func_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AST-mirrored summary JSON for a single file.")
    parser.add_argument("--repo", required=True, help="Repo name under index root.")
    parser.add_argument("--file", required=True, help="Relative file path in repo (e.g. src/app.py).")
    parser.add_argument("--root", default="summary_index_function_level", help="Summary index root directory.")
    parser.add_argument("--max_len", type=int, default=80, help="Max summary preview length.")
    args = parser.parse_args()

    inspect_ast(args.repo, args.file, args.root, args.max_len)


if __name__ == "__main__":
    main()
