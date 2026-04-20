#!/usr/bin/env python
"""
Smoke test for examples/experiments/ — verifies all scripts and notebook
imports resolve after the spatial_neural_adapter -> spatial_adapter rename.

- .py files: loaded via importlib (top-level code runs; __main__ blocks don't)
- .ipynb files: all `import` / `from X import Y` statements from every code
  cell are concatenated and exec'd in a fresh namespace

Standalone — does not require pytest.

Usage:
    conda activate spatial-adapter
    python tools/smoke_examples.py
"""


import ast
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "examples" / "experiments"


def _visible(path: Path) -> bool:
    return not any(part.startswith((".", "__")) for part in path.parts)


def _collect() -> tuple[list[Path], list[Path]]:
    py = sorted(p for p in EXPERIMENTS_DIR.rglob("*.py") if _visible(p))
    ipynb = sorted(p for p in EXPERIMENTS_DIR.rglob("*.ipynb") if _visible(p))
    return py, ipynb


def _ensure_repo_on_syspath() -> None:
    s = str(REPO_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def _import_script(script: Path) -> None:
    # Use the natural dotted name (e.g. examples.experiments.base) so any
    # cross-references inside the package resolve to the same module object.
    # A fake prefix like "_smoke_." causes aliasing bugs with dataclasses /
    # enums when the real dotted name is also imported transitively.
    rel = script.relative_to(REPO_ROOT)
    mod_name = ".".join(rel.with_suffix("").parts)
    importlib.import_module(mod_name)


def _extract_imports(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                out.append(ast.unparse(node))
    return "\n".join(out)


def _run_notebook(nb: Path) -> str | None:
    src = _extract_imports(nb)
    if not src.strip():
        return "no import statements"
    exec(compile(src, str(nb), "exec"), {"__name__": "__smoke__"})
    return None


def _run(
    label: str, items: list[Path], work: Callable[[Path], str | None]
) -> tuple[int, int, int]:
    print(f"\n=== {label} ({len(items)} file{'s' if len(items) != 1 else ''}) ===")
    passed = skipped = failed = 0
    for p in items:
        rel = p.relative_to(REPO_ROOT).as_posix()
        t0 = time.perf_counter()
        try:
            maybe_skip = work(p)
            dt = time.perf_counter() - t0
            if maybe_skip:
                print(f"  SKIP  {rel}  ({maybe_skip})")
                skipped += 1
            else:
                print(f"  PASS  {rel}  ({dt:.2f}s)")
                passed += 1
        except BaseException:
            dt = time.perf_counter() - t0
            print(f"  FAIL  {rel}  ({dt:.2f}s)")
            traceback.print_exc(limit=6)
            failed += 1
    return passed, skipped, failed


def main() -> int:
    _ensure_repo_on_syspath()
    py_files, nb_files = _collect()

    p1, s1, f1 = _run(
        "Python scripts", py_files, lambda p: (_import_script(p), None)[1]
    )
    p2, s2, f2 = _run("Notebook imports", nb_files, _run_notebook)

    total = p1 + s1 + f1 + p2 + s2 + f2
    print("\n" + "=" * 60)
    print(
        f"SUMMARY: {p1 + p2} passed, {s1 + s2} skipped, {f1 + f2} failed  ({total} total)"
    )
    print("=" * 60)
    return 1 if (f1 + f2) else 0


if __name__ == "__main__":
    sys.exit(main())
