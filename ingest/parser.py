# =============================================================================
# ingest/parser.py — AST Parsing & Import Dependency Extraction
# =============================================================================

import ast
import re
from typing import Optional

from .config import (
    logger,
    IGNORED_DIRECTORIES,
    IGNORED_FILENAMES,
    IGNORED_SUFFIXES,
)

# ── AST Parsing Dependencies ───────────────────────────────────────────────
try:
    import tree_sitter_go as tsgo
    from tree_sitter import Language, Parser

    GO_LANGUAGE = Language(tsgo.language())
    go_parser = Parser(GO_LANGUAGE)
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter or tree-sitter-go not installed. Go function parsing will be skipped.")


def _is_noise_file(filename: str) -> bool:
    lower_path = filename.lower().replace("\\", "/")
    path_parts = set(lower_path.split("/"))
    if path_parts.intersection(IGNORED_DIRECTORIES):
        return True
    name = lower_path.rsplit("/", 1)[-1]
    if name in IGNORED_FILENAMES:
        return True
    return name.endswith(IGNORED_SUFFIXES)


def parse_python_ast(filepath: str, source_code: str) -> dict:
    if not filepath.endswith(".py") or not source_code:
        return {"functions": [], "calls": []}

    functions = []
    calls = []

    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_code = ast.get_source_segment(source_code, node) or ""
                functions.append({
                    "name": node.name,
                    "id": f"{filepath}::{node.name}",
                    "start": node.lineno,
                    "end": node.end_lineno,
                    "code": func_code
                })
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        calls.append((node.name, child.func.id, None))
                    elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        parts = []
                        attr_node = child.func
                        while isinstance(attr_node, ast.Attribute):
                            parts.append(attr_node.attr)
                            attr_node = attr_node.value
                        if isinstance(attr_node, ast.Name):
                            parts.append(attr_node.id)
                        parts.reverse()
                        qualified_name = ".".join(parts)
                        calls.append((node.name, child.func.attr, qualified_name))
    except Exception:
        pass

    return {"functions": functions, "calls": calls}


def parse_go_ast(filepath: str, source_code: str) -> dict:
    if not TREE_SITTER_AVAILABLE or not filepath.endswith(".go") or not source_code:
        return {"functions": [], "calls": []}

    try:
        tree = go_parser.parse(bytes(source_code, "utf8"))
    except Exception as exc:
        logger.debug("Failed to parse Go AST for %s: %s", filepath, exc)
        return {"functions": [], "calls": []}

    functions = []
    calls = []

    def get_text(node):
        return source_code[node.start_byte:node.end_byte]

    def walk(node, current_func=None):
        new_func = current_func

        # 1. Identify Function and Method Declarations
        if node.type in ['function_declaration', 'method_declaration']:
            name_node = node.child_by_field_name('name')
            if name_node:
                name = get_text(name_node)
                new_func = name  # Track current function scope for call-edge detection
                functions.append({
                    "name":  name,
                    "id":    f"{filepath}::{name}",
                    "start": node.start_point[0] + 1,
                    "end":   node.end_point[0] + 1,
                    "code":  get_text(node),
                })

        # 2. Identify Function Calls within a function
        elif node.type == 'call_expression' and current_func:
            func_node = node.children[0]
            if func_node.type == 'identifier':  # e.g., foo()
                callee_name = get_text(func_node)
                calls.append((current_func, callee_name, None))
            elif func_node.type == 'selector_expression':  # e.g., pkg.foo() or obj.foo()
                parts = []
                for child in func_node.children:
                    if child.type == 'field_identifier':
                        parts.append(get_text(child))
                    elif child.type == 'identifier':
                        parts.insert(0, get_text(child))
                if parts:
                    callee_name = parts[-1]  # bare name for intra-repo matching
                    callee_qualified = ".".join(parts)  # e.g., "lipgloss.NewStyle"
                    calls.append((current_func, callee_name, callee_qualified))

        # Recurse through children
        for child in node.children:
            walk(child, new_func)

    walk(tree.root_node)
    return {"functions": functions, "calls": calls}


def _append_dependency(dependencies: list, filepath: str, target_module: str):
    if not target_module:
        return
    edge = (filepath, "DEPENDS_ON", target_module)
    if edge not in dependencies:
        dependencies.append(edge)


def _go_file_kind(filepath: str) -> Optional[str]:
    name = filepath.lower().replace("\\", "/").rsplit("/", 1)[-1]
    if name == "go.mod":    return "go_mod"
    if name == "go.work":   return "go_work"
    if name.endswith(".go"): return "go_source"
    return None


def _extract_go_dependencies(filepath: str, lines: list, patch_mode: bool = False) -> list:
    dependencies = []
    in_import_block = in_require_block = in_use_block = in_replace_block = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if patch_mode:
            if not line or line[0] not in {"+", " "}:
                continue
            line = line[1:].lstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        if stripped.startswith("import ("):
            in_import_block = True; continue
        if in_import_block:
            if stripped.startswith(")"):
                in_import_block = False; continue
            m = re.search(r'["`]\s*([^"`]+?)\s*["`]', stripped)
            if m: _append_dependency(dependencies, filepath, m.group(1))
            continue
        if stripped.startswith("import "):
            m = re.search(r'["`]\s*([^"`]+?)\s*["`]', stripped)
            if m: _append_dependency(dependencies, filepath, m.group(1))
            continue

        if stripped.startswith("require ("):
            in_require_block = True; continue
        if stripped.startswith("use ("):
            in_use_block = True; continue
        if stripped.startswith("replace ("):
            in_replace_block = True; continue

        if in_require_block:
            if stripped.startswith(")"):
                in_require_block = False; continue
            parts = stripped.split()
            if len(parts) >= 2: _append_dependency(dependencies, filepath, parts[0])
            continue
        if in_use_block:
            if stripped.startswith(")"):
                in_use_block = False; continue
            _append_dependency(dependencies, filepath, stripped)
            continue
        if in_replace_block:
            if stripped.startswith(")"):
                in_replace_block = False; continue
            if "=>" in stripped:
                left, right = [p.strip() for p in stripped.split("=>", 1)]
                _append_dependency(dependencies, filepath, left.split()[0] if left else "")
                _append_dependency(dependencies, filepath, right.split()[0] if right else "")
            continue

        if stripped.startswith("require "):
            parts = stripped.split()
            if len(parts) >= 3: _append_dependency(dependencies, filepath, parts[1])
        if stripped.startswith("use "):
            parts = stripped.split(None, 1)
            if len(parts) == 2: _append_dependency(dependencies, filepath, parts[1].strip())
        if stripped.startswith("replace ") and "=>" in stripped:
            body = stripped[len("replace "):].strip()
            left, right = [p.strip() for p in body.split("=>", 1)]
            _append_dependency(dependencies, filepath, left.split()[0] if left else "")
            _append_dependency(dependencies, filepath, right.split()[0] if right else "")

    return dependencies


def _extract_generic_dependencies(filepath: str, lines: list, patch_mode: bool = False) -> list:
    dependencies = []
    patterns = [
        r"^\s*from\s+([a-zA-Z0-9_./@+-]+)\s+import",
        r"^\s*import\s+([a-zA-Z0-9_./@+-]+)",
        r"from\s+['\"]([^'\"]+)['\"]",
        r"require\(['\"]([^'\"]+)['\"]\)",
        r"#include\s*[<\"]([^>\"]+)[>\"]",
        r"use\s+([a-zA-Z0-9_:]+)",
    ]
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if patch_mode:
            if not line or line[0] not in {"+", " "}:
                continue
            line = line[1:].lstrip()
        stripped = line.strip()
        if not stripped or (stripped.startswith("#") and not stripped.startswith("#include")):
            continue
        for pattern in patterns:
            m = re.search(pattern, stripped)
            if m:
                _append_dependency(dependencies, filepath, m.group(1))
    return dependencies


def extract_imports_from_source(filepath: str, source_code: str) -> list:
    kind = _go_file_kind(filepath)
    lines = source_code.split("\n")
    if kind in {"go_mod", "go_work", "go_source"}:
        return _extract_go_dependencies(filepath, lines, patch_mode=False)
    return _extract_generic_dependencies(filepath, lines, patch_mode=False)


def extract_file_dependencies(compact_files: list) -> list:
    """Legacy additive-only extraction. Kept for backwards compatibility."""
    dependencies = []
    for item in compact_files:
        source_file = item.get("filename")
        patch = item.get("patch", "")
        if not patch or not source_file:
            continue
        kind = _go_file_kind(source_file)
        lines = patch.split("\n")
        if kind in {"go_mod", "go_work", "go_source"}:
            new_deps = _extract_go_dependencies(source_file, lines, patch_mode=True)
        else:
            new_deps = _extract_generic_dependencies(source_file, lines, patch_mode=True)
        for dep in new_deps:
            if dep not in dependencies:
                dependencies.append(dep)
    return dependencies


def extract_temporal_dependencies(compact_files: list) -> dict:
    """Split a git diff into added and removed import dependencies."""
    added: list = []
    removed: list = []

    for item in compact_files:
        source_file = item.get("filename")
        patch = item.get("patch", "")
        if not patch or not source_file:
            continue

        kind = _go_file_kind(source_file)
        extractor = (
            _extract_go_dependencies
            if kind in {"go_mod", "go_work", "go_source"}
            else _extract_generic_dependencies
        )

        addition_lines: list[str] = []
        deletion_lines: list[str] = []

        for raw_line in patch.split("\n"):
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                addition_lines.append(raw_line[1:])
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                deletion_lines.append(raw_line[1:])

        for dep in extractor(source_file, addition_lines, patch_mode=False):
            if dep not in added:
                added.append(dep)

        for dep in extractor(source_file, deletion_lines, patch_mode=False):
            if dep not in removed:
                removed.append(dep)

    return {"added": added, "removed": removed}


def get_modified_functions(patch_text: str, filepath: str, ast_data: dict) -> list:
    if not patch_text or not ast_data.get("functions"):
        return []

    modified_lines = set()

    for line in patch_text.split("\n"):
        if line.startswith("@@"):
            match = re.search(r'\+(\d+)(?:,(\d+))? @@', line)
            if match:
                start_line = int(match.group(1))
                line_count = int(match.group(2) or 1)
                for i in range(start_line, start_line + line_count):
                    modified_lines.add(i)

    modified_funcs = set()
    for func in ast_data["functions"]:
        func_range = set(range(func["start"], func["end"] + 1))
        if modified_lines.intersection(func_range):
            modified_funcs.add(func["id"])

    return list(modified_funcs)
