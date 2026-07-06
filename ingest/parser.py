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
    _GO_TEST_SUFFIXES,
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
    # Skip Go test files — their TestXxx/BenchmarkXxx functions pollute the call graph.
    if name.endswith(_GO_TEST_SUFFIXES):
        return True
    return name.endswith(IGNORED_SUFFIXES)


def parse_python_ast(filepath: str, source_code: str) -> dict:
    if not filepath.endswith(".py") or not source_code:
        return {"functions": [], "calls": [], "types": [], "variables": [], "directives": []}

    functions = []
    calls = []

    try:
        tree = ast.parse(source_code)

        # Only capture top-level functions and class-level methods.
        # Nested inner functions are skipped to prevent duplicate IDs when two files
        # share common helper names (e.g. `_flush`, `_retry`).
        seen_lines: set = set()
        func_nodes: list = []

        # Collect FunctionDef/AsyncFunctionDef directly inside Module or ClassDef
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_nodes.append((node, node.name))
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_nodes.append((child, f"{node.name}.{child.name}"))

        for node, scoped_name in func_nodes:
            if node.lineno in seen_lines:
                continue
            seen_lines.add(node.lineno)
            func_code = ast.get_source_segment(source_code, node) or ""
            functions.append({
                "name": scoped_name,
                "id": f"{filepath}::{scoped_name}",
                "start": node.lineno,
                "end": node.end_lineno,
                "code": func_code
            })
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    calls.append((scoped_name, child.func.id, None))
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
                    calls.append((scoped_name, child.func.attr, qualified_name))
    except Exception:
        pass

    return {"functions": functions, "calls": calls, "types": [], "variables": [], "directives": []}


def parse_go_ast(filepath: str, source_code: str) -> dict:
    if not TREE_SITTER_AVAILABLE or not filepath.endswith(".go") or not source_code:
        return {"functions": [], "calls": [], "types": [], "variables": [], "directives": []}

    try:
        raw_bytes = source_code if isinstance(source_code, bytes) else bytes(source_code, "utf8")
        tree = go_parser.parse(raw_bytes)
    except Exception as exc:
        logger.debug("Failed to parse Go AST for %s: %s", filepath, exc)
        return {"functions": [], "calls": [], "types": [], "variables": [], "directives": []}

    functions = []
    calls = []
    types = []
    variables = []
    directives = []
    func_channel_map = {}

    def get_text(node):
        return raw_bytes[node.start_byte:node.end_byte].decode("utf8", errors="replace")

    for line in source_code.splitlines():
        line_str = line.strip()
        if line_str.startswith("//go:embed ") or line_str.startswith("//go:build ") or line_str.startswith("//go:generate "):
            parts = line_str.split(None, 1)
            if len(parts) == 2:
                directives.append({"directive": parts[0][2:], "args": parts[1].strip()})

    def walk(node, current_func=None, call_context="SYNC"):
        new_func = current_func
        new_context = call_context

        if node.type == 'go_statement':
            new_context = "GOROUTINE"
        elif node.type == 'defer_statement':
            new_context = "DEFER"

        # 1. Identify Function and Method Declarations
        if node.type in ['function_declaration', 'method_declaration']:
            name_node = node.child_by_field_name('name')
            if name_node:
                name = get_text(name_node)
                receiver_name = ""
                recv_node = None
                if node.type == 'method_declaration':
                    recv_node = node.child_by_field_name('receiver')
                    if recv_node:
                        recv_text = get_text(recv_node)
                        m = re.search(r'\*?(?:[A-Za-z0-9_]+\.)?([A-Za-z0-9_]+)(?:\[.*?\])?\s*\)', recv_text)
                        if m:
                            receiver_name = m.group(1)
                
                scoped_name = f"{receiver_name}.{name}" if receiver_name else name
                new_func = scoped_name
                func_channel_map[scoped_name] = {"sent": set(), "recv": set()}
                functions.append({
                    "name":  scoped_name,
                    "id":    f"{filepath}::{scoped_name}",
                    "start": node.start_point[0] + 1,
                    "end":   node.end_point[0] + 1,
                    "code":  get_text(node),
                    "is_exported": name[0].isupper() if name else False,
                    "is_pointer_receiver": "*" in get_text(recv_node) if recv_node else False,
                })

        # 2. Identify Types (Structs, Interfaces, Type Aliases)
        elif node.type == 'type_spec':
            name_node = node.child_by_field_name('name')
            type_node = node.child_by_field_name('type')
            if name_node and type_node:
                type_name = get_text(name_node)
                kind = "TYPE"
                fields = []
                methods = []
                embedded_types = []
                tags = []

                if type_node.type == 'struct_type':
                    kind = "STRUCT"
                    for child in type_node.children:
                        if child.type == 'field_declaration_list':
                            for field in child.children:
                                if field.type == 'field_declaration':
                                    f_names = [get_text(c) for c in field.children if c.type == 'field_identifier']
                                    f_type_node = next((c for c in field.children if c.type not in ['field_identifier', 'raw_string_literal', 'comment']), None)
                                    f_type = get_text(f_type_node) if f_type_node else ""
                                    tag_node = next((c for c in field.children if c.type == 'raw_string_literal'), None)
                                    f_tag = get_text(tag_node) if tag_node else ""
                                    if f_tag and f_tag not in tags:
                                        tags.append(f_tag)

                                    if f_names:
                                        for fn in f_names:
                                            fields.append({"name": fn, "type": f_type, "tag": f_tag, "is_embedded": False})
                                    else:
                                        embed_name = f_type.lstrip("*").split(".")[-1]
                                        if embed_name:
                                            embedded_types.append(embed_name)
                                            fields.append({"name": embed_name, "type": f_type, "tag": f_tag, "is_embedded": True})

                elif type_node.type == 'interface_type':
                    kind = "INTERFACE"

                    # A1: recursive walk — handles grammars that insert an intermediate
                    # `interface_body` node (or similar) between `interface_type` and
                    # the actual `method_elem` / `method_spec` leaves.
                    _IFACE_SKIP_TYPES = frozenset({'{', '}', ';', 'interface', 'comment'})

                    def _collect_iface_members(n):
                        for c in n.children:
                            if c.type in ('method_spec', 'method_elem'):
                                m_name_node = c.child_by_field_name('name')
                                if not m_name_node:
                                    m_name_node = next(
                                        (gc for gc in c.children
                                         if gc.type in ('field_identifier', 'identifier')),
                                        None,
                                    )
                                if m_name_node:
                                    methods.append({
                                        "name":      get_text(m_name_node),
                                        "signature": get_text(c),
                                    })
                            elif c.type in ('type_identifier', 'selector_expression', 'qualified_type'):
                                embed_name = get_text(c).split('.')[-1]
                                if embed_name and embed_name not in embedded_types:
                                    embedded_types.append(embed_name)
                            elif c.type not in _IFACE_SKIP_TYPES:
                                # Recurse into unknown intermediate container nodes
                                _collect_iface_members(c)

                    _collect_iface_members(type_node)

                    # A3: regex fallback — if tree-sitter found no methods at all
                    # (grammar mismatch or empty interface), parse the raw source text.
                    if not methods:
                        raw_iface = get_text(type_node)
                        # Match exported method names: CapitalLetter followed by identifier + '('
                        for m in re.finditer(r'\b([A-Z][A-Za-z0-9_]*)\s*\(', raw_iface):
                            m_name = m.group(1)
                            # Skip type-like names that appear in composite literals
                            if m_name in ('True', 'False', 'Nil'):
                                continue
                            line_start = raw_iface.rfind('\n', 0, m.start()) + 1
                            line_end   = raw_iface.find('\n', m.start())
                            sig = raw_iface[line_start: line_end if line_end != -1 else len(raw_iface)].strip()
                            methods.append({"name": m_name, "signature": sig})

                # A2: prepend the `type` keyword so stored code is always valid Go syntax.
                # get_text(node) on a type_spec returns "TypeName struct/interface {...}"
                # without the leading `type` keyword — add it back for readability.
                type_code = "type " + get_text(node)

                types.append({
                    "name": type_name,
                    "id": f"{filepath}::{type_name}",
                    "kind": kind,
                    "start": node.start_point[0] + 1,
                    "end":   node.end_point[0] + 1,
                    "code":  type_code,
                    "fields": fields,
                    "methods": methods,
                    "embedded_types": embedded_types,
                    "tags": tags,
                    "is_exported": type_name[0].isupper() if type_name else False,
                })

        # 3. Identify Package-Level Variables and Constants
        elif node.type in ['var_declaration', 'const_declaration'] and current_func is None:
            kind = "VAR" if node.type == 'var_declaration' else "CONST"
            for child in node.children:
                if child.type in ['var_spec', 'const_spec', 'value_spec']:
                    for id_node in child.children:
                        if id_node.type == 'identifier':
                            v_name = get_text(id_node)
                            variables.append({
                                "name": v_name,
                                "id": f"{filepath}::{v_name}",
                                "kind": kind,
                                "start": child.start_point[0] + 1,
                                "end": child.end_point[0] + 1,
                                "code": get_text(child),
                                "is_exported": v_name[0].isupper() if v_name else False,
                            })

        # 4. Identify Function Calls and Channel Operations within a function
        elif current_func:
            if node.type == 'call_expression':
                func_node = node.children[0]
                if func_node.type == 'identifier':
                    callee_name = get_text(func_node)
                    calls.append((current_func, callee_name, None, new_context))
                elif func_node.type == 'selector_expression':
                    parts = []
                    for child in func_node.children:
                        if child.type == 'field_identifier':
                            parts.append(get_text(child))
                        elif child.type == 'identifier':
                            parts.insert(0, get_text(child))
                    if parts:
                        callee_name = parts[-1]
                        callee_qualified = ".".join(parts)
                        calls.append((current_func, callee_name, callee_qualified, new_context))
            elif node.type == 'send_statement' and node.children:
                chan_name = get_text(node.children[0]).split("[")[0].strip()
                if chan_name and current_func in func_channel_map:
                    func_channel_map[current_func]["sent"].add(chan_name)
            elif node.type == 'receive_expression' or (node.type == 'unary_expression' and get_text(node).startswith('<-')):
                chan_expr = get_text(node).lstrip("<-").strip().split("[")[0].strip()
                if chan_expr and current_func in func_channel_map:
                    func_channel_map[current_func]["recv"].add(chan_expr)

        # Recurse through children
        for child in node.children:
            walk(child, new_func, new_context)

    walk(tree.root_node)

    for f in functions:
        fname = f["name"]
        if fname in func_channel_map:
            f["channels_sent"] = sorted(list(func_channel_map[fname]["sent"]))
            f["channels_received"] = sorted(list(func_channel_map[fname]["recv"]))

    return {"functions": functions, "calls": calls, "types": types, "variables": variables, "directives": directives}


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

        # Split diff lines into addition/deletion buckets, keeping the
        # +/- prefix so that patch_mode=True can correctly filter them.
        addition_lines: list[str] = []
        deletion_lines: list[str] = []

        for raw_line in patch.split("\n"):
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                addition_lines.append(raw_line)   # keep '+' prefix for patch_mode
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                deletion_lines.append(raw_line)   # keep '-' prefix for patch_mode

        # Use patch_mode=True so _extract_*_dependencies strips the prefix
        # and only considers lines starting with '+' (or ' ' for context).
        # Previously this used patch_mode=False on pre-stripped lines, which
        # caused the extractor to try to parse blank-stripped content as
        # raw source, missing multi-line import blocks.
        for dep in extractor(source_file, addition_lines, patch_mode=True):
            if dep not in added:
                added.append(dep)

        # For deletions, temporarily treat '-' lines as '+' lines so the
        # patch_mode=True filter picks them up.
        deletion_as_additions = [
            "+" + line[1:] if line.startswith("-") else line
            for line in deletion_lines
        ]
        for dep in extractor(source_file, deletion_as_additions, patch_mode=True):
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
