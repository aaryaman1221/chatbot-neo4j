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


def _unroll_python_call(node: ast.AST) -> Optional[str]:
    """Helper to unroll compound Python call attributes cleanly without breaking on indexes."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        prefix = _unroll_python_call(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def parse_python_ast(filepath: str, source_code: str) -> dict:
    if not filepath.endswith(".py") or not source_code:
        return {"functions": [], "calls": [], "types": [], "variables": [], "directives": []}

    functions = []
    calls = []

    try:
        tree = ast.parse(source_code)
        seen_lines = set()
        func_nodes = []

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
            is_private = scoped_name.split(".")[-1].startswith("_")
            functions.append({
                "name": scoped_name,
                "id": f"{filepath}::{scoped_name}",
                "start": node.lineno,
                "end": node.end_lineno,
                "code": func_code,
                "is_exported": not is_private,
                "is_pointer_receiver": "." in scoped_name,  # If it's a class method, it modifies an instance block
                "channels_sent": [],
                "channels_received": []
            })
            
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    qualified_name = _unroll_python_call(child.func)
                    if qualified_name:
                        bare_name = qualified_name.split(".")[-1]
                        calls.append((scoped_name, bare_name, qualified_name))
    except Exception:
        pass

    return {"functions": functions, "calls": calls, "types": [], "variables": [], "directives": []}


def _normalize_go_type(type_str: str) -> str:
    """
    Cleans a Go type string to extract the base type name.
    """
    if not type_str:
        return ""
    # Strip channel prefix
    type_str = re.sub(r'^(?:chan\s+|<-chan\s+|chan\s*<-)', '', type_str)
    # Strip slice/array decorators
    type_str = re.sub(r'^\[\d*\]', '', type_str)
    # Strip pointer decorator
    type_str = type_str.lstrip('*')
    # Strip map decorators
    m_map = re.match(r'^map\[[^\]]+\](.*)', type_str)
    if m_map:
        return _normalize_go_type(m_map.group(1).strip())
    # Strip package prefix
    if '.' in type_str:
        type_str = type_str.split('.')[-1]
    return type_str.strip()


def parse_field_tag(tag_str: str) -> dict:
    """
    Parses a Go struct field tag string (e.g. `json:"id" db:"user_id"`)
    into a key-value dictionary.
    """
    if not tag_str:
        return {}
    tag_str = tag_str.strip('`"')
    matches = re.findall(r'(\w+):"([^"]+)"', tag_str)
    parsed = {}
    for key, val in matches:
        base_val = val.split(',')[0].strip()
        parsed[key] = base_val
    return parsed


def parse_go_ast(filepath: str, source_code: str) -> dict:
    if not TREE_SITTER_AVAILABLE or not filepath.endswith(".go") or not source_code:
        return {"functions": [], "calls": [], "types": [], "variables": [], "directives": []}

    try:
        raw_bytes = source_code if isinstance(source_code, bytes) else bytes(source_code, "utf8")
        tree = go_parser.parse(raw_bytes)
    except Exception as exc:
        logger.debug("Failed to parse Go AST for %s: %s", filepath, exc)
        return {"functions": [], "calls": [], "types": [], "variables": [], "directives": []}

    functions, calls, types, variables, directives = [], [], [], [], []
    func_channel_map = {}
    
    # Track extra Go-specific properties
    func_return_types_map = {}
    func_accepts_context_map = {}
    func_locks_map = {}
    func_err_source = {}
    func_propagated_errors = {}
    func_type_parameters_map = {}
    
    is_test_file = filepath.endswith(_GO_TEST_SUFFIXES)

    def get_text(node):
        return raw_bytes[node.start_byte:node.end_byte].decode("utf8", errors="replace")

    for line in source_code.splitlines():
        line_str = line.strip()
        if line_str.startswith(("//go:embed ", "//go:build ", "//go:generate ")):
            parts = line_str.split(None, 1)
            if len(parts) == 2:
                directives.append({"directive": parts[0][2:], "args": parts[1].strip()})

    def find_call_expression(n):
        if n.type == 'call_expression':
            return n
        for child in n.children:
            res = find_call_expression(child)
            if res:
                return res
        return None

    def extract_type_parameters(node):
        tp_node = next((c for c in node.children if c.type == 'type_parameter_list'), None)
        params = []
        if tp_node:
            for c in tp_node.children:
                if c.type == 'type_parameter_declaration':
                    ident_node = next((gc for gc in c.children if gc.type == 'identifier'), None)
                    constraint_node = next((gc for gc in c.children if gc.type == 'type_constraint'), None)
                    param_name = get_text(ident_node) if ident_node else ""
                    constraint_name = get_text(constraint_node) if constraint_node else ""
                    if not constraint_name:
                        non_ident = [gc for gc in c.children if gc.type != 'identifier' and gc.type not in (',', ' ', 'comment')]
                        if non_ident:
                            constraint_name = get_text(non_ident[-1])
                    if param_name:
                        params.append({"param": param_name, "constraint": constraint_name})
        return params

    def walk(node, current_func=None, call_context="SYNC"):
        new_func = current_func
        new_context = call_context

        if node.type == 'go_statement':
            new_context = "GOROUTINE"
        elif node.type == 'defer_statement':
            new_context = "DEFER"

        # 1. Functional Structures Tracking
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
                        m = re.search(r'\b([A-Za-z0-9_]+)(?:\[.*?\])?\s*\)', recv_text)
                        if m:
                            receiver_name = m.group(1)
                
                scoped_name = f"{receiver_name}.{name}" if receiver_name else name
                new_func = scoped_name
                func_channel_map[scoped_name] = {"sent": set(), "recv": set()}
                func_locks_map[scoped_name] = []
                func_err_source[scoped_name] = {}
                func_propagated_errors[scoped_name] = set()
                
                # Check for context parameter
                accepts_ctx = False
                param_list = node.child_by_field_name('parameters') or next((c for c in node.children if c.type == 'parameter_list'), None)
                if param_list:
                    for param_decl in param_list.children:
                        if param_decl.type == 'parameter_declaration':
                            type_child = next((c for c in param_decl.children if c.type not in ('identifier', 'comment', ',', ' ')), None)
                            if type_child:
                                type_text = get_text(type_child)
                                if 'context.Context' in type_text or type_text == 'Context':
                                    accepts_ctx = True
                func_accepts_context_map[scoped_name] = accepts_ctx

                # Extract return types
                return_types = []
                input_param_node = None
                block_node = None
                children = node.children
                input_param_idx = -1
                block_idx = -1
                for idx, child in enumerate(children):
                    if child.type == 'parameter_list' and input_param_node is None:
                        input_param_node = child
                        input_param_idx = idx
                    elif child.type == 'block':
                        block_node = child
                        block_idx = idx
                        
                if input_param_node is not None and block_node is not None:
                    return_nodes = children[input_param_idx + 1 : block_idx]
                    for ret_node in return_nodes:
                        if ret_node.type == 'parameter_list':
                            for param_decl in ret_node.children:
                                if param_decl.type == 'parameter_declaration':
                                    type_child = None
                                    for c in param_decl.children:
                                        if c.type not in ('identifier', 'comment', ',', ' '):
                                            type_child = c
                                    if type_child:
                                        normalized = _normalize_go_type(get_text(type_child))
                                        if normalized:
                                            return_types.append(normalized)
                                    else:
                                        normalized = _normalize_go_type(get_text(param_decl))
                                        if normalized:
                                            return_types.append(normalized)
                        elif ret_node.type not in ('comment', 'space', ',', ' '):
                            normalized = _normalize_go_type(get_text(ret_node))
                            if normalized:
                                return_types.append(normalized)
                func_return_types_map[scoped_name] = return_types

                # Extract type parameters (generics)
                func_type_parameters_map[scoped_name] = extract_type_parameters(node)

                # Determine if it is a test function
                is_test_func = is_test_file and (name.startswith("Test") or name.startswith("Benchmark") or name.startswith("Fuzz"))

                functions.append({
                    "name":  scoped_name,
                    "id":    f"{filepath}::{scoped_name}",
                    "start": node.start_point[0] + 1,
                    "end":   node.end_point[0] + 1,
                    "code":  get_text(node),
                    "is_exported": name[0].isupper() if name else False,
                    "is_pointer_receiver": "*" in get_text(recv_node) if recv_node else False,
                    "is_test": is_test_func,
                })

        # 2. Type Schema Evaluation Blocks
        elif node.type == 'type_spec':
            name_node = node.child_by_field_name('name')
            type_node = node.child_by_field_name('type')
            if name_node and type_node:
                type_name = get_text(name_node)
                kind, fields, methods, embedded_types, tags = "TYPE", [], [], [], []
                tag_mappings = {}

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
                                    
                                    # Parse Go field tags
                                    f_parsed_tags = parse_field_tag(f_tag)
                                    if f_tag and f_tag not in tags:
                                        tags.append(f_tag)

                                    if f_names:
                                        for fn in f_names:
                                            fields.append({
                                                "name": fn,
                                                "type": f_type,
                                                "tag": f_tag,
                                                "parsed_tags": f_parsed_tags,
                                                "is_embedded": False
                                            })
                                            for tk, tv in f_parsed_tags.items():
                                                if tk not in tag_mappings:
                                                    tag_mappings[tk] = {}
                                                tag_mappings[tk][tv] = fn
                                    else:
                                        embed_name = f_type.lstrip("*").split(".")[-1]
                                        if embed_name:
                                            embedded_types.append(embed_name)
                                            fields.append({
                                                "name": embed_name,
                                                "type": f_type,
                                                "tag": f_tag,
                                                "parsed_tags": f_parsed_tags,
                                                "is_embedded": True
                                            })
                                            for tk, tv in f_parsed_tags.items():
                                                if tk not in tag_mappings:
                                                    tag_mappings[tk] = {}
                                                tag_mappings[tk][tv] = embed_name

                elif type_node.type in ['interface_type', 'interface_body']:
                    kind = "INTERFACE"
                    _IFACE_SKIP_TYPES = frozenset({'{', '}', ';', 'interface', 'comment'})

                    def _collect_iface_members(n):
                        for c in n.children:
                            if c.type in ('method_spec', 'method_elem'):
                                m_name_node = c.child_by_field_name('name') or next(
                                    (gc for gc in c.children if gc.type in ('field_identifier', 'identifier')), None
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
                                _collect_iface_members(c)

                    _collect_iface_members(type_node)

                    if not methods:
                        raw_iface = get_text(type_node)
                        for m in re.finditer(r'\b([A-Z][A-Za-z0-9_]*)\s*\(', raw_iface):
                            m_name = m.group(1)
                            if m_name in ('True', 'False', 'Nil'):
                                continue
                            line_start = raw_iface.rfind('\n', 0, m.start()) + 1
                            line_end   = raw_iface.find('\n', m.start())
                            sig = raw_iface[line_start: line_end if line_end != -1 else len(raw_iface)].strip()
                            methods.append({"name": m_name, "signature": sig})

                # Extract generic type parameters
                type_params = extract_type_parameters(node)

                types.append({
                    "name": type_name,
                    "id": f"{filepath}::{type_name}",
                    "kind": kind,
                    "start": node.start_point[0] + 1,
                    "end":   node.end_point[0] + 1,
                    "code":  "type " + get_text(node),
                    "fields": fields,
                    "methods": methods,
                    "embedded_types": embedded_types,
                    "tags": tags,
                    "tag_mappings": tag_mappings,
                    "type_parameters": type_params,
                    "is_exported": type_name[0].isupper() if type_name else False,
                })

        # 3. Variable/Constants Space Track
        elif node.type in ['var_declaration', 'const_declaration'] and current_func is None:
            kind = "VAR" if node.type == 'var_declaration' else "CONST"
            for child_spec in node.children:
                if child_spec.type in ['var_spec', 'const_spec', 'value_spec']:
                    # Check if contains a channel type
                    chan_node = next((c for c in child_spec.children if c.type == 'channel_type'), None)
                    var_kind = kind
                    chan_elem = ""
                    if chan_node:
                        var_kind = "CHAN"
                        elem_node = next((c for c in chan_node.children if c.type not in ('chan', 'comment', 'space', ' ', '<-')), None)
                        if elem_node:
                            chan_elem = _normalize_go_type(get_text(elem_node))

                    for id_node in child_spec.children:
                        if id_node.type == 'identifier':
                            v_name = get_text(id_node)
                            variables.append({
                                "name": v_name,
                                "id": f"{filepath}::{v_name}",
                                "kind": var_kind,
                                "chan_elem_type": chan_elem,
                                "start": child_spec.start_point[0] + 1,
                                "end": child_spec.end_point[0] + 1,
                                "code": get_text(child_spec),
                                "is_exported": v_name[0].isupper() if v_name else False,
                            })

        # 4. Expressions & Dynamic Operations Processing
        elif current_func:
            if node.type == 'call_expression':
                func_node = node.children[0]
                callee_name = ""
                callee_qual = None
                
                if func_node.type == 'identifier':
                    callee_name = get_text(func_node)
                elif func_node.type == 'selector_expression':
                    parts = []
                    for child in func_node.children:
                        if child.type in ['field_identifier', 'identifier']:
                            parts.append(get_text(child))
                    if parts:
                        callee_name = parts[-1]
                        callee_qual = ".".join(parts)

                # Mutex lock / unlock tracking
                if func_node.type == 'selector_expression':
                    obj_node = func_node.child_by_field_name('operand') or func_node.children[0]
                    field_node = func_node.child_by_field_name('field') or func_node.children[-1]
                    if obj_node and field_node:
                        action = get_text(field_node)
                        if action in ('Lock', 'RLock', 'Unlock', 'RUnlock'):
                            obj_name = get_text(obj_node)
                            act_str = f"defer_{action.lower()}" if new_context == "DEFER" else action.lower()
                            if current_func in func_locks_map:
                                func_locks_map[current_func].append(f"{act_str}:{obj_name}")

                if callee_name:
                    # Check context propagation & cancellation scope creation
                    propagates_ctx = False
                    creates_cancel = False
                    
                    if callee_name in ('WithCancel', 'WithTimeout', 'WithDeadline', 'WithCancelCause') and callee_qual and callee_qual.startswith('context.'):
                        creates_cancel = True
                        
                    arg_list = next((c for c in node.children if c.type == 'argument_list'), None)
                    if arg_list:
                        for arg in arg_list.children:
                            if arg.type in ('identifier', 'selector_expression', 'pointer_expression', 'unary_expression'):
                                arg_text = get_text(arg).lower()
                                if 'ctx' in arg_text or arg_text == 'context':
                                    propagates_ctx = True
                                    break
                                    
                    calls.append((current_func, callee_name, callee_qual, new_context, propagates_ctx, creates_cancel))

            elif node.type == 'send_statement' and node.children:
                chan_name = get_text(node.children[0]).split("[")[0].strip()
                if chan_name and current_func in func_channel_map:
                    func_channel_map[current_func]["sent"].add(chan_name)
            elif node.type in ['receive_expression', 'unary_expression'] and get_text(node).startswith('<-'):
                chan_expr = get_text(node).lstrip("<-").strip().split("[")[0].strip()
                if chan_expr and current_func in func_channel_map:
                    func_channel_map[current_func]["recv"].add(chan_expr)

            # Error path tracing: check assignments of calls to error variables
            elif node.type in ('short_var_declaration', 'assignment_statement'):
                op_idx = -1
                for idx, c in enumerate(node.children):
                    if c.type in (':=', '='):
                        op_idx = idx
                        break
                
                if op_idx != -1:
                    left_nodes = node.children[:op_idx]
                    right_nodes = node.children[op_idx+1:]
                    
                    callee = None
                    for rn in right_nodes:
                        call_node = find_call_expression(rn)
                        if call_node:
                            f_node = call_node.children[0]
                            callee = get_text(f_node).split('.')[-1]
                            break
                    
                    if callee:
                        for ln in left_nodes:
                            def collect_identifiers(n, idents):
                                if n.type == 'identifier':
                                    idents.append(get_text(n))
                                for c in n.children:
                                    collect_identifiers(c, idents)
                            idents = []
                            collect_identifiers(ln, idents)
                            for var_name in idents:
                                if 'err' in var_name.lower():
                                    if current_func in func_err_source:
                                        func_err_source[current_func][var_name] = callee

            elif node.type == 'return_statement':
                ret_text = get_text(node)
                # Check returns of error variables
                if current_func in func_err_source:
                    for var_name, callee in func_err_source[current_func].items():
                        if var_name in ret_text:
                            if current_func in func_propagated_errors:
                                func_propagated_errors[current_func].add(callee)
                
                # Check direct returns of calls
                call_node = find_call_expression(node)
                if call_node:
                    callee = get_text(call_node.children[0]).split('.')[-1]
                    if current_func in func_propagated_errors:
                        func_propagated_errors[current_func].add(callee)

        for child in node.children:
            walk(child, new_func, new_context)

    walk(tree.root_node)

    # Backfill extended properties to functions
    for f in functions:
        fname = f["name"]
        if fname in func_channel_map:
            f["channels_sent"] = sorted(list(func_channel_map[fname]["sent"]))
            f["channels_received"] = sorted(list(func_channel_map[fname]["recv"]))
            
        f["return_types"] = func_return_types_map.get(fname, [])
        f["accepts_context"] = func_accepts_context_map.get(fname, False)
        f["lock_sequence"] = func_locks_map.get(fname, [])
        f["propagated_errors"] = sorted(list(func_propagated_errors.get(fname, set())))
        f["type_parameters"] = func_type_parameters_map.get(fname, [])

    return {"functions": functions, "calls": calls, "types": types, "variables": variables, "directives": directives}


def _append_dependency(dependencies: list, filepath: str, target_module: str):
    if not target_module: return
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
            if not line or line[0] not in {"+", " "}: continue
            line = line[1:].lstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("//"): continue

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
            if not line or line[0] not in {"+", " "}: continue
            line = line[1:].lstrip()
        stripped = line.strip()
        if not stripped or (stripped.startswith("#") and not stripped.startswith("#include")):
            continue
        for pattern in patterns:
            m = re.search(pattern, stripped)
            if m: _append_dependency(dependencies, filepath, m.group(1))
    return dependencies


def extract_imports_from_source(filepath: str, source_code: str) -> list:
    kind = _go_file_kind(filepath)
    lines = source_code.split("\n")
    if kind in {"go_mod", "go_work", "go_source"}:
        return _extract_go_dependencies(filepath, lines, patch_mode=False)
    return _extract_generic_dependencies(filepath, lines, patch_mode=False)


def extract_file_dependencies(compact_files: list) -> list:
    dependencies = []
    for item in compact_files:
        source_file = item.get("filename")
        patch = item.get("patch", "")
        if not patch or not source_file: continue
        kind = _go_file_kind(source_file)
        lines = patch.split("\n")
        new_deps = _extract_go_dependencies(source_file, lines, patch_mode=True) if kind in {"go_mod", "go_work", "go_source"} else _extract_generic_dependencies(source_file, lines, patch_mode=True)
        for dep in new_deps:
            if dep not in dependencies: dependencies.append(dep)
    return dependencies


def extract_temporal_dependencies(compact_files: list) -> dict:
    added, removed = [], []
    for item in compact_files:
        source_file = item.get("filename")
        patch = item.get("patch", "")
        if not patch or not source_file: continue

        kind = _go_file_kind(source_file)
        extractor = _extract_go_dependencies if kind in {"go_mod", "go_work", "go_source"} else _extract_generic_dependencies
        addition_lines, deletion_lines = [], []

        for raw_line in patch.split("\n"):
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                addition_lines.append(raw_line)
            elif raw_line.startswith("-") and not raw_line.startswith("---"):
                deletion_lines.append(raw_line)

        for dep in extractor(source_file, addition_lines, patch_mode=True):
            if dep not in added: added.append(dep)

        deletion_as_additions = ["+" + line[1:] if line.startswith("-") else line for line in deletion_lines]
        for dep in extractor(source_file, deletion_as_additions, patch_mode=True):
            if dep not in removed: removed.append(dep)

    return {"added": added, "removed": removed}


def get_modified_functions(patch_text: str, filepath: str, ast_data: dict) -> list:
    if not patch_text or not ast_data.get("functions"): return []
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