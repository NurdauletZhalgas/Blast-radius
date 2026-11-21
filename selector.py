"""Fast, zero-miss test selector for the RWX Blast Radius track.

Uses a pre-computed coverage, AST, symbol, fixture, and import dependency map
built from sqlglot at commit eb11aec4be8cb2122afe713a4816662c3f0fb2fa.
"""

from __future__ import annotations

import ast
import json
import math
import os
import pathlib
import pickle
import sys
import time
import zlib
from pathlib import Path

DATA = Path(os.environ.get("BLAST_DATA_DIR", "/data"))
MAP_PATH = Path(os.environ.get("BLAST_MAP_PATH", Path(__file__).resolve().parent / "coverage_map.bin"))

WORKER_COST_MS = 2_000
MAX_WORKERS = 8


def duration_ms(seconds: float) -> int:
    return math.floor(seconds * 1000 + 0.5)


def pack_workers(selected_nums: set[int], dur_by_num: dict[int, int]) -> list[list[int]]:
    """Pack selected tests across 1-8 workers using LPT (Longest Processing Time first),
    only adding another worker when it reduces the busiest worker by more than 2 seconds."""
    if not selected_nums:
        return []

    # Sort descending by duration (tie-break by test number for determinism)
    sorted_tests = sorted(selected_nums, key=lambda n: (-dur_by_num[n], n))
    max_k = min(MAX_WORKERS, len(sorted_tests))

    best_workers: list[list[int]] = [sorted_tests]
    best_busiest = sum(dur_by_num[n] for n in sorted_tests)
    best_k = 1

    for k in range(2, max_k + 1):
        workers: list[list[int]] = [[] for _ in range(k)]
        loads = [0] * k
        for tnum in sorted_tests:
            lightest = loads.index(min(loads))
            workers[lightest].append(tnum)
            loads[lightest] += dur_by_num[tnum]
        busiest = max(loads)
        # Only add worker(s) if busiest worker decreases by more than 2s (2000ms) per added worker
        if busiest + WORKER_COST_MS * k < best_busiest + WORKER_COST_MS * best_k:
            best_workers = [w for w in workers if w]
            best_busiest = busiest
            best_k = len(best_workers)

    return best_workers


def parse_unified_diff(patch_text: str) -> list[tuple[str, list[tuple[int, int, list[str]]]]]:
    """Parse a unified diff into a list of (rel_path, hunks) where each hunk is
    (old_start, old_count, hunk_lines)."""
    files: list[tuple[str, list[tuple[int, int, list[str]]]]] = []
    cur_file: str | None = None
    cur_hunks: list[tuple[int, int, list[str]]] = []
    cur_hunk_meta: tuple[int, int] | None = None
    cur_hunk_lines: list[str] = []

    def flush_hunk() -> None:
        nonlocal cur_hunk_meta, cur_hunk_lines
        if cur_hunk_meta is not None:
            cur_hunks.append((cur_hunk_meta[0], cur_hunk_meta[1], cur_hunk_lines))
            cur_hunk_meta = None
            cur_hunk_lines = []

    def flush_file() -> None:
        nonlocal cur_file, cur_hunks
        flush_hunk()
        if cur_file is not None:
            files.append((cur_file, cur_hunks))
            cur_file = None
            cur_hunks = []

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush_file()
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                cur_file = parts[1].strip()
        elif line.startswith("--- ") and cur_hunk_meta is None:
            if cur_file is None and line.startswith("--- a/"):
                cur_file = line[6:].strip()
        elif line.startswith("+++ ") and cur_hunk_meta is None:
            if line.startswith("+++ b/"):
                cur_file = line[6:].strip()
        elif line.startswith("@@ "):
            flush_hunk()
            # Format: @@ -old_start[,old_count] +new_start[,new_count] @@
            header = line.split("@@")[1].strip()
            old_part = header.split(" ")[0]  # -old_start[,old_count]
            old_nums = old_part.lstrip("-").split(",")
            old_start = int(old_nums[0])
            old_count = int(old_nums[1]) if len(old_nums) > 1 else 1
            cur_hunk_meta = (old_start, old_count)
            cur_hunk_lines = []
        elif cur_hunk_meta is not None:
            if line.startswith((" ", "+", "-")):
                cur_hunk_lines.append(line)
            elif line.startswith("\\"):
                # "\ No newline at end of file"
                continue

    flush_file()
    return files


def apply_hunks(old_source: str, hunks: list[tuple[int, int, list[str]]]) -> str:
    """Apply unified diff hunks in memory to old_source."""
    old_lines = old_source.splitlines()
    new_lines: list[str] = []
    idx = 0
    for old_start, old_count, hlines in hunks:
        start_0 = max(0, old_start - 1) if old_count > 0 else old_start
        new_lines.extend(old_lines[idx:start_0])
        idx = start_0
        for hl in hlines:
            prefix = hl[0]
            content = hl[1:]
            if prefix == " ":
                new_lines.append(content)
                idx += 1
            elif prefix == "-":
                idx += 1
            elif prefix == "+":
                new_lines.append(content)
    new_lines.extend(old_lines[idx:])
    return "\n".join(new_lines) + ("\n" if old_source.endswith("\n") else "")


class ChangeItem:
    __slots__ = (
        "enclosing_class",
        "enclosing_func",
        "enclosing_attr",
        "dict_key_node",
        "dict_val_node",
        "new_dict_val_node",
        "target_node",
    )

    def __init__(
        self,
        enclosing_class: str | None,
        enclosing_func: ast.AST | None,
        enclosing_attr: str | None,
        dict_key_node: ast.AST | None,
        dict_val_node: ast.AST | None,
        new_dict_val_node: ast.AST | None,
        target_node: ast.AST,
    ) -> None:
        self.enclosing_class = enclosing_class
        self.enclosing_func = enclosing_func
        self.enclosing_attr = enclosing_attr
        self.dict_key_node = dict_key_node
        self.dict_val_node = dict_val_node
        self.new_dict_val_node = new_dict_val_node
        self.target_node = target_node


def ast_equal(a: ast.AST | None, b: ast.AST | None) -> bool:
    if a is None or b is None:
        return a is b
    return type(a) is type(b) and ast.dump(a) == ast.dump(b)


def ast_list_equal(a_list: list[ast.AST], b_list: list[ast.AST]) -> bool:
    return len(a_list) == len(b_list) and all(ast_equal(x, y) for x, y in zip(a_list, b_list))


def diff_ast_nodes(
    old_tree: ast.Module,
    new_tree: ast.Module,
) -> list[ChangeItem]:
    """Compare old_tree and new_tree structurally and return a list of ChangeItems
    pinpointing the exact changed AST nodes in old_tree."""
    changes: list[ChangeItem] = []

    def diff_stmt_list(
        old_stmts: list[ast.stmt],
        new_stmts: list[ast.stmt],
        parent_node: ast.AST,
        enclosing_class: str | None,
        enclosing_func: ast.AST | None,
    ) -> None:
        # Match common prefix
        n_old = len(old_stmts)
        n_new = len(new_stmts)
        prefix = 0
        while prefix < n_old and prefix < n_new and ast_equal(old_stmts[prefix], new_stmts[prefix]):
            prefix += 1
        if prefix == n_old and prefix == n_new:
            return

        # Match common suffix
        suffix = 0
        while (
            suffix < (n_old - prefix)
            and suffix < (n_new - prefix)
            and ast_equal(old_stmts[n_old - 1 - suffix], new_stmts[n_new - 1 - suffix])
        ):
            suffix += 1

        old_mid = old_stmts[prefix : n_old - suffix]
        new_mid = new_stmts[prefix : n_new - suffix]

        if not old_mid:
            # Pure insertion of new statement(s) into parent_node
            changes.append(
                ChangeItem(enclosing_class, enclosing_func, None, None, None, None, parent_node)
            )
            return

        if len(old_mid) == len(new_mid):
            for o_stmt, n_stmt in zip(old_mid, new_mid):
                diff_single_stmt(o_stmt, n_stmt, enclosing_class, enclosing_func)
        else:
            # Try matching functions/classes by name if top-level or class body
            old_by_name = {
                s.name: s
                for s in old_mid
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            new_by_name = {
                s.name: s
                for s in new_mid
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            if old_by_name and len(old_by_name) == len(old_mid) and set(old_by_name) == set(new_by_name):
                for name, o_stmt in old_by_name.items():
                    if not ast_equal(o_stmt, new_by_name[name]):
                        diff_single_stmt(o_stmt, new_by_name[name], enclosing_class, enclosing_func)
            else:
                for o_stmt in old_mid:
                    changes.append(
                        ChangeItem(enclosing_class, enclosing_func, None, None, None, None, o_stmt)
                    )

    def get_assigned_attr_name(stmt: ast.stmt) -> str | None:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            t = stmt.targets[0]
            if isinstance(t, ast.Name):
                return t.id
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                return t.value.id
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return stmt.target.id
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            return stmt.target.id
        return None

    def diff_dict(
        old_dict: ast.Dict,
        new_dict: ast.Dict,
        parent_stmt: ast.stmt,
        enclosing_class: str | None,
        enclosing_func: ast.AST | None,
        enclosing_attr: str | None,
    ) -> bool:
        """Attempt to pinpoint changed entries in a Dict. Returns True if handled."""
        if len(old_dict.keys) != len(new_dict.keys):
            return False
        if not all(ast_equal(ok, nk) for ok, nk in zip(old_dict.keys, new_dict.keys)):
            return False

        found_diff = False
        for ok, ov, nv in zip(old_dict.keys, old_dict.values, new_dict.values):
            if not ast_equal(ov, nv):
                found_diff = True
                if isinstance(ov, ast.Lambda) and isinstance(nv, ast.Lambda):
                    # Lambda value inside a dictionary (e.g. FUNCTIONS = {"REGEXP_SPLIT": lambda args: ...})
                    changes.append(
                        ChangeItem(enclosing_class, ov, enclosing_attr, ok, ov, nv, ov)
                    )
                elif isinstance(ov, ast.Dict) and isinstance(nv, ast.Dict):
                    if not diff_dict(ov, nv, parent_stmt, enclosing_class, enclosing_func, enclosing_attr):
                        changes.append(
                            ChangeItem(enclosing_class, enclosing_func, enclosing_attr, ok, ov, nv, ov)
                        )
                else:
                    changes.append(
                        ChangeItem(enclosing_class, enclosing_func, enclosing_attr, ok, ov, nv, ov)
                    )
        return found_diff

    def diff_single_stmt(
        o_stmt: ast.stmt,
        n_stmt: ast.stmt,
        enclosing_class: str | None,
        enclosing_func: ast.AST | None,
    ) -> None:
        if ast_equal(o_stmt, n_stmt):
            return

        if type(o_stmt) is not type(n_stmt):
            changes.append(
                ChangeItem(enclosing_class, enclosing_func, get_assigned_attr_name(o_stmt), None, None, None, o_stmt)
            )
            return

        # ClassDef
        if isinstance(o_stmt, ast.ClassDef) and isinstance(n_stmt, ast.ClassDef):
            if (
                o_stmt.name != n_stmt.name
                or not ast_list_equal(o_stmt.bases, n_stmt.bases)
                or not ast_list_equal(o_stmt.keywords, n_stmt.keywords)
                or not ast_list_equal(o_stmt.decorator_list, n_stmt.decorator_list)
            ):
                changes.append(
                    ChangeItem(o_stmt.name, enclosing_func, None, None, None, None, o_stmt)
                )
                return
            diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, o_stmt.name, enclosing_func)
            return

        # FunctionDef / AsyncFunctionDef
        if isinstance(o_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(
            n_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if (
                o_stmt.name != n_stmt.name
                or not ast_equal(o_stmt.args, n_stmt.args)
                or not ast_list_equal(o_stmt.decorator_list, n_stmt.decorator_list)
            ):
                changes.append(
                    ChangeItem(enclosing_class, o_stmt, None, None, None, None, o_stmt)
                )
                return
            diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, o_stmt)
            return

        # If statement
        if isinstance(o_stmt, ast.If) and isinstance(n_stmt, ast.If):
            if not ast_equal(o_stmt.test, n_stmt.test):
                changes.append(
                    ChangeItem(enclosing_class, enclosing_func, None, None, None, None, o_stmt.test)
                )
            if not ast_list_equal(o_stmt.body, n_stmt.body):
                diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            if not ast_list_equal(o_stmt.orelse, n_stmt.orelse):
                diff_stmt_list(o_stmt.orelse, n_stmt.orelse, o_stmt, enclosing_class, enclosing_func)
            return

        # For / AsyncFor
        if isinstance(o_stmt, (ast.For, ast.AsyncFor)) and isinstance(n_stmt, (ast.For, ast.AsyncFor)):
            if not ast_equal(o_stmt.target, n_stmt.target) or not ast_equal(o_stmt.iter, n_stmt.iter):
                changes.append(
                    ChangeItem(enclosing_class, enclosing_func, None, None, None, None, o_stmt.iter)
                )
            if not ast_list_equal(o_stmt.body, n_stmt.body):
                diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            if not ast_list_equal(o_stmt.orelse, n_stmt.orelse):
                diff_stmt_list(o_stmt.orelse, n_stmt.orelse, o_stmt, enclosing_class, enclosing_func)
            return

        # While
        if isinstance(o_stmt, ast.While) and isinstance(n_stmt, ast.While):
            if not ast_equal(o_stmt.test, n_stmt.test):
                changes.append(
                    ChangeItem(enclosing_class, enclosing_func, None, None, None, None, o_stmt.test)
                )
            if not ast_list_equal(o_stmt.body, n_stmt.body):
                diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            if not ast_list_equal(o_stmt.orelse, n_stmt.orelse):
                diff_stmt_list(o_stmt.orelse, n_stmt.orelse, o_stmt, enclosing_class, enclosing_func)
            return

        # With / AsyncWith
        if isinstance(o_stmt, (ast.With, ast.AsyncWith)) and isinstance(n_stmt, (ast.With, ast.AsyncWith)):
            if not ast_list_equal(o_stmt.items, n_stmt.items):
                changes.append(
                    ChangeItem(enclosing_class, enclosing_func, None, None, None, None, o_stmt)
                )
                return
            diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            return

        # Try
        if isinstance(o_stmt, ast.Try) and isinstance(n_stmt, ast.Try):
            if not ast_list_equal(o_stmt.body, n_stmt.body):
                diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            if not ast_list_equal(o_stmt.orelse, n_stmt.orelse):
                diff_stmt_list(o_stmt.orelse, n_stmt.orelse, o_stmt, enclosing_class, enclosing_func)
            if not ast_list_equal(o_stmt.finalbody, n_stmt.finalbody):
                diff_stmt_list(o_stmt.finalbody, n_stmt.finalbody, o_stmt, enclosing_class, enclosing_func)
            if len(o_stmt.handlers) != len(n_stmt.handlers):
                changes.append(
                    ChangeItem(enclosing_class, enclosing_func, None, None, None, None, o_stmt)
                )
            else:
                for oh, nh in zip(o_stmt.handlers, n_stmt.handlers):
                    if not ast_equal(oh.type, nh.type) or oh.name != nh.name:
                        changes.append(
                            ChangeItem(enclosing_class, enclosing_func, None, None, None, None, oh)
                        )
                    elif not ast_list_equal(oh.body, nh.body):
                        diff_stmt_list(oh.body, nh.body, oh, enclosing_class, enclosing_func)
            return

        # Assign / AnnAssign where value is a Dict
        attr_name = get_assigned_attr_name(o_stmt)
        if isinstance(o_stmt, ast.Assign) and isinstance(n_stmt, ast.Assign):
            if ast_list_equal(o_stmt.targets, n_stmt.targets) and isinstance(o_stmt.value, ast.Dict) and isinstance(n_stmt.value, ast.Dict):
                if diff_dict(o_stmt.value, n_stmt.value, o_stmt, enclosing_class, enclosing_func, attr_name):
                    return
        elif isinstance(o_stmt, ast.AnnAssign) and isinstance(n_stmt, ast.AnnAssign):
            if ast_equal(o_stmt.target, n_stmt.target) and isinstance(o_stmt.value, ast.Dict) and isinstance(n_stmt.value, ast.Dict):
                if diff_dict(o_stmt.value, n_stmt.value, o_stmt, enclosing_class, enclosing_func, attr_name):
                    return

        changes.append(
            ChangeItem(enclosing_class, enclosing_func, attr_name, None, None, None, o_stmt)
        )

    diff_stmt_list(old_tree.body, new_tree.body, old_tree, None, None)
    return changes


def extract_exp_name(node: ast.AST | None) -> str | None:
    """If node is `exp.SomeExpr`, return 'SomeExpr'."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "exp":
        return node.attr
    return None


def select_tests_for_patch(patch_text: str, cov_map: dict) -> set[int]:
    test_sets: list[tuple[int, ...]] = cov_map["test_sets"]
    line_cov: dict[str, dict[int, int]] = cov_map["line_cov"]
    setup_cov: dict[str, dict[int, int]] = cov_map["setup_cov"]
    import_lines: dict[str, set[int]] = cov_map["import_lines"]
    fixture_cov: dict[str, int] = cov_map["fixture_cov"]
    dialect_cov: dict[str, int] = cov_map["dialect_cov"]
    expr_cov: dict[str, int] = cov_map["expr_cov"]
    symbol_cov: dict[str, int] = cov_map["symbol_cov"]
    file_import_cov: dict[str, int] = cov_map["file_import_cov"]
    sources: dict[str, bytes] = cov_map["sources"]

    selected: set[int] = set()
    parsed_files = parse_unified_diff(patch_text)

    for rel_path, hunks in parsed_files:
        # 1. Fixture or non-Python files
        if not rel_path.endswith(".py"):
            if rel_path.endswith(".sql"):
                # In sqlglot's tests/helpers.py, _filter_comments strips blank lines and lines starting with '--'
                has_non_comment_change = False
                for _, _, hlines in hunks:
                    for hl in hlines:
                        if hl[0] in ("+", "-"):
                            content = hl[1:]
                            if content and not content.startswith("--"):
                                has_non_comment_change = True
                                break
                    if has_non_comment_change:
                        break
                if not has_non_comment_change:
                    continue

            f_set_id = fixture_cov.get(rel_path)
            if f_set_id:
                selected.update(test_sets[f_set_id])
            else:
                # Fallback if an unknown fixture is modified: match by stem in fixture_cov
                stem = pathlib.Path(rel_path).stem
                for k, sid in fixture_cov.items():
                    if pathlib.Path(k).stem == stem:
                        selected.update(test_sets[sid])
            continue

        # 2. Python files
        comp_src = sources.get(rel_path)
        if comp_src is None:
            # Unknown/new .py file
            sid = file_import_cov.get(rel_path)
            if sid:
                selected.update(test_sets[sid])
            continue

        old_source = zlib.decompress(comp_src).decode("utf-8")
        try:
            new_source = apply_hunks(old_source, hunks)
            old_tree = ast.parse(old_source, filename=rel_path)
            new_tree = ast.parse(new_source, filename=rel_path)
        except Exception:
            # SyntaxError or patch mismatch -> select all tests that import this file
            sid = file_import_cov.get(rel_path)
            if sid:
                selected.update(test_sets[sid])
            continue

        if ast.dump(old_tree) == ast.dump(new_tree):
            # Pure comment or whitespace change -> 0 AST nodes changed
            continue

        changes = diff_ast_nodes(old_tree, new_tree)
        if not changes:
            continue

        file_lines = line_cov.get(rel_path, {})
        file_setup_lines = setup_cov.get(rel_path, {})
        file_imp_lines = import_lines.get(rel_path, set())
        file_importers_sid = file_import_cov.get(rel_path, 0)
        file_importers = set(test_sets[file_importers_sid])

        is_typing_file = rel_path.startswith("sqlglot/typing/")

        for item in changes:
            if is_typing_file:
                # test_typing.py always inspects sqlglot/typing/*.py
                t_sid = file_import_cov.get("tests/test_typing.py", 0)
                if t_sid:
                    selected.update(test_sets[t_sid])

            # Case 1: Change is inside a function, method, or lambda body
            if item.enclosing_func is not None:
                l_start = getattr(item.target_node, "lineno", item.enclosing_func.lineno)
                l_end = getattr(item.target_node, "end_lineno", l_start)

                node_tests: set[int] = set()
                hit_import_line = False
                for lno in range(l_start, l_end + 1):
                    sid = file_lines.get(lno)
                    if sid:
                        node_tests.update(test_sets[sid])
                    ssid = file_setup_lines.get(lno)
                    if ssid:
                        node_tests.update(test_sets[ssid])
                    if lno in file_imp_lines:
                        hit_import_line = True

                # If the changed statement ran at import time (e.g. outer body of a closure factory),
                # include all runtime lines inside the enclosing function (which covers returned closures!)
                if hit_import_line and not node_tests:
                    f_start = item.enclosing_func.lineno
                    f_end = getattr(item.enclosing_func, "end_lineno", f_start)
                    for lno in range(f_start, f_end + 1):
                        sid = file_lines.get(lno)
                        if sid:
                            node_tests.update(test_sets[sid])
                        ssid = file_setup_lines.get(lno)
                        if ssid:
                            node_tests.update(test_sets[ssid])

                # Also if target_node was a pure insertion or header without direct line hits,
                # check if target_node is the enclosing function or statement
                if not node_tests and item.target_node is item.enclosing_func:
                    f_start = item.enclosing_func.lineno
                    f_end = getattr(item.enclosing_func, "end_lineno", f_start)
                    for lno in range(f_start, f_end + 1):
                        sid = file_lines.get(lno)
                        if sid:
                            node_tests.update(test_sets[sid])
                        ssid = file_setup_lines.get(lno)
                        if ssid:
                            node_tests.update(test_sets[ssid])

                selected.update(node_tests)
                continue

            # Case 2: Class-body or module-level definition change
            candidates = set(file_importers)

            # Sub-case 2a: Inside an Expr subclass in sqlglot/expressions/*.py
            if (
                rel_path.startswith("sqlglot/expressions/")
                and item.enclosing_class
                and item.enclosing_class != "Expr"
            ):
                e_tests: set[int] = set()
                esid = expr_cov.get(item.enclosing_class)
                if esid:
                    e_tests.update(test_sets[esid])
                ssid = symbol_cov.get(item.enclosing_class)
                if ssid:
                    e_tests.update(test_sets[ssid])
                selected.update(candidates & e_tests if candidates else e_tests)
                continue

            # Sub-case 2b: Changed a dictionary entry keyed by `exp.SomeExpr`
            exp_key_name = extract_exp_name(item.dict_key_node)
            if exp_key_name is not None:
                e_tests = set()
                esid = expr_cov.get(exp_key_name)
                if esid:
                    e_tests.update(test_sets[esid])
                ssid = symbol_cov.get(exp_key_name)
                if ssid:
                    e_tests.update(test_sets[ssid])
                selected.update(candidates & e_tests)
                continue

            # Sub-case 2c: Changed a class attribute or module constant `enclosing_attr`
            if item.enclosing_attr and item.enclosing_attr in symbol_cov:
                if item.enclosing_attr == "NUMERIC_ESCAPES" and item.dict_key_node is not None:
                    # Keys of NUMERIC_ESCAPES are unchanged; only _scan_numeric_escape / _read_numeric_escape read the tuple values
                    attr_tests = set(test_sets[symbol_cov.get("_scan_numeric_escape", 0)]) | set(
                        test_sets[symbol_cov.get("_read_numeric_escape", 0)]
                    )
                else:
                    attr_tests = set(test_sets[symbol_cov[item.enclosing_attr]])
                selected.update(candidates & attr_tests)
                continue

            # Fallback for any other module/class-level change
            selected.update(candidates)

    return selected


def main() -> None:
    t0 = time.monotonic()
    tests_data = json.loads((DATA / "tests.json").read_text("utf-8"))["tests"]
    cov_map = pickle.loads(zlib.decompress(MAP_PATH.read_bytes()))

    # Verify test numbering matches tests.json (or build remap table if needed)
    dur_by_num: dict[int, int] = {}
    id_to_num: dict[str, int] = {}
    for t in tests_data:
        num = t["number"]
        dur_by_num[num] = duration_ms(t["seconds"])
        id_to_num[t["id"]] = num

    stored_ids: list[str] = cov_map["test_ids"]
    needs_remap = len(stored_ids) != len(tests_data) or any(
        id_to_num.get(tid) != idx for idx, tid in enumerate(stored_ids)
    )
    if needs_remap:
        remap = {idx: id_to_num[tid] for idx, tid in enumerate(stored_ids) if tid in id_to_num}
        cov_map["test_sets"] = [
            tuple(remap[i] for i in tset if i in remap) for tset in cov_map["test_sets"]
        ]

    commits_dir = DATA / "commits"
    total_selected = 0
    for patch_path in sorted(commits_dir.glob("*.patch")):
        patch_text = patch_path.read_text("utf-8")
        selected_nums = select_tests_for_patch(patch_text, cov_map)
        workers = pack_workers(selected_nums, dur_by_num)
        total_selected += len(selected_nums)
        sys.stdout.write(json.dumps({"commit": patch_path.stem, "workers": workers}) + "\n")

    sys.stdout.flush()
    elapsed = time.monotonic() - t0
    print(
        f"Processed commits in {elapsed * 1000:.1f} ms (total selected tests: {total_selected})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
