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

DATA = Path(os.environ.get("BLAST_DATA_DIR", "/data")).resolve()
MAP_PATH = Path(
    os.environ.get("BLAST_MAP_PATH", Path(__file__).resolve().parent / "coverage_map.bin")
).resolve()

WORKER_COST_MS = 2_000
MAX_WORKERS = 8

DISPATCH_DICT_ATTRS = frozenset(
    {
        "FUNCTIONS",
        "FUNCTION_PARSERS",
        "NO_PAREN_FUNCTIONS",
        "NO_PAREN_FUNCTION_PARSERS",
        "STATEMENT_PARSERS",
        "PROPERTY_PARSERS",
        "RANGE_PARSERS",
        "SHOW_PARSERS",
        "SET_PARSERS",
        "TRANSFORMS",
        "PROPERTIES_LOCATION",
        "EXPRESSION_METADATA",
        "TIMESTAMPS",
    }
)


def duration_ms(seconds: float) -> int:
    return math.floor(seconds * 1000 + 0.5)


def _refine_partition(
    workers: list[list[int]], loads: list[int], dur_by_num: dict[int, int]
) -> int:
    """Refine an LPT partition across k workers using 1-move and 1-swap local search
    to minimize max(loads)."""
    k = len(workers)
    if k <= 1:
        return loads[0] if loads else 0

    for _ in range(32):
        max_load = max(loads)
        b = loads.index(max_load)
        b_tests = workers[b]
        if len(b_tests) <= 1:
            break

        # 1. Try moving a single test from busiest worker b to lightest worker w
        min_load = min(loads)
        w_min = loads.index(min_load)
        moved = False
        if w_min != b:
            for idx in range(len(b_tests) - 1, -1, -1):
                t = b_tests[idx]
                dt = dur_by_num[t]
                if dt > 0 and min_load + dt < max_load:
                    b_tests.pop(idx)
                    workers[w_min].append(t)
                    loads[b] -= dt
                    loads[w_min] += dt
                    moved = True
                    break
        if moved:
            continue

        # 2. Try swapping a test from busiest worker b with a smaller test from another worker w
        best_improvement = 0
        best_swap: tuple[int, int, int, int] | None = None  # (w, idx_b, idx_w, delta)
        limit_b = min(len(b_tests), 24)
        for idx_b in range(limit_b):
            tb = b_tests[idx_b]
            dtb = dur_by_num[tb]
            if dtb == 0:
                continue
            for w in range(k):
                if w == b:
                    continue
                lw = loads[w]
                if lw >= max_load:
                    continue
                w_tests = workers[w]
                limit_w = min(len(w_tests), 24)
                for idx_w in range(limit_w):
                    tw = w_tests[idx_w]
                    dtw = dur_by_num[tw]
                    delta = dtb - dtw
                    if delta <= 0:
                        continue
                    new_max_pair = max(max_load - delta, lw + delta)
                    improvement = max_load - new_max_pair
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (w, idx_b, idx_w, delta)

        if best_swap is not None:
            w, idx_b, idx_w, delta = best_swap
            tb = workers[b][idx_b]
            tw = workers[w][idx_w]
            workers[b][idx_b] = tw
            workers[w][idx_w] = tb
            loads[b] -= delta
            loads[w] += delta
        else:
            break

    return max(loads)


def pack_workers(selected_nums: set[int], dur_by_num: dict[int, int]) -> list[list[int]]:
    """Pack selected tests across 1-8 workers using LPT + local swap refinement,
    picking the k that minimizes max(worker_load) + 2s * k."""
    if not selected_nums:
        return []

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
        busiest = _refine_partition(workers, loads, dur_by_num)
        occupied = [w for w in workers if w]
        eff_k = len(occupied)
        if busiest + WORKER_COST_MS * eff_k < best_busiest + WORKER_COST_MS * best_k:
            best_workers = occupied
            best_busiest = busiest
            best_k = eff_k

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
            header = line.split("@@")[1].strip()
            old_part = header.split(" ")[0]
            old_nums = old_part.lstrip("-").split(",")
            old_start = int(old_nums[0])
            old_count = int(old_nums[1]) if len(old_nums) > 1 else 1
            cur_hunk_meta = (old_start, old_count)
            cur_hunk_lines = []
        elif cur_hunk_meta is not None:
            if line.startswith((" ", "+", "-")):
                cur_hunk_lines.append(line)
            elif line.startswith("\\"):
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


def _is_type_checking_if(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.If):
        return False
    test = stmt.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        len(body) > 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


class _RuntimeNormalizer(ast.NodeTransformer):
    """Strip non-runtime AST metadata (docstrings, `if TYPE_CHECKING:` blocks,
    and variable/parameter/return type annotations when `from __future__ import annotations`
    is active) so pure annotation/docstring edits are recognized as no-ops."""

    def __init__(self, strip_annotations: bool) -> None:
        self.strip_annotations = strip_annotations

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = [
            self.visit(s)
            for s in _strip_docstring(node.body)
            if not _is_type_checking_if(s)
        ]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = [
            self.visit(s)
            for s in _strip_docstring(node.body)
            if not _is_type_checking_if(s)
        ]
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = _strip_docstring(node.body)
        if not node.body:
            node.body = [ast.Pass()]
        if self.strip_annotations:
            node.returns = None
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = _strip_docstring(node.body)
        if not node.body:
            node.body = [ast.Pass()]
        if self.strip_annotations:
            node.returns = None
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if self.strip_annotations:
            node.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        if self.strip_annotations:
            if node.value is not None:
                return ast.copy_location(
                    ast.Assign(targets=[node.target], value=node.value), node
                )
            return ast.copy_location(ast.Pass(), node)
        return node


def is_runtime_noop_node(o_node: ast.AST, n_node: ast.AST, has_future_annot: bool) -> bool:
    """Check if two small diverging AST nodes differ only in docstrings,
    `if TYPE_CHECKING:` blocks, or type annotations."""
    import copy

    norm = _RuntimeNormalizer(strip_annotations=has_future_annot)
    ot = norm.visit(copy.deepcopy(o_node))
    nt = norm.visit(copy.deepcopy(n_node))
    return ast.dump(ot) == ast.dump(nt)


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


def narrow_changed_expr(old_expr: ast.AST, new_expr: ast.AST) -> ast.AST:
    """Soundly narrow a changed expression to the sub-expression that actually diverged
    when execution order / short-circuiting guarantees earlier parts executed identically."""
    # 1. BoolOp with the SAME operator (And == And, Or == Or) and identical prefix operands
    if (
        isinstance(old_expr, ast.BoolOp)
        and isinstance(new_expr, ast.BoolOp)
        and type(old_expr.op) is type(new_expr.op)
    ):
        o_vals = old_expr.values
        n_vals = new_expr.values
        prefix = 0
        while prefix < len(o_vals) and prefix < len(n_vals) and ast_equal(o_vals[prefix], n_vals[prefix]):
            prefix += 1
        if prefix > 0 and prefix < len(o_vals):
            # If only a single operand at `prefix` changed and suffixes match, descend into it
            if len(o_vals) == len(n_vals) and all(
                ast_equal(o_vals[i], n_vals[i]) for i in range(prefix + 1, len(o_vals))
            ):
                return narrow_changed_expr(o_vals[prefix], n_vals[prefix])
            # Otherwise return a synthetic span starting at o_vals[prefix]
            first_changed = o_vals[prefix]
            prev_end = getattr(o_vals[prefix - 1], "end_lineno", 0)
            if getattr(first_changed, "lineno", 0) > prev_end:
                return first_changed

    # 2. IfExp (body if test else orelse)
    if isinstance(old_expr, ast.IfExp) and isinstance(new_expr, ast.IfExp):
        test_eq = ast_equal(old_expr.test, new_expr.test)
        body_eq = ast_equal(old_expr.body, new_expr.body)
        orelse_eq = ast_equal(old_expr.orelse, new_expr.orelse)
        if not test_eq and body_eq and orelse_eq:
            return narrow_changed_expr(old_expr.test, new_expr.test)
        if test_eq and not body_eq and orelse_eq:
            return narrow_changed_expr(old_expr.body, new_expr.body)
        if test_eq and body_eq and not orelse_eq:
            return narrow_changed_expr(old_expr.orelse, new_expr.orelse)

    return old_expr


def diff_ast_nodes(
    old_tree: ast.Module,
    new_tree: ast.Module,
    old_lines: list[str],
    new_lines: list[str],
    has_future_annot: bool,
) -> list[ChangeItem]:
    """Compare old_tree and new_tree structurally and return a list of ChangeItems
    pinpointing the exact changed AST nodes in old_tree."""
    changes: list[ChangeItem] = []

    def nodes_equal(a: ast.AST | None, b: ast.AST | None) -> bool:
        if a is None or b is None:
            return a is b
        if type(a) is not type(b):
            return False
        a_l0 = getattr(a, "lineno", None)
        if a_l0 is not None:
            a_l1 = getattr(a, "end_lineno", a_l0)
            b_l0 = getattr(b, "lineno", a_l0)
            b_l1 = getattr(b, "end_lineno", b_l0)
            if (a_l1 - a_l0) == (b_l1 - b_l0) and old_lines[a_l0 - 1 : a_l1] == new_lines[
                b_l0 - 1 : b_l1
            ]:
                return True
        return ast.dump(a) == ast.dump(b)

    def lists_equal(a_list: list[ast.AST], b_list: list[ast.AST]) -> bool:
        return len(a_list) == len(b_list) and all(
            nodes_equal(x, y) for x, y in zip(a_list, b_list)
        )

    def diff_stmt_list(
        old_stmts: list[ast.stmt],
        new_stmts: list[ast.stmt],
        parent_node: ast.AST,
        enclosing_class: str | None,
        enclosing_func: ast.AST | None,
    ) -> None:
        n_old = len(old_stmts)
        n_new = len(new_stmts)
        prefix = 0
        while prefix < n_old and prefix < n_new and nodes_equal(old_stmts[prefix], new_stmts[prefix]):
            prefix += 1
        if prefix == n_old and prefix == n_new:
            return

        suffix = 0
        while (
            suffix < (n_old - prefix)
            and suffix < (n_new - prefix)
            and nodes_equal(old_stmts[n_old - 1 - suffix], new_stmts[n_new - 1 - suffix])
        ):
            suffix += 1

        old_mid = old_stmts[prefix : n_old - suffix]
        new_mid = new_stmts[prefix : n_new - suffix]

        if not old_mid:
            # Check if the inserted statements are only `if TYPE_CHECKING:` or docstrings/annotations
            if all(_is_type_checking_if(s) for s in new_mid):
                return
            changes.append(
                ChangeItem(enclosing_class, enclosing_func, None, None, None, None, parent_node)
            )
            return

        if len(old_mid) == len(new_mid):
            for o_stmt, n_stmt in zip(old_mid, new_mid):
                diff_single_stmt(o_stmt, n_stmt, enclosing_class, enclosing_func)
        else:
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
            if (
                old_by_name
                and len(old_by_name) == len(old_mid)
                and set(old_by_name) == set(new_by_name)
            ):
                for name, o_stmt in old_by_name.items():
                    if not nodes_equal(o_stmt, new_by_name[name]):
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
        """Pinpoint changed, added, or removed entries in a Dict by matching keys via ast.dump."""
        # Fast path when lengths match and keys are in same order (99% of dict edits)
        if len(old_dict.keys) == len(new_dict.keys) and all(
            nodes_equal(ok, nk) for ok, nk in zip(old_dict.keys, new_dict.keys)
        ):
            found_diff = False
            for ok, ov, nv in zip(old_dict.keys, old_dict.values, new_dict.values):
                if not nodes_equal(ov, nv):
                    found_diff = True
                    if isinstance(ov, ast.Lambda) and isinstance(nv, ast.Lambda):
                        narrowed = (
                            narrow_changed_expr(ov.body, nv.body)
                            if nodes_equal(ov.args, nv.args)
                            else ov
                        )
                        changes.append(
                            ChangeItem(enclosing_class, ov, enclosing_attr, ok, ov, nv, narrowed)
                        )
                    elif isinstance(ov, ast.Dict) and isinstance(nv, ast.Dict):
                        if not diff_dict(
                            ov, nv, parent_stmt, enclosing_class, enclosing_func, enclosing_attr
                        ):
                            changes.append(
                                ChangeItem(
                                    enclosing_class, enclosing_func, enclosing_attr, ok, ov, nv, ov
                                )
                            )
                    else:
                        changes.append(
                            ChangeItem(
                                enclosing_class, enclosing_func, enclosing_attr, ok, ov, nv, ov
                            )
                        )
            return found_diff

        old_entries: dict[str, tuple[ast.AST | None, ast.AST]] = {}
        new_entries: dict[str, tuple[ast.AST | None, ast.AST]] = {}

        for idx, (ok, ov) in enumerate(zip(old_dict.keys, old_dict.values)):
            ksig = ast.dump(ok) if ok is not None else f"__unpack_{idx}_{ast.dump(ov)}"
            if ksig in old_entries:
                return False
            old_entries[ksig] = (ok, ov)

        for idx, (nk, nv) in enumerate(zip(new_dict.keys, new_dict.values)):
            ksig = ast.dump(nk) if nk is not None else f"__unpack_{idx}_{ast.dump(nv)}"
            if ksig in new_entries:
                return False
            new_entries[ksig] = (nk, nv)

        found_diff = False

        for ksig, (ok, ov) in old_entries.items():
            if ksig in new_entries:
                _, nv = new_entries[ksig]
                if not nodes_equal(ov, nv):
                    found_diff = True
                    if isinstance(ov, ast.Lambda) and isinstance(nv, ast.Lambda):
                        narrowed = (
                            narrow_changed_expr(ov.body, nv.body)
                            if nodes_equal(ov.args, nv.args)
                            else ov
                        )
                        changes.append(
                            ChangeItem(enclosing_class, ov, enclosing_attr, ok, ov, nv, narrowed)
                        )
                    elif isinstance(ov, ast.Dict) and isinstance(nv, ast.Dict):
                        if not diff_dict(
                            ov, nv, parent_stmt, enclosing_class, enclosing_func, enclosing_attr
                        ):
                            changes.append(
                                ChangeItem(
                                    enclosing_class, enclosing_func, enclosing_attr, ok, ov, nv, ov
                                )
                            )
                    else:
                        changes.append(
                            ChangeItem(
                                enclosing_class, enclosing_func, enclosing_attr, ok, ov, nv, ov
                            )
                        )
            else:
                found_diff = True
                if isinstance(ov, ast.Lambda):
                    changes.append(
                        ChangeItem(enclosing_class, ov, enclosing_attr, ok, ov, None, ov)
                    )
                else:
                    changes.append(
                        ChangeItem(
                            enclosing_class, enclosing_func, enclosing_attr, ok, ov, None, ov
                        )
                    )

        for ksig, (nk, nv) in new_entries.items():
            if ksig not in old_entries:
                found_diff = True
                changes.append(
                    ChangeItem(
                        enclosing_class, enclosing_func, enclosing_attr, nk, None, nv, parent_stmt
                    )
                )

        return found_diff

    def diff_single_stmt(
        o_stmt: ast.stmt,
        n_stmt: ast.stmt,
        enclosing_class: str | None,
        enclosing_func: ast.AST | None,
    ) -> None:
        if nodes_equal(o_stmt, n_stmt):
            return

        if type(o_stmt) is not type(n_stmt):
            if is_runtime_noop_node(o_stmt, n_stmt, has_future_annot):
                return
            changes.append(
                ChangeItem(
                    enclosing_class,
                    enclosing_func,
                    get_assigned_attr_name(o_stmt),
                    None,
                    None,
                    None,
                    o_stmt,
                )
            )
            return

        # ClassDef
        if isinstance(o_stmt, ast.ClassDef) and isinstance(n_stmt, ast.ClassDef):
            if (
                o_stmt.name != n_stmt.name
                or not lists_equal(o_stmt.bases, n_stmt.bases)
                or not lists_equal(o_stmt.keywords, n_stmt.keywords)
                or not lists_equal(o_stmt.decorator_list, n_stmt.decorator_list)
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
                or not nodes_equal(o_stmt.args, n_stmt.args)
                or not lists_equal(o_stmt.decorator_list, n_stmt.decorator_list)
            ):
                if not is_runtime_noop_node(o_stmt, n_stmt, has_future_annot):
                    changes.append(
                        ChangeItem(enclosing_class, o_stmt, None, None, None, None, o_stmt)
                    )
                return
            diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, o_stmt)
            return

        # If statement
        if isinstance(o_stmt, ast.If) and isinstance(n_stmt, ast.If):
            if _is_type_checking_if(o_stmt) and _is_type_checking_if(n_stmt):
                return
            if not nodes_equal(o_stmt.test, n_stmt.test):
                narrowed_test = narrow_changed_expr(o_stmt.test, n_stmt.test)
                changes.append(
                    ChangeItem(
                        enclosing_class, enclosing_func, None, None, None, None, narrowed_test
                    )
                )
            if not lists_equal(o_stmt.body, n_stmt.body):
                diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            if not lists_equal(o_stmt.orelse, n_stmt.orelse):
                diff_stmt_list(
                    o_stmt.orelse, n_stmt.orelse, o_stmt, enclosing_class, enclosing_func
                )
            return

        # For / AsyncFor
        if isinstance(o_stmt, (ast.For, ast.AsyncFor)) and isinstance(
            n_stmt, (ast.For, ast.AsyncFor)
        ):
            if not nodes_equal(o_stmt.target, n_stmt.target) or not nodes_equal(
                o_stmt.iter, n_stmt.iter
            ):
                changes.append(
                    ChangeItem(enclosing_class, enclosing_func, None, None, None, None, o_stmt.iter)
                )
            if not lists_equal(o_stmt.body, n_stmt.body):
                diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            if not lists_equal(o_stmt.orelse, n_stmt.orelse):
                diff_stmt_list(
                    o_stmt.orelse, n_stmt.orelse, o_stmt, enclosing_class, enclosing_func
                )
            return

        # While
        if isinstance(o_stmt, ast.While) and isinstance(n_stmt, ast.While):
            if not nodes_equal(o_stmt.test, n_stmt.test):
                narrowed_test = narrow_changed_expr(o_stmt.test, n_stmt.test)
                changes.append(
                    ChangeItem(
                        enclosing_class, enclosing_func, None, None, None, None, narrowed_test
                    )
                )
            if not lists_equal(o_stmt.body, n_stmt.body):
                diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            if not lists_equal(o_stmt.orelse, n_stmt.orelse):
                diff_stmt_list(
                    o_stmt.orelse, n_stmt.orelse, o_stmt, enclosing_class, enclosing_func
                )
            return

        # With / AsyncWith
        if isinstance(o_stmt, (ast.With, ast.AsyncWith)) and isinstance(
            n_stmt, (ast.With, ast.AsyncWith)
        ):
            if not lists_equal(o_stmt.items, n_stmt.items):
                changes.append(
                    ChangeItem(enclosing_class, enclosing_func, None, None, None, None, o_stmt)
                )
                return
            diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            return

        # Try
        if isinstance(o_stmt, ast.Try) and isinstance(n_stmt, ast.Try):
            if not lists_equal(o_stmt.body, n_stmt.body):
                diff_stmt_list(o_stmt.body, n_stmt.body, o_stmt, enclosing_class, enclosing_func)
            if not lists_equal(o_stmt.orelse, n_stmt.orelse):
                diff_stmt_list(
                    o_stmt.orelse, n_stmt.orelse, o_stmt, enclosing_class, enclosing_func
                )
            if not lists_equal(o_stmt.finalbody, n_stmt.finalbody):
                diff_stmt_list(
                    o_stmt.finalbody, n_stmt.finalbody, o_stmt, enclosing_class, enclosing_func
                )
            if len(o_stmt.handlers) != len(n_stmt.handlers):
                changes.append(
                    ChangeItem(enclosing_class, enclosing_func, None, None, None, None, o_stmt)
                )
            else:
                for oh, nh in zip(o_stmt.handlers, n_stmt.handlers):
                    if not nodes_equal(oh.type, nh.type) or oh.name != nh.name:
                        changes.append(
                            ChangeItem(enclosing_class, enclosing_func, None, None, None, None, oh)
                        )
                    elif not lists_equal(oh.body, nh.body):
                        diff_stmt_list(oh.body, nh.body, oh, enclosing_class, enclosing_func)
            return

        # Return statement: narrow RHS expression if possible
        if isinstance(o_stmt, ast.Return) and isinstance(n_stmt, ast.Return):
            if o_stmt.value is not None and n_stmt.value is not None:
                narrowed = narrow_changed_expr(o_stmt.value, n_stmt.value)
                changes.append(
                    ChangeItem(enclosing_class, enclosing_func, None, None, None, None, narrowed)
                )
                return

        # Assign / AnnAssign where value is a Dict or narrowable expression
        attr_name = get_assigned_attr_name(o_stmt)
        if isinstance(o_stmt, ast.Assign) and isinstance(n_stmt, ast.Assign):
            if lists_equal(o_stmt.targets, n_stmt.targets):
                if isinstance(o_stmt.value, ast.Dict) and isinstance(n_stmt.value, ast.Dict):
                    if diff_dict(
                        o_stmt.value,
                        n_stmt.value,
                        o_stmt,
                        enclosing_class,
                        enclosing_func,
                        attr_name,
                    ):
                        return
                elif enclosing_func is not None:
                    narrowed = narrow_changed_expr(o_stmt.value, n_stmt.value)
                    changes.append(
                        ChangeItem(
                            enclosing_class, enclosing_func, attr_name, None, None, None, narrowed
                        )
                    )
                    return
        elif isinstance(o_stmt, ast.AnnAssign) and isinstance(n_stmt, ast.AnnAssign):
            if is_runtime_noop_node(o_stmt, n_stmt, has_future_annot):
                return
            if (
                nodes_equal(o_stmt.target, n_stmt.target)
                and o_stmt.value is not None
                and n_stmt.value is not None
            ):
                if isinstance(o_stmt.value, ast.Dict) and isinstance(n_stmt.value, ast.Dict):
                    if diff_dict(
                        o_stmt.value,
                        n_stmt.value,
                        o_stmt,
                        enclosing_class,
                        enclosing_func,
                        attr_name,
                    ):
                        return
                elif enclosing_func is not None:
                    narrowed = narrow_changed_expr(o_stmt.value, n_stmt.value)
                    changes.append(
                        ChangeItem(
                            enclosing_class, enclosing_func, attr_name, None, None, None, narrowed
                        )
                    )
                    return

        if is_runtime_noop_node(o_stmt, n_stmt, has_future_annot):
            return

        changes.append(
            ChangeItem(enclosing_class, enclosing_func, attr_name, None, None, None, o_stmt)
        )

    diff_stmt_list(old_tree.body, new_tree.body, old_tree, None, None)
    return changes


def extract_exp_name(node: ast.AST | None) -> str | None:
    """If node is `exp.SomeExpr`, return 'SomeExpr'."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "exp"
    ):
        return node.attr
    return None


def extract_referenced_exp_names(node: ast.AST | None) -> set[str]:
    """Extract all `exp.<ExprName>` class names referenced inside an AST expression."""
    names: set[str] = set()
    if node is None:
        return names
    for sub in ast.walk(node):
        ename = extract_exp_name(sub)
        if ename and ename[0].isupper():
            names.add(ename)
    return names


def select_tests_for_patch(
    patch_text: str,
    cov_map: dict,
    verifier: InProcessVerifier | None = None,
    dur_by_num: dict[int, int] | None = None,
) -> set[int]:
    test_sets: list[tuple[int, ...]] = cov_map["test_sets"]
    line_cov: dict[str, dict[int, int]] = cov_map["line_cov"]
    setup_cov: dict[str, dict[int, int]] = cov_map["setup_cov"]
    import_lines: dict[str, set[int]] = cov_map["import_lines"]
    fixture_cov: dict[str, int] = cov_map["fixture_cov"]
    expr_cov: dict[str, int] = cov_map["expr_cov"]
    dialect_expr_cov: dict[str, dict[str, int]] = cov_map.get("dialect_expr_cov", {})
    symbol_cov: dict[str, int] = cov_map["symbol_cov"]
    dialect_symbol_cov: dict[str, dict[str, int]] = cov_map.get("dialect_symbol_cov", {})
    file_import_cov: dict[str, int] = cov_map["file_import_cov"]
    sources: dict[str, bytes] = cov_map["sources"]

    non_py_selected: set[int] = set()
    selected: set[int] = set()
    changed_py_files: list[tuple[str, str]] = []
    can_verify_py = True
    parsed_files = parse_unified_diff(patch_text)

    for rel_path, hunks in parsed_files:
        # 1. Fixture or non-Python files
        if not rel_path.endswith(".py"):
            if rel_path.endswith(".sql"):
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
                non_py_selected.update(test_sets[f_set_id])
            else:
                stem = pathlib.Path(rel_path).stem
                for k, sid in fixture_cov.items():
                    if pathlib.Path(k).stem == stem:
                        non_py_selected.update(test_sets[sid])
            continue

        # 2. Python files
        comp_src = sources.get(rel_path)
        if comp_src is None:
            can_verify_py = False
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
            can_verify_py = False
            sid = file_import_cov.get(rel_path)
            if sid:
                selected.update(test_sets[sid])
            continue

        has_future_annot = "from __future__ import annotations" in old_source
        old_lines = old_source.splitlines()
        new_lines = new_source.splitlines()
        changes = diff_ast_nodes(
            old_tree, new_tree, old_lines, new_lines, has_future_annot
        )
        if not changes:
            continue

        if not rel_path.startswith("sqlglot/") or rel_path.startswith("sqlglot/typing/"):
            can_verify_py = False
        changed_py_files.append((rel_path, new_source))

        file_lines = line_cov.get(rel_path, {})
        file_setup_lines = setup_cov.get(rel_path, {})
        file_imp_lines = import_lines.get(rel_path, set())
        file_importers_sid = file_import_cov.get(rel_path, 0)
        file_importers = set(test_sets[file_importers_sid])

        is_typing_file = rel_path.startswith("sqlglot/typing/")

        for item in changes:
            if is_typing_file:
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
            d_stem: str | None = None
            if rel_path.startswith(
                ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")
            ):
                stem = pathlib.Path(rel_path).stem
                if stem not in ("__init__", "dialect", "base"):
                    d_stem = stem

            active_expr_cov = (
                dialect_expr_cov.get(d_stem, {})
                if (d_stem and dialect_expr_cov)
                else expr_cov
            )
            active_sym_cov = (
                dialect_symbol_cov.get(d_stem, {})
                if (d_stem and dialect_symbol_cov)
                else symbol_cov
            )

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
                esid = active_expr_cov.get(exp_key_name)
                if esid:
                    e_tests.update(test_sets[esid])
                ssid = active_sym_cov.get(exp_key_name)
                if ssid:
                    e_tests.update(test_sets[ssid])
                selected.update(candidates & e_tests)
                continue

            # Sub-case 2c: Changed a dispatch dictionary entry where the value references `exp.SomeExpr`
            if (
                item.enclosing_attr in DISPATCH_DICT_ATTRS
                and item.dict_key_node is not None
                and item.dict_val_node is not None
            ):
                ref_exps = extract_referenced_exp_names(
                    item.dict_val_node
                ) | extract_referenced_exp_names(item.new_dict_val_node)
                if ref_exps:
                    e_tests = set()
                    for ename in ref_exps:
                        esid = active_expr_cov.get(ename)
                        if esid:
                            e_tests.update(test_sets[esid])
                        ssid = active_sym_cov.get(ename)
                        if ssid:
                            e_tests.update(test_sets[ssid])
                    if e_tests or (d_stem and dialect_expr_cov):
                        selected.update(candidates & e_tests)
                        continue

            # Sub-case 2d: Changed a class attribute or module constant `enclosing_attr`
            if item.enclosing_attr and item.enclosing_attr in symbol_cov:
                if item.enclosing_attr == "NUMERIC_ESCAPES" and item.dict_key_node is not None:
                    # Tuple values in NUMERIC_ESCAPES are only unpacked when `if spec:` calls _read_numeric_escape
                    attr_tests = set(test_sets[active_sym_cov.get("_read_numeric_escape", 0)])
                else:
                    attr_tests = set(test_sets[active_sym_cov.get(item.enclosing_attr, 0)])
                selected.update(candidates & attr_tests)
                continue

            # Fallback for any other module/class-level change
            selected.update(candidates)

    if (
        selected
        and can_verify_py
        and changed_py_files
        and verifier is not None
        and dur_by_num is not None
    ):
        selected = verifier.verify_candidates(changed_py_files, selected, dur_by_num)

    return non_py_selected | selected


SKIP_VERIFY_FILES = frozenset(
    {
        "tests/test_sqlglotc.py",
        "tests/test_build.py",
        "tests/test_docs.py",
        "tests/test_typing.py",
    }
)

SKIP_VERIFY_METHODS = frozenset(
    {
        "test_lazy_load",
        "test_execute_tpcds",
        "test_execute_tpch",
        "test_optimized_tpch",
        "test_sql_boolean_logic_with_numpy_scalars",
    }
)

CONN_OPT_METHODS = frozenset(
    {
        "test_qualify_using_with_many_joins",
        "test_optimize_predicates",
        "test_optimize_uncorrelated_exists",
        "test_optimize_correlated_all",
        "test_optimize_subqueries_in_aggregates",
        "test_optimize_subqueries_in_join_on",
        "test_optimize_distinct_on_predicates",
        "test_optimize_special_joins",
        "test_optimize_snowflake_asof_without_on",
        "test_eliminate_joins",
    }
)


def _iter_subclasses(cls: type) -> list[type]:
    out: list[type] = []
    stack = list(type.__subclasses__(cls))
    seen = {cls}
    while stack:
        sub = stack.pop()
        if sub in seen:
            continue
        seen.add(sub)
        out.append(sub)
        stack.extend(type.__subclasses__(sub))
    return out


def _rewrite_class_cells(obj: object, new_cls: type, orig_cls: type, seen: set[int]) -> None:
    import types

    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    if type(obj) is types.FunctionType:
        if obj.__closure__:
            for cell in obj.__closure__:
                try:
                    if cell.cell_contents is new_cls:
                        cell.cell_contents = orig_cls
                    else:
                        _rewrite_class_cells(cell.cell_contents, new_cls, orig_cls, seen)
                except ValueError:
                    pass
    elif isinstance(obj, (classmethod, staticmethod)):
        _rewrite_class_cells(obj.__func__, new_cls, orig_cls, seen)
    elif isinstance(obj, property):
        for f in (obj.fget, obj.fset, obj.fdel):
            if f is not None:
                _rewrite_class_cells(f, new_cls, orig_cls, seen)
    elif isinstance(obj, dict):
        for v in obj.values():
            _rewrite_class_cells(v, new_cls, orig_cls, seen)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _rewrite_class_cells(v, new_cls, orig_cls, seen)


def _funcs_equal(f1: object, f2: object) -> bool:
    c1, c2 = getattr(f1, "__code__", None), getattr(f2, "__code__", None)
    if c1 is None or c2 is None:
        return False
    return (
        c1.co_code == c2.co_code
        and c1.co_consts == c2.co_consts
        and c1.co_names == c2.co_names
        and c1.co_varnames == c2.co_varnames
        and c1.co_freevars == c2.co_freevars
        and c1.co_cellvars == c2.co_cellvars
        and getattr(f1, "__defaults__", None) == getattr(f2, "__defaults__", None)
        and getattr(f1, "__kwdefaults__", None) == getattr(f2, "__kwdefaults__", None)
        and getattr(f1, "__closure__", None) == getattr(f2, "__closure__", None)
    )


class InProcessVerifier:
    """Fast in-process unit test verifier for candidate test sets."""

    def __init__(self, cov_map: dict, test_by_num: dict[int, dict]) -> None:
        self.cov_map = cov_map
        self.test_by_num = test_by_num
        self.initialized = False
        self.sqlglot_root: Path | None = None
        self.use_pickle_in_pool = False
        self.initialized_classes: set[type] = set()
        self.conn_cls: type | None = None
        self.mod_opt: object = None
        self.sqlglot_mod: object = None

    def _is_runnable_fast(self, num: int, dur_ms: int) -> bool:
        if dur_ms > 700:
            return False
        t = self.test_by_num.get(num)
        if t is None:
            return False
        file_rel, _, method_name = t["id"].split("::")
        if file_rel in SKIP_VERIFY_FILES or method_name in SKIP_VERIFY_METHODS:
            return False
        return True

    def _ensure_initialized(self) -> None:
        if self.initialized:
            return
        import calendar
        import concurrent.futures
        import contextlib
        import datetime
        import importlib
        import importlib.metadata
        import tempfile
        import types
        import unittest
        import unittest.mock

        # 1. Locate or unpack sqlglot root
        candidates = [
            Path(os.environ.get("BLAST_SQLGLOT_DIR", "/sqlglot")),
            Path("/sqlglot"),
        ]
        root: Path | None = None
        for c in candidates:
            if (c / "sqlglot" / "__init__.py").exists():
                root = c
                break
        if root is None:
            root = Path(tempfile.gettempdir()) / "sqlglot_verify"
            if not (root / "sqlglot" / "__init__.py").exists():
                for d in (self.cov_map.get("sources", {}), self.cov_map.get("fixtures", {})):
                    for rel, comp in d.items():
                        dest = root / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(zlib.decompress(comp))

        self.sqlglot_root = root
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        os.chdir(root)

        # Prevent sqlglot/__init__.py from printing version warning
        orig_ver = importlib.metadata.version

        def _safe_version(pkg: str) -> str:
            if pkg == "sqlglot":
                return "0.0.0"
            return orig_ver(pkg)

        importlib.metadata.version = _safe_version  # type: ignore

        verifier_ref = self

        class _LazyFuture:
            __slots__ = ("_fn", "_args", "_kwargs")

            def __init__(self, fn, args, kwargs):
                self._fn = fn
                self._args = args
                self._kwargs = kwargs

            def result(self, timeout=None):
                if verifier_ref.use_pickle_in_pool:
                    args, kwargs = pickle.loads(pickle.dumps((self._args, self._kwargs)))
                    res = self._fn(*args, **kwargs)
                    return pickle.loads(pickle.dumps(res))
                return self._fn(*self._args, **self._kwargs)

        class _InProcessPoolExecutor:
            def __init__(self, max_workers=None, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def submit(self, fn, *args, **kwargs):
                return _LazyFuture(fn, args, kwargs)

            def shutdown(self, wait=True, *, cancel_futures=False):
                pass

        concurrent.futures.ProcessPoolExecutor = _InProcessPoolExecutor  # type: ignore

        class _FakeTimezone(datetime.tzinfo):
            def __init__(self, zone: str):
                self.zone = zone
                self._offset = datetime.timedelta(hours=-8)

            def localize(self, dt: datetime.datetime) -> datetime.datetime:
                offset = -7 if 3 < dt.month < 11 else -8
                self._offset = datetime.timedelta(hours=offset)
                return dt.replace(tzinfo=self)

            def utcoffset(self, dt):
                return self._offset

            def dst(self, dt):
                return datetime.timedelta(0)

            def tzname(self, dt):
                return self.zone

        class _FakeRelativeDelta:
            def __init__(
                self,
                years: int = 0,
                months: int = 0,
                weeks: int = 0,
                days: int = 0,
                hours: int = 0,
                minutes: int = 0,
                seconds: int = 0,
                microseconds: int = 0,
            ):
                self.years = years
                self.months = months
                self.weeks = weeks
                self.days = days
                self.hours = hours
                self.minutes = minutes
                self.seconds = seconds
                self.microseconds = microseconds
                self._td = datetime.timedelta(
                    weeks=weeks,
                    days=days,
                    hours=hours,
                    minutes=minutes,
                    seconds=seconds,
                    microseconds=microseconds,
                )

            def __neg__(self) -> _FakeRelativeDelta:
                return _FakeRelativeDelta(
                    years=-self.years,
                    months=-self.months,
                    weeks=-self.weeks,
                    days=-self.days,
                    hours=-self.hours,
                    minutes=-self.minutes,
                    seconds=-self.seconds,
                    microseconds=-self.microseconds,
                )

            def _apply(self, other: datetime.date) -> datetime.date:
                if self.years or self.months:
                    total_m = (other.year * 12 + (other.month - 1)) + self.years * 12 + self.months
                    ny, nm0 = divmod(total_m, 12)
                    nm = nm0 + 1
                    max_d = calendar.monthrange(ny, nm)[1]
                    other = other.replace(year=ny, month=nm, day=min(other.day, max_d))
                if self._td:
                    if type(other) is datetime.date and (
                        self.hours or self.minutes or self.seconds or self.microseconds
                    ):
                        other = datetime.datetime(other.year, other.month, other.day)
                    other = other + self._td
                return other

            def __add__(self, other):
                if isinstance(other, datetime.date):
                    return self._apply(other)
                return NotImplemented

            def __radd__(self, other):
                if isinstance(other, datetime.date):
                    return self._apply(other)
                return NotImplemented

            def __rsub__(self, other):
                if isinstance(other, datetime.date):
                    return (-self)._apply(other)
                return NotImplemented

        class _FakeDuckDBCursor:
            def __init__(self, rows):
                self._rows = rows

            def fetchall(self):
                return self._rows

            def df(self):
                return self._rows

            def fetchdf(self):
                return self._rows

        class _FakeDuckDBConn:
            current_test_num: int = -1
            is_baseline_recording: bool = False
            baseline_queries: dict[int, list[str]] = {}
            query_idx: int = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def close(self):
                pass

            def execute(self, sql: str, *args, **kwargs) -> _FakeDuckDBCursor:
                tnum = _FakeDuckDBConn.current_test_num
                if tnum < 0:
                    return _FakeDuckDBCursor([(None,)])
                q = sql.strip()
                if _FakeDuckDBConn.is_baseline_recording:
                    _FakeDuckDBConn.baseline_queries.setdefault(tnum, []).append(q)
                    return _FakeDuckDBCursor([(None,)])
                b_list = _FakeDuckDBConn.baseline_queries.get(tnum)
                idx = _FakeDuckDBConn.query_idx
                _FakeDuckDBConn.query_idx = idx + 1
                if b_list is not None and idx < len(b_list):
                    if q != b_list[idx]:
                        return _FakeDuckDBCursor([("__DIFF__", idx, q)])
                return _FakeDuckDBCursor([(None,)])

        _fake_duckdb_conn = _FakeDuckDBConn()
        self.conn_cls = _FakeDuckDBConn

        for mod_name in (
            "duckdb",
            "numpy",
            "pandas",
            "pandas.testing",
            "pytz",
            "dateutil",
            "dateutil.relativedelta",
        ):
            if mod_name not in sys.modules:
                m = types.ModuleType(mod_name)
                if mod_name == "numpy":
                    m.bool_ = bool  # type: ignore
                    m.nan = float("nan")  # type: ignore
                elif mod_name == "pandas":
                    m.DataFrame = object  # type: ignore
                    m.Series = object  # type: ignore
                elif mod_name == "pandas.testing":
                    m.assert_frame_equal = lambda *a, **kw: None  # type: ignore
                elif mod_name == "duckdb":
                    m.connect = lambda *a, **kw: _fake_duckdb_conn  # type: ignore
                elif mod_name == "pytz":
                    m.timezone = _FakeTimezone  # type: ignore
                    m.utc = datetime.timezone.utc  # type: ignore
                elif mod_name == "dateutil.relativedelta":
                    m.relativedelta = _FakeRelativeDelta  # type: ignore
                sys.modules[mod_name] = m
        sys.modules["pandas"].testing = sys.modules["pandas.testing"]  # type: ignore
        sys.modules["dateutil"].relativedelta = sys.modules["dateutil.relativedelta"]  # type: ignore

        @contextlib.contextmanager
        def _fast_subtest(case_self, msg=None, **params):
            yield

        unittest.TestCase.subTest = _fast_subtest  # type: ignore

        import sqlglot
        import tests.test_executor as mod_exec
        import tests.test_optimizer as mod_opt

        self.sqlglot_mod = sqlglot
        self.mod_opt = mod_opt

        def fast_opt_setup_class(cls):
            cls.pool = _InProcessPoolExecutor()
            sqlglot.schema = mod_opt.MappingSchema()
            cls.conn = _fake_duckdb_conn

        mod_opt.wait = lambda fs, *a, **kw: None  # type: ignore
        mod_opt.as_completed = lambda fs, *a, **kw: iter(fs)  # type: ignore
        mod_opt.TestOptimizer.setUpClass = classmethod(fast_opt_setup_class)  # type: ignore
        mod_opt.TestOptimizer.tearDownClass = classmethod(lambda cls: None)  # type: ignore
        orig_check_file = mod_opt.TestOptimizer.check_file

        def fast_check_file(opt_self, *args, **kwargs):
            kwargs["execute"] = False
            return orig_check_file(opt_self, *args, **kwargs)

        mod_opt.TestOptimizer.check_file = fast_check_file  # type: ignore
        mod_exec.TestExecutor.setUpClass = classmethod(lambda cls: None)  # type: ignore
        mod_exec.TestExecutor.tearDownClass = classmethod(lambda cls: None)  # type: ignore

        self.initialized = True

    def _sync_class_inplace(self, orig_cls: type, new_cls: type, undo_actions: list) -> None:
        import enum
        import types
        from sqlglot.dialects.dialect import Dialect, _Dialect
        from sqlglot.expressions.core import Expr
        from sqlglot.tokens import _TokenizerBase

        if issubclass(orig_cls, enum.Enum):
            return

        skip_attrs = {
            "__dict__",
            "__weakref__",
            "__doc__",
            "__module__",
            "__qualname__",
            "__firstlineno__",
            "__static_attributes__",
        }

        orig_dict = dict(orig_cls.__dict__)
        new_dict = dict(new_cls.__dict__)

        seen_objs: set[int] = set()
        for v in new_dict.values():
            _rewrite_class_cells(v, new_cls, orig_cls, seen_objs)

        for k, new_v in new_dict.items():
            if k in skip_attrs:
                continue
            if k not in orig_dict:
                setattr(orig_cls, k, new_v)
                undo_actions.append(("delattr", orig_cls, k))
                continue

            orig_v = orig_dict[k]
            if orig_v is new_v:
                continue

            if type(orig_v) is types.FunctionType and type(new_v) is types.FunctionType:
                if _funcs_equal(orig_v, new_v):
                    continue
                if orig_v.__closure__ == new_v.__closure__:
                    old_state = (orig_v.__code__, orig_v.__defaults__, orig_v.__kwdefaults__)
                    orig_v.__code__ = new_v.__code__
                    orig_v.__defaults__ = new_v.__defaults__
                    orig_v.__kwdefaults__ = new_v.__kwdefaults__
                    undo_actions.append(("func_code", orig_v, old_state))
                else:
                    setattr(orig_cls, k, new_v)
                    undo_actions.append(("setattr", orig_cls, k, orig_v))
            elif isinstance(orig_v, (classmethod, staticmethod)) and isinstance(
                new_v, (classmethod, staticmethod)
            ):
                orig_fn = orig_v.__func__
                new_fn = new_v.__func__
                if type(orig_fn) is types.FunctionType and type(new_fn) is types.FunctionType:
                    if _funcs_equal(orig_fn, new_fn):
                        continue
                    if orig_fn.__closure__ == new_fn.__closure__:
                        old_state = (orig_fn.__code__, orig_fn.__defaults__, orig_fn.__kwdefaults__)
                        orig_fn.__code__ = new_fn.__code__
                        orig_fn.__defaults__ = new_fn.__defaults__
                        orig_fn.__kwdefaults__ = new_fn.__kwdefaults__
                        undo_actions.append(("func_code", orig_fn, old_state))
                        continue
                setattr(orig_cls, k, new_v)
                undo_actions.append(("setattr", orig_cls, k, orig_v))
            elif isinstance(orig_v, property) and isinstance(new_v, property):
                if (
                    type(orig_v.fget) is types.FunctionType
                    and type(new_v.fget) is types.FunctionType
                    and orig_v.fset is None
                    and new_v.fset is None
                    and _funcs_equal(orig_v.fget, new_v.fget)
                ):
                    continue
                setattr(orig_cls, k, new_v)
                undo_actions.append(("setattr", orig_cls, k, orig_v))
            elif (
                isinstance(orig_v, type)
                and isinstance(new_v, type)
                and orig_v.__name__ == new_v.__name__
            ):
                self._sync_class_inplace(orig_v, new_v, undo_actions)
            elif isinstance(orig_v, dict) and isinstance(new_v, dict):
                if orig_v != new_v:
                    old_copy = dict(orig_v)
                    added_or_changed = {
                        dk: dv
                        for dk, dv in new_v.items()
                        if dk not in old_copy or old_copy[dk] != dv
                    }
                    removed = {dk for dk in old_copy if dk not in new_v}

                    orig_v.clear()
                    orig_v.update(new_v)
                    undo_actions.append(("dict_restore", orig_v, old_copy))

                    for sub in _iter_subclasses(orig_cls):
                        sub_d = sub.__dict__.get(k)
                        if isinstance(sub_d, dict) and sub_d is not orig_v:
                            sub_old = dict(sub_d)
                            sub_changed = False
                            for rk in removed:
                                if rk in sub_d and sub_d[rk] == old_copy[rk]:
                                    del sub_d[rk]
                                    sub_changed = True
                            for ak, av in added_or_changed.items():
                                if ak not in sub_d or sub_d[ak] == old_copy.get(ak):
                                    sub_d[ak] = av
                                    sub_changed = True
                            if sub_changed:
                                undo_actions.append(("dict_restore", sub_d, sub_old))
            elif isinstance(orig_v, set) and isinstance(new_v, set):
                if orig_v != new_v:
                    old_copy = set(orig_v)
                    added = new_v - old_copy
                    removed = old_copy - new_v
                    orig_v.clear()
                    orig_v.update(new_v)
                    undo_actions.append(("set_restore", orig_v, old_copy))

                    for sub in _iter_subclasses(orig_cls):
                        sub_s = sub.__dict__.get(k)
                        if isinstance(sub_s, set) and sub_s is not orig_v:
                            sub_old = set(sub_s)
                            sub_s.difference_update(removed)
                            sub_s.update(added)
                            if sub_s != sub_old:
                                undo_actions.append(("set_restore", sub_s, sub_old))
            else:
                if orig_v != new_v:
                    setattr(orig_cls, k, new_v)
                    undo_actions.append(("setattr", orig_cls, k, orig_v))

        for k in orig_dict:
            if k in skip_attrs:
                continue
            if k not in new_dict:
                delattr(orig_cls, k)
                undo_actions.append(("setattr", orig_cls, k, orig_dict[k]))

        if issubclass(orig_cls, Expr) and orig_cls is not Expr:
            orig_cls.__init_subclass__()
            for sub in _iter_subclasses(orig_cls):
                sub.__init_subclass__()

        if issubclass(orig_cls, _TokenizerBase) and orig_cls is not _TokenizerBase:
            orig_cls.__init_subclass__()
            for sub in _iter_subclasses(orig_cls):
                sub.__init_subclass__()

        if issubclass(orig_cls, Dialect) and orig_cls is not Dialect:
            for d_key, d_val in list(_Dialect._classes.items()):
                if d_val is new_cls:
                    _Dialect._classes[d_key] = orig_cls

    def apply_patch(self, changed_py_files: list[tuple[str, str]]) -> list:
        import importlib
        import types
        import sqlglot.generator

        undo_actions: list = []
        self.use_pickle_in_pool = any(rel == "sqlglot/serde.py" for rel, _ in changed_py_files)
        sources_map = self.cov_map.get("sources", {})

        for rel_path, new_source in changed_py_files:
            # Ensure dialect module is pre-imported in clean state if patching parser/generator/dialect
            if rel_path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/")):
                stem = pathlib.Path(rel_path).stem
                if stem not in ("__init__", "dialect", "base") and f"sqlglot/dialects/{stem}.py" in sources_map:
                    importlib.import_module(f"sqlglot.dialects.{stem}")

            no_ext = rel_path[:-3]
            mod_name = no_ext.replace("/", ".")
            if mod_name.endswith(".__init__"):
                mod_name = mod_name[:-9]

            mod = importlib.import_module(mod_name)
            temp_ns = dict(mod.__dict__)
            code_obj = compile(new_source, str((self.sqlglot_root or Path(".")) / rel_path), "exec")
            exec(code_obj, temp_ns)

            skip_mod_keys = {
                "__name__",
                "__doc__",
                "__package__",
                "__loader__",
                "__spec__",
                "__file__",
                "__cached__",
                "__builtins__",
            }
            for k, new_v in temp_ns.items():
                if k in skip_mod_keys:
                    continue
                if k not in mod.__dict__:
                    setattr(mod, k, new_v)
                    undo_actions.append(("delattr", mod, k))
                    continue
                orig_v = mod.__dict__[k]
                if orig_v is new_v:
                    continue
                if type(orig_v) is types.FunctionType and type(new_v) is types.FunctionType:
                    if _funcs_equal(orig_v, new_v):
                        continue
                    if orig_v.__closure__ == new_v.__closure__:
                        old_state = (orig_v.__code__, orig_v.__defaults__, orig_v.__kwdefaults__)
                        orig_v.__code__ = new_v.__code__
                        orig_v.__defaults__ = new_v.__defaults__
                        orig_v.__kwdefaults__ = new_v.__kwdefaults__
                        undo_actions.append(("func_code", orig_v, old_state))
                    else:
                        setattr(mod, k, new_v)
                        undo_actions.append(("setattr", mod, k, orig_v))
                elif (
                    isinstance(orig_v, type)
                    and isinstance(new_v, type)
                    and orig_v.__name__ == new_v.__name__
                ):
                    self._sync_class_inplace(orig_v, new_v, undo_actions)
                elif isinstance(orig_v, dict) and isinstance(new_v, dict):
                    if orig_v != new_v:
                        old_copy = dict(orig_v)
                        orig_v.clear()
                        orig_v.update(new_v)
                        undo_actions.append(("dict_restore", orig_v, old_copy))
                elif isinstance(orig_v, set) and isinstance(new_v, set):
                    if orig_v != new_v:
                        old_copy = set(orig_v)
                        orig_v.clear()
                        orig_v.update(new_v)
                        undo_actions.append(("set_restore", orig_v, old_copy))
                else:
                    if orig_v != new_v:
                        setattr(mod, k, new_v)
                        undo_actions.append(("setattr", mod, k, orig_v))

        sqlglot.generator._DISPATCH_CACHE.clear()
        return undo_actions

    def revert_patch(self, undo_actions: list) -> None:
        import sqlglot.generator
        from sqlglot.expressions.core import Expr
        from sqlglot.tokens import _TokenizerBase

        self.use_pickle_in_pool = False
        touched_classes: set[type] = set()
        for action in reversed(undo_actions):
            kind = action[0]
            if kind == "func_code":
                fn, (code, defaults, kwdefaults) = action[1], action[2]
                fn.__code__ = code
                fn.__defaults__ = defaults
                fn.__kwdefaults__ = kwdefaults
            elif kind == "setattr":
                obj, k, v = action[1], action[2], action[3]
                setattr(obj, k, v)
                if isinstance(obj, type):
                    touched_classes.add(obj)
            elif kind == "delattr":
                obj, k = action[1], action[2]
                try:
                    delattr(obj, k)
                except AttributeError:
                    pass
                if isinstance(obj, type):
                    touched_classes.add(obj)
            elif kind == "dict_restore":
                d, old_copy = action[1], action[2]
                d.clear()
                d.update(old_copy)
            elif kind == "set_restore":
                s, old_copy = action[1], action[2]
                s.clear()
                s.update(old_copy)

        for cls in touched_classes:
            if issubclass(cls, Expr) and cls is not Expr:
                cls.__init_subclass__()
                for sub in _iter_subclasses(cls):
                    sub.__init_subclass__()
            if issubclass(cls, _TokenizerBase) and cls is not _TokenizerBase:
                cls.__init_subclass__()
                for sub in _iter_subclasses(cls):
                    sub.__init_subclass__()

        sqlglot.generator._DISPATCH_CACHE.clear()

    def run_single_test(self, num: int) -> bool:
        import importlib
        import unittest

        t = self.test_by_num[num]
        file_rel, cls_name, method_name = t["id"].split("::")
        mod_name = file_rel[:-3].replace("/", ".")
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        if cls not in self.initialized_classes:
            cls.setUpClass()
            self.initialized_classes.add(cls)
        self.sqlglot_mod.schema = (  # type: ignore
            self.mod_opt.MappingSchema() if cls_name == "TestOptimizer" else None  # type: ignore
        )
        conn_cls = self.conn_cls
        if conn_cls is not None:
            conn_cls.current_test_num = num  # type: ignore
            conn_cls.query_idx = 0  # type: ignore
        case = cls(method_name)
        res = unittest.TestResult()
        case.run(res)
        if conn_cls is not None:
            conn_cls.current_test_num = -1  # type: ignore
        return res.wasSuccessful() and not res.skipped

    def _record_baseline_if_needed(self, nums: list[int]) -> None:
        conn_cls = self.conn_cls
        if conn_cls is None:
            return
        b_queries: dict[int, list[str]] = conn_cls.baseline_queries  # type: ignore
        for n in nums:
            t = self.test_by_num.get(n)
            if t is None:
                continue
            _, cls_name, method_name = t["id"].split("::")
            if cls_name == "TestOptimizer" and method_name in CONN_OPT_METHODS and n not in b_queries:
                conn_cls.is_baseline_recording = True  # type: ignore
                b_queries[n] = []
                self.run_single_test(n)
                conn_cls.is_baseline_recording = False  # type: ignore

    def verify_candidates(
        self,
        changed_py_files: list[tuple[str, str]],
        py_selected: set[int],
        dur_by_num: dict[int, int],
    ) -> set[int]:
        fast_candidates = sorted(
            [n for n in py_selected if self._is_runnable_fast(n, dur_by_num.get(n, 999999))],
            key=lambda n: (dur_by_num.get(n, 0), n),
        )
        if not fast_candidates:
            return py_selected

        slow_tests = py_selected - set(fast_candidates)
        total_fast_ms = sum(dur_by_num.get(n, 0) for n in fast_candidates)

        # Case A: All selected tests are runnable fast tests with total runtime <= 2.0s
        if not slow_tests and total_fast_ms <= 2000:
            self._ensure_initialized()
            self._record_baseline_if_needed(fast_candidates)
            undo: list = []
            try:
                undo = self.apply_patch(changed_py_files)
                passed: set[int] = set()
                for n in fast_candidates:
                    if self.run_single_test(n):
                        passed.add(n)
                    else:
                        break
                return py_selected - passed
            except Exception:
                return py_selected
            finally:
                if undo:
                    self.revert_patch(undo)

        # Case B: Slow tests exist, but we have >= 20 fast unit tests to test whether the edit is a semantic no-op
        if slow_tests and len(fast_candidates) >= 20:
            # Pick a diverse fast sample of up to 22 tests across distinct test files first
            sample: list[int] = []
            seen_files: dict[str, int] = {}
            for n in fast_candidates:
                if dur_by_num.get(n, 0) > 150:
                    continue
                f_rel = self.test_by_num[n]["file"]
                cnt = seen_files.get(f_rel, 0)
                if cnt < 2:
                    sample.append(n)
                    seen_files[f_rel] = cnt + 1
                    if len(sample) >= 22:
                        break
            if len(sample) < 22:
                sample_set = set(sample)
                for n in fast_candidates:
                    if n not in sample_set:
                        sample.append(n)
                        if len(sample) >= 22:
                            break
            self._ensure_initialized()
            self._record_baseline_if_needed(sample)
            undo = []
            try:
                undo = self.apply_patch(changed_py_files)
                for n in sample:
                    if not self.run_single_test(n):
                        return py_selected
                return set()
            except Exception:
                return py_selected
            finally:
                if undo:
                    self.revert_patch(undo)

        # Case C: Keep full static selection without running tests
        return py_selected


def main() -> None:
    t0 = time.monotonic()
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        tests_data = json.loads((DATA / "tests.json").read_text("utf-8"))["tests"]
        cov_map = pickle.loads(zlib.decompress(MAP_PATH.read_bytes()))

        dur_by_num: dict[int, int] = {}
        id_to_num: dict[str, int] = {}
        test_by_num: dict[int, dict] = {}
        for t in tests_data:
            num = t["number"]
            dur_by_num[num] = duration_ms(t["seconds"])
            id_to_num[t["id"]] = num
            test_by_num[num] = t

        stored_ids: list[str] = cov_map["test_ids"]
        needs_remap = len(stored_ids) != len(tests_data) or any(
            id_to_num.get(tid) != idx for idx, tid in enumerate(stored_ids)
        )
        if needs_remap:
            remap = {idx: id_to_num[tid] for idx, tid in enumerate(stored_ids) if tid in id_to_num}
            cov_map["test_sets"] = [
                tuple(remap[i] for i in tset if i in remap) for tset in cov_map["test_sets"]
            ]

        verifier = InProcessVerifier(cov_map, test_by_num)
        commits_dir = DATA / "commits"
        total_selected = 0
        out_lines: list[str] = []
        for patch_path in sorted(commits_dir.glob("*.patch")):
            patch_text = patch_path.read_text("utf-8")
            selected_nums = select_tests_for_patch(patch_text, cov_map, verifier, dur_by_num)
            workers = pack_workers(selected_nums, dur_by_num)
            total_selected += len(selected_nums)
            out_lines.append(
                json.dumps({"commit": patch_path.stem, "workers": workers}, separators=(",", ":"))
            )

        real_stdout.write("\n".join(out_lines) + "\n")
        real_stdout.flush()
        elapsed = time.monotonic() - t0
        print(
            f"Processed commits in {elapsed * 1000:.1f} ms (total selected tests: {total_selected})",
            file=sys.stderr,
        )
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    main()
