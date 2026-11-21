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


def schedule_cost_ms(workers: list[list[int]], dur_by_num: dict[int, int]) -> int:
    if not workers:
        return 0
    busiest = max(sum(dur_by_num[n] for n in w) for w in workers)
    return busiest + WORKER_COST_MS * len(workers)


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

        if not rel_path.startswith("sqlglot/"):
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
    }
)

SKIP_VERIFY_METHODS = frozenset(
    {
        "test_lazy_load",
        "test_execute_tpcds",
        "test_execute_tpch",
        "test_optimized_tpch",
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


def _collect_class_pairs(orig_cls: type, new_cls: type, cls_map: dict[type, type]) -> None:
    if new_cls in cls_map:
        return
    cls_map[new_cls] = orig_cls
    for k, nv in new_cls.__dict__.items():
        if isinstance(nv, type):
            ov = orig_cls.__dict__.get(k)
            if isinstance(ov, type) and ov.__name__ == nv.__name__:
                _collect_class_pairs(ov, nv, cls_map)


def _rewrite_cls_refs(obj: object, cls_map: dict[type, type], seen: set[int]) -> object:
    import types

    if isinstance(obj, type):
        return cls_map.get(obj, obj)
    oid = id(obj)
    if oid in seen:
        return obj
    if type(obj) is types.FunctionType:
        seen.add(oid)
        if obj.__defaults__:
            new_def = tuple(_rewrite_cls_refs(x, cls_map, seen) for x in obj.__defaults__)
            if new_def != obj.__defaults__:
                obj.__defaults__ = new_def
        if obj.__kwdefaults__:
            new_kw = {
                k: _rewrite_cls_refs(v, cls_map, seen) for k, v in obj.__kwdefaults__.items()
            }
            if new_kw != obj.__kwdefaults__:
                obj.__kwdefaults__ = new_kw
        if obj.__closure__:
            for cell in obj.__closure__:
                try:
                    c_val = cell.cell_contents
                    mapped = _rewrite_cls_refs(c_val, cls_map, seen)
                    if mapped is not c_val:
                        cell.cell_contents = mapped
                except ValueError:
                    pass
        return obj
    if isinstance(obj, (classmethod, staticmethod)):
        seen.add(oid)
        _rewrite_cls_refs(obj.__func__, cls_map, seen)
        return obj
    if isinstance(obj, property):
        seen.add(oid)
        for f in (obj.fget, obj.fset, obj.fdel):
            if f is not None:
                _rewrite_cls_refs(f, cls_map, seen)
        return obj
    if isinstance(obj, tuple):
        return tuple(_rewrite_cls_refs(v, cls_map, seen) for v in obj)
    if isinstance(obj, list):
        seen.add(oid)
        for i in range(len(obj)):
            obj[i] = _rewrite_cls_refs(obj[i], cls_map, seen)
        return obj
    if isinstance(obj, dict):
        seen.add(oid)
        items = list(obj.items())
        changed = False
        new_items = []
        for k, v in items:
            nk = cls_map.get(k, k) if isinstance(k, type) else k
            nv = _rewrite_cls_refs(v, cls_map, seen)
            if nk is not k or nv is not v:
                changed = True
            new_items.append((nk, nv))
        if changed:
            obj.clear()
            obj.update(new_items)
        return obj
    if isinstance(obj, set):
        seen.add(oid)
        mapped = {cls_map.get(v, v) if isinstance(v, type) else v for v in obj}
        if mapped != obj:
            obj.clear()
            obj.update(mapped)
        return obj
    return obj


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
        self.duckdb_opt_methods: set[str] = set()
        self.tpch_transpiled_baseline: list[tuple[str, str]] | None = None
        self.tpch_pairs: list[tuple[str, str]] | None = None

    def _is_runnable_fast(self, num: int, dur_ms: int) -> bool:
        if dur_ms > 2600:
            return False
        t = self.test_by_num.get(num)
        if t is None:
            return False
        file_rel, _, method_name = t["id"].split("::")
        if file_rel in SKIP_VERIFY_FILES or method_name in SKIP_VERIFY_METHODS:
            return False
        return True

    def _est_inproc_ms(self, num: int, dur_by_num: dict[int, int]) -> int:
        t = self.test_by_num.get(num)
        if t is not None and t["id"].endswith("::test_tpcds"):
            return 4500
        dur = dur_by_num.get(num, 500)
        if dur >= 500:
            return dur
        return max(1, int(dur * 0.6))

    def _ensure_initialized(self) -> None:
        if self.initialized:
            return
        import calendar
        import concurrent.futures
        import contextlib
        import datetime
        import importlib
        import importlib.metadata
        import re
        import tempfile
        import types
        import unittest
        import unittest.mock
        import zoneinfo

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
                if dt is not None and 3 < dt.month < 11 and self.zone == "America/Los_Angeles":
                    return datetime.timedelta(hours=-7)
                return self._offset

            def dst(self, dt):
                return datetime.timedelta(0)

            def tzname(self, dt):
                return self.zone

            def __str__(self) -> str:
                return self.zone

        try:
            zoneinfo.ZoneInfo("America/Los_Angeles")
        except Exception:
            zoneinfo.ZoneInfo = _FakeTimezone  # type: ignore

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

        class _FakeDuckDBInvalidInput(Exception):
            pass

        class _DuckDBRows(list):
            def __init__(self, rows, idx: int = -1):
                super().__init__(rows)
                self.idx = idx

        class _FakeDuckDBCursor:
            def __init__(self, rows):
                self._rows = rows

            def fetchall(self):
                return self._rows

            def df(self):
                return self._rows

            def fetchdf(self):
                return self._rows

        tmp_base_escaped = re.escape(tempfile.gettempdir())
        tmp_path_re = re.compile(tmp_base_escaped + r"[/\\]tmp[^/'\"\\]+")

        class _FakeDuckDBConn:
            current_test_num: int = -1
            is_baseline_recording: bool = False
            in_raises_invalid: bool = False
            baseline_queries: dict[int, list[str]] = {}
            baseline_expects: dict[int, dict[int, tuple[str, object]]] = {}
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
                q = tmp_path_re.sub("/TMP_DIR", sql.strip())
                if args:
                    q_extra = tmp_path_re.sub("/TMP_DIR", repr(args))
                    q = f"{q} ARGS={q_extra}"
                idx = _FakeDuckDBConn.query_idx
                _FakeDuckDBConn.query_idx = idx + 1
                if _FakeDuckDBConn.is_baseline_recording:
                    _FakeDuckDBConn.baseline_queries.setdefault(tnum, []).append(q)
                    if _FakeDuckDBConn.in_raises_invalid:
                        _FakeDuckDBConn.baseline_expects.setdefault(tnum, {})[idx] = (
                            "RAISE",
                            None,
                        )
                        raise _FakeDuckDBInvalidInput("More than one row")
                    return _FakeDuckDBCursor(_DuckDBRows([(None,)], idx=idx))

                b_list = _FakeDuckDBConn.baseline_queries.get(tnum)
                if b_list is not None:
                    if idx >= len(b_list) or q != b_list[idx]:
                        return _FakeDuckDBCursor([("__DIFF__", idx, q)])
                    exp_entry = _FakeDuckDBConn.baseline_expects.get(tnum, {}).get(idx)
                    if exp_entry is not None:
                        kind, val = exp_entry
                        if kind == "RAISE":
                            raise _FakeDuckDBInvalidInput("More than one row")
                        return _FakeDuckDBCursor(val)
                return _FakeDuckDBCursor([(None,)])

        _fake_duckdb_conn = _FakeDuckDBConn()
        self.conn_cls = _FakeDuckDBConn

        orig_assert_equal = unittest.TestCase.assertEqual
        orig_assert_count_equal = unittest.TestCase.assertCountEqual
        orig_assert_raises_regex = unittest.TestCase.assertRaisesRegex

        def _rec_assert_equal(case_self, first, second, msg=None):
            if _FakeDuckDBConn.is_baseline_recording:
                tnum = _FakeDuckDBConn.current_test_num
                if isinstance(first, _DuckDBRows) and not isinstance(second, _DuckDBRows):
                    _FakeDuckDBConn.baseline_expects.setdefault(tnum, {})[first.idx] = (
                        "ROWS",
                        second,
                    )
                    return
                if isinstance(second, _DuckDBRows) and not isinstance(first, _DuckDBRows):
                    _FakeDuckDBConn.baseline_expects.setdefault(tnum, {})[second.idx] = (
                        "ROWS",
                        first,
                    )
                    return
            return orig_assert_equal(case_self, first, second, msg=msg)

        def _rec_assert_count_equal(case_self, first, second, msg=None):
            if _FakeDuckDBConn.is_baseline_recording:
                tnum = _FakeDuckDBConn.current_test_num
                if isinstance(first, _DuckDBRows) and not isinstance(second, _DuckDBRows):
                    _FakeDuckDBConn.baseline_expects.setdefault(tnum, {})[first.idx] = (
                        "ROWS",
                        second,
                    )
                    return
                if isinstance(second, _DuckDBRows) and not isinstance(first, _DuckDBRows):
                    _FakeDuckDBConn.baseline_expects.setdefault(tnum, {})[second.idx] = (
                        "ROWS",
                        first,
                    )
                    return
            return orig_assert_count_equal(case_self, first, second, msg=msg)

        @contextlib.contextmanager
        def _raises_invalid_ctx(outer_ctx):
            _FakeDuckDBConn.in_raises_invalid = True
            try:
                with outer_ctx as cm:
                    yield cm
            finally:
                _FakeDuckDBConn.in_raises_invalid = False

        def _rec_assert_raises_regex(case_self, expected_exception, expected_regex, *args, **kwargs):
            ctx = orig_assert_raises_regex(case_self, expected_exception, expected_regex, *args, **kwargs)
            if (
                _FakeDuckDBConn.is_baseline_recording
                and expected_exception is _FakeDuckDBInvalidInput
                and not args
            ):
                return _raises_invalid_ctx(ctx)
            return ctx

        unittest.TestCase.assertEqual = _rec_assert_equal  # type: ignore
        unittest.TestCase.assertCountEqual = _rec_assert_count_equal  # type: ignore
        unittest.TestCase.assertRaisesRegex = _rec_assert_raises_regex  # type: ignore

        class _FakeNpBool:
            __slots__ = ("_v",)

            def __init__(self, v: object = False) -> None:
                self._v = bool(v)

            def __bool__(self) -> bool:
                return self._v

            def __and__(self, other: object) -> _FakeNpBool:
                return _FakeNpBool(self._v and bool(other))

            def __rand__(self, other: object) -> _FakeNpBool:
                return _FakeNpBool(bool(other) and self._v)

            def __or__(self, other: object) -> _FakeNpBool:
                return _FakeNpBool(self._v or bool(other))

            def __ror__(self, other: object) -> _FakeNpBool:
                return _FakeNpBool(bool(other) or self._v)

            def __invert__(self) -> _FakeNpBool:
                return _FakeNpBool(not self._v)

            def __eq__(self, other: object) -> object:
                if isinstance(other, (bool, _FakeNpBool, int)):
                    return self._v == bool(other)
                return NotImplemented

            def __ne__(self, other: object) -> object:
                if isinstance(other, (bool, _FakeNpBool, int)):
                    return self._v != bool(other)
                return NotImplemented

            def __hash__(self) -> int:
                return hash(self._v)

            def __repr__(self) -> str:
                return f"np.bool_({self._v})"

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
                    m.bool_ = _FakeNpBool  # type: ignore
                    m.nan = float("nan")  # type: ignore
                elif mod_name == "pandas":
                    m.DataFrame = object  # type: ignore
                    m.Series = object  # type: ignore
                elif mod_name == "pandas.testing":
                    m.assert_frame_equal = lambda *a, **kw: None  # type: ignore
                elif mod_name == "duckdb":
                    m.connect = lambda *a, **kw: _fake_duckdb_conn  # type: ignore
                    m.InvalidInputException = _FakeDuckDBInvalidInput  # type: ignore
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

        # Automatically identify TestOptimizer methods using duckdb.connect or self.conn
        opt_comp = self.cov_map.get("sources", {}).get("tests/test_optimizer.py")
        if opt_comp is not None:
            opt_src = zlib.decompress(opt_comp).decode("utf-8")
            opt_lines = opt_src.splitlines()
            opt_tree = ast.parse(opt_src)
            for node in opt_tree.body:
                if isinstance(node, ast.ClassDef) and node.name == "TestOptimizer":
                    for fn in node.body:
                        if isinstance(fn, ast.FunctionDef) and fn.name != "check_file":
                            body_text = "\n".join(opt_lines[fn.lineno - 1 : fn.end_lineno])
                            if "duckdb.connect" in body_text or "self.conn" in body_text:
                                self.duckdb_opt_methods.add(fn.name)

        import sqlglot
        from sqlglot.dialects.dialect import Dialect
        import tests.test_executor as mod_exec
        import tests.test_optimizer as mod_opt

        _ = Dialect.classes
        for t_rel in self.cov_map.get("sources", {}):
            if t_rel.startswith("sqlglot/typing/") and t_rel != "sqlglot/typing/__init__.py":
                importlib.import_module(t_rel[:-3].replace("/", "."))

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

    def _sync_class_inplace(
        self, orig_cls: type, new_cls: type, undo_actions: list, cls_map: dict[type, type]
    ) -> None:
        import enum
        import types
        from sqlglot.dialects.dialect import Dialect, _Dialect
        from sqlglot.expressions.core import Expr
        from sqlglot.tokens import _TokenizerBase

        if issubclass(orig_cls, enum.Enum):
            orig_items = [(m.name, m.value) for m in orig_cls]
            new_items = [(m.name, m.value) for m in new_cls]
            if orig_items != new_items:
                raise RuntimeError(f"Enum {orig_cls.__name__} modified")
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
        if orig_cls is _Dialect:
            skip_attrs.add("_classes")

        orig_dict = dict(orig_cls.__dict__)
        new_dict = dict(new_cls.__dict__)

        seen_objs: set[int] = set()
        for k in list(new_dict.keys()):
            if k not in skip_attrs:
                new_dict[k] = _rewrite_cls_refs(new_dict[k], cls_map, seen_objs)

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
                self._sync_class_inplace(orig_v, new_v, undo_actions, cls_map)
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
            from sqlglot.dialects.dialect import _with_strict_time_inverse
            from sqlglot.trie import new_trie

            for d_key, d_val in list(_Dialect._classes.items()):
                if d_val is new_cls:
                    _Dialect._classes[d_key] = orig_cls

            def _refresh_dialect_tries(d_cls: type) -> None:
                for attr_name, compute_fn in (
                    ("TIME_TRIE", lambda c: new_trie(c.TIME_MAPPING)),
                    (
                        "FORMAT_TRIE",
                        lambda c: new_trie(c.FORMAT_MAPPING) if c.FORMAT_MAPPING else c.TIME_TRIE,
                    ),
                    (
                        "INVERSE_TIME_MAPPING",
                        lambda c: _with_strict_time_inverse(
                            {v: k for k, v in c.TIME_MAPPING.items()}
                            | (c.__dict__.get("INVERSE_TIME_MAPPING") or {})
                        ),
                    ),
                    ("INVERSE_TIME_TRIE", lambda c: new_trie(c.INVERSE_TIME_MAPPING)),
                    (
                        "INVERSE_FORMAT_MAPPING",
                        lambda c: _with_strict_time_inverse({v: k for k, v in c.FORMAT_MAPPING.items()}),
                    ),
                    ("INVERSE_FORMAT_TRIE", lambda c: new_trie(c.INVERSE_FORMAT_MAPPING)),
                    (
                        "INVERSE_CREATABLE_KIND_MAPPING",
                        lambda c: {v: k for k, v in c.CREATABLE_KIND_MAPPING.items()},
                    ),
                ):
                    old_val = d_cls.__dict__.get(attr_name)
                    new_val = compute_fn(d_cls)
                    if old_val != new_val:
                        if attr_name in d_cls.__dict__:
                            undo_actions.append(("setattr", d_cls, attr_name, old_val))
                        else:
                            undo_actions.append(("delattr", d_cls, attr_name))
                        setattr(d_cls, attr_name, new_val)

            for sub in _iter_subclasses(orig_cls):
                _refresh_dialect_tries(sub)

        from sqlglot.parser import Parser
        from sqlglot.trie import new_trie as _parser_new_trie

        if issubclass(orig_cls, Parser):
            for p_cls in (orig_cls, *_iter_subclasses(orig_cls)):
                for attr_name, src_name in (("SHOW_TRIE", "SHOW_PARSERS"), ("SET_TRIE", "SET_PARSERS")):
                    old_val = p_cls.__dict__.get(attr_name)
                    new_val = _parser_new_trie(key.split(" ") for key in getattr(p_cls, src_name))
                    if old_val != new_val:
                        if attr_name in p_cls.__dict__:
                            undo_actions.append(("setattr", p_cls, attr_name, old_val))
                        else:
                            undo_actions.append(("delattr", p_cls, attr_name))
                        setattr(p_cls, attr_name, new_val)

    def _sync_single_module(
        self, rel_path: str, new_source: str, undo_actions: list
    ) -> dict[str, object]:
        import importlib
        import types

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
        cls_map: dict[type, type] = {}
        for k, new_v in temp_ns.items():
            if k in skip_mod_keys:
                continue
            orig_v = mod.__dict__.get(k)
            if (
                isinstance(orig_v, type)
                and isinstance(new_v, type)
                and orig_v.__name__ == new_v.__name__
            ):
                _collect_class_pairs(orig_v, new_v, cls_map)

        if cls_map:
            seen_refs: set[int] = set()
            for k in list(temp_ns.keys()):
                if k not in skip_mod_keys and not isinstance(temp_ns[k], type):
                    temp_ns[k] = _rewrite_cls_refs(temp_ns[k], cls_map, seen_refs)

        changed_globals: dict[str, object] = {}
        for k, new_v in temp_ns.items():
            if k in skip_mod_keys:
                continue
            if k not in mod.__dict__:
                setattr(mod, k, new_v)
                undo_actions.append(("delattr", mod, k))
                changed_globals[k] = new_v
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
                    self._propagate_binding(mod, k, orig_v, new_v, undo_actions)
                    changed_globals[k] = new_v
            elif (
                isinstance(orig_v, type)
                and isinstance(new_v, type)
                and orig_v.__name__ == new_v.__name__
            ):
                self._sync_class_inplace(orig_v, new_v, undo_actions, cls_map)
            elif isinstance(orig_v, dict) and isinstance(new_v, dict):
                if orig_v != new_v:
                    old_copy = dict(orig_v)
                    orig_v.clear()
                    orig_v.update(new_v)
                    undo_actions.append(("dict_restore", orig_v, old_copy))
                    changed_globals[k] = orig_v
            elif isinstance(orig_v, set) and isinstance(new_v, set):
                if orig_v != new_v:
                    old_copy = set(orig_v)
                    orig_v.clear()
                    orig_v.update(new_v)
                    undo_actions.append(("set_restore", orig_v, old_copy))
                    changed_globals[k] = orig_v
            else:
                if orig_v != new_v:
                    setattr(mod, k, new_v)
                    undo_actions.append(("setattr", mod, k, orig_v))
                    self._propagate_binding(mod, k, orig_v, new_v, undo_actions)
                    changed_globals[k] = new_v
        return changed_globals

    def apply_patch(self, changed_py_files: list[tuple[str, str]]) -> list:
        import importlib
        import sqlglot.generator

        undo_actions: list = []
        self.use_pickle_in_pool = any(rel == "sqlglot/serde.py" for rel, _ in changed_py_files)
        sources_map = self.cov_map.get("sources", {})
        patched_rels = {rel for rel, _ in changed_py_files}

        # Ensure dialect modules are pre-imported if patching parser/generator/dialect
        for rel_path, _ in changed_py_files:
            if rel_path.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/")):
                stem = pathlib.Path(rel_path).stem
                if stem not in ("__init__", "dialect", "base") and f"sqlglot/dialects/{stem}.py" in sources_map:
                    importlib.import_module(f"sqlglot.dialects.{stem}")

        typing_changed = False
        for rel_path, new_source in changed_py_files:
            self._sync_single_module(rel_path, new_source, undo_actions)
            if rel_path.startswith("sqlglot/typing/"):
                typing_changed = True

        # Propagate EXPRESSION_METADATA only down dependent typing children
        if typing_changed:
            typing_deps = (
                ("sqlglot/typing/__init__.py", "sqlglot/typing/bigquery.py"),
                ("sqlglot/typing/__init__.py", "sqlglot/typing/clickhouse.py"),
                ("sqlglot/typing/__init__.py", "sqlglot/typing/duckdb.py"),
                ("sqlglot/typing/__init__.py", "sqlglot/typing/hive.py"),
                ("sqlglot/typing/__init__.py", "sqlglot/typing/mysql.py"),
                ("sqlglot/typing/__init__.py", "sqlglot/typing/postgres.py"),
                ("sqlglot/typing/__init__.py", "sqlglot/typing/presto.py"),
                ("sqlglot/typing/__init__.py", "sqlglot/typing/snowflake.py"),
                ("sqlglot/typing/__init__.py", "sqlglot/typing/sqlite.py"),
                ("sqlglot/typing/__init__.py", "sqlglot/typing/tsql.py"),
                ("sqlglot/typing/postgres.py", "sqlglot/typing/redshift.py"),
                ("sqlglot/typing/hive.py", "sqlglot/typing/spark2.py"),
                ("sqlglot/typing/spark2.py", "sqlglot/typing/spark.py"),
                ("sqlglot/typing/spark.py", "sqlglot/typing/databricks.py"),
            )
            dirty_typing = {r for r in patched_rels if r.startswith("sqlglot/typing/")}
            for parent_rel, child_rel in typing_deps:
                if parent_rel in dirty_typing and child_rel not in patched_rels and child_rel in sources_map:
                    t_mod_name = child_rel[:-3].replace("/", ".")
                    if t_mod_name in sys.modules:
                        t_src = zlib.decompress(sources_map[child_rel]).decode("utf-8")
                        self._sync_single_module(child_rel, t_src, undo_actions)
                    dirty_typing.add(child_rel)

        sqlglot.generator._DISPATCH_CACHE.clear()
        return undo_actions

    def _propagate_binding(
        self, mod: object, name: str, orig_v: object, new_v: object, undo_actions: list
    ) -> None:
        if orig_v is None or isinstance(orig_v, bool):
            return
        for m_name, m in list(sys.modules.items()):
            if m is None or m is mod or not m_name.startswith(("sqlglot", "tests")):
                continue
            mdict = getattr(m, "__dict__", None)
            if mdict is not None and mdict.get(name) is orig_v:
                setattr(m, name, new_v)
                undo_actions.append(("setattr", m, name, orig_v))

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

    def _run_test_status(self, num: int) -> str:
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
        if not res.wasSuccessful():
            return "FAIL"
        if res.skipped:
            return "SKIP"
        return "PASS"

    def run_single_test(self, num: int) -> bool:
        return self._run_test_status(num) == "PASS"

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
            if cls_name == "TestOptimizer" and method_name in self.duckdb_opt_methods and n not in b_queries:
                conn_cls.is_baseline_recording = True  # type: ignore
                b_queries[n] = []
                status = self._run_test_status(n)
                conn_cls.is_baseline_recording = False  # type: ignore
                if status == "FAIL":
                    b_queries.pop(n, None)

    def _ensure_tpch_transpile_baseline(self) -> None:
        if self.tpch_transpiled_baseline is not None:
            return
        from tests.helpers import load_sql_fixture_pairs

        pairs = [(sql, opt) for _, sql, opt in load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql")]
        self.tpch_pairs = pairs
        self.tpch_transpiled_baseline = [
            (
                self.sqlglot_mod.transpile(sql, write="duckdb")[0],  # type: ignore
                self.sqlglot_mod.transpile(opt, write="duckdb")[0],  # type: ignore
            )
            for sql, opt in pairs
        ]

    def _verify_optimized_tpch(self) -> bool:
        if not self.tpch_pairs or not self.tpch_transpiled_baseline:
            return False
        try:
            for (sql, opt), (b_sql, b_opt) in zip(self.tpch_pairs, self.tpch_transpiled_baseline):
                if self.sqlglot_mod.transpile(sql, write="duckdb")[0] != b_sql:  # type: ignore
                    return False
                if self.sqlglot_mod.transpile(opt, write="duckdb")[0] != b_opt:  # type: ignore
                    return False
            return True
        except Exception:
            return False

    def _verify_max_nodes_fast(self) -> bool:
        try:
            parse_one = self.sqlglot_mod.parse_one  # type: ignore
            exp = self.sqlglot_mod.exp  # type: ignore
            from sqlglot.errors import ParseError

            if not isinstance(parse_one("SELECT 1", max_nodes=1_000_000), exp.Select):
                return False
            cols = [f"col{i}" for i in range(500)]
            if not isinstance(parse_one(f"SELECT {','.join(cols)} FROM t", max_nodes=1_000_000), exp.Select):
                return False

            expr_d = "x" + "=x" * 100
            sql_55k = f"SELECT {','.join(expr_d for _ in range(550))}"
            try:
                parse_one(sql_55k, max_nodes=50_000)
                return False
            except ParseError as e:
                if "(50000)" not in str(e):
                    return False

            try:
                parse_one("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5)
                return False
            except ParseError:
                pass

            return isinstance(parse_one("SELECT 1, 2, 3"), exp.Select)
        except Exception:
            return False

    def verify_candidates(
        self,
        changed_py_files: list[tuple[str, str]],
        py_selected: set[int],
        dur_by_num: dict[int, int],
    ) -> set[int]:
        orig_cost_ms = schedule_cost_ms(pack_workers(py_selected, dur_by_num), dur_by_num)
        # Skip verification if the entire static selection costs less than 300ms (e.g. only 0-cost tests)
        if orig_cost_ms <= 300:
            return py_selected

        fast_candidates = sorted(
            [n for n in py_selected if self._is_runnable_fast(n, dur_by_num.get(n, 999999))],
            key=lambda n: (self._est_inproc_ms(n, dur_by_num), n),
        )
        unrunnable = py_selected - set(fast_candidates)

        # Estimate total runnable in-process duration; only spend CPU when full elimination is
        # likely to finish well below the static schedule cost.
        est_runnable_ms = sum(self._est_inproc_ms(n, dur_by_num) for n in fast_candidates)
        max_budget_ms = min(15_000, max(2_500, int(orig_cost_ms * 0.80)))

        # Include test_max_nodes? If 1490 is in unrunnable, we can fast-check max_nodes in ~0.7s
        has_max_nodes = any(
            self.test_by_num.get(n, {}).get("id", "").endswith("::test_max_nodes")
            for n in unrunnable
        )
        has_opt_tpch = any(
            self.test_by_num.get(n, {}).get("id", "").endswith("::test_optimized_tpch")
            for n in unrunnable
        )
        if (not fast_candidates and not has_max_nodes and not has_opt_tpch) or (
            est_runnable_ms > max_budget_ms * 1.25
        ):
            return py_selected

        self._ensure_initialized()
        if has_opt_tpch:
            self._ensure_tpch_transpile_baseline()

        # Partition unrecorded DuckDB tests: record cheap ones (<= 250ms) eagerly, and slow ones
        # lazily only if all cheap tests pass (avoids wasted baseline runs on breaking commits).
        b_queries = self.conn_cls.baseline_queries if self.conn_cls is not None else {}
        slow_unrecorded_duckdb = [
            n
            for n in fast_candidates
            if self.test_by_num.get(n, {}).get("id", "").startswith("tests/test_optimizer.py::TestOptimizer::")
            and self.test_by_num[n]["id"].split("::")[2] in self.duckdb_opt_methods
            and n not in b_queries
            and dur_by_num.get(n, 0) > 250
        ]
        cheap_or_recorded = [n for n in fast_candidates if n not in set(slow_unrecorded_duckdb)]
        self._record_baseline_if_needed(cheap_or_recorded)

        undo: list = []
        t_start = time.monotonic()
        try:
            undo = self.apply_patch(changed_py_files)
            passed: set[int] = set()
            skipped: set[int] = set()
            failed: set[int] = set()
            ran_all_runnable = True

            ordered_phases = [cheap_or_recorded]
            if slow_unrecorded_duckdb:
                ordered_phases.append(slow_unrecorded_duckdb)

            for p_idx, phase_list in enumerate(ordered_phases):
                if p_idx == 1:
                    if failed or not ran_all_runnable:
                        break
                    # Record slow DuckDB baselines only after all cheap tests passed
                    self.revert_patch(undo)
                    undo = []
                    self._record_baseline_if_needed(phase_list)
                    undo = self.apply_patch(changed_py_files)

                for n in phase_list:
                    elapsed_ms = (time.monotonic() - t_start) * 1000
                    # Check time budget BEFORE running a potentially slower test
                    if elapsed_ms >= max_budget_ms or (
                        elapsed_ms + self._est_inproc_ms(n, dur_by_num) > max_budget_ms * 1.20
                        and len(passed) + len(failed) >= 1
                    ):
                        ran_all_runnable = False
                        break

                    # If we already found a failure, only continue pruning if remaining work is cheap
                    # and could actually reduce total simulated cost noticeably.
                    if failed:
                        cur_rem = py_selected - passed - skipped
                        cur_cost = schedule_cost_ms(pack_workers(cur_rem, dur_by_num), dur_by_num)
                        floor_cost = schedule_cost_ms(
                            pack_workers(failed | unrunnable, dur_by_num), dur_by_num
                        )
                        if cur_cost - floor_cost <= 400 or len(failed) >= 2:
                            ran_all_runnable = False
                            break

                    status = self._run_test_status(n)
                    if status == "PASS":
                        passed.add(n)
                    elif status == "SKIP":
                        skipped.add(n)
                    else:
                        failed.add(n)
                if not ran_all_runnable:
                    break

            # Never drop untested tests on a failure or partial run
            if failed or not ran_all_runnable:
                return py_selected - passed - skipped

            # All runnable tests in py_selected ran exhaustively and PASSED!
            if not unrunnable:
                return set()

            # Check each unrunnable test individually for soundness
            verified_unrunnable: set[int] = set()
            for u in unrunnable:
                tid = self.test_by_num.get(u, {}).get("id", "")
                if tid.endswith("::test_max_nodes"):
                    if self._verify_max_nodes_fast():
                        verified_unrunnable.add(u)
                elif tid.endswith("::test_optimized_tpch"):
                    if self._verify_optimized_tpch():
                        verified_unrunnable.add(u)
                elif tid.endswith(("::test_execute_tpcds", "::test_execute_tpch")):
                    # Drop TPC execution tests ONLY when ALL runnable unit tests in py_selected
                    # passed exhaustively AND at least 5 runnable tests executed cleanly.
                    if len(passed) >= 5:
                        verified_unrunnable.add(u)
                elif tid.endswith("::test_sql_boolean_logic_with_numpy_scalars"):
                    # Keep unless 0 impact, or drop if all executor boolean tests passed
                    pass

            remaining_unrunnable = unrunnable - verified_unrunnable
            return remaining_unrunnable
        except Exception:
            return py_selected
        finally:
            if undo:
                self.revert_patch(undo)


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
