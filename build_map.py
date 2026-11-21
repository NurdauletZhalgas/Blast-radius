#!/usr/bin/env python3
"""One-time script to build a compact coverage, AST, symbol, fixture, and import map
from sqlglot at commit eb11aec4be8cb2122afe713a4816662c3f0fb2fa.

Can be run via Docker:
  docker build -f Dockerfile.map -t blast-radius-map .
  docker run --rm -v "${PWD}:/out" blast-radius-map
"""

from __future__ import annotations

import argparse
import ast
import builtins
import concurrent.futures
import gc
import importlib
import importlib.machinery
import io
import json
import os
import pathlib
import pickle
import subprocess
import sys
import tarfile
import time
import types
import unittest
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import Future

PINNED_COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
REPO_URL = "https://github.com/tobymao/sqlglot.git"
TARBALL_URL = f"https://github.com/tobymao/sqlglot/archive/{PINNED_COMMIT}.tar.gz"


def ensure_sqlglot(sqlglot_dir: pathlib.Path) -> pathlib.Path:
    if (sqlglot_dir / "sqlglot" / "__init__.py").exists():
        print(f"Using existing sqlglot checkout at {sqlglot_dir}", file=sys.stderr)
        return sqlglot_dir

    sqlglot_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print(f"Cloning {REPO_URL} into {sqlglot_dir}...", file=sys.stderr)
        subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(sqlglot_dir)], check=True)
        subprocess.run(
            ["git", "-C", str(sqlglot_dir), "checkout", PINNED_COMMIT],
            check=True,
        )
        return sqlglot_dir
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"Downloading tarball {TARBALL_URL}...", file=sys.stderr)
        tar_path = sqlglot_dir.parent / "sqlglot.tar.gz"
        urllib.request.urlretrieve(TARBALL_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(sqlglot_dir.parent)
        extracted = sqlglot_dir.parent / f"sqlglot-{PINNED_COMMIT}"
        if sqlglot_dir.exists():
            sqlglot_dir.rmdir()
        extracted.rename(sqlglot_dir)
        return sqlglot_dir


class InProcessPoolExecutor:
    """Replaces ProcessPoolExecutor during coverage collection so worker functions
    (like TestOptimizer.check_file and TestExecutor._mp_execute) execute in-process
    and have their exact line/dialect/expression coverage traced."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, /, *args, **kwargs):
        f = Future()
        try:
            args, kwargs = pickle.loads(pickle.dumps((args, kwargs)))
            res = fn(*args, **kwargs)
            res = pickle.loads(pickle.dumps(res))
            f.set_result(res)
        except BaseException as exc:
            f.set_exception(exc)
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def build_map(sqlglot_root: pathlib.Path, tests_json_path: pathlib.Path, output_path: pathlib.Path) -> None:
    t0 = time.monotonic()
    sqlglot_root = sqlglot_root.resolve()
    root_prefix = str(sqlglot_root).replace("\\", "/")
    if not root_prefix.endswith("/"):
        root_prefix += "/"

    os.environ["SKIP_INTEGRATION"] = "0"
    sys.path.insert(0, str(sqlglot_root))

    # Patch ProcessPoolExecutor before importing tests
    concurrent.futures.ProcessPoolExecutor = InProcessPoolExecutor  # type: ignore

    tests_meta = json.loads(tests_json_path.read_text("utf-8"))["tests"]
    test_ids = [t["id"] for t in tests_meta]
    print(f"Loaded {len(test_ids)} tests from {tests_json_path}", file=sys.stderr)

    # Set up sys.monitoring for per-test line coverage
    mon = sys.monitoring
    TOOL_ID = mon.COVERAGE_ID
    mon.use_tool_id(TOOL_ID, "blast_radius_cov")

    code_rel_path: dict[types.CodeType, str] = {}
    registered_codes: set[types.CodeType] = set()
    cur_lines: set[tuple[str, int]] = set()

    def normalize_rel(filename: str) -> str | None:
        norm = filename.replace("\\", "/")
        if norm.startswith(root_prefix):
            return norm[len(root_prefix):]
        # Also resolve symlinks/relative paths if needed
        try:
            resolved = str(pathlib.Path(filename).resolve()).replace("\\", "/")
            if resolved.startswith(root_prefix):
                return resolved[len(root_prefix):]
        except Exception:
            pass
        return None

    def register_code(code: types.CodeType) -> None:
        if type(code) is not types.CodeType or code in registered_codes:
            return
        registered_codes.add(code)
        rel = normalize_rel(code.co_filename)
        if rel is not None and (
            rel.startswith("sqlglot/") or rel.startswith("tests/") or rel.startswith("sqlglotc/")
        ):
            code_rel_path[code] = rel
            mon.set_local_events(TOOL_ID, code, mon.events.LINE)
            for const in code.co_consts:
                if type(const) is types.CodeType:
                    register_code(const)

    def on_line(code: types.CodeType, line_number: int):
        rel = code_rel_path.get(code)
        if rel is not None:
            cur_lines.add((rel, line_number))
        return mon.DISABLE

    mon.register_callback(TOOL_ID, mon.events.LINE, on_line)

    # Hook SourceFileLoader.get_code and builtins.compile to auto-register code objects
    orig_get_code = importlib.machinery.SourceFileLoader.get_code
    orig_compile = builtins.compile

    def hooked_get_code(self, fullname):
        code = orig_get_code(self, fullname)
        if code is not None:
            register_code(code)
        return code

    def hooked_compile(*args, **kwargs):
        code = orig_compile(*args, **kwargs)
        if isinstance(code, types.CodeType):
            register_code(code)
        return code

    importlib.machinery.SourceFileLoader.get_code = hooked_get_code  # type: ignore
    builtins.compile = hooked_compile  # type: ignore

    def scan_gc_functions() -> None:
        for obj in gc.get_objects():
            if type(obj) is types.FunctionType:
                register_code(obj.__code__)

    # Pre-import all sqlglot modules and record import-time lines
    import_lines: dict[str, set[int]] = defaultdict(set)
    cur_lines.clear()
    for p in sorted((sqlglot_root / "sqlglot").rglob("*.py")):
        rel = p.relative_to(sqlglot_root).with_suffix("")
        mod_name = ".".join(rel.parts)
        if mod_name.endswith(".__init__"):
            mod_name = mod_name[:-9]
        if mod_name.endswith(".__main__"):
            continue
        importlib.import_module(mod_name)

    import sqlglot
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.expressions.core import Expr

    # Force loading all dialects
    _ = Dialect.classes
    scan_gc_functions()

    for rel, line in cur_lines:
        import_lines[rel].add(line)
    print(
        f"Pre-imported sqlglot ({len(code_rel_path)} code objects, "
        f"{sum(len(v) for v in import_lines.values())} import-time lines)",
        file=sys.stderr,
    )

    # Hook open() to track fixture and file reads
    orig_open = builtins.open
    cur_opens: set[str] = set()

    def hooked_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            rel = normalize_rel(os.fspath(file))
            if rel is not None:
                cur_opens.add(rel)
        return orig_open(file, *args, **kwargs)

    builtins.open = hooked_open
    io.open = hooked_open

    # Hook Dialect usage and track active dialect(s) at runtime
    cur_dialects: set[str] = set()
    cur_active_dialects: tuple[str, ...] = ()
    test_all_lines: set[tuple[str, int]] = set()
    cur_lines_by_dialect: dict[str, set[tuple[str, int]]] = defaultdict(set)
    cur_exprs_by_dialect: dict[str, set[str]] = defaultdict(set)
    dialect_mro_cache: dict[type, tuple[str, ...]] = {}
    switch_count = 0

    def get_dialect_names(d_cls: type) -> tuple[str, ...]:
        names = dialect_mro_cache.get(d_cls)
        if names is None:
            s: list[str] = []
            for base in d_cls.__mro__:
                if isinstance(base, type) and issubclass(base, Dialect) and base is not Dialect:
                    mod_tail = base.__module__.split(".")[-1]
                    if mod_tail not in s:
                        s.append(mod_tail)
                    bname = base.__name__.lower()
                    if bname not in s:
                        s.append(bname)
            names = tuple(s)
            dialect_mro_cache[d_cls] = names
        return names

    def switch_active_dialect(d_cls: type) -> None:
        nonlocal cur_active_dialects, switch_count
        names = get_dialect_names(d_cls)
        cur_dialects.update(names)
        if names != cur_active_dialects:
            if cur_lines:
                test_all_lines.update(cur_lines)
                for d in cur_active_dialects:
                    cur_lines_by_dialect[d].update(cur_lines)
                cur_lines.clear()
            cur_active_dialects = names
            switch_count += 1
            if switch_count <= 250:
                mon.restart_events()

    orig_dialect_init = Dialect.__init__

    def hooked_dialect_init(self, *args, **kwargs):
        switch_active_dialect(type(self))
        return orig_dialect_init(self, *args, **kwargs)

    Dialect.__init__ = hooked_dialect_init  # type: ignore

    orig_get_or_raise = Dialect.get_or_raise.__func__  # type: ignore

    @classmethod  # type: ignore
    def hooked_get_or_raise(cls, dialect=None):
        res = orig_get_or_raise(cls, dialect)
        if isinstance(res, Dialect):
            switch_active_dialect(type(res))
        elif isinstance(res, type) and issubclass(res, Dialect):
            switch_active_dialect(res)
        return res

    Dialect.get_or_raise = hooked_get_or_raise  # type: ignore

    orig_tokenizer_fn = Dialect.tokenizer
    orig_parser_fn = Dialect.parser
    orig_generator_fn = Dialect.generator

    def hooked_tokenizer(self, *args, **kwargs):
        switch_active_dialect(type(self))
        return orig_tokenizer_fn(self, *args, **kwargs)

    def hooked_parser(self, *args, **kwargs):
        switch_active_dialect(type(self))
        return orig_parser_fn(self, *args, **kwargs)

    def hooked_generator(self, *args, **kwargs):
        switch_active_dialect(type(self))
        return orig_generator_fn(self, *args, **kwargs)

    Dialect.tokenizer = hooked_tokenizer  # type: ignore
    Dialect.parser = hooked_parser  # type: ignore
    Dialect.generator = hooked_generator  # type: ignore

    # Hook Expr.__init__ to track expression classes instantiated per test (and per active dialect)
    cur_exprs: set[str] = set()
    orig_expr_init = Expr.__init__
    expr_mro_cache: dict[type, tuple[str, ...]] = {}

    def hooked_expr_init(self, **args):
        cls = type(self)
        names = expr_mro_cache.get(cls)
        if names is None:
            names = tuple(
                b.__name__
                for b in cls.__mro__
                if isinstance(b, type) and issubclass(b, Expr) and b is not Expr
            )
            expr_mro_cache[cls] = names
        cur_exprs.update(names)
        for d in cur_active_dialects:
            cur_exprs_by_dialect[d].update(names)
        return orig_expr_init(self, **args)

    Expr.__init__ = hooked_expr_init  # type: ignore

    # Group tests by (file, ClassName)
    groups: list[tuple[str, str, list[tuple[int, str]]]] = []
    for t in tests_meta:
        num = t["number"]
        file_rel, cls_name, method_name = t["id"].split("::")
        if not groups or groups[-1][0] != file_rel or groups[-1][1] != cls_name:
            groups.append((file_rel, cls_name, []))
        groups[-1][2].append((num, method_name))

    # Coverage accumulators: item -> set of test numbers
    line_to_tests: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    dialect_line_to_tests: dict[str, dict[str, dict[int, set[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    setup_class_line_to_tests: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    fixture_to_tests: dict[str, set[int]] = defaultdict(set)
    dialect_to_tests: dict[str, set[int]] = defaultdict(set)
    expr_to_tests: dict[str, set[int]] = defaultdict(set)
    dialect_expr_to_tests: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    # Pre-import all test modules
    test_modules: dict[str, types.ModuleType] = {}
    for file_rel, _, _ in groups:
        mod_name = ".".join(pathlib.Path(file_rel).with_suffix("").parts)
        cur_lines.clear()
        mod = importlib.import_module(mod_name)
        test_modules[file_rel] = mod
        for rel, line in cur_lines:
            import_lines[rel].add(line)
    scan_gc_functions()

    failed_tests: list[tuple[int, str, str]] = []
    cwd = os.getcwd()
    os.chdir(sqlglot_root)
    try:
        for g_idx, (file_rel, cls_name, methods) in enumerate(groups, start=1):
            mod = test_modules[file_rel]
            cls = getattr(mod, cls_name)
            class_test_nums = {num for num, _ in methods}

            # Trace setUpClass
            cur_lines.clear()
            test_all_lines.clear()
            cur_opens.clear()
            cur_dialects.clear()
            cur_exprs.clear()
            cur_active_dialects = ()
            cur_lines_by_dialect.clear()
            cur_exprs_by_dialect.clear()
            switch_count = 0
            mon.restart_events()

            cls.setUpClass()
            scan_gc_functions()

            for rel, line in (cur_lines | test_all_lines):
                setup_class_line_to_tests[rel][line].update(class_test_nums)
            for opened_rel in cur_opens:
                # Special handling for TestExecutor.setUpClass which loads tpc-h and tpc-ds fixtures
                if cls_name == "TestExecutor" and "tpc-h" in opened_rel:
                    for num, m_name in methods:
                        if "tpch" in m_name:
                            fixture_to_tests[opened_rel].add(num)
                elif cls_name == "TestExecutor" and "tpc-ds" in opened_rel:
                    for num, m_name in methods:
                        if "tpcds" in m_name:
                            fixture_to_tests[opened_rel].add(num)
                else:
                    fixture_to_tests[opened_rel].update(class_test_nums)

            # Run each test method in the class
            for num, method_name in methods:
                cur_lines.clear()
                test_all_lines.clear()
                cur_opens.clear()
                cur_dialects.clear()
                cur_exprs.clear()
                cur_active_dialects = ()
                cur_lines_by_dialect.clear()
                cur_exprs_by_dialect.clear()
                switch_count = 0
                mon.restart_events()

                case = cls(method_name)
                res = unittest.TestResult()
                case.run(res)

                if cur_lines:
                    test_all_lines.update(cur_lines)
                    for d in cur_active_dialects:
                        cur_lines_by_dialect[d].update(cur_lines)
                    cur_lines.clear()

                if not res.wasSuccessful():
                    err_msg = str(res.errors or res.failures)[:200]
                    failed_tests.append((num, f"{file_rel}::{cls_name}::{method_name}", err_msg))

                for d_name, d_lines in cur_lines_by_dialect.items():
                    for rel, line in d_lines:
                        dialect_line_to_tests[d_name][rel][line].add(num)

                for d_name, d_exprs in cur_exprs_by_dialect.items():
                    for e_name in d_exprs:
                        dialect_expr_to_tests[d_name][e_name].add(num)

                for rel, line in test_all_lines:
                    line_to_tests[rel][line].add(num)
                    # If a dialect file was executed directly, also record dialect usage
                    if rel.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
                        stem = pathlib.Path(rel).stem
                        if stem not in ("__init__", "dialect", "base"):
                            cur_dialects.add(stem)
                            dialect_line_to_tests[stem][rel][line].add(num)

                for opened_rel in cur_opens:
                    fixture_to_tests[opened_rel].add(num)
                for d_name in cur_dialects:
                    dialect_to_tests[d_name].add(num)
                for e_name in cur_exprs:
                    expr_to_tests[e_name].add(num)

            try:
                cls.tearDownClass()
                cls.doClassCleanups()
            except Exception as e:
                print(f"Warning in tearDownClass for {cls_name}: {e}", file=sys.stderr)

            # Ensure all dialects are reloaded if TestDialectEntryPoints cleared Dialect._classes
            if len(Dialect._classes) < 20:
                _ = Dialect.classes

            if g_idx % 10 == 0 or g_idx == len(groups):
                elapsed = time.monotonic() - t0
                print(f"[{g_idx}/{len(groups)}] Finished {file_rel} ({elapsed:.1f}s)", file=sys.stderr)
    finally:
        os.chdir(cwd)
        builtins.open = orig_open
        io.open = orig_open
        Dialect.__init__ = orig_dialect_init  # type: ignore
        Dialect.get_or_raise = classmethod(orig_get_or_raise)  # type: ignore
        Dialect.tokenizer = orig_tokenizer_fn  # type: ignore
        Dialect.parser = orig_parser_fn  # type: ignore
        Dialect.generator = orig_generator_fn  # type: ignore
        Expr.__init__ = orig_expr_init  # type: ignore
        importlib.machinery.SourceFileLoader.get_code = orig_get_code  # type: ignore
        builtins.compile = orig_compile  # type: ignore
        mon.set_events(TOOL_ID, 0)
        mon.free_tool_id(TOOL_ID)

    if failed_tests:
        print(f"WARNING: {len(failed_tests)} tests failed during map build:", file=sys.stderr)
        for num, tid, msg in failed_tests[:10]:
            print(f"  #{num} {tid}: {msg}", file=sys.stderr)
    else:
        print("All 1620 tests passed during coverage trace!", file=sys.stderr)

    # Collect all Python source files in sqlglot/, tests/, sqlglotc/
    py_sources: dict[str, str] = {}
    compressed_sources: dict[str, bytes] = {}
    for folder in ("sqlglot", "tests", "sqlglotc"):
        folder_path = sqlglot_root / folder
        if not folder_path.exists():
            continue
        for p in sorted(folder_path.rglob("*.py")):
            rel = str(p.relative_to(sqlglot_root)).replace("\\", "/")
            text = p.read_text("utf-8")
            py_sources[rel] = text
            compressed_sources[rel] = zlib.compress(text.encode("utf-8"), level=9)

    # Collect non-TPC fixture files in tests/fixtures/ for in-process candidate test verification
    compressed_fixtures: dict[str, bytes] = {}
    fixtures_root = sqlglot_root / "tests" / "fixtures"
    if fixtures_root.exists():
        for p in sorted(fixtures_root.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(sqlglot_root)).replace("\\", "/")
            if "/tpc-h/" in rel or "/tpc-ds/" in rel:
                continue
            compressed_fixtures[rel] = zlib.compress(p.read_bytes(), level=9)

    # Special case: test_lazy_load (#214) runs subprocess `import sqlglot; assert len(sqlglot.Dialect._classes) == 1`
    lazy_load_nums = {
        t["number"] for t in tests_meta if t["id"].endswith("::test_lazy_load")
    }
    # And test_no_redundant_expression_metadata (#1619) inspects all sqlglot/typing/*.py
    typing_test_nums = {
        t["number"] for t in tests_meta if t["file"] == "tests/test_typing.py"
    }

    # Build static import dependency graph and symbol -> runtime test coverage
    mod_to_rel: dict[str, str] = {}
    for rel in py_sources:
        no_ext = rel[:-3] if rel.endswith(".py") else rel
        mod_name = no_ext.replace("/", ".")
        if mod_name.endswith(".__init__"):
            mod_to_rel[mod_name[:-9]] = rel
        mod_to_rel[mod_name] = rel

    # direct_imports[rel_a] = set of rel_b imported by rel_a
    direct_imports: dict[str, set[str]] = defaultdict(set)
    raw_symbol_cov: dict[str, set[int]] = defaultdict(set)
    raw_dialect_symbol_cov: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    for rel, text in py_sources.items():
        tree = ast.parse(text, filename=rel)
        pkg_parts = rel.split("/")[:-1]
        file_line_cov = line_to_tests.get(rel, {})
        file_setup_cov = setup_class_line_to_tests.get(rel, {})

        # Skip Tokenizer.__init__ and TokenizerCore.__init__ / _extract_string preamble
        # when indexing raw_symbol_cov, as they merely copy all attributes on every call.
        skip_sym_lines: set[int] = set()
        if rel in ("sqlglot/tokens.py", "sqlglot/tokenizer_core.py"):
            for n in ast.walk(tree):
                if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "_extract_string"):
                    end_l = getattr(n, "end_lineno", n.lineno)
                    if n.name == "_extract_string":
                        # Only skip the local variable caching preamble (first 22 lines)
                        end_l = min(end_l, n.lineno + 22)
                    skip_sym_lines.update(range(n.lineno, end_l + 1))

        # Find all runtime function/lambda line ranges so we only index symbol reads at runtime lines
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = mod_to_rel.get(alias.name)
                    if target:
                        direct_imports[rel].add(target)
                    parts = alias.name.split(".")
                    for i in range(1, len(parts)):
                        parent_target = mod_to_rel.get(".".join(parts[:i]))
                        if parent_target:
                            direct_imports[rel].add(parent_target)
            elif isinstance(node, ast.ImportFrom):
                base_mod = node.module or ""
                if node.level > 0:
                    prefix = pkg_parts[: len(pkg_parts) - node.level + 1]
                    base_mod = ".".join(prefix + ([base_mod] if base_mod else []))
                if base_mod:
                    target = mod_to_rel.get(base_mod)
                    if target:
                        direct_imports[rel].add(target)
                    for alias in node.names:
                        sub_target = mod_to_rel.get(f"{base_mod}.{alias.name}")
                        if sub_target:
                            direct_imports[rel].add(sub_target)

            # Index symbol usages on runtime-covered lines
            lineno = getattr(node, "lineno", None)
            if lineno is not None and lineno not in skip_sym_lines:
                cov = file_line_cov.get(lineno)
                scov = file_setup_cov.get(lineno)
                sym = None
                if isinstance(node, ast.Name):
                    sym = node.id
                elif isinstance(node, ast.Attribute):
                    sym = node.attr
                if sym:
                    if cov:
                        raw_symbol_cov[sym].update(cov)
                    if scov:
                        raw_symbol_cov[sym].update(scov)
                    for d_name, d_files in dialect_line_to_tests.items():
                        dcov = d_files.get(rel, {}).get(lineno)
                        if dcov:
                            raw_dialect_symbol_cov[d_name][sym].update(dcov)

    # Propagate metaclass / TokenizerCore derived attribute coverages to their source class attributes
    DERIVED_ATTR_MAP = {
        "TIME_MAPPING": ("TIME_TRIE", "FORMAT_TRIE", "INVERSE_TIME_MAPPING", "INVERSE_TIME_TRIE", "format_time"),
        "FORMAT_MAPPING": ("FORMAT_TRIE", "INVERSE_FORMAT_MAPPING", "INVERSE_FORMAT_TRIE"),
        "INVERSE_TIME_MAPPING": ("INVERSE_TIME_TRIE",),
        "CREATABLE_KIND_MAPPING": ("INVERSE_CREATABLE_KIND_MAPPING",),
        "SUPPORTED_JSON_PATH_PARTS": ("TRANSFORMS",),
        "UNESCAPED_SEQUENCES": ("ESCAPED_SEQUENCES", "IDENTIFIER_ESCAPED_SEQUENCES", "unescaped_sequences"),
        "DATE_PART_MAPPING": ("VALID_INTERVAL_UNITS",),
        "NORMALIZATION_STRATEGY": ("normalization_strategy",),
        "QUOTES": ("_QUOTES", "_FORMAT_STRINGS", "_KEYWORD_TRIE", "QUOTE_START", "QUOTE_END", "quotes"),
        "IDENTIFIERS": ("_IDENTIFIERS", "IDENTIFIER_START", "IDENTIFIER_END", "identifiers"),
        "BIT_STRINGS": ("_FORMAT_STRINGS", "BIT_START", "BIT_END", "has_bit_strings"),
        "HEX_STRINGS": ("_FORMAT_STRINGS", "HEX_START", "HEX_END", "has_hex_strings"),
        "BYTE_STRINGS": ("_FORMAT_STRINGS", "BYTE_START", "BYTE_END"),
        "UNICODE_STRINGS": ("_FORMAT_STRINGS", "UNICODE_START", "UNICODE_END"),
        "RAW_STRINGS": ("_FORMAT_STRINGS", "format_strings"),
        "NATIONAL_RAW_STRINGS": ("_FORMAT_STRINGS", "format_strings"),
        "HEREDOC_STRINGS": ("_FORMAT_STRINGS", "format_strings"),
        "STRING_ESCAPES": (
            "_STRING_ESCAPES",
            "BYTE_STRING_ESCAPES",
            "_BYTE_STRING_ESCAPES",
            "STRINGS_SUPPORT_ESCAPED_SEQUENCES",
            "UNESCAPED_SEQUENCES",
            "ESCAPED_SEQUENCES",
            "string_escapes",
        ),
        "BYTE_STRING_ESCAPES": ("_BYTE_STRING_ESCAPES", "BYTE_STRINGS_SUPPORT_ESCAPED_SEQUENCES", "byte_string_escapes"),
        "IDENTIFIER_ESCAPES": ("_IDENTIFIER_ESCAPES", "IDENTIFIER_ESCAPED_SEQUENCES", "identifier_escapes"),
        "COMMENTS": ("_COMMENTS", "_KEYWORD_TRIE", "comments"),
        "KEYWORDS": ("_KEYWORD_TRIE", "SUPPORTS_COLUMN_JOIN_MARKS", "keyword_trie"),
        "SINGLE_TOKENS": ("_KEYWORD_TRIE",),
        "NUMERIC_ESCAPES": ("_scan_numeric_escape", "_read_numeric_escape"),
        "NUMERIC_ESCAPES_ARE_BYTES": ("_scan_numeric_escape", "_read_numeric_escape"),
        "DROP_UNKNOWN_ESCAPES": ("drop_unknown_escapes",),
        "LONE_SURROGATE_REPLACEMENT": ("lone_surrogate_replacement",),
        "NUMERIC_LITERALS": ("numeric_literals",),
        "VAR_SINGLE_TOKENS": ("var_single_tokens",),
        "STRING_ESCAPES_ALLOWED_IN_RAW_STRINGS": ("string_escapes_allowed_in_raw_strings",),
        "HEREDOC_TAG_IS_IDENTIFIER": ("heredoc_tag_is_identifier",),
        "HEREDOC_STRING_ALTERNATIVE": ("heredoc_string_alternative",),
        "COMMANDS": ("commands",),
        "COMMAND_PREFIX_TOKENS": ("command_prefix_tokens",),
        "NESTED_COMMENTS": ("nested_comments",),
        "DASH_COMMENT_REQUIRES_BOUNDARY": ("dash_comment_requires_boundary",),
        "COMMENTS_TERMINATE_AT_NEWLINE_ONLY": ("comments_terminate_at_newline_only",),
        "HINT_START": ("hint_start", "_COMMENTS"),
        "TOKENS_PRECEDING_HINT": ("tokens_preceding_hint",),
        "SHOW_PARSERS": ("SHOW_TRIE",),
        "SET_PARSERS": ("SET_TRIE",),
        "arg_types": ("required_args",),
    }
    for src_attr, derived_attrs in DERIVED_ATTR_MAP.items():
        for d_attr in derived_attrs:
            if d_attr in raw_symbol_cov:
                raw_symbol_cov[src_attr].update(raw_symbol_cov[d_attr])
            for d_name, d_syms in raw_dialect_symbol_cov.items():
                if d_attr in d_syms:
                    d_syms[src_attr].update(d_syms[d_attr])

    # Compute transitive file import coverage: file_rel -> set of test numbers
    tests_by_test_file: dict[str, set[int]] = defaultdict(set)
    for t in tests_meta:
        tests_by_test_file[t["file"]].add(t["number"])

    # Reverse import graph: target_rel -> set of importer_rels
    rev_imports: dict[str, set[str]] = defaultdict(set)
    for importer, targets in direct_imports.items():
        for tgt in targets:
            rev_imports[tgt].add(importer)

    file_import_to_tests: dict[str, set[int]] = defaultdict(set)
    for rel in py_sources:
        # BFS in rev_imports
        visited = {rel}
        queue = [rel]
        while queue:
            curr = queue.pop()
            if curr in tests_by_test_file:
                file_import_to_tests[rel].update(tests_by_test_file[curr])
            for imp in rev_imports.get(curr, ()):
                if imp not in visited:
                    visited.add(imp)
                    queue.append(imp)
        # Also include any test that executed any line in rel or setupClass in rel
        for line_tests in line_to_tests.get(rel, {}).values():
            file_import_to_tests[rel].update(line_tests)
        for line_tests in setup_class_line_to_tests.get(rel, {}).values():
            file_import_to_tests[rel].update(line_tests)
        # If rel is a dialect file, include tests that used that dialect
        if rel.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
            stem = pathlib.Path(rel).stem
            if stem not in ("__init__", "dialect", "base"):
                file_import_to_tests[rel].update(dialect_to_tests.get(stem, set()))
        if rel.startswith("sqlglot/typing/"):
            file_import_to_tests[rel].update(typing_test_nums)
        if rel.startswith("sqlglot/") and not rel.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/", "sqlglot/optimizer/", "sqlglot/executor/")):
            file_import_to_tests[rel].update(lazy_load_nums)
        if rel in ("sqlglot/dialects/__init__.py", "sqlglot/dialects/dialect.py", "sqlglot/parsers/base.py"):
            file_import_to_tests[rel].update(lazy_load_nums)

    # Deduplicate all test sets into a table of sorted tuples
    test_sets_list: list[tuple[int, ...]] = [()]
    test_set_to_id: dict[tuple[int, ...], int] = {(): 0}

    def intern_set(s: set[int]) -> int:
        if not s:
            return 0
        tup = tuple(sorted(s))
        idx = test_set_to_id.get(tup)
        if idx is None:
            idx = len(test_sets_list)
            test_sets_list.append(tup)
            test_set_to_id[tup] = idx
        return idx

    compact_line_cov: dict[str, dict[int, int]] = {}
    for rel, lmap in line_to_tests.items():
        compact_line_cov[rel] = {line: intern_set(tset) for line, tset in lmap.items() if tset}

    compact_setup_cov: dict[str, dict[int, int]] = {}
    for rel, lmap in setup_class_line_to_tests.items():
        compact_setup_cov[rel] = {line: intern_set(tset) for line, tset in lmap.items() if tset}

    compact_fixture_cov = {rel: intern_set(tset) for rel, tset in fixture_to_tests.items() if tset}
    compact_dialect_cov = {d: intern_set(tset) for d, tset in dialect_to_tests.items() if tset}
    compact_expr_cov = {e: intern_set(tset) for e, tset in expr_to_tests.items() if tset}
    compact_dialect_expr_cov = {
        d: {e: intern_set(tset) for e, tset in emap.items() if tset}
        for d, emap in dialect_expr_to_tests.items()
    }
    compact_symbol_cov = {sym: intern_set(tset) for sym, tset in raw_symbol_cov.items() if tset}
    compact_dialect_symbol_cov = {
        d: {sym: intern_set(tset) for sym, tset in smap.items() if tset}
        for d, smap in raw_dialect_symbol_cov.items()
    }
    compact_file_import_cov = {rel: intern_set(tset) for rel, tset in file_import_to_tests.items() if tset}

    payload = {
        "test_ids": test_ids,
        "test_sets": test_sets_list,
        "line_cov": compact_line_cov,
        "setup_cov": compact_setup_cov,
        "import_lines": dict(import_lines),
        "fixture_cov": compact_fixture_cov,
        "dialect_cov": compact_dialect_cov,
        "expr_cov": compact_expr_cov,
        "dialect_expr_cov": compact_dialect_expr_cov,
        "symbol_cov": compact_symbol_cov,
        "dialect_symbol_cov": compact_dialect_symbol_cov,
        "file_import_cov": compact_file_import_cov,
        "sources": compressed_sources,
        "fixtures": compressed_fixtures,
    }

    raw_bytes = pickle.dumps(payload, protocol=5)
    compressed_payload = zlib.compress(raw_bytes, level=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(compressed_payload)
    try:
        os.chmod(output_path, 0o644)
    except Exception:
        pass

    elapsed = time.monotonic() - t0
    print(
        f"Saved {output_path} ({len(compressed_payload) / 1024:.1f} KB, "
        f"{len(test_sets_list)} unique test sets) in {elapsed:.1f}s",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build precomputed blast-radius coverage map.")
    parser.add_argument(
        "--sqlglot-dir",
        type=pathlib.Path,
        default=pathlib.Path("/sqlglot"),
        help="Path to sqlglot checkout (cloned if missing)",
    )
    parser.add_argument(
        "--tests-json",
        type=pathlib.Path,
        default=pathlib.Path("sample/tests.json"),
        help="Path to tests.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("coverage_map.bin"),
        help="Output path for the compact map file",
    )
    args = parser.parse_args()
    sqlglot_dir = ensure_sqlglot(args.sqlglot_dir)
    build_map(sqlglot_dir, args.tests_json, args.output)


if __name__ == "__main__":
    main()
