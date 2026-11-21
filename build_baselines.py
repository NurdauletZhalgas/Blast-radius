"""Build-time precomputation of deterministic baselines from the pinned sqlglot commit.

Run during Docker image build (not charged at scoring time). Produces /app/baselines.bin
containing:
- DuckDB baseline query recordings for TestOptimizer methods
- TPC-H parse & DuckDB-transpile baselines (for test_execute_tpch & test_optimized_tpch)
- TPC-DS parse, DuckDB-transpile, optimize, Plan & PythonExecutor codegen hashes
  (for sound independent verification of test_execute_tpcds)
- Dialect inheritance map (for lazy dialect loading during verification)
"""
from __future__ import annotations

import ast
import gc
import hashlib
import os
import pathlib
import pickle
import sys
import time
import zlib

APP_DIR = pathlib.Path(os.environ.get("BLAST_APP_DIR", pathlib.Path(__file__).resolve().parent))
SQLGLOT_DIR = pathlib.Path(os.environ.get("BLAST_SQLGLOT_DIR", "/sqlglot"))
OUT_PATH = pathlib.Path(os.environ.get("BLAST_BASELINES_PATH", APP_DIR / "baselines.bin"))


def plan_py_fingerprint(plan: object, py_exec: object) -> bytes:
    """Compute a deterministic, memory-address-free SHA-256 digest of Plan DAG + Python codegen."""
    py_exec._subquery_plans.clear()  # type: ignore[attr-defined]
    py_exec._plan_names_by_sql.clear()  # type: ignore[attr-defined]
    gen = py_exec.generator  # type: ignore[attr-defined]

    def gen_expr(expr: object) -> str:
        if not expr:
            return ""
        replaced = py_exec._replace_subqueries(expr)  # type: ignore[attr-defined]
        return gen.generate(replaced)

    def hash_str(h: hashlib._Hash, s: str) -> None:
        b = s.encode("utf-8")
        h.update(len(b).to_bytes(4, "little"))
        h.update(b)

    memo: dict[int, bytes] = {}

    def step_fp(st: object) -> bytes:
        sid = id(st)
        if sid in memo:
            return memo[sid]
        h = hashlib.sha256()
        src = getattr(st, "source", None)
        src_s = src.sql() if hasattr(src, "sql") else str(src)
        op = getattr(st, "op", None)
        op_s = op.__name__ if isinstance(op, type) else str(op)
        hash_str(
            h,
            f"{type(st).__name__}|{getattr(st, 'name', '')}|{src_s}|"
            f"{getattr(st, 'source_name', '')}|{getattr(st, 'left', '')}|{getattr(st, 'right', '')}|"
            f"{getattr(st, 'limit', '')}|{getattr(st, 'offset', '')}|"
            f"{getattr(st, 'distinct', '')}|{op_s}",
        )
        cond = getattr(st, "condition", None)
        if cond is not None:
            hash_str(h, "C:" + gen_expr(cond))
        for proj in getattr(st, "projections", ()):
            hash_str(h, "P:" + gen_expr(proj))
        joins = getattr(st, "joins", None)
        if joins:
            for jname in sorted(joins):
                j = joins[jname]
                hash_str(h, f"J:{jname}|{j.get('side')}|{j.get('method')}")
                if j.get("condition") is not None:
                    hash_str(h, "JC:" + gen_expr(j["condition"]))
                for k in j.get("source_key", ()):
                    hash_str(h, "JSK:" + gen_expr(k))
                for k in j.get("join_key", ()):
                    hash_str(h, "JJK:" + gen_expr(k))
        grp = getattr(st, "group", None)
        if grp:
            for gk in grp:
                hash_str(h, "GK:" + str(gk))
                hash_str(h, "GV:" + gen_expr(grp[gk]))
        for agg in getattr(st, "aggregations", ()):
            hash_str(h, "A:" + gen_expr(agg))
        for opnd in getattr(st, "operands", ()):
            hash_str(h, "O:" + gen_expr(opnd))
        for sk in getattr(st, "key", ()):
            hash_str(h, "S:" + gen_expr(sk))
        dep_fps = sorted(step_fp(dep) for dep in getattr(st, "dependencies", ()))
        for dfp in dep_fps:
            h.update(dfp)
        digest = h.digest()
        memo[sid] = digest
        return digest

    root_h = hashlib.sha256()
    root_h.update(step_fp(plan.root))  # type: ignore[attr-defined]
    processed_subplans: set[str] = set()
    while True:
        new_names = sorted(k for k in py_exec._subquery_plans if k not in processed_subplans)
        if not new_names:
            break
        for sq_name in new_names:
            processed_subplans.add(sq_name)
            sq_plan, _ = py_exec._subquery_plans[sq_name]
            hash_str(root_h, "SQ:" + sq_name)
            root_h.update(step_fp(sq_plan.root))
    return root_h.digest()


def main() -> None:
    t0 = time.monotonic()
    sys.path.insert(0, str(APP_DIR))
    sys.path.insert(0, str(SQLGLOT_DIR))
    os.chdir(SQLGLOT_DIR)

    import selector

    cov_map = pickle.loads(zlib.decompress((APP_DIR / "coverage_map.bin").read_bytes()))
    # Build minimal test_by_num from cov_map["test_ids"]
    test_by_num = {
        idx: {"number": idx, "id": tid, "file": tid.split("::")[0]}
        for idx, tid in enumerate(cov_map["test_ids"])
    }

    # 1. Discover DuckDB TestOptimizer methods from source AST
    duckdb_opt_methods: set[str] = set()
    opt_comp = cov_map.get("sources", {}).get("tests/test_optimizer.py")
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
                            duckdb_opt_methods.add(fn.name)

    # 2. Initialize InProcessVerifier (without prebuilt baselines) and record DuckDB baselines
    os.environ["BLAST_DISABLE_PREBUILT_BASELINES"] = "1"
    verifier = selector.InProcessVerifier(cov_map, test_by_num)
    verifier._ensure_initialized()

    duckdb_nums = [
        num
        for num, t in test_by_num.items()
        if t["id"].startswith("tests/test_optimizer.py::TestOptimizer::")
        and t["id"].split("::")[2] in duckdb_opt_methods
    ]
    verifier._record_baseline_if_needed(duckdb_nums)
    b_queries = verifier.conn_cls.baseline_queries  # type: ignore[union-attr]
    b_expects = verifier.conn_cls.baseline_expects  # type: ignore[union-attr]
    duckdb_baselines: dict[str, list[str]] = {}
    duckdb_expects: dict[str, dict[int, tuple[str, object]]] = {}
    for num in duckdb_nums:
        mname = test_by_num[num]["id"].split("::")[2]
        if num in b_queries:
            duckdb_baselines[mname] = list(b_queries[num])
        if num in b_expects:
            duckdb_expects[mname] = dict(b_expects[num])

    # 3. Dialect subclass map (stem -> list of descendant dialect module names)
    from sqlglot.dialects.dialect import Dialect

    _ = Dialect.classes
    dialect_subclasses: dict[str, list[str]] = {}
    all_dialects = list(set(Dialect.classes.values()))
    for d_cls in all_dialects:
        d_mod = getattr(d_cls, "__module__", "")
        if not d_mod.startswith("sqlglot.dialects.") or d_mod == "sqlglot.dialects.dialect":
            continue
        for base in d_cls.__mro__[1:]:
            b_mod = getattr(base, "__module__", "")
            if b_mod.startswith("sqlglot.dialects.") and b_mod != "sqlglot.dialects.dialect":
                b_stem = b_mod.rsplit(".", 1)[-1]
                lst = dialect_subclasses.setdefault(b_stem, [])
                if d_mod not in lst:
                    lst.append(d_mod)
    for k in dialect_subclasses:
        dialect_subclasses[k].sort()

    # 4. TPC-H baselines (for test_execute_tpch & test_optimized_tpch)
    from sqlglot import exp, find_tables, parse_one, transpile
    from sqlglot.executor.python import PythonExecutor
    from sqlglot.generator import Generator
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import MappingSchema
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    tpch_raw = list(load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
    tpch_pairs = [(sql, opt) for _, sql, opt in tpch_raw]
    tpch_parse_baseline = [parse_one(sql).sql() for sql, _ in tpch_pairs]
    tpch_transpile_baseline = [
        (transpile(sql, write="duckdb")[0], transpile(opt, write="duckdb")[0])
        for sql, opt in tpch_pairs
    ]

    # 5. TPC-DS baselines + per-query line coverage (using sys.monitoring)
    root_prefix = str(SQLGLOT_DIR.resolve()).replace("\\", "/").lower()
    if not root_prefix.endswith("/"):
        root_prefix += "/"
    code_to_rel: dict[object, str] = {}

    def _get_rel(code: object) -> str:
        r = code_to_rel.get(code)
        if r is None:
            fn = getattr(code, "co_filename", "").replace("\\", "/")
            fl = fn.lower()
            if fl.startswith(root_prefix + "sqlglot/"):
                r = fn[len(root_prefix) :]
            else:
                r = ""
            code_to_rel[code] = r
        return r

    mon = sys.monitoring
    TOOL_ID = 2
    DISABLE = mon.DISABLE
    mon.use_tool_id(TOOL_ID, "qcov_build")
    cur_lines: set[tuple[str, int]] = set()

    def _on_line(code: object, line: int) -> object:
        r = _get_rel(code)
        if r:
            cur_lines.add((r, line))
        return DISABLE

    mon.register_callback(TOOL_ID, mon.events.LINE, _on_line)

    qmask_to_id: dict[int, int] = {0: 0}
    qmask_table: list[int] = [0]

    def _compress_qmap(raw: dict[tuple[str, int], int]) -> dict[str, dict[int, int]]:
        out: dict[str, dict[int, int]] = {}
        for (rel, lno), mask in raw.items():
            mid = qmask_to_id.get(mask)
            if mid is None:
                mid = len(qmask_table)
                qmask_to_id[mask] = mid
                qmask_table.append(mask)
            out.setdefault(rel, {})[lno] = mid
        return out

    tpcds_raw = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    tpcds_sqls = [sql for _, sql, _ in tpcds_raw]
    tpcds_parse_baseline = [parse_one(sql).sql() for sql in tpcds_sqls]
    tpcds_exec_indices = sorted(
        [i for i, (m, _, _) in enumerate(tpcds_raw) if m.get("execute")],
        key=lambda i: (len(tpcds_sqls[i]), i),
    )
    tpcds_exec_set = set(tpcds_exec_indices)

    mon.set_events(TOOL_ID, mon.events.LINE)
    cur_lines.clear()
    mon.restart_events()
    ms0 = MappingSchema(TPCDS_SCHEMA)
    tpcds_schema_lines: set[tuple[str, int]] = set(cur_lines)

    py_exec0 = PythonExecutor()
    tpcds_exec_fingerprints: dict[int, tuple[str, tuple[str, ...], str, bytes]] = {}
    raw_tpcds_exec_qcov: dict[tuple[str, int], int] = {}
    for idx in tpcds_exec_indices:
        sql = tpcds_sqls[idx]
        cur_lines.clear()
        mon.restart_events()
        expr = parse_one(sql)
        duck_sql = expr.sql(dialect="duckdb")
        tbls = tuple(sorted(t.name for t in find_tables(expr)))
        opt_expr = optimize(expr, ms0, leave_tables_isolated=True)
        pl = Plan(opt_expr)
        fp = plan_py_fingerprint(pl, py_exec0)
        bit = 1 << idx
        for k in cur_lines:
            raw_tpcds_exec_qcov[k] = raw_tpcds_exec_qcov.get(k, 0) | bit
        tpcds_exec_fingerprints[idx] = (duck_sql, tbls, opt_expr.sql(), fp)

    raw_tpcds_parse_qcov: dict[tuple[str, int], int] = {}
    for idx, sql in enumerate(tpcds_sqls):
        if idx in tpcds_exec_set:
            continue
        cur_lines.clear()
        mon.restart_events()
        _ = parse_one(sql).sql()
        bit = 1 << idx
        for k in cur_lines:
            raw_tpcds_parse_qcov[k] = raw_tpcds_parse_qcov.get(k, 0) | bit

    mon.set_events(TOOL_ID, 0)

    # 5b. Trace per-query line coverage for all major TestOptimizer fixture tests
    import tests.test_optimizer as mod_opt

    verifier._ensure_test_module_patched("tests.test_optimizer", mod_opt)
    fixture_qcov_raw: dict[str, dict[tuple[str, int], int]] = {}
    fixture_counts: dict[str, int] = {}
    fixture_schema_lines: dict[str, set[tuple[str, int]]] = {}
    fixture_method_files: dict[str, list[str]] = {}

    FIXTURE_METHODS = (
        "test_tpcds",
        "test_annotate_funcs",
        "test_qualify_columns",
        "test_simplify",
        "test_optimize",
        "test_tpch",
        "test_merge_subqueries",
        "test_annotate_types",
        "test_unnest_subqueries",
        "test_pushdown_predicates",
        "test_pushdown_projection",
        "test_eliminate_joins",
        "test_eliminate_subqueries",
        "test_normalize",
        "test_canonicalize",
        "test_isolate_table_selects",
        "test_qualify_tables",
        "test_optimize_joins",
        "test_eliminate_ctes",
    )

    orig_check_file_patched = mod_opt.TestOptimizer.check_file
    orig_load_pairs = mod_opt.load_sql_fixture_pairs
    active_method: list[str] = [""]

    def _traced_check_file(opt_self, file, func, pretty=False, execute=None, only=None, **kwargs):
        fname = f"optimizer/{file}.sql"
        if only is not None:
            return orig_check_file_patched(
                opt_self, file, func, pretty=pretty, execute=False, only=only, **kwargs
            )
        mname = active_method[0]
        if mname:
            flist = fixture_method_files.setdefault(mname, [])
            if fname not in flist:
                flist.append(fname)
        raw_map = fixture_qcov_raw.setdefault(fname, {})
        schema_set = fixture_schema_lines.setdefault(fname, set())
        pairs_list = list(orig_load_pairs(fname))
        fixture_counts[fname] = len(pairs_list)

        if file in ("tpc-ds/tpc-ds", "tpc-h/tpc-h") and isinstance(kwargs.get("schema"), dict):
            mon.set_events(TOOL_ID, mon.events.LINE)
            cur_lines.clear()
            mon.restart_events()
            kwargs["schema"] = mod_opt.MappingSchema(kwargs["schema"])
            schema_set.update(cur_lines)
            mon.set_events(TOOL_ID, 0)

        import json as _json
        from tests.helpers import string_to_bool as _s2b

        mon.set_events(TOOL_ID, mon.events.LINE)
        for q_idx, (meta, sql, expected) in enumerate(pairs_list):
            dialect = meta.get("dialect")
            leave_tables_isolated = meta.get("leave_tables_isolated")
            validate_qualify_columns = meta.get("validate_qualify_columns")
            canonicalize_table_aliases = meta.get("canonicalize_table_aliases")
            func_kwargs = kwargs.copy()
            if schema_meta := meta.get("schema"):
                func_kwargs["schema"] = _json.loads(schema_meta)
            if leave_tables_isolated is not None:
                func_kwargs["leave_tables_isolated"] = _s2b(leave_tables_isolated)
            if validate_qualify_columns is not None:
                func_kwargs["validate_qualify_columns"] = _s2b(validate_qualify_columns)
            if dialect:
                func_kwargs["dialect"] = dialect
            if canonicalize_table_aliases is not None:
                func_kwargs["canonicalize_table_aliases"] = _s2b(canonicalize_table_aliases)

            cur_lines.clear()
            mon.restart_events()
            optimized = mod_opt.parse_and_optimize(func, sql, dialect, **func_kwargs)
            actual = optimized.sql(pretty=pretty, dialect=dialect)
            bit = 1 << q_idx
            for k in cur_lines:
                raw_map[k] = raw_map.get(k, 0) | bit
            opt_self.assertEqual(expected, actual)
        mon.set_events(TOOL_ID, 0)

    def _traced_load_pairs(filename):
        mname = active_method[0]
        if mname not in ("test_annotate_funcs", "test_annotate_types"):
            yield from orig_load_pairs(filename)
            return
        flist = fixture_method_files.setdefault(mname, [])
        if filename not in flist:
            flist.append(filename)
        raw_map = fixture_qcov_raw.setdefault(filename, {})
        pairs_list = list(orig_load_pairs(filename))
        fixture_counts[filename] = len(pairs_list)
        mon.set_events(TOOL_ID, mon.events.LINE)
        try:
            for q_idx, item in enumerate(pairs_list):
                cur_lines.clear()
                mon.restart_events()
                yield item
                bit = 1 << q_idx
                for k in cur_lines:
                    raw_map[k] = raw_map.get(k, 0) | bit
        finally:
            mon.set_events(TOOL_ID, 0)

    mod_opt.TestOptimizer.check_file = _traced_check_file  # type: ignore
    mod_opt.load_sql_fixture_pairs = _traced_load_pairs  # type: ignore
    try:
        opt_num_by_method = {
            t["id"].split("::")[2]: num
            for num, t in test_by_num.items()
            if t["id"].startswith("tests/test_optimizer.py::TestOptimizer::")
        }
        for mname in FIXTURE_METHODS:
            num = opt_num_by_method.get(mname)
            if num is None:
                continue
            active_method[0] = mname
            st = verifier._run_test_status(num)
            if st == "FAIL":
                raise RuntimeError(f"Fixture trace failed for {mname}")
    finally:
        active_method[0] = ""
        mod_opt.TestOptimizer.check_file = orig_check_file_patched  # type: ignore
        mod_opt.load_sql_fixture_pairs = orig_load_pairs  # type: ignore
        mon.set_events(TOOL_ID, 0)
        mon.free_tool_id(TOOL_ID)

    tpcds_exec_qcov = _compress_qmap(raw_tpcds_exec_qcov)
    tpcds_parse_qcov = _compress_qmap(raw_tpcds_parse_qcov)
    fixture_qcov = {fname: _compress_qmap(rmap) for fname, rmap in fixture_qcov_raw.items()}

    env_invariants = (
        dict(Generator.TIME_PART_SINGULARS),
        frozenset(t.name for t in exp.DataType.TEXT_TYPES),
        frozenset(t.name for t in exp.DataType.NUMERIC_TYPES),
    )

    # 6. Measure exact in-process CPU time of every runnable test on this container
    gc.collect()
    gc.freeze()
    inproc_ms_by_id: dict[str, int] = {}
    clean_failing_ids: set[str] = set()
    for num, t_info in test_by_num.items():
        if not verifier._is_runnable_fast(num, 0):
            continue
        t_start = time.process_time()
        st = verifier._run_test_status(num)
        dt_ms = max(1, int(round((time.process_time() - t_start) * 1000)))
        tid = t_info["id"]
        if st == "FAIL":
            clean_failing_ids.add(tid)
        else:
            inproc_ms_by_id[tid] = dt_ms

    verifier.tpcds_sqls = tpcds_sqls
    verifier.tpcds_parse_baseline = tpcds_parse_baseline
    verifier.tpcds_exec_indices = tpcds_exec_indices
    verifier.tpcds_exec_fingerprints = tpcds_exec_fingerprints
    verifier.env_invariants = env_invariants
    t_tpcds = time.process_time()
    ok_tpcds = verifier._verify_execute_tpcds([])
    tpcds_exec_ms = max(100, int(round((time.process_time() - t_tpcds) * 1000)))
    if not ok_tpcds:
        raise RuntimeError("Clean _verify_execute_tpcds failed during build_baselines")

    data = {
        "duckdb_opt_methods": duckdb_opt_methods,
        "duckdb_baselines": duckdb_baselines,
        "duckdb_expects": duckdb_expects,
        "dialect_subclasses": dialect_subclasses,
        "tpch_pairs": tpch_pairs,
        "tpch_parse_baseline": tpch_parse_baseline,
        "tpch_transpile_baseline": tpch_transpile_baseline,
        "tpcds_sqls": tpcds_sqls,
        "tpcds_parse_baseline": tpcds_parse_baseline,
        "tpcds_exec_indices": tpcds_exec_indices,
        "tpcds_exec_fingerprints": tpcds_exec_fingerprints,
        "tpcds_schema_lines": tpcds_schema_lines,
        "tpcds_exec_qcov": tpcds_exec_qcov,
        "tpcds_parse_qcov": tpcds_parse_qcov,
        "qmask_table": qmask_table,
        "fixture_qcov": fixture_qcov,
        "fixture_counts": fixture_counts,
        "fixture_schema_lines": fixture_schema_lines,
        "fixture_method_files": fixture_method_files,
        "env_invariants": env_invariants,
        "inproc_ms_by_id": inproc_ms_by_id,
        "clean_failing_ids": clean_failing_ids,
        "tpcds_exec_ms": tpcds_exec_ms,
    }
    raw_bytes = zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), level=9)
    OUT_PATH.write_bytes(raw_bytes)
    print(
        f"Wrote {OUT_PATH} ({len(raw_bytes)} bytes) in {time.monotonic() - t0:.2f}s: "
        f"duckdb={len(duckdb_baselines)}/{len(duckdb_opt_methods)}, "
        f"tpch={len(tpch_pairs)}, tpcds_exec={len(tpcds_exec_indices)}/{len(tpcds_sqls)} ({tpcds_exec_ms}ms), "
        f"qcov_files={len(fixture_qcov)}, qmasks={len(qmask_table)}, "
        f"dialect_parents={len(dialect_subclasses)}, "
        f"timed={len(inproc_ms_by_id)}, clean_fails={len(clean_failing_ids)}",
        file=sys.stderr,
    )


def add_compiled_timings() -> None:
    """Measure exact in-process CPU timings and compiled-mode skips under /sqlglot_c
    and merge them into existing baselines.bin."""
    t0 = time.monotonic()
    sys.path.insert(0, str(APP_DIR))
    sys.path.insert(0, str(SQLGLOT_DIR))
    os.chdir(SQLGLOT_DIR)

    import selector

    cov_map = pickle.loads(zlib.decompress((APP_DIR / "coverage_map.bin").read_bytes()))
    data = pickle.loads(zlib.decompress(OUT_PATH.read_bytes()))
    test_by_num = {
        idx: {"number": idx, "id": tid, "file": tid.split("::")[0]}
        for idx, tid in enumerate(cov_map["test_ids"])
    }

    compiled_files: set[str] = set()
    for p in (SQLGLOT_DIR / "sqlglot").rglob("*.so"):
        stem = p.name.split(".")[0]
        if stem.endswith("__mypyc"):
            continue
        rel = p.relative_to(SQLGLOT_DIR).with_name(stem + ".py").as_posix()
        compiled_files.add(rel)

    verifier = selector.InProcessVerifier(cov_map, test_by_num)
    verifier._ensure_initialized()
    gc.collect()
    gc.freeze()

    inproc_ms_c_by_id: dict[str, int] = {}
    clean_failing_c_ids: set[str] = set()
    compiled_skipped_ids: set[str] = set()
    for num, t_info in test_by_num.items():
        if not verifier._is_runnable_fast(num, 0):
            continue
        t_start = time.process_time()
        st = verifier._run_test_status(num)
        dt_ms = max(1, int(round((time.process_time() - t_start) * 1000)))
        tid = t_info["id"]
        if st == "FAIL":
            clean_failing_c_ids.add(tid)
        elif st == "SKIP":
            compiled_skipped_ids.add(tid)
            inproc_ms_c_by_id[tid] = dt_ms
        else:
            inproc_ms_c_by_id[tid] = dt_ms

    t_tpcds = time.process_time()
    ok_tpcds = verifier._verify_execute_tpcds([])
    tpcds_exec_c_ms = max(100, int(round((time.process_time() - t_tpcds) * 1000)))
    if not ok_tpcds:
        raise RuntimeError("Clean compiled _verify_execute_tpcds failed during build_baselines")

    data["compiled_files"] = compiled_files
    data["inproc_ms_c_by_id"] = inproc_ms_c_by_id
    data["clean_failing_c_ids"] = clean_failing_c_ids
    data["compiled_skipped_ids"] = compiled_skipped_ids
    data["tpcds_exec_c_ms"] = tpcds_exec_c_ms

    raw_bytes = zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), level=9)
    OUT_PATH.write_bytes(raw_bytes)
    print(
        f"Updated {OUT_PATH} with compiled timings ({len(raw_bytes)} bytes) in {time.monotonic() - t0:.2f}s: "
        f"compiled_so={len(compiled_files)}, timed_c={len(inproc_ms_c_by_id)}, "
        f"tpcds_exec_c={tpcds_exec_c_ms}ms, skipped_c={len(compiled_skipped_ids)}, "
        f"clean_fails_c={len(clean_failing_c_ids)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    if "--compiled-timings" in sys.argv:
        add_compiled_timings()
    else:
        main()
