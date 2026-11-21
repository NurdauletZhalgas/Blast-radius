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

    # 5. TPC-DS baselines (for sound independent verification of test_execute_tpcds)
    tpcds_raw = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    tpcds_sqls = [sql for _, sql, _ in tpcds_raw]
    tpcds_parse_baseline = [parse_one(sql).sql() for sql in tpcds_sqls]
    tpcds_exec_indices = sorted(
        [i for i, (m, _, _) in enumerate(tpcds_raw) if m.get("execute")],
        key=lambda i: (len(tpcds_sqls[i]), i),
    )
    ms0 = MappingSchema(TPCDS_SCHEMA)
    py_exec0 = PythonExecutor()
    tpcds_exec_fingerprints: dict[int, tuple[str, tuple[str, ...], str, bytes]] = {}
    for idx in tpcds_exec_indices:
        sql = tpcds_sqls[idx]
        expr = parse_one(sql)
        duck_sql = expr.sql(dialect="duckdb")
        tbls = tuple(sorted(t.name for t in find_tables(expr)))
        opt_expr = optimize(expr, ms0, leave_tables_isolated=True)
        pl = Plan(opt_expr)
        fp = plan_py_fingerprint(pl, py_exec0)
        tpcds_exec_fingerprints[idx] = (duck_sql, tbls, opt_expr.sql(), fp)

    env_invariants = (
        dict(Generator.TIME_PART_SINGULARS),
        frozenset(t.name for t in exp.DataType.TEXT_TYPES),
        frozenset(t.name for t in exp.DataType.NUMERIC_TYPES),
    )

    # 6. Measure exact in-process CPU time of every runnable test on this container
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
        "env_invariants": env_invariants,
        "inproc_ms_by_id": inproc_ms_by_id,
        "clean_failing_ids": clean_failing_ids,
    }
    raw_bytes = zlib.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), level=9)
    OUT_PATH.write_bytes(raw_bytes)
    print(
        f"Wrote {OUT_PATH} ({len(raw_bytes)} bytes) in {time.monotonic() - t0:.2f}s: "
        f"duckdb={len(duckdb_baselines)}/{len(duckdb_opt_methods)}, "
        f"tpch={len(tpch_pairs)}, tpcds_exec={len(tpcds_exec_indices)}/{len(tpcds_sqls)}, "
        f"dialect_parents={len(dialect_subclasses)}, "
        f"timed={len(inproc_ms_by_id)}, clean_fails={len(clean_failing_ids)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
