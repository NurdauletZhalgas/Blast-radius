# Multi-stage build: compile sqlglotc mypyc extensions in full python:3.13 builder
# (no apt-get needed; only PyPI + pinned Docker Hub images), then copy compiled
# /sqlglot_c and tiny runtime helpers into the pinned python:3.13-slim scorer image.
FROM python:3.13@sha256:9ffe84122ce74ebdffe2e0e6b946c9dd95d0bf3463f6eb1006b4dc6fefe36d01 AS builder
RUN pip install --no-cache-dir mypy==2.4.1 setuptools==84.2.1 librt==0.17.0 mypy_extensions==1.2.0 typing_extensions==4.16.0
COPY coverage_map.bin /app/coverage_map.bin
RUN python -c "import pathlib, pickle, zlib; m = pickle.loads(zlib.decompress(pathlib.Path('/app/coverage_map.bin').read_bytes())); r = pathlib.Path('/sqlglot_c'); [((r / k).parent.mkdir(parents=True, exist_ok=True), (r / k).write_bytes(zlib.decompress(v))) for d in (m.get('sources', {}), m.get('fixtures', {})) for k, v in d.items()]"
RUN python -c "import os, sys; from setuptools import setup; from mypyc.build import mypycify; os.chdir('/sqlglot_c'); sub = lambda p: sorted(f'sqlglot/{p}/{f}' for f in os.listdir(f'sqlglot/{p}') if f.endswith('.py') and f != '__init__.py'); files = ['--ignore-missing-imports','sqlglot/errors.py','sqlglot/generator.py','sqlglot/helper.py','sqlglot/parser.py','sqlglot/time.py','sqlglot/tokenizer_core.py','sqlglot/trie.py', *sub('expressions'), *sub('generators'), *sub('parsers'), 'sqlglot/optimizer/scope.py','sqlglot/optimizer/resolver.py','sqlglot/optimizer/isolate_table_selects.py','sqlglot/optimizer/normalize_identifiers.py','sqlglot/optimizer/qualify.py','sqlglot/optimizer/qualify_tables.py','sqlglot/optimizer/qualify_columns.py','sqlglot/optimizer/simplify.py','sqlglot/optimizer/annotate_types.py','sqlglot/optimizer/canonicalize_internal_names.py','sqlglot/executor/table.py']; os.environ['MYPYPATH']='/sqlglot_c'; sys.argv=['setup.py','build_ext','--inplace','-j','4']; setup(name='sqlglotc', ext_modules=mypycify(files, opt_level='1', debug_level='0', separate=True))" && rm -rf /sqlglot_c/build

# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image we labeled every commit in: its Python and SQLite
# decide which sqlglot tests run, so keep it if your image runs them.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY --from=builder /usr/local/lib/python3.13/site-packages/librt /usr/local/lib/python3.13/site-packages/librt
COPY --from=builder /usr/local/lib/python3.13/site-packages/mypy_extensions.py /usr/local/lib/python3.13/site-packages/mypy_extensions.py
COPY --from=builder /usr/local/lib/python3.13/site-packages/typing_extensions.py /usr/local/lib/python3.13/site-packages/typing_extensions.py
COPY --from=builder /sqlglot_c /sqlglot_c
COPY coverage_map.bin /app/coverage_map.bin
RUN python -c "import pathlib, pickle, zlib; m = pickle.loads(zlib.decompress(pathlib.Path('/app/coverage_map.bin').read_bytes())); r = pathlib.Path('/sqlglot'); [((r / k).parent.mkdir(parents=True, exist_ok=True), (r / k).write_bytes(zlib.decompress(v))) for d in (m.get('sources', {}), m.get('fixtures', {})) for k, v in d.items()]" && python -m compileall -q /sqlglot /sqlglot_c && chmod -R a+rX /sqlglot /sqlglot_c
COPY build_baselines.py /app/build_baselines.py
COPY selector.py /app/selector.py
RUN python /app/build_baselines.py && BLAST_SQLGLOT_DIR=/sqlglot_c python /app/build_baselines.py --compiled-timings && python -m compileall -q /app && chmod -R a+rX /app
CMD ["python", "/app/selector.py"]
