# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image we labeled every commit in: its Python and SQLite
# decide which sqlglot tests run, so keep it if your image runs them.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY coverage_map.bin /app/coverage_map.bin
COPY selector.py /app/selector.py
RUN python -c "import pathlib, pickle, zlib; m = pickle.loads(zlib.decompress(pathlib.Path('/app/coverage_map.bin').read_bytes())); r = pathlib.Path('/sqlglot'); [((r / k).parent.mkdir(parents=True, exist_ok=True), (r / k).write_bytes(zlib.decompress(v))) for d in (m.get('sources', {}), m.get('fixtures', {})) for k, v in d.items()]" && chmod -R a+rX /app /sqlglot
CMD ["python", "/app/selector.py"]
