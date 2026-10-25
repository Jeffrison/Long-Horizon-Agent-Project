"""Source fingerprints independent of Git working-tree cleanliness."""
import hashlib
from pathlib import Path


def hash_sources(paths,root):
    root=Path(root)
    digest=hashlib.sha256()
    for path in sorted(set(Path(p) for p in paths)):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        # Ignore checkout line-ending conversion, retain all semantic source text.
        digest.update(path.read_text(encoding="utf-8").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def source_fingerprint():
    root=Path(__file__).resolve().parents[1]
    paths=[p for folder in ("agent","tools") for p in (root/folder).rglob("*.py")]
    paths += [root/"eval"/name for name in
              ("data_loader.py","metrics.py","exact_math.py","scoring.py","run_eval.py","report.py","provenance.py")]
    return hash_sources(paths,root)
