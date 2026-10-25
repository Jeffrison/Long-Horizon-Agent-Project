"""Check preserved experiment bytes against the complete artifact manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(directory):
    root=Path(directory).resolve()
    manifest=root/"artifact-sha256.json"
    expected=json.loads(manifest.read_text(encoding="utf-8"))
    for name,digest in expected.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid artifact path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p!=manifest}
    if actual!=set(expected):
        raise ValueError("Artifact manifest does not cover the exact file inventory")
    print(json.dumps({"verified_files":len(expected),"all_checksums_match":True},indent=2))
    return len(expected)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",nargs="?",default="reports")
    verify(parser.parse_args().directory)
