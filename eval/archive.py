"""Archive completed runs with raw evidence, hashes, and a local environment snapshot."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

from eval.report import write_report
from eval.verify_run import verify


def archive(source, destination):
    source, destination = Path(source), Path(destination)
    summary = write_report(source)
    if not summary["complete"]:
        raise ValueError("Formal archive requires a complete run")
    verify(source)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", "summary.json", "report.md"):
        shutil.copyfile(source/name, destination/name)
    for name in ("verification.json", "failure-analysis.json", "failure-analysis.md",
                 "scoring-differences.json", "accuracy.png", "accuracy.svg"):
        if (source/name).exists():
            shutil.copyfile(source/name, destination/name)
    for name in ("episodes.jsonl", "started.jsonl"):
        with (destination/(name+".gz")).open("wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as compressed:
                compressed.write((source/name).read_bytes())
    packages = {}
    for package in ("requests", "python-dotenv", "sympy", "numpy", "scipy", "pytest"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": packages,
                   "note": "Observed local versions, not a claim that optional GPU training dependencies were installed."}
    (destination/"environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(destination.iterdir()) if path.is_file()}
    (destination/"sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps({"archive":str(destination), "files":len(hashes),
                      "episodes":summary["completed_episodes"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive(args.source, args.output)
