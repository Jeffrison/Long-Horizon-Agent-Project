"""Download a fixed Qwen checkpoint from its official HF/ModelScope publishers."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
FILES = {
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["hf", "modelscope"], default="modelscope")
    parser.add_argument("--output", default="outputs/models/qwen2.5-0.5b-instruct")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base = (f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/{REVISION}/"
            if args.source == "hf" else
            "https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/resolve/master/")
    for filename, expected in FILES.items():
        target = output / filename
        if target.exists() and file_hash(target) == expected:
            print("verified existing", filename, flush=True)
            continue
        part = output / (filename + ".part")
        digest, total = hashlib.sha256(), 0
        with urlopen(Request(base + filename, headers={"Range": "bytes=0-"}), timeout=30) as response, part.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if total % (64 * 1024 * 1024) == 0:
                    print(filename, total // (1024 * 1024), "MiB", flush=True)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"{filename}: checksum mismatch; .part was not activated")
        part.replace(target)
        print("verified downloaded", filename, flush=True)
    (output / "source_manifest.json").write_text(json.dumps({
        "repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": REVISION,
        "source": args.source, "sha256": FILES}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
