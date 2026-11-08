"""Fetch pinned Solidity dependencies into lib/ (gitignored) without git or global writes.

reactive-lib-omni declares UNLICENSED upstream, so it is not committed to this repository;
each builder downloads it from the official repo at the pinned commit.
"""
from pathlib import Path
import io
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEPS = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))

for name, (repo, commit) in DEPS.items():
    target = ROOT / "lib" / name
    if target.exists():
        continue
    raw = urllib.request.urlopen(f"https://codeload.github.com/{repo}/zip/{commit}").read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for item in archive.infolist():
            rel = Path(*Path(item.filename).parts[1:])
            if not rel.parts or item.is_dir():
                continue
            out = (target / rel).resolve()
            if not out.is_relative_to(target.resolve()):
                raise ValueError("Unsafe archive path")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(item))
print("Pinned dependencies ready")
