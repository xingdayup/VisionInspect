"""Explicit, bounded asset preparation; never used implicitly by model execution."""

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visioninspect.common import read_json, sha256, write_json

MODEL_REVISION = "491b427b45c94c7fb0e78b5474cc919aff584bbf"


def fetch(url, path, expected_size, expected_sha=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        path.exists()
        and path.stat().st_size == expected_size
        and (not expected_sha or sha256(path) == expected_sha)
    ):
        return dict(
            path=path.relative_to(ROOT).as_posix(),
            bytes=expected_size,
            sha256=sha256(path),
            reused=True,
        )
    if path.exists():
        raise ValueError(
            f"Existing asset does not match expected metadata; will not overwrite: {path}"
        )
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            with (
                urllib.request.urlopen(url, timeout=30) as response,
                partial.open("wb") as target,
            ):
                length = int(response.headers.get("Content-Length", expected_size))
                if length != expected_size:
                    raise ValueError(f"Length mismatch: {url}")
                total = 0
                while block := response.read(256 * 1024):
                    total += len(block)
                    if total > expected_size:
                        raise ValueError(f"Download exceeded preflight size: {url}")
                    target.write(block)
            if total != expected_size or (
                expected_sha and sha256(partial) != expected_sha
            ):
                raise ValueError(f"Size/hash mismatch: {url}")
            partial.replace(path)
            return dict(
                path=path.relative_to(ROOT).as_posix(),
                bytes=total,
                sha256=sha256(path),
                reused=False,
            )
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def weights():
    url = f"https://huggingface.co/timm/resnet18.a1_in1k/resolve/{MODEL_REVISION}/model.safetensors"
    with urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD"), timeout=30
    ) as response:
        size = int(response.headers["Content-Length"])
        checksum = response.headers.get("x-linked-etag", "").strip('"')
    if size != 46807446:
        raise ValueError("Weight size changed from approved budget")
    record = fetch(
        url,
        ROOT / "weights/resnet18.a1_in1k/model.safetensors",
        size,
        checksum if len(checksum) == 64 else None,
    )
    write_json(
        ROOT / "weights/resnet18.a1_in1k/source.json",
        dict(
            model="timm/resnet18.a1_in1k",
            revision=MODEL_REVISION,
            license="Apache-2.0",
            url=url,
            **record,
        ),
    )
    print(json.dumps(record))


def bottle():
    metadata = read_json(ROOT / "logs/mirror.txt")
    revision = metadata["sha"]
    if metadata.get("cardData", {}).get("license") != "cc-by-nc-sa-4.0":
        raise ValueError("Mirror dataset license does not match official license")
    license_text = (ROOT / "logs/mirror-license.txt").read_text(encoding="utf-8")
    if "Attribution-NonCommercial-ShareAlike" not in license_text:
        raise ValueError("Mirror license file does not match official license")
    tree_url = f"https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/{revision}/bottle?recursive=true&limit=1000"
    with urllib.request.urlopen(tree_url, timeout=30) as response:
        if response.headers.get("Link"):
            raise ValueError(
                "Unexpected pagination; must enumerate all files before download"
            )
        tree = json.load(response)
    files = [entry for entry in tree if entry["type"] == "file"]
    total = sum(entry["size"] for entry in files)
    if total > 1000000000 or len(files) != 357:
        raise ValueError(f"Unexpected mirror budget/file count: {total}, {len(files)}")
    if any(
        not entry["path"].startswith("bottle/") or ".." in Path(entry["path"]).parts
        for entry in files
    ):
        raise ValueError("Unsafe or non-bottle download path")
    preflight = dict(
        provider="User-authorized third-party mirror; not officially hosted",
        repository="foersben/mvtec-ad",
        revision=revision,
        license="CC-BY-NC-SA-4.0",
        official_source="https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        total_bytes=total,
        total_files=len(files),
        tree=tree,
    )
    write_json(ROOT / "logs/bottle-download-preflight.json", preflight)

    def download(entry):
        url = f"https://huggingface.co/datasets/foersben/mvtec-ad/resolve/{revision}/{entry['path']}"
        checksum = entry.get("lfs", {}).get("oid")
        record = fetch(
            url, ROOT / "data/mvtec_ad" / entry["path"], entry["size"], checksum
        )
        if not checksum:
            body = (ROOT / "data/mvtec_ad" / entry["path"]).read_bytes()
            blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
            if blob != entry["oid"]:
                raise ValueError(f"Git blob checksum mismatch: {entry['path']}")
        return dict(source_path=entry["path"], git_blob_oid=entry.get("oid"), **record)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for index, result in enumerate(pool.map(download, files)):
            results.append(result)
            if (index + 1) % 25 == 0:
                print(f"downloaded/verified {index + 1}/{len(files)}", flush=True)
    write_json(
        ROOT / "data/mvtec_ad/source.json",
        dict(**{k: v for k, v in preflight.items() if k != "tree"}, files=results),
    )
    print(f"BOTTLE_READY files={len(files)} bytes={total}", flush=True)


def preflight():
    """Fetch only metadata/attribution for the explicitly user-authorized mirror."""
    with urllib.request.urlopen(
        "https://huggingface.co/api/datasets/foersben/mvtec-ad", timeout=30
    ) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    if metadata.get("cardData", {}).get("license") != "cc-by-nc-sa-4.0":
        raise ValueError("Dataset card license changed; review required")
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "logs/mirror.txt", metadata)
    resources = {
        "bottle-tree": f"https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/{revision}/bottle?recursive=true&limit=1000",
        "mirror-card": f"https://huggingface.co/datasets/foersben/mvtec-ad/raw/{revision}/README.md",
        "mirror-license": f"https://huggingface.co/datasets/foersben/mvtec-ad/raw/{revision}/bottle/license.txt",
        "mirror-readme": f"https://huggingface.co/datasets/foersben/mvtec-ad/raw/{revision}/bottle/readme.txt",
    }
    for name, url in resources.items():
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read().decode("utf-8")
        (ROOT / "logs" / f"{name}.txt").write_text(body, encoding="utf-8")
    files = [
        entry
        for entry in read_json(ROOT / "logs/bottle-tree.txt")
        if entry["type"] == "file"
    ]
    print(
        json.dumps(
            dict(
                revision=revision,
                files=len(files),
                bytes=sum(entry["size"] for entry in files),
                license="CC-BY-NC-SA-4.0",
                note="Preflight only; no dataset images downloaded",
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=["preflight", "weights", "bottle"])
    args = parser.parse_args()
    {"preflight": preflight, "weights": weights, "bottle": bottle}[args.asset]()
