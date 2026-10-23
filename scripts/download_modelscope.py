"""Explicit, resumable Qwen asset preparation from its verified ModelScope repository."""

import argparse
import hashlib
import json
import shutil
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        if urllib.parse.urlsplit(url).scheme != "https":
            raise ValueError("Non-HTTPS redirect refused")
        return super().redirect_request(request, response, code, message, headers, url)


def open_url(request):
    return urllib.request.build_opener(HTTPSRedirect()).open(request, timeout=60)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf8")
    temporary.replace(path)


def metadata(url):
    with open_url(url) as response:
        body = response.read(2 * 1024**2 + 1)
    if len(body) > 2 * 1024**2:
        raise ValueError("Metadata response too large")
    result = json.loads(body)
    if result.get("Code") != 200:
        raise ValueError("ModelScope metadata request was not successful")
    return result["Data"]


def preflight(root, spec, minimum_free_gib):
    if minimum_free_gib < 25:
        raise ValueError("A minimum of 25 GiB is required for this approved deployment")
    base = f"https://modelscope.cn/api/v1/models/{spec['model']}"
    info = metadata(base)
    if info.get("License") != "apache-2.0":
        raise ValueError("ModelScope license changed; stop for review")
    query = urllib.parse.urlencode(
        dict(Revision=spec["modelscope_revision"], Recursive="true")
    )
    remote = {
        item["Path"]: item for item in metadata(base + "/repo/files?" + query)["Files"]
    }
    directory = root / "weights/qwen3-vl-8b-instruct"
    items = []
    for item in spec["files"]:
        name = item["path"]
        entry = remote.get(name)
        if (
            Path(name).name != name
            or not entry
            or (entry["Size"], entry["Sha256"]) != (item["bytes"], item["sha256"])
        ):
            raise ValueError(f"Pinned ModelScope identity mismatch: {name}")
        target = directory / name
        partial = target.with_suffix(target.suffix + ".part")
        if target.is_symlink() or partial.is_symlink():
            raise ValueError("Linked model asset refused")
        if target.exists():
            if (
                target.stat().st_size != item["bytes"]
                or sha256(target) != item["sha256"]
            ):
                raise FileExistsError(
                    f"Existing asset differs; it will not be overwritten: {name}"
                )
            continue
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item["bytes"]:
            raise ValueError(f"Unexpected partial size: {name}")
        items.append(dict(item, offset=offset))
    remaining = sum(i["bytes"] - i["offset"] for i in items)
    free = shutil.disk_usage(root).free
    required = max(minimum_free_gib * GIB, remaining + 8 * GIB)
    report = dict(
        model=spec["model"],
        backend="ModelScope",
        revision=spec["modelscope_revision"],
        license=spec["license"],
        remaining_download_bytes=remaining,
        partial_bytes=sum(i["offset"] for i in items),
        free_disk_bytes=free,
        minimum_free_disk_bytes=required,
        reserve_bytes=8 * GIB,
        files=items,
        passed=free >= required,
    )
    save(root / "logs/modelscope-preflight.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2),
        flush=True,
    )
    if not report["passed"]:
        raise RuntimeError("Insufficient disk space; no model files downloaded")
    return items


def transfer(root, spec, items, workers, budget):
    if sum(i["bytes"] - i["offset"] for i in items) > budget:
        raise RuntimeError("Model download budget exceeded")
    directory = root / "weights/qwen3-vl-8b-instruct"
    directory.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    usage = {"response_payload_bytes": 0, "budget_bytes": budget, "files": []}

    def fetch(item):
        target = directory / item["path"]
        partial = target.with_suffix(target.suffix + ".part")
        last = time.monotonic()
        query = urllib.parse.urlencode(
            dict(Revision=spec["modelscope_revision"], FilePath=item["path"])
        )
        url = f"https://modelscope.cn/api/v1/models/{spec['model']}/repo?" + query
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == item["bytes"]:
                break
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={offset}-{item['bytes'] - 1}"}
            )
            try:
                with open_url(request) as response, partial.open("ab") as output:
                    expected = f"bytes {offset}-{item['bytes'] - 1}/{item['bytes']}"
                    if offset and (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected
                    ):
                        raise ValueError(
                            "Server ignored the resume range; refusing a duplicate full download"
                        )
                    while block := response.read(1024**2):
                        if offset + len(block) > item["bytes"]:
                            raise ValueError("Asset response exceeded pinned size")
                        if shutil.disk_usage(root).free - len(block) < 8 * GIB:
                            raise RuntimeError("8 GiB disk reserve reached")
                        with lock:
                            if usage["response_payload_bytes"] + len(block) > budget:
                                raise RuntimeError("Download budget reached")
                            usage["response_payload_bytes"] += len(block)
                        output.write(block)
                        offset += len(block)
                        if time.monotonic() - last >= 10:
                            print(
                                f"{item['path']}: {offset:,}/{item['bytes']:,} bytes",
                                flush=True,
                            )
                            last = time.monotonic()
                if offset == item["bytes"]:
                    break
                if attempt == 2:
                    raise ValueError(f"Truncated model file: {item['path']}")
            except (OSError, IncompleteRead) as error:
                if isinstance(error, ssl.SSLCertVerificationError) or isinstance(
                    getattr(error, "reason", None), ssl.SSLCertVerificationError
                ):
                    raise
                if isinstance(error, urllib.error.HTTPError) and error.code not in {
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise
                if attempt == 2:
                    raise
                print(
                    f"Resuming {item['path']} after {type(error).__name__}", flush=True
                )
                time.sleep(attempt + 1)
        if partial.stat().st_size != item["bytes"] or sha256(partial) != item["sha256"]:
            raise ValueError(f"Final SHA256 mismatch; partial retained: {item['path']}")
        partial.replace(target)
        with lock:
            usage["files"].append(dict(path=item["path"], sha256=item["sha256"]))
        print("VERIFIED", item["path"], flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, items))
    finally:
        save(root / "logs/modelscope-download.json", usage)
    for item in spec["files"]:
        target = directory / item["path"]
        if (
            not target.is_file()
            or target.stat().st_size != item["bytes"]
            or sha256(target) != item["sha256"]
        ):
            raise ValueError("Model assets are incomplete")
    save(
        directory / "source.json",
        dict(
            model=spec["model"],
            revision=spec["revision"],
            license=spec["license"],
            artifact_bytes=spec["artifact_bytes"],
            files=spec["origin_files"],
            download_backend="ModelScope",
            modelscope_revision=spec["modelscope_revision"],
            modelscope_url=f"https://modelscope.cn/models/{spec['model']}",
        ),
    )
    print(
        "All model files verified; source.json ready. Model inference has not been validated.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "download"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=int, default=50)
    parser.add_argument("--download-budget-bytes", type=int, default=18_000_000_000)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this script in the GPU server terminal")
    if args.download_budget_bytes <= 0:
        parser.error("Download budget must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / "configs/qwen3_vl_assets.json").read_text("utf8"))
    items = preflight(root, spec, args.minimum_free_gib)
    if args.command == "download":
        transfer(root, spec, items, args.workers, args.download_budget_bytes)


if __name__ == "__main__":
    main()
