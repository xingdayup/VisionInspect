"""Install only manifest-listed migration assets; never overwrite differing local files."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def install(package, root=ROOT):
    manifest = json.loads((package / "transfer-manifest.json").read_text("utf8"))
    archive = package / "visioninspect-assets.tar.gz"
    recorded = next(
        item for item in manifest["archives"] if item["filename"] == archive.name
    )
    if (
        archive.stat().st_size != recorded["bytes"]
        or digest(archive) != recorded["sha256"]
    ):
        raise ValueError("Archive checksum mismatch")
    entries = {item["path"]: item for item in manifest["files"]}
    if (
        len(entries) != len(manifest["files"])
        or sum(item["bytes"] for item in entries.values()) > 500 * 1024**2
    ):
        raise ValueError("Invalid asset manifest budget")
    seen = set()
    installed = 0
    reused = 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or ":" in member.name
            ):
                raise ValueError("Unsafe archive entry")
            if not name.parts or name.parts[0] not in {
                "data",
                "weights",
                "outputs",
                "frontend",
            }:
                raise ValueError("Archive cannot modify application or Git files")
            if member.name not in entries or member.name in seen:
                raise ValueError("Unexpected or duplicate archive entry")
            seen.add(member.name)
            expected = entries[member.name]
            if member.size != expected["bytes"]:
                raise ValueError("Entry size mismatch")
            target = (root / member.name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Entry escapes target root")
            if target.is_file():
                if (
                    target.stat().st_size != member.size
                    or digest(target) != expected["sha256"]
                ):
                    raise FileExistsError(f"Existing asset differs: {member.name}")
                reused += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".transfer-part")
            with source.extractfile(member) as stream, temporary.open("xb") as output:
                remaining = member.size
                while remaining:
                    block = stream.read(min(4 * 1024**2, remaining))
                    if not block:
                        raise ValueError("Truncated archive entry")
                    output.write(block)
                    remaining -= len(block)
            if digest(temporary) != expected["sha256"]:
                raise ValueError("Entry checksum mismatch; partial retained")
            temporary.replace(target)
            installed += 1
    if seen != set(entries):
        raise ValueError("Missing archive entries")
    result = dict(
        passed=True,
        commit=manifest["commit"],
        installed_files=installed,
        reused_files=reused,
        source=str(package.resolve()),
        gpu_model_prepared=False,
    )
    (root / ".cache").mkdir(exist_ok=True)
    (root / ".cache/transfer-installed.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    print(json.dumps(install(parser.parse_args().package.resolve()), indent=2))
