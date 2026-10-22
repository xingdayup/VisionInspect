"""Create a Git bundle and a separate checksummed, bounded local asset archive."""

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
import zipfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/transfer")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise RuntimeError("Commit and audit all source changes before packaging")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_git_content.py")],
        cwd=ROOT,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    target = args.output.resolve() / f"{time.strftime('%Y%m%d-%H%M%S')}-{commit[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    roots = [
        "data/mvtec_ad/bottle",
        "data/manifests/bottle.json",
        "weights/profiles/bottle.json",
        "weights/profiles/bottle-references.npz",
        "weights/resnet18.a1_in1k",
        "outputs/20261021-232056-fit-ae72d8/model.ckpt",
        "outputs/20261021-232056-fit-ae72d8/calibration.json",
        "frontend/dist",
    ]
    files = []
    for name in roots:
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"Required transfer asset missing: {name}")
        files.extend(
            sorted(p for p in path.rglob("*") if p.is_file())
            if path.is_dir()
            else [path]
        )
    if any(
        p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()) for p in files
    ):
        raise ValueError("Unsafe transfer asset path")
    total = sum(p.stat().st_size for p in files)
    if total > 500 * 1024**2:
        raise RuntimeError(
            "Asset archive input exceeds 500 MiB; inspect before expanding transfer"
        )
    print(
        json.dumps(
            dict(
                commit=commit,
                files=len(files),
                asset_input_bytes=total,
                destination=str(target),
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    bundle = target / "visioninspect.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "--all"], cwd=ROOT, check=True
    )
    archive = target / "visioninspect-assets.tar.gz"
    records = []
    with tarfile.open(archive, "w:gz", compresslevel=3) as output:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            output.add(path, arcname=name, recursive=False)
            records.append(
                dict(path=name, bytes=path.stat().st_size, sha256=digest(path))
            )
    result = dict(
        schema_version=1,
        commit=commit,
        asset_input_bytes=total,
        files=records,
        archives=[
            dict(filename=p.name, bytes=p.stat().st_size, sha256=digest(p))
            for p in (bundle, archive)
        ],
        excluded=[
            "virtual environments",
            "node_modules",
            "Qwen weights",
            "service history",
            "keys",
        ],
        license_notice="bottle CC-BY-NC-SA-4.0; noncommercial research, private transfer only; see docs/DATASETS.md",
        gpu_model_prepared=False,
    )
    (target / "transfer-manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    (target / "START-HERE.txt").write_text(
        "Upload these three transfer files into your own PAI workspace.\n"
        "git clone visioninspect.bundle VisionInspect\ncd VisionInspect\n"
        "/usr/local/bin/python3 -B scripts/install_assets.py --package ..\n"
        "bash scripts/setup_gpu.sh\nSee README.md for CUDA budget checks and explicit GPU verification.\n",
        encoding="utf8",
    )
    portable = target / "visioninspect-pai-migration.zip"
    with zipfile.ZipFile(portable, "w", compression=zipfile.ZIP_STORED) as output:
        for name in [
            "visioninspect.bundle",
            "visioninspect-assets.tar.gz",
            "transfer-manifest.json",
            "START-HERE.txt",
        ]:
            output.write(target / name, arcname=name)
    (target / "visioninspect-pai-migration.zip.sha256").write_text(
        digest(portable) + "  " + portable.name + "\n", encoding="ascii"
    )
    print(target, flush=True)


if __name__ == "__main__":
    main()
