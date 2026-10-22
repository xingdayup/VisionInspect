from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * (attempt + 1))


def read_json(path: Path):
    # A filesystem can briefly deny a reader during an atomic replacement.
    # Corrupt JSON and genuine missing files still fail immediately.
    for attempt in range(6):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * (attempt + 1))


def configure_environment() -> None:
    # Only this process and its children; never persists host environment settings.
    for name, relative in {
        "HF_HOME": ".cache/huggingface",
        "TORCH_HOME": ".cache/torch",
        "UV_CACHE_DIR": ".cache/uv",
        "PIP_CACHE_DIR": ".cache/pip",
        "TEMP": ".tmp",
        "TMP": ".tmp",
    }.items():
        destination = ROOT / relative
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(destination)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    # Python may have cached its temp directory before environment configuration.
    tempfile.tempdir = str(ROOT / ".tmp")


@contextmanager
def offline(allow_loopback_ports=()):
    """Fail closed on Python network connections, including accidental model downloads."""
    ports = frozenset(allow_loopback_ports)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection

    def permitted(address):
        return (
            isinstance(address, tuple)
            and len(address) >= 2
            and address[0] == "127.0.0.1"
            and address[1] in ports
        )

    def denied(*args, **kwargs):
        raise RuntimeError(
            "Network disabled during baseline execution; prepare local assets first"
        )

    def connect(instance, address):
        return original_connect(instance, address) if permitted(address) else denied()

    def connect_ex(instance, address):
        return (
            original_connect_ex(instance, address) if permitted(address) else denied()
        )

    def create(address, *args, **kwargs):
        return (
            original_create(address, *args, **kwargs)
            if permitted(address)
            else denied()
        )

    with (
        patch.object(socket.socket, "connect", connect),
        patch.object(socket.socket, "connect_ex", connect_ex),
        patch.object(socket, "create_connection", create),
    ):
        yield


def load_config(path: Path) -> dict:
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    runtime = config["runtime"]
    if runtime["device"] != "cuda" or runtime["precision"] != "float32":
        raise ValueError(
            "GPU deployment requires CUDA / float32; CPU fallback is disabled"
        )
    if runtime["batch_size"] != 1 or runtime["num_workers"] != 0:
        raise ValueError("GPU inspection requires batch_size=1, num_workers=0")
    if config["evaluation"]["quantile_method"] != "higher":
        raise ValueError("Calibration must use higher quantile")
    return config


def identity(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
