"""Shared paths, atomic JSON access and process-level environment guards."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    """Resolve a config path against the project root unless it is absolute."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    """Hash a file in 1 MiB blocks so large checkpoints do not load into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically, retrying the rename on Windows sharing violations."""
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


def read_json(path: Path) -> Any:
    """Read JSON, tolerating the brief window of an in-progress replacement.

    Corrupt JSON and genuinely missing files still fail immediately.
    """
    for attempt in range(6):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * (attempt + 1))


def configure_environment() -> None:
    """Point every cache at the project and force offline model loading.

    Only this process and its children are affected; host settings never change.
    """
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
    # Python may have cached its temp directory before this configuration.
    tempfile.tempdir = str(ROOT / ".tmp")


@contextmanager
def offline(allow_loopback_ports: tuple[int, ...] = ()) -> Iterator[None]:
    """Fail closed on Python network connections, including accidental downloads.

    Only 127.0.0.1 on the listed ports is reachable, which is exactly what the
    local vLLM socket needs and nothing more.
    """
    ports = frozenset(allow_loopback_ports)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection

    def permitted(address: Any) -> bool:
        return (
            isinstance(address, tuple)
            and len(address) >= 2
            and address[0] == "127.0.0.1"
            and address[1] in ports
        )

    def denied(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Network disabled during baseline execution; prepare local assets first"
        )

    def connect(instance: socket.socket, address: Any) -> Any:
        return original_connect(instance, address) if permitted(address) else denied()

    def connect_ex(instance: socket.socket, address: Any) -> Any:
        return (
            original_connect_ex(instance, address) if permitted(address) else denied()
        )

    def create(address: Any, *args: Any, **kwargs: Any) -> Any:
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
    """Load a runtime config and reject anything that would imply CPU fallback."""
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
    """Return a stable digest for a JSON-serializable structure."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
