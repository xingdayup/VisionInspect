from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def configure_environment() -> None:
    # Only this process and its children; never persists host environment settings.
    for name, relative in {'HF_HOME': '.cache/huggingface', 'TORCH_HOME': '.cache/torch',
                           'UV_CACHE_DIR': '.cache/uv', 'PIP_CACHE_DIR': '.cache/pip',
                           'TEMP': '.tmp', 'TMP': '.tmp'}.items():
        destination = ROOT / relative
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(destination)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['PYTHONUTF8'] = '1'
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    # Python may have cached its temp directory before environment configuration.
    tempfile.tempdir = str(ROOT / '.tmp')


@contextmanager
def offline():
    """Fail closed on Python network connections, including accidental model downloads."""
    def denied(*args, **kwargs):
        raise RuntimeError('Network disabled during baseline execution; prepare local assets first')
    with patch.object(socket.socket, 'connect', denied), patch.object(socket.socket, 'connect_ex', denied), \
         patch.object(socket, 'create_connection', denied):
        yield


def load_config(path: Path) -> dict:
    import yaml
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported config schema')
    runtime = config['runtime']
    if runtime['device'] != 'cpu' or runtime['precision'] != 'float32':
        raise ValueError('Phase 1 requires CPU / float32')
    if runtime['batch_size'] != 1 or runtime['num_workers'] != 0:
        raise ValueError('Phase 1 requires batch_size=1, num_workers=0')
    if config['evaluation']['quantile_method'] != 'higher':
        raise ValueError('Calibration must use higher quantile')
    return config


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
