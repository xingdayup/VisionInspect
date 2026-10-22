"""Probe control-flow regression with fake packages; no GPU compatibility claim."""

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import pai_vllm


def runtime(monkeypatch, native_error=None):
    imported = []

    def import_native(name):
        imported.append(name)
        if name != "vllm._C_stable_libtorch":
            raise ModuleNotFoundError(f"No module named {name}", name=name)
        if native_error is not None:
            raise native_error
        return SimpleNamespace()

    metadata = SimpleNamespace(
        version=lambda name: pai_vllm.EXPECTED[name], distributions=lambda: []
    )
    monkeypatch.setattr(
        pai_vllm,
        "importlib",
        SimpleNamespace(metadata=metadata, import_module=import_native),
    )
    monkeypatch.setattr(
        pai_vllm,
        "sys",
        SimpleNamespace(
            platform="linux", version=sys.version, executable=sys.executable
        ),
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        get_device_name=lambda index: "simulated GPU",
        get_device_capability=lambda index: (8, 6),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="simulated")),
    )
    for name in ["vllm", "vllm.model_executor", "vllm.model_executor.models"]:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.model_executor.models"].ModelRegistry = SimpleNamespace(
        get_supported_archs=lambda: ["Qwen3VLForConditionalGeneration"]
    )
    return imported


def test_pinned_runtime_without_legacy_extension_can_pass_probe(tmp_path, monkeypatch):
    imported = runtime(monkeypatch)
    pai_vllm.check(tmp_path)
    assert imported == ["vllm._C_stable_libtorch"]
    assert pai_vllm.read(tmp_path / "logs/runtime-check.json")["passed"]


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError(
            "Stable native extension is absent", name="vllm._C_stable_libtorch"
        ),
        OSError("Native extension has an unresolved shared-library symbol"),
    ],
)
def test_native_extension_failure_is_not_hidden(tmp_path, monkeypatch, error):
    runtime(monkeypatch, native_error=error)
    with pytest.raises(type(error), match=str(error)):
        pai_vllm.check(tmp_path)
    assert not (tmp_path / "logs/runtime-check.json").exists()


def test_root_comes_from_environment_when_flag_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONINSPECT_GPU_ROOT", str(tmp_path / "gpu"))
    assert pai_vllm.resolve_root(None) == (tmp_path / "gpu").resolve()


def test_explicit_root_wins_over_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONINSPECT_GPU_ROOT", str(tmp_path / "gpu"))
    explicit = tmp_path / "explicit"
    assert pai_vllm.resolve_root(explicit) == explicit.resolve()


def test_root_without_flag_or_environment_is_rejected(monkeypatch):
    monkeypatch.delenv("VISIONINSPECT_GPU_ROOT", raising=False)
    with pytest.raises(ValueError, match="VISIONINSPECT_GPU_ROOT"):
        pai_vllm.resolve_root(None)


def test_download_explains_missing_runtime_check_before_any_network_call(tmp_path):
    with pytest.raises(RuntimeError, match="setup_vllm"):
        pai_vllm.download(tmp_path)


def test_download_rejects_a_failed_runtime_check(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/runtime-check.json").write_text('{"passed": false}', "utf8")
    with pytest.raises(RuntimeError, match="did not pass"):
        pai_vllm.download(tmp_path)


def test_server_environment_puts_the_venv_bin_on_path(monkeypatch):
    # The interpreter is a symlink to the base install, so the venv bin must be
    # taken from sys.prefix rather than from the resolved executable.
    monkeypatch.setenv("PATH", "/usr/bin")
    entries = pai_vllm.server_environment()["PATH"].split(os.pathsep)
    expected = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    assert entries[0] == str(expected)
    assert entries[-1] == "/usr/bin"


def test_cuda_toolkit_bin_is_used_only_when_nvcc_exists(tmp_path, monkeypatch):
    fake = tmp_path / "cuda"
    (fake / "bin").mkdir(parents=True)
    monkeypatch.setenv("CUDA_HOME", str(fake))
    assert pai_vllm.cuda_toolkit_bin() is None
    (fake / "bin/nvcc").write_text("", encoding="utf8")
    assert pai_vllm.cuda_toolkit_bin() == str(fake / "bin")
