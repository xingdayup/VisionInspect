"""Probe control-flow regression with fake packages; no GPU compatibility claim."""
import sys
from types import ModuleType, SimpleNamespace
import pytest
from scripts import pai_vllm


def runtime(monkeypatch, native_error=None):
    imported=[]
    def import_native(name):
        imported.append(name)
        if name!='vllm._C_stable_libtorch':
            raise ModuleNotFoundError(f'No module named {name}',name=name)
        if native_error is not None:
            raise native_error
        return SimpleNamespace()
    metadata=SimpleNamespace(version=lambda name:pai_vllm.EXPECTED[name],distributions=lambda:[])
    monkeypatch.setattr(pai_vllm,'importlib',SimpleNamespace(metadata=metadata,import_module=import_native))
    monkeypatch.setattr(pai_vllm,'sys',SimpleNamespace(platform='linux',version=sys.version,executable=sys.executable))
    cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,
        get_device_name=lambda index:'simulated GPU',get_device_capability=lambda index:(8,6))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda,version=SimpleNamespace(cuda='simulated')))
    for name in ['vllm','vllm.model_executor','vllm.model_executor.models']:
        module=ModuleType(name);module.__path__=[]
        monkeypatch.setitem(sys.modules,name,module)
    sys.modules['vllm.model_executor.models'].ModelRegistry=SimpleNamespace(
        get_supported_archs=lambda:['Qwen3VLForConditionalGeneration'])
    return imported


def test_pinned_runtime_without_legacy_extension_can_pass_probe(tmp_path,monkeypatch):
    imported=runtime(monkeypatch)
    pai_vllm.check(tmp_path)
    assert imported==['vllm._C_stable_libtorch']
    assert pai_vllm.read(tmp_path/'logs/runtime-check.json')['passed']


@pytest.mark.parametrize('error',[
    ModuleNotFoundError('Stable native extension is absent',name='vllm._C_stable_libtorch'),
    OSError('Native extension has an unresolved shared-library symbol'),
])
def test_native_extension_failure_is_not_hidden(tmp_path,monkeypatch,error):
    runtime(monkeypatch,native_error=error)
    with pytest.raises(type(error),match=str(error)):
        pai_vllm.check(tmp_path)
    assert not (tmp_path/'logs/runtime-check.json').exists()
