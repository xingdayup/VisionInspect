from pathlib import Path
import pytest
from visioninspect.common import read_json,write_json


def test_windows_transient_reader_share_violation_is_bounded(tmp_path,monkeypatch):
    path=tmp_path/'progress.json';write_json(path,{'stage':'completed'})
    original=Path.read_text;calls=[]
    def read(instance,*args,**kwargs):
        calls.append(1)
        if len(calls)<3:raise PermissionError('Injected Windows sharing violation')
        return original(instance,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    assert read_json(path)=={'stage':'completed'} and len(calls)==3


def test_permanent_denial_and_corruption_are_not_hidden(tmp_path,monkeypatch):
    path=tmp_path/'progress.json';path.write_text('corrupt','utf8')
    with pytest.raises(ValueError):read_json(path)
    calls=[]
    def denied(*a,**k):calls.append(1);raise PermissionError('Injected permanent denial')
    monkeypatch.setattr(Path,'read_text',denied)
    with pytest.raises(PermissionError):read_json(path)
    assert len(calls)==6


def test_windows_atomic_replacement_retries_without_truncating_original(tmp_path,monkeypatch):
    path=tmp_path/'progress.json';write_json(path,{'stage':'queued'})
    original=Path.replace;calls=[]
    def replace(instance,target):
        calls.append(1)
        if len(calls)<3:
            assert read_json(target)=={'stage':'queued'}
            raise PermissionError('Injected Windows sharing violation')
        return original(instance,target)
    monkeypatch.setattr(Path,'replace',replace)
    write_json(path,{'stage':'completed'})
    assert read_json(path)=={'stage':'completed'} and len(calls)==3
