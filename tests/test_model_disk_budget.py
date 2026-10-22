import io
import json
from types import SimpleNamespace
import pytest
from scripts import pai_vllm


@pytest.mark.parametrize(
    "minimum,free,passed",
    [
        (50, 49, False),
        (25, 26, True),
        (25, 24, False),
    ],
)
def test_explicit_model_disk_budget_preserves_default_and_reserve(
    tmp_path, monkeypatch, minimum, free, passed
):
    monkeypatch.setattr(pai_vllm, "urlopen", lambda *a, **k: io.BytesIO(b"{}"))
    monkeypatch.setattr(
        pai_vllm, "inspect_metadata", lambda data: [dict(size=17_545_914_364)]
    )
    monkeypatch.setattr(
        pai_vllm.shutil, "disk_usage", lambda path: SimpleNamespace(free=free * 1024**3)
    )
    if passed:
        pai_vllm.preflight(tmp_path, minimum)
    else:
        with pytest.raises(RuntimeError, match="disk budget exceeded"):
            pai_vllm.preflight(tmp_path, minimum)
    report = json.loads((tmp_path / "logs/model-preflight.json").read_text())
    assert report["minimum_free_disk_bytes"] == minimum * 1024**3
    assert report["minimum_reserve_bytes"] == 8 * 1024**3
    assert not report["model_weights_downloaded"]


def test_model_disk_override_cannot_consume_the_required_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(pai_vllm, "urlopen", lambda *a, **k: io.BytesIO(b"{}"))
    monkeypatch.setattr(
        pai_vllm, "inspect_metadata", lambda data: [dict(size=17_545_914_364)]
    )
    monkeypatch.setattr(
        pai_vllm.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3)
    )
    with pytest.raises(ValueError, match="8 GiB reserve"):
        pai_vllm.preflight(tmp_path, 24)
