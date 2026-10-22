import hashlib
import pytest
from scripts import prepare_assets
from visioninspect.common import offline


def test_existing_verified_asset_is_reused_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_assets, "ROOT", tmp_path)
    path = tmp_path / "asset.bin"
    path.write_bytes(b"verified")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with offline():
        result = prepare_assets.fetch("https://example.com/asset", path, 8, checksum)
    assert result["reused"] is True
    assert path.read_bytes() == b"verified"


def test_different_existing_asset_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_assets, "ROOT", tmp_path)
    path = tmp_path / "asset.bin"
    path.write_bytes(b"user-content")
    with offline(), pytest.raises(ValueError, match="will not overwrite"):
        prepare_assets.fetch("https://example.com/asset", path, 3, "0" * 64)
    assert path.read_bytes() == b"user-content"
