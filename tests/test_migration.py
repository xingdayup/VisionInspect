"""Portable assets retain frozen identities and reject unsafe or changed contents."""

import hashlib
import io
import json
import shutil
import tarfile
import pytest
from visioninspect.data import check_data, load_manifest
from visioninspect.common import sha256
from scripts.install_assets import install


def test_explicit_relocation_preserves_manifest_identity_and_checks_content(
    tiny_dataset, tmp_path, monkeypatch
):
    config, root = tiny_dataset
    manifest = check_data(config)
    before = sha256(__import__("pathlib").Path(config["dataset"]["manifest"]))
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    config["dataset"]["root"] = str(relocated)
    with pytest.raises(ValueError, match="root"):
        load_manifest(config, ("train",))
    monkeypatch.setenv("VISIONINSPECT_RELOCATED_ASSETS", "1")
    same, records = load_manifest(config, ("train",))
    assert same["digest"] == manifest["digest"] and records
    assert check_data(config)["digest"] == manifest["digest"]
    assert sha256(__import__("pathlib").Path(config["dataset"]["manifest"])) == before
    (relocated / records[0]["path"]).write_bytes(b"INTENTIONAL CORRUPTION FIXTURE")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(config, ("train",))


def package(tmp_path, name="data/fixture.txt", content=b"EXPLICIT TEST FIXTURE"):
    source = tmp_path / "package"
    source.mkdir()
    archive = source / "visioninspect-assets.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        output.addfile(member, io.BytesIO(content))
    manifest = dict(
        commit="unit-test",
        files=[
            dict(
                path=name,
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        ],
        archives=[
            dict(
                filename=archive.name,
                bytes=archive.stat().st_size,
                sha256=sha256(archive),
            )
        ],
    )
    (source / "transfer-manifest.json").write_text(json.dumps(manifest), "utf8")
    return source


def test_transfer_installs_reuses_and_refuses_overwrite(tmp_path):
    source = package(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    assert install(source, target)["installed_files"] == 1
    assert install(source, target)["reused_files"] == 1
    (target / "data/fixture.txt").write_bytes(b"USER MODIFICATION")
    with pytest.raises(FileExistsError):
        install(source, target)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "data\\escape", "C:/escape"]
)
def test_transfer_rejects_escaping_members(tmp_path, name):
    source = package(tmp_path, name)
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError):
        install(source, target)
