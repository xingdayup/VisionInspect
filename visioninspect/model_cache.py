"""One CUDA detector and its reference index, owned by one inference process."""

from __future__ import annotations

from pathlib import Path

from .common import identity, project_path


def file_stamp(path: str | Path) -> list[object]:
    """Describe a file by identity-relevant metadata, or by path alone if absent."""
    path = Path(path).resolve()
    try:
        stat = path.stat()
        return [
            str(path),
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_ino,
        ]
    except FileNotFoundError:
        return [str(path), None]


class ModelCache:
    """Hold at most one detector plus its reference index.

    Frozen assets are never re-read mid-run: bind_detector compares a digest
    over the registration, the calibration profile and the on-disk file stamps,
    and drops cached state whenever that digest changes.
    """

    def __init__(self) -> None:
        self.detector_key: str | None = None
        self.detector: object | None = None
        self.references: tuple[object, ...] | None = None

    def bind_detector(self, registration: dict, profile: dict) -> None:
        """Drop cached state when the bound model assets have changed."""
        paths = [
            project_path(registration["fit_run"]) / "model.ckpt",
            project_path(registration["profile"]),
            project_path(registration["references"]),
        ]
        key = identity(
            dict(
                registration=registration,
                profile=profile,
                files=[file_stamp(p) for p in paths],
            )
        )
        if key != self.detector_key:
            self.detector = None
            self.references = None
            self.detector_key = key

    def status(self) -> dict[str, object]:
        """Report only what is cached, never the model weights themselves."""
        return dict(
            detector_loaded=self.detector is not None,
            references_loaded=self.references is not None,
            device="cuda:0",
        )
