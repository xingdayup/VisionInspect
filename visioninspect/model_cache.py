"""One CUDA detector and its reference index, owned by one inference process."""
from pathlib import Path
from .common import identity,project_path


def file_stamp(path):
    path=Path(path).resolve()
    try:
        stat=path.stat()
        return [str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino]
    except FileNotFoundError:
        return [str(path),None]


class ModelCache:
    def __init__(self):
        self.detector_key=None;self.detector=None;self.references=None

    def bind_detector(self,registration,profile):
        paths=[project_path(registration['fit_run'])/'model.ckpt',
               project_path(registration['profile']),project_path(registration['references'])]
        key=identity(dict(registration=registration,profile=profile,files=[file_stamp(p) for p in paths]))
        if key!=self.detector_key:
            self.detector=None;self.references=None;self.detector_key=key

    def status(self):
        return dict(detector_loaded=self.detector is not None,references_loaded=self.references is not None,device='cuda:0')
