"""One detector slot and one optional VLM slot, owned by one inference process."""
from pathlib import Path
from .common import identity,project_path,read_json


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
        self.vlm_key=None;self.vlm=None

    def bind_detector(self,registration,profile):
        paths=[project_path(registration['fit_run'])/'model.ckpt',
               project_path(registration['profile']),project_path(registration['references'])]
        key=identity(dict(registration=registration,profile=profile,files=[file_stamp(p) for p in paths]))
        if key!=self.detector_key:
            self.detector=None;self.references=None;self.detector_key=key

    def get_vlm(self,directory,max_new_tokens):
        from .vlm import LocalVLM,local_asset_path
        directory=Path(directory);source=directory/'source.json';paths=[source]
        if source.is_file():
            paths += [local_asset_path(directory,item['path']) for item in read_json(source)['files']]
        key=identity(dict(files=[file_stamp(p) for p in paths],max_new_tokens=max_new_tokens))
        if key!=self.vlm_key:
            self.vlm=None;self.vlm_key=key
            self.vlm=LocalVLM(directory,max_new_tokens)
        return self.vlm

    def status(self):
        return dict(detector_loaded=self.detector is not None,references_loaded=self.references is not None,
                    vlm_loaded=self.vlm is not None and self.vlm.model is not None)
