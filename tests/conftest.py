import copy
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from visioninspect.common import ROOT, load_config, configure_environment

configure_environment()


@pytest.fixture(autouse=True)
def model_assets(tmp_path,monkeypatch):
    """Synthetic metadata only, so API/tool tests never need users' model assets."""
    from visioninspect.common import identity,sha256,write_json
    from visioninspect import service,api
    fit=tmp_path/'fixture-fit';fit.mkdir();(fit/'model.ckpt').write_bytes(b'UNIT TEST ONLY, NOT A MODEL')
    references=tmp_path/'references.npz';references.write_bytes(b'UNIT TEST ONLY')
    profile=dict(category='bottle',digest='',image_threshold=10.,review_lower=9.,review_upper=11.,
        pixel_threshold=7.,reference_similarity_floor=.8,manifest_digest='fixture-manifest',
        checkpoint_sha256=sha256(fit/'model.ckpt'),reference_index_sha256=sha256(references),
        quality=dict(sharpness_floor=1.,brightness_low=0.,brightness_high=255.),data_root=str(tmp_path))
    profile.pop('digest');profile['digest']=identity(profile)
    path=tmp_path/'profile.json';write_json(path,profile)
    registration=dict(fit_run=str(fit),profile=str(path),references=str(references),
        config=str(ROOT/'configs/patchcore.yaml'),data_root=str(tmp_path))
    value={'bottle':dict(label='测试瓶口',patchcore=registration)}
    monkeypatch.setattr(service,'registry',lambda:value)
    monkeypatch.setattr(api,'registry',lambda:value)


@pytest.fixture
def gpu_service_fixture(monkeypatch):
    """API routing tests simulate hardware presence, never GPU numerical execution."""
    monkeypatch.setattr('visioninspect.api.cuda_available',lambda:True)


@pytest.fixture
def tiny_dataset(tmp_path):
    """Synthetic fixtures only; never reported as benchmark results."""
    config=copy.deepcopy(load_config(ROOT/'configs/patchcore.yaml'))
    root=tmp_path/'synthetic_bottle'
    settings=config['dataset']
    settings.update(root=str(root),manifest=str(tmp_path/'manifest.json'),image_size=32,
                    expected_train=20,expected_test_good=2,expected_test_anomaly=6)
    generator=np.random.default_rng(7)
    groups=[('train','good',20),('test','good',2)]+[('test',d,2) for d in settings['defects']]
    for partition,category,count in groups:
        directory=root/partition/category
        directory.mkdir(parents=True)
        for index in range(count):
            array=generator.integers(0,255,(32,32,3),dtype=np.uint8)
            Image.fromarray(array).save(directory/f'{index:03}.png')
            if category!='good':
                mask_dir=root/'ground_truth'/category
                mask_dir.mkdir(parents=True,exist_ok=True)
                mask=np.zeros((32,32),dtype=np.uint8);mask[8:16,8:16]=255
                Image.fromarray(mask).save(mask_dir/f'{index:03}_mask.png')
    return config,root
