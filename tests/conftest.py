import copy
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from visioninspect.common import ROOT, load_config, configure_environment

configure_environment()


@pytest.fixture
def tiny_dataset(tmp_path):
    """Synthetic fixtures only; never reported as benchmark results."""
    config=copy.deepcopy(load_config(ROOT/'configs/patchcore_cpu.yaml'))
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
