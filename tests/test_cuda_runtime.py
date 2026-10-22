"""CUDA placement/roundtrip tests; synthetic tensors are not benchmark results."""
from io import BytesIO
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from visioninspect.api import create_app
from visioninspect.common import ROOT,load_config
from visioninspect.model import Detector,cuda_device,setup


def test_missing_cuda_fails_before_checkpoint_is_read(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def forbidden(*args,**kwargs):raise AssertionError('Checkpoint must not be read without CUDA')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(RuntimeError,match='CPU fallback is disabled'):Detector.load(tmp_path/'absent.ckpt')


def test_gpu_config_rejects_cpu_device(tmp_path):
    text=(ROOT/'configs/patchcore.yaml').read_text('utf8')
    path=tmp_path/'invalid.yaml';path.write_text(text.replace('device: cuda','device: cpu'),'utf8')
    with pytest.raises(ValueError,match='CPU fallback is disabled'):load_config(path)


@pytest.mark.parametrize('mode',['fixed','dynamic'])
def test_retired_workflow_is_rejected(tmp_path,mode):
    image=BytesIO();Image.new('RGB',(256,256)).save(image,format='PNG')
    with TestClient(create_app(tmp_path)) as client:
        response=client.post('/api/tasks',files={'file':('fixture.png',image.getvalue(),'image/png')},data={'mode':mode})
        assert response.status_code==422


def test_api_without_cuda_never_enqueues_cpu_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr('visioninspect.api.cuda_available',lambda:False)
    image=BytesIO();Image.new('RGB',(256,256)).save(image,format='PNG')
    with TestClient(create_app(tmp_path)) as client:
        response=client.post('/api/tasks',files={'file':('fixture.png',image.getvalue(),'image/png')},data={'mode':'baseline'})
        assert response.status_code==503 and 'CUDA' in response.json()['detail']
        assert client.get('/api/tasks').json()['total']==0
        assert client.get('/ready').status_code==503


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires real NVIDIA CUDA; never substitutes CPU')
def test_real_cuda_model_memory_bank_input_and_checkpoint_roundtrip(tmp_path):
    config=load_config(ROOT/'configs/patchcore.yaml');setup(config)
    detector=Detector(config)
    # An untrained model and random memory bank test device plumbing only.
    detector.model.memory_bank=torch.rand((16,384),device=cuda_device())
    detector.model.eval();image=np.zeros((256,256,3),dtype=np.uint8)
    assert detector.tensor(image).device.type=='cuda'
    assert detector.model.memory_bank.device.type=='cuda'
    assert all(parameter.device.type=='cuda' for parameter in detector.model.parameters())
    score,heat,_=detector.predict(image)
    detector.save(tmp_path/'fixture.ckpt',dict(purpose='CUDA ENGINEERING TEST ONLY'))
    loaded,_=Detector.load(tmp_path/'fixture.ckpt')
    other_score,other_heat,_=loaded.predict(image)
    np.testing.assert_allclose(other_score,score,rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(other_heat,heat,rtol=1e-5,atol=1e-6)
    assert heat.shape==(256,256) and np.isfinite(heat).all()
