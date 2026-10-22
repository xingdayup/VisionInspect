"""Input geometry and actual call boundaries, not VLM semantic accuracy claims."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from visioninspect.common import read_json,sha256
from visioninspect.vlm import LocalVLM,prepare_payload,INPUT_PROTOCOL
from visioninspect.tools import ToolFailure

@pytest.mark.parametrize('reference_present',[True,False])
def test_payload_keeps_source_roles_separate_and_does_not_modify_images(reference_present):
    query=Image.new('RGB',(900,900),'red');reference=Image.new('RGB',(900,900),'blue') if reference_present else None
    images,messages,audit=prepare_payload(query,reference,'bottle')
    assert len(images)==(2 if reference_present else 1)
    assert sum(item['type']=='image' for item in messages[0]['content'])==len(images)
    assert images[0].size==(512,512) and images[0].getpixel((256,256))==(255,0,0)
    assert query.size==(900,900)
    if reference_present:
        assert images[1].getpixel((256,256))==(0,0,255) and reference.size==(900,900)
        assert 'Image 1: inspected bottle' in audit['prompt'] and 'Image 2: normal training reference' in audit['prompt']
        assert 'Left:' not in audit['prompt'] and 'Right:' not in audit['prompt']

def test_rectangular_source_is_letterboxed_without_geometric_distortion():
    images,_,audit=prepare_payload(Image.new('RGB',(800,400),'red'),None,'bottle')
    assert images[0].getpixel((256,127))==(240,240,240)
    assert images[0].getpixel((256,128))==(255,0,0)
    assert images[0].getpixel((256,383))==(255,0,0)
    assert images[0].getpixel((256,384))==(240,240,240)
    assert audit['source_dimensions']==[[800,400]]

class BoundaryProcessor:
    image_token_id=9;image_seq_len=64
    image_processor=SimpleNamespace(image_mean=[.5]*3,image_std=[.5]*3)
    def apply_chat_template(self,messages,add_generation_prompt):
        self.messages=messages;assert add_generation_prompt
        return 'UNIT TEST IMAGE PLACEHOLDERS'
    def __call__(self,*,text,images,return_tensors,do_image_splitting):
        import torch
        assert return_tensors=='pt' and do_image_splitting is False
        assert len(images)==sum(item['type']=='image' for item in self.messages[0]['content'])
        self.images=images
        pixels=torch.stack([torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/127.5-1 for image in images])[None]
        return dict(pixel_values=pixels,input_ids=torch.tensor([[9]*(64*len(images))+[1,2]]))
    def batch_decode(self,values,skip_special_tokens):
        assert skip_special_tokens
        return ['UNIT TEST ONLY']

def fake_ready_vlm(monkeypatch,tmp_path):
    import torch
    instance=LocalVLM(tmp_path);instance.processor=BoundaryProcessor()
    instance.model=SimpleNamespace(generate=lambda **kwargs:torch.cat([kwargs['input_ids'],torch.tensor([[3]])],dim=1))
    monkeypatch.setattr(instance,'load',lambda:None)
    return instance

def test_generation_boundary_and_saved_encoder_inputs_match_roles(monkeypatch,tmp_path):
    instance=fake_ready_vlm(monkeypatch,tmp_path)
    result=instance.describe(Image.new('RGB',(900,900),'red'),Image.new('RGB',(900,900),'blue'),audit_directory=tmp_path/'audit')
    audit=read_json(tmp_path/'audit/input.json')
    assert audit['protocol']==INPUT_PROTOCOL and audit['pixel_values_shape']==[1,2,3,512,512]
    assert audit['image_tokens']==128 and not audit['region_crops_sent']
    for index,color in enumerate([(255,0,0),(0,0,255)]):
        record=audit['images'][index]
        with Image.open(tmp_path/'audit'/record['encoder_path']) as image:assert image.getpixel((256,256))==color
        assert sha256(tmp_path/'audit'/record['path'])==record['sha256']
    assert result['candidate']=='unknown' and result['verified'] is False
    assert INPUT_PROTOCOL in result['backend']

def test_image_placeholder_mismatch_fails_before_generation(monkeypatch,tmp_path):
    instance=fake_ready_vlm(monkeypatch,tmp_path)
    instance.processor.image_seq_len=63
    with pytest.raises(ToolFailure,match='占位符'):
        instance.describe(Image.new('RGB',(900,900)),Image.new('RGB',(900,900)))

def test_tool_passes_inspection_reference_and_task_local_audit_directory(tmp_path):
    import yaml
    from visioninspect.common import ROOT
    from visioninspect.tools import ToolKit
    registration=yaml.safe_load((ROOT/'configs/models.yaml').read_text('utf8'))['models']['bottle']['patchcore']
    policy=yaml.safe_load((ROOT/'configs/agent.yaml').read_text('utf8'))
    query=tmp_path/'query.png';Image.new('RGB',(900,900),'red').save(query)
    output=tmp_path/'task';(output/'evidence').mkdir(parents=True)
    Image.new('RGB',(900,900),'blue').save(output/'evidence/reference_1.png')
    class RecordingVLM:
        def describe(self,image,reference,category,audit_directory):
            assert image.getpixel((10,10))==(255,0,0) and reference.getpixel((10,10))==(0,0,255)
            assert category=='bottle' and audit_directory==output/'vlm_input'
            return dict(candidate='unknown',evidence=[],uncertain=True,available=True,backend='UNIT TEST ONLY',raw_text='',verified=False)
    result=ToolKit(registration,policy,output,RecordingVLM()).invoke('classify_defect_candidate',dict(image_path=str(query),category='bottle'))
    assert result['candidate']=='unknown' and result['available']
