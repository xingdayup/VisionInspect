"""CPU adapter around the pinned Anomalib EfficientAdModel; no auto-downloading Trainer."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize,center_crop,rgb_to_grayscale
from .common import project_path,read_json,sha256,identity

def enforce_fit_budget(config,preflight):
    if preflight.get('cpu_threads')!=config['runtime']['threads'] or preflight.get('config_digest')!=identity(config):
        raise ValueError('EfficientAD compute preflight does not match frozen config')
    seconds=float(preflight['step_p95_seconds'])*config['training']['steps']
    if not np.isfinite(seconds) or seconds<=0:raise ValueError('Invalid EfficientAD throughput estimate')
    if seconds>config['runtime']['fit_timeout_seconds']:
        raise RuntimeError(f'EfficientAD fit resource budget exceeded before training: compute estimate {seconds:.0f}s > limit {config["runtime"]["fit_timeout_seconds"]}s')
    return seconds

class EfficientAD:
    def __init__(self,config):
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel,EfficientAdModelSize
        self.config=config;self.size=config['model']['image_size'];self.teacher_ready=False
        if self.size!=256:raise ValueError('EfficientAD CPU protocol requires 256x256')
        self.model=EfficientAdModel(teacher_out_channels=config['model']['teacher_out_channels'],
                  model_size=EfficientAdModelSize.S,padding=config['model']['padding'],pad_maps=config['model']['pad_maps']).cpu().float()
        self.model.teacher.requires_grad_(False).eval()

    def load_teacher(self):
        path=project_path(self.config['model']['teacher_weights']);metadata=read_json(project_path(self.config['model']['teacher_metadata']))
        digest=sha256(path)
        if digest!=metadata['teacher_sha256']:raise ValueError('EfficientAD teacher checksum mismatch')
        self.model.teacher.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.teacher_ready=True;return digest

    def tensor(self,array,regularizer=False):
        # EfficientAD performs ImageNet normalization inside teacher/student/AE.
        image=torch.from_numpy(np.array(array,copy=True)).permute(2,0,1).float().div_(255)
        if regularizer:
            image=resize(image,[512,512],antialias=True)
            if torch.rand(()).item()<.3:image=rgb_to_grayscale(image,num_output_channels=3)
            image=center_crop(image,[256,256])
        else:image=resize(image,[256,256],antialias=True)
        return image.unsqueeze(0)

    def optimizer(self):
        return torch.optim.Adam(list(self.model.student.parameters())+list(self.model.ae.parameters()),
                    lr=self.config['training']['learning_rate'],weight_decay=self.config['training']['weight_decay'])

    def training_step(self,batch,regularizer,optimizer):
        if batch.shape!=(1,3,256,256) or regularizer.shape!=(1,3,256,256):raise ValueError('Expected RGB 256x256 batch=1')
        self.model.train();self.model.teacher.eval();optimizer.zero_grad(set_to_none=True)
        losses=self.model(batch,regularizer)
        if not all(torch.isfinite(value).item() for value in losses):raise ValueError('Non-finite EfficientAD loss')
        loss=sum(losses);loss.backward();optimizer.step()
        return [float(value.detach()) for value in losses]

    def teacher_statistics(self,root,records):
        if not records or any(r['split']!='train' or r['label']!=0 for r in records):raise ValueError('Teacher statistics require normal train only')
        sums=squares=None;count=0
        with torch.inference_mode():
            for record in records:
                with Image.open(root/record['path']) as image:batch=self.tensor(np.asarray(image.convert('RGB')))
                features=self.model.teacher(batch).double()
                one=features.sum(dim=(0,2,3));two=features.square().sum(dim=(0,2,3))
                sums=one if sums is None else sums+one;squares=two if squares is None else squares+two
                count+=features.shape[0]*features.shape[2]*features.shape[3]
            mean=sums/count;std=(squares/count-mean.square()).clamp_min(1e-12).sqrt()
            self.model.mean_std['mean'].copy_(mean.float()[None,:,None,None])
            self.model.mean_std['std'].copy_(std.float()[None,:,None,None])

    def fit(self,preflight_path,progress=None):
        if not self.teacher_ready:raise ValueError('Load verified pretrained teacher first')
        enforce_fit_budget(self.config,read_json(preflight_path))
        from .data import load_manifest
        import psutil
        manifest,records=load_manifest(self.config,('train',));root=Path(manifest['root'])
        _,regularizer_root,regularizer_records=authorized_regularizer(project_path(self.config['training']['regularizer_manifest']),
                              {record['sha256'] for record in manifest['records']})
        started=time.perf_counter()
        self.teacher_statistics(root,records)
        optimizer=self.optimizer();steps=self.config['training']['steps']
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,int(.95*steps)),gamma=.1)
        order=[];ood_order=[];history=[]
        for index in range(steps):
            if time.perf_counter()-started>self.config['runtime']['fit_timeout_seconds']:raise TimeoutError('EfficientAD fit time budget exceeded')
            if psutil.Process().memory_info().rss>self.config['runtime']['memory_limit_gib']*1024**3:raise MemoryError('EfficientAD fit memory budget exceeded')
            if index%len(records)==0:order=torch.randperm(len(records)).tolist()
            if index%len(regularizer_records)==0:ood_order=torch.randperm(len(regularizer_records)).tolist()
            normal=records[order[index%len(records)]];ood=regularizer_records[ood_order[index%len(regularizer_records)]]
            with Image.open(root/normal['path']) as image:batch=self.tensor(np.asarray(image.convert('RGB')))
            with Image.open(regularizer_root/ood['path']) as image:penalty=self.tensor(np.asarray(image.convert('RGB')),regularizer=True)
            loss=self.training_step(batch,penalty,optimizer);scheduler.step()
            if index==0 or (index+1)%100==0:
                record=dict(step=index+1,losses=loss);history.append(record)
                if progress:progress(record)
        self.model.eval();return history

    def normal_map_quantiles(self,root,records):
        if not records or any(r['split']!='validation' or r['label']!=0 for r in records):raise ValueError('Map quantiles require normal validation only')
        self.model.eval();student=[];autoencoder=[]
        with torch.inference_mode():
            for record in records:
                with Image.open(root/record['path']) as image:batch=self.tensor(np.asarray(image.convert('RGB')))
                st,ae=self.model.get_maps(batch,normalize=False);student.append(st.flatten());autoencoder.append(ae.flatten())
            result={}
            for suffix,maps in [('st',student),('ae',autoencoder)]:
                values=torch.cat(maps)
                result['qa_'+suffix]=float(torch.quantile(values,.9));result['qb_'+suffix]=float(torch.quantile(values,.995))
        self.apply_quantiles(result);return result

    def apply_quantiles(self,values):
        for suffix in ('st','ae'):
            if not np.isfinite([values['qa_'+suffix],values['qb_'+suffix]]).all() or values['qb_'+suffix]<=values['qa_'+suffix]:
                raise ValueError('Invalid normal-validation map quantiles')
        with torch.no_grad():
            for key,value in values.items():self.model.quantiles[key].copy_(torch.tensor(value,dtype=torch.float32))

    def predict(self,array):
        self.model.eval();start=time.perf_counter()
        with torch.inference_mode():output=self.model(self.tensor(array))
        score=float(output.pred_score.item());heatmap=output.anomaly_map.squeeze().numpy().astype(np.float32)
        if not np.isfinite(score) or heatmap.shape!=(256,256) or not np.isfinite(heatmap).all():raise ValueError('Invalid EfficientAD prediction')
        return score,heatmap,time.perf_counter()-start

    def save(self,path,metadata):
        torch.save(dict(schema_version=1,backend='efficientad_s',config=self.config,state_dict=self.model.state_dict(),metadata=metadata),path)

    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['schema_version']!=1 or state['backend']!='efficientad_s':raise ValueError('Not an EfficientAD checkpoint')
        instance=cls(state['config']);instance.model.load_state_dict(state['state_dict'],strict=True)
        instance.teacher_ready=True;instance.model.eval();return instance,state['metadata']

def authorized_regularizer(path,bottle_hashes):
    """Validate a user's provenance assertion and file integrity; never grants image rights."""
    manifest=read_json(path)
    if manifest.get('digest')!=identity({k:v for k,v in manifest.items() if k!='digest'}):raise ValueError('Regularizer manifest checksum mismatch')
    if manifest.get('authorized_for_research') is not True or not all(manifest.get(key) for key in ('source_url','license','authorization_note')):
        raise ValueError('Explicit source/rights record required for regularizer images')
    root=project_path(manifest['root']);records=manifest['records']
    if not records:raise ValueError('Regularizer images missing; automatic download disabled')
    for record in records:
        image=(root/record['path']).resolve()
        if not image.is_relative_to(root) or sha256(image)!=record['sha256']:raise ValueError('Regularizer image changed or unsafe path')
        if record['sha256'] in bottle_hashes:raise ValueError('MVTec images cannot replace unrelated regularizer images')
        try:
            with Image.open(image) as opened:opened.verify()
        except (OSError,ValueError) as error:raise ValueError('Regularizer image cannot be decoded') from error
    return manifest,root,records
