"""Optional CPU VLM; local files only, no remote code and no final-decision authority."""
from pathlib import Path
from PIL import Image,ImageOps
from .common import read_json,sha256,write_json
from .tools import ToolFailure

INPUT_PROTOCOL='separate-images-v2'

def candidate_prompt(category, has_reference):
    if category == 'bottle':
        return ('Image 1: inspected bottle. Image 2: normal training reference. Describe visible differences at the bottle rim in one short sentence. Do not infer causes. If unclear say unknown.'
                if has_reference else 'Describe visible irregularities at this bottle rim in one short sentence. Do not infer causes or product safety. If unclear say unknown.')
    products = {'metal_nut':'metal nut', 'capsule':'capsule', 'pill':'pill'}
    if category not in products:
        raise ToolFailure('schema_error','当前类别没有已定义的候选描述提示词')
    product = products[category]
    return (f'Image 1: inspected {product}. Image 2: normal training reference. Describe visible surface differences in one short sentence. Do not infer causes or product safety. If unclear say unknown.'
            if has_reference else f'Describe visible surface irregularities of this {product} in one short sentence. Do not infer causes or product safety. If unclear say unknown.')

def prepare_payload(image:Image.Image,reference:Image.Image|None,category):
    """One square encoder input per source, preserving aspect ratio without cropping."""
    sources=[image]+([reference] if reference is not None else [])
    images=[ImageOps.pad(source.convert('RGB'),(512,512),method=Image.Resampling.LANCZOS,
                        color=(240,240,240)) for source in sources]
    prompt=candidate_prompt(category,reference is not None)
    messages=[dict(role='user',content=[dict(type='image') for _ in images]+[dict(type='text',text=prompt)])]
    return images,messages,dict(protocol=INPUT_PROTOCOL,category=category,prompt=prompt,
               source_dimensions=[list(source.size) for source in sources],
               prepared_dimensions=[list(image.size) for image in images],
               roles=['inspection']+(['normal_reference_1'] if reference is not None else []),
               aspect_ratio_preserved=True,region_crops_sent=False)

def local_asset_path(directory:Path,recorded_path:str):
    name=recorded_path.replace('\\','/').rsplit('/',1)[-1]
    if name in {'','.','..'}:raise ToolFailure('vlm_unavailable','VLM 资产路径无效')
    return directory/name

class LocalVLM:
    def __init__(self,directory:Path,max_new_tokens=48):
        self.directory=directory;self.max_new_tokens=max_new_tokens;self.model=None;self.processor=None

    def load(self):
        if self.model is not None:return
        source=self.directory/'source.json'
        if not source.is_file():raise ToolFailure('vlm_unavailable','本地 VLM 资产未准备')
        metadata=read_json(source)
        if metadata['revision']!='a7da5b986cb59b408707209984f360a5f4ad7e47':raise ToolFailure('vlm_unavailable','VLM revision 不匹配')
        for item in metadata['files']:
            path=local_asset_path(self.directory,item['path'])
            if not path.is_file() or sha256(path)!=item['sha256']:raise ToolFailure('vlm_unavailable','VLM 文件校验失败')
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText
        torch.set_num_threads(4)
        self.processor=AutoProcessor.from_pretrained(str(self.directory),local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForImageTextToText.from_pretrained(str(self.directory),local_files_only=True,
                          trust_remote_code=False,use_safetensors=True,dtype=torch.float32,attn_implementation='eager').cpu().eval()

    def describe(self,image:Image.Image,reference:Image.Image|None=None,category='bottle',audit_directory:Path|None=None):
        images,messages,audit=prepare_payload(image,reference,category)
        self.load()
        text=self.processor.apply_chat_template(messages,add_generation_prompt=True)
        inputs=self.processor(text=text,images=images,return_tensors='pt',do_image_splitting=False)
        import torch
        shape=list(inputs['pixel_values'].shape)
        tokens=int((inputs['input_ids']==self.processor.image_token_id).sum())
        if shape!=[1,len(images),3,512,512] or tokens!=len(images)*self.processor.image_seq_len:
            raise ToolFailure('vlm_unavailable','VLM 图像数量、尺寸或占位符与输入不一致')
        audit.update(pixel_values_shape=shape,image_tokens=tokens,chat_template=text,
                     do_image_splitting=False,model_revision='a7da5b986cb59b408707209984f360a5f4ad7e47')
        if audit_directory is not None:
            audit_directory.mkdir(parents=True,exist_ok=True);records=[]
            mean=torch.tensor(self.processor.image_processor.image_mean).reshape(3,1,1)
            std=torch.tensor(self.processor.image_processor.image_std).reshape(3,1,1)
            for index,(prepared,role) in enumerate(zip(images,audit['roles'],strict=True)):
                path=audit_directory/f'input_{index+1}.png';prepared.save(path)
                encoded=inputs['pixel_values'][0,index].detach().cpu()*std+mean
                visible=(encoded.clamp(0,1).permute(1,2,0)*255).round().to(torch.uint8).numpy()
                encoder_path=audit_directory/f'encoder_{index+1}.png';Image.fromarray(visible).save(encoder_path)
                records.append(dict(role=role,path=path.name,sha256=sha256(path),
                                    encoder_path=encoder_path.name,encoder_sha256=sha256(encoder_path)))
            audit['images']=records;write_json(audit_directory/'input.json',audit)
        with torch.inference_mode():
            result=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False,use_cache=True)
        raw=self.processor.batch_decode(result[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
        return dict(candidate='unknown',evidence=[raw] if raw else [],uncertain=True,available=True,
                    backend=f'SmolVLM-500M CPU a7da5b9 / {INPUT_PROTOCOL}',raw_text=raw,verified=False)
