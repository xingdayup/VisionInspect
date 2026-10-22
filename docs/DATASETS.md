# 数据、权重与复用来源

当前 GPU 产品仅注册 MVTec AD `bottle`。保留既有合法资产，不隐式下载数据集、骨干或 VLM。

## MVTec AD bottle

- [官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)，许可 CC BY-NC-SA 4.0，仅非商业研究；保留署名、版权、许可与修改说明。
- 官方提供下载表单，前期连接失败后，用户明确授权评估 [foersben/mvtec-ad 镜像](https://huggingface.co/datasets/foersben/mvtec-ad)。固定 revision：`c75b39616f84db43677bcc8228caaafaf5096d7f`。
- 已核验数据卡、bottle/license.txt、readme.txt；许可与官方一致。未取得官方逐文件哈希，不能声称逐字节等同官方归档。
- 已有 357 个文件，156,570,806 字节：209 张正常训练图、20 张正常测试图、63 张异常测试图、63 张对应掩膜及两份说明。
- 异常类型 broken_large 20、broken_small 22、contamination 21。原图与掩膜为 900×900。
- 训练正常样本按 seed 42 固定分为训练 168 张、验证 41 张；测试集独立保留。GPU 正式拟合、校准和评估另建运行目录。
- 下载核验镜像 LFS SHA256 或 Git blob SHA1，并记录本地 SHA256、来源、revision 和实际大小。来源清单为本地 `data/mvtec_ad/source.json`，划分及文件哈希为 `data/manifests/bottle.json`。

数据检查拒绝文件缺失、解码错误、尺寸/掩膜不符、内容变化和划分不一致；不会触发 Anomalib 全集下载。迁移通过显式路径重定位保留原清单身份。

署名：Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger, “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection”, CVPR 2019。

## ResNet 骨干

[timm/resnet18.a1_in1k](https://huggingface.co/timm/resnet18.a1_in1k)，Apache-2.0，固定 revision `491b427b45c94c7fb0e78b5474cc919aff584bbf`。仅取得 `model.safetensors`：46,807,446 字节，SHA256 `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`。

这是 ImageNet-1k 预训练权重，不等于取得 ImageNet 图像数据。初始化使用 `pre_trained=False`，再严格加载本地参数。来源记录位于 `weights/resnet18.a1_in1k/source.json`。

## GPU VLM

[Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，Apache-2.0，固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。四份 BF16 权重合计 17,534,339,512 字节；元数据、文件大小及 SHA256 固定在 `scripts/pai_vllm.py`。

当前只完成平台原生运行时与架构检查，尚未验证本项目 Qwen 权重加载及生成。显式预检许可和至少 50 GiB 可用空间后才允许下载；GPU 迁移包不包含 Qwen。推理只访问已校验的本地模型和本机 vLLM。

## 算法与历史资产

Anomalib 2.7.0，Apache-2.0，参考源码 commit `9fb1337c0ac866797f4eedcaa0972d0f24a84f19`。本项目使用 PatchcoreModel、k-center-greedy、异常图与 AUPRO，另外实现资源控制、校验、校准及输出管理；未复制 AnomalyClaw / VELM 代码。

旧 metal_nut 数据、SmolVLM、EfficientAD 权重和实验输出未物理删除，当前产品不加载、不迁移这些资产。其原始来源、许可与检查记录在 Git 标签 `pre-gpu-only-20261022` 的历史文档中。capsule / pill 仅有前期元数据核验，不描述为已下载。扩大类别或取得其他模型仍需逐项检查许可和资源。