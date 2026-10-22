# 来源、许可证与使用范围

本项目默认非商业研究。代码、模型、数据分别遵守各自许可，不能把一个库的许可套用到全部资产。没有创建公开仓库或再分发数据/权重。

| 使用项 | 固定版本 / revision | 来源及许可 |
|---|---|---|
| Anomalib | 2.7.0 | [官方仓库](https://github.com/open-edge-platform/anomalib)，Apache-2.0；调用真实 PatchcoreModel、AUPRO，未拷贝整套训练工程 |
| PyTorch / torchvision | 2.12.0+cpu / 0.27.0+cpu | [PyTorch](https://github.com/pytorch/pytorch)，BSD 风格；CPU 官方 wheel 来源与哈希见锁文件 |
| timm | 1.0.22 | [官方仓库](https://github.com/huggingface/pytorch-image-models)，Apache-2.0 |
| ResNet18 权重 | 491b427b45c94c7fb0e78b5474cc919aff584bbf | [timm/resnet18.a1_in1k](https://huggingface.co/timm/resnet18.a1_in1k)，模型卡 Apache-2.0；46,807,446 字节，SHA256 80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612 |
| MVTec AD bottle / metal_nut | 镜像 c75b39616f84db43677bcc8228caaafaf5096d7f | [官方数据介绍](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)，CC BY-NC-SA 4.0；用户授权的 [foersben 镜像](https://huggingface.co/datasets/foersben/mvtec-ad/tree/c75b39616f84db43677bcc8228caaafaf5096d7f) 非官方托管；bottle 已实验，metal_nut 仅完成下载校验，详情见 DATASETS.md |
| SmolVLM-500M-Instruct | a7da5b986cb59b408707209984f360a5f4ad7e47 | [发布方模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)，Apache-2.0；主权重 1,015,025,832 字节，SHA256 d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2；完成校验后才允许推理 |
| Transformers | 5.20.2 | [官方仓库](https://github.com/huggingface/transformers)，Apache-2.0，纯本地加载，不启用 trust_remote_code |
| FastAPI / React / Vite | 0.115.6 / 19.3.1 / 8.3.5 | 官方包及 lock 中记录的分发；MIT 许可 |
| Radix Toggle Group | 1.2.1 | [官方组件文档](https://www.radix-ui.com/primitives/docs/components/toggle-group)，MIT；使用证据视图切换的键盘与单选语义，版本及传递依赖锁定在前端 lock 中 |
| EfficientAD Small teacher | 官方 Anomalib release，归档 SHA c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a | [Anomalib 发布资产](https://github.com/open-edge-platform/anomalib/releases/tag/efficientad_pretrained_weights)，发布方和 [原作者仓库](https://github.com/nelson1425/EfficientAD) 为 Apache-2.0；39,960,466 字节归档、10,779,695 字节 Small teacher，来源记录在 weights/efficientad/source.json |

完整 Python 传递依赖见 requirements.txt / requirements-linux.txt，前端见 frontend/package-lock.json。安装包各自的 LICENSE 保留在分发中；此表不是对所有传递依赖的重新许可。

EfficientAD teacher 已下载并核验，完整训练与评估仍未进行。实际 Anomalib 2.7.0 训练损失需要额外通用图像作为惩罚输入，不能仅凭模型名断言必须下载完整 ImageNet 或 Imagenette；二者仅是可考虑的数据来源。数据集图像的使用权需独立确认，Imagenette 仓库代码许可不自动覆盖所有原始图像权利。本地限定路径检查未建立授权数据清单，未下载此类数据或接受许可协议。详见 DATASETS.md。

metal_nut / capsule / pill 已获用户逐类扩展授权，但每次下载仍须检查该类许可与预算。当前只新增 metal_nut 到盘；没有把它或 VisA、LOCO 混入既有 bottle 结果。项目自身代码尚未选择对外发布许可，不默认继承上游许可证。
