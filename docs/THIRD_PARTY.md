# 第三方来源与许可

本项目默认非商业研究。程序、预训练权重和数据分别遵守自身许可。项目尚未选择对外发布许可证，不自动继承依赖许可。

| 使用项 | 版本 / 来源 | 许可与范围 |
|---|---|---|
| PyTorch / torchvision | 2.13.0 / 0.28.0，官方 PyPI CUDA 分发 | BSD 风格；CUDA 组件各自遵守 NVIDIA 分发条款 |
| Anomalib | 2.7.0，https://github.com/open-edge-platform/anomalib | Apache-2.0；直接调用 PatchcoreModel，未复制训练框架 |
| timm | 1.0.22，https://github.com/huggingface/pytorch-image-models | Apache-2.0 |
| FrEIA | 0.2，https://github.com/VLL-HD/FrEIA | MIT；Anomalib 的传递依赖，已核对本地分发 LICENSE |
| LangGraph | 1.2.16，https://github.com/langchain-ai/langgraph | MIT |
| vLLM | 0.27.1，https://github.com/vllm-project/vllm | Apache-2.0；独立 GPU Python 只读复用 |
| Qwen3-VL-8B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | Apache-2.0；revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b，17,534,339,512 字节 BF16 权重，显式准备后才加载 |
| ResNet18 权重 | https://huggingface.co/timm/resnet18.a1_in1k | 模型卡 Apache-2.0；revision 491b427b45c94c7fb0e78b5474cc919aff584bbf，46,807,446 字节，SHA256 80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612 |
| MVTec AD bottle | https://www.mvtec.com/research-teaching/datasets/mvtec-ad | CC BY-NC-SA 4.0；用户授权的 foersben/mvtec-ad 非官方镜像，固定 revision c75b39616f84db43677bcc8228caaafaf5096d7f，图像与许可文件逐一校验 |
| FastAPI / React / Vite / Radix | Python / frontend 锁文件记录版本 | MIT，保留分发内 LICENSE |

GPU 应用完整依赖及分发哈希在 requirements.txt，前端在 frontend/package-lock.json。FrEIA 唯一获准的源码归档为 34,270 字节，SHA256 e9f1732b4a238b85b0ad3bc14b279a653a6063d14152028e164b989582026bdc；setup 使用锁定 setuptools、关闭构建隔离，不临时安装未锁定构建依赖。其他包只允许 wheel。

旧 SmolVLM、EfficientAD 与 CPU 分支已从当前源码移除，其历史许可记录可在 Git 标签 pre-gpu-only-20261022 查看。物理旧权重未删除，不随 GPU 迁移包再分发。MVTec 图片使用权不由模型许可证覆盖，不默认支持商业用途。
