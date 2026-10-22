# 数据、权重与复用来源

## MVTec AD bottle

- 原始来源：https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- 许可：CC BY-NC-SA 4.0，仅非商业研究；保留版权、署名、原始许可与修改说明。当前不重新分发数据。
- 官方页面本次仍显示下载表单，本机 curl/PowerShell TLS 握手失败。没有提交个人信息，没有改变代理或系统证书设置。
- 用户随后明确授权评估第三方镜像：https://huggingface.co/datasets/foersben/mvtec-ad
- 镜像固定 revision：`c75b39616f84db43677bcc8228caaafaf5096d7f`。
- 已核验镜像 README 数据卡、bottle/license.txt、readme.txt，许可与官方一致。镜像托管者的声明不能替代官方归档校验；目前未获得官方逐文件哈希，不能宣称逐字节等同官方版本。
- bottle 清单：357 个文件，156,570,806 字节。包含 209 张正常训练图、20 张正常测试图、63 张异常测试图、63 张掩膜，以及两份说明文件。
- 异常：broken_large 20、broken_small 22、contamination 21；不推断该类别含划痕等其他标签。
- 下载逐文件核验大小和镜像 LFS SHA256；另外记录本地 SHA256、固定 revision、来源及实际字节数。清单位于本地 `data/mvtec_ad/source.json`，不提交原始数据。
- 完整检查后另生成 `data/manifests/bottle.json`，保存真实 split、文件名和哈希。数据变化会在拟合/校准/测试前被拒绝。

下载前元数据预检使用标准库 GET：

```text
https://huggingface.co/api/datasets/foersben/mvtec-ad
https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/<revision>/bottle?recursive=true&limit=1000
https://huggingface.co/datasets/foersben/mvtec-ad/raw/<revision>/README.md
https://huggingface.co/datasets/foersben/mvtec-ad/raw/<revision>/bottle/license.txt
https://huggingface.co/datasets/foersben/mvtec-ad/raw/<revision>/bottle/readme.txt
```

应将元数据分别保存到本地日志中的 mirror.txt、bottle-tree.txt、mirror-card.txt、mirror-license.txt、mirror-readme.txt 后审查。准备脚本会重新以固定 revision 获取完整树，拒绝分页未处理、超预算、路径越界或非 bottle 文件，不调用 Hugging Face 的全库下载。

原始论文署名：Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger, “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection”, CVPR 2019。

## 骨干权重

- 模型卡：https://huggingface.co/timm/resnet18.a1_in1k
- Apache-2.0；ImageNet-1k 预训练权重，不下载 ImageNet 数据。
- revision：`491b427b45c94c7fb0e78b5474cc919aff584bbf`。
- 仅 model.safetensors，46,807,446 字节；SHA256 `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`。
- 本地 source.json 记录 URL、revision、字节数、许可和哈希。模型初始化使用 pre_trained=False，再严格加载本地特征层参数，禁止隐式联网。

## 算法代码

Anomalib 2.7.0（Apache-2.0），参考源码固定 commit `9fb1337c0ac866797f4eedcaa0972d0f24a84f19`。本项目调用其 PatchcoreModel、k-center-greedy、异常图生成与 AUPRO 实现，自行实现资源控制、校验、校准、命令行和输出管理。未复制 AnomalyClaw / VELM 代码。Phase 1 时尚未运行 VLM；后续已完成本地 SmolVLM 与 C/D 对照，见 COMPARISON.md。

## 2026-10-22 多类别准备状态

用户已允许 metal_nut → capsule → pill 逐类扩展，要求每类下载前核验许可和资源。同一固定 revision 下每类 license.txt 均已检查为 CC BY-NC-SA 4.0。下表为 metadata 字节数，不含依赖、模型或实验输出；不代表三类均已下载。

| 类别 | 文件 / 字节 | train good | test good / anomaly | 当前状态 |
|---|---:|---:|---:|---|
| metal_nut | 430 / 165,862,477 | 220 | 22 / 93 | 已下载，逐文件哈希与完整解码通过；原图 / mask 均 700×700，正常图像 RGB；尚未运行严格 check-data、生成 split 或拟合评估 |
| capsule | 462 / 404,753,963 | 219 | 23 / 109 | 仅元数据与许可核验，未下载 |
| pill | 577 / 275,077,521 | 267 | 26 / 141 | 仅元数据与许可核验，未下载 |

metal_nut 异常类型实际目录为 bent 25、color 22、flip 23、scratch 23，并有 93 张对应掩膜；provenance 位于 data/mvtec_ad/metal_nut-source.json，不覆盖 bottle 的 source.json。配置 configs/patchcore_cpu_metal_nut.yaml 只记录已核实的数据结构和沿用的实验参数；存在配置不等于训练完成或服务支持。

执行命令：`.venv/Scripts/python.exe -B scripts/prepare_category.py metal_nut`，退出 0，完整 stdout/stderr 在 logs/metal_nut-download.log。下载前快照在 logs/metal_nut-download-preflight.json。每个文件核验镜像 LFS SHA256 或 Git blob SHA1，并保存本地 SHA256。此后本次目录整理没有继续下载数据或启动新实验。

## 本地可选 VLM

HuggingFaceTB/SmolVLM-500M-Instruct，Apache-2.0，固定 revision `a7da5b986cb59b408707209984f360a5f4ad7e47`。13 个本地资产共 1,019,893,574 字节，主权重 1,015,025,832 字节，SHA256 `d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2`。已核验并在 Python socket 禁用条件下推理；只输出未经验证的候选描述，不提供正式缺陷类型分类准确率。

## EfficientAD：区分三类输入

1. **预训练 teacher 权重**：已取得官方 Anomalib release 归档 39,960,466 字节，SHA256 `c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a`；仅提取 Small teacher 10,779,695 字节，SHA256 `a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d`。该权重不等于正常训练数据或正则化图像。
2. **MVTec 正常训练 / 验证图像**：用于 teacher 特征统计、student / AE 训练与正常验证 map 分位数、图像阈值校准。bottle 数据已具备，但尚未进行正式 EfficientAD 训练。
3. **额外通用图像**：安装的 Anomalib 2.7.0 低层 EfficientAdModel.compute_losses 会对 batch_imagenet 调用 student，完整训练路径不能把 None 当成有效正则化输入；推理路径不需要这些额外图像。原作者允许禁用惩罚项的变体，若使用必须单列消融，不能冒称当前完整 B 组。没有自动调用会获取 Imagenette 的 Lightning 训练钩子。

用户未提供授权的 ImageNet / Imagenette 路径。实际执行 `scripts/audit_regularization_data.py`：检查项目 data、datasets、.cache、.cache/huggingface/datasets，以及 E:/data、E:/datasets、D:/data、D:/datasets、用户 data / datasets 下常见 ImageNet / Imagenette 名称和显式数据路径环境变量。候选目录未发现，`data/manifests/imagenette-authorized.json` 不存在。记录在 logs/regularization-data-audit.json；仅覆盖已列路径，不能据此声称全机绝无相关数据，也不会从目录名推定授权。

已完成 teacher 加载、低层 CPU 适配、三项损失有限性、teacher 冻结与 checkpoint 重载检查。outputs/20261022-003242-efficientad-smoke-205125 是 **合成张量架构 / 吞吐冒烟**：7 个优化步骤，预热 2 步后计时 5 步，p95 2.0346 秒，进程树峰值 1,122,672,640 字节，39.10 秒退出 0。不是数据集训练、不是正式 B 组，也没有测试指标。按 70,000 步估算纯计算约 39.6 小时，尚不含真实图像读写和校准，超过现有 60 分钟拟合预算；不会自动缩减步数。

后续完整实验须先建立有来源、适用许可、授权说明、文件哈希与完整性的通用图像清单，再核对下载 / 存储资源和计算预算，确认可执行后补齐正式训练、校准与评估。不得用随机图片或 MVTec 测试图替代正式正则化数据，也不自动获取完整 ImageNet / Imagenette。默认非商业研究，不提交个人信息、不调用付费 API。
