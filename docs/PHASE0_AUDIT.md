# VisionInspect-Agent：Phase 0 项目与环境审计

审计日期：2026-10-21，Asia/Shanghai。状态：仅完成项目、环境、上游源码与许可审计；等待用户确认 Phase 1。交接文档已完整阅读，未修改。本文中“实测 / 源码核验”和“建议 / 预算”分开记录。

## 1. 已完成的工作与结论

1. 项目根目录为 `E:\秋招\项目\VisionInspect`，审计开始时只有交接文档，无代码、Git 仓库、数据、模型或专属环境；不存在可直接运行的项目。
2. Windows 10、i9-9900K、约 64 GB 内存足以开展单类别 CPU PatchCore 实验。P620 仅 2 GB 显存，初次空闲 710 MiB，不作为首个基线的运行设备。
3. 当前裸 `python` 指向 Conda base，裸 `pip` 指向另一安装路径；现有 `test` 环境存在 Pillow DLL 运行错误，系统 Python 的 torch/torchvision 版本也不匹配。建议只创建项目 `.venv`，不修补或安装到这些已有环境。
4. Anomalib 2.7.0 是已发布版本，CPU wheel 路径可访问；拟固定 Python 3.12.12、torch 2.12.0+cpu / torchvision 0.27.0+cpu，先做完整依赖求解与导入验证。
5. AnomalyClaw 的真实许可是 **CC BY-NC 4.0**，不能因为 GitHub 检测为 `NOASSERTION` 而当作无限制代码。MVTec AD 是 **CC BY-NC-SA 4.0**。二者均需明确非商业限制。
6. Anomalib 的 MVTecAD 默认验证集等于测试集，且缺少类别目录会自动下载全集；Phase 1 必须覆盖这些默认行为。

主机实测、错误信息与命令见 [ENVIRONMENT.md](ENVIRONMENT.md)；原始本地检查输出见 [phase0_local.log](audit/phase0_local.log)。没有安装、下载模型或数据，没有模型效果指标。

## 2. 参考仓库：许可、依赖、权重与真实入口

本节是 README 和源码的静态核验，**未克隆、安装或执行任何参考仓库**。文件链接固定到本次核验的 commit，避免将未来默认分支变化混入结果。所有仓库本次元数据均未标为 archived。

### Anomalib：首阶段算法基础设施

- 仓库：[open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)；审计固定 release `lib/v2.7.0`，commit `9fb1337c0ac866797f4eedcaa0972d0f24a84f19`，发布于 2026-10-06。
- [LICENSE](https://github.com/open-edge-platform/anomalib/blob/9fb1337c0ac866797f4eedcaa0972d0f24a84f19/LICENSE)：Apache-2.0。复用时保留许可证、版权与适用 NOTICE；模型、数据的许可分别核验。
- [pyproject.toml](https://github.com/open-edge-platform/anomalib/blob/9fb1337c0ac866797f4eedcaa0972d0f24a84f19/pyproject.toml)：Python >=3.10；CPU extra 要求 torch >=2.10、torchvision >=0.25；lightning >=2.6 且排除 2.6.2/2.6.3，torchmetrics >=1.8.2；还包括 timm、Kornia、OpenCV headless、scikit-image/learn、Pandas 等。不能照旧教程安装低版本 torch 后声称支持当前 Anomalib。
- [Patchcore 实际入口](https://github.com/open-edge-platform/anomalib/blob/9fb1337c0ac866797f4eedcaa0972d0f24a84f19/src/anomalib/models/image/patchcore/lightning_model.py)：`Patchcore` + `MVTecAD` + `Engine.fit()`；支持 `Engine.predict()` / `Engine.test()`；CLI 为 `anomalib train --model Patchcore --data anomalib.data.MVTecAD`。这些是上游入口，未在本机执行。
- 默认是 `wide_resnet50_2`、layer2/layer3、coreset 0.1、邻居数 9；训练是提取预训练特征并做 k-center-greedy 子采样，不是从零训练 CNN。首次创建 `pre_trained=True` 模型可能通过 timm 下载骨干权重。
- 首阶段只装固定版本核心包与明确的 CPU torch/torchvision；不装 `[full]`、VLM、Studio、OpenVINO 或视频组件。完整锁文件和 wheel 兼容性留待 Phase 1 求解确认。

### AnomalyClaw：后续 Agent 设计参考

- 固定 commit `1a0c4b4e8e25d8a231f694b264a6669925b6d0b6`。
- [LICENSE](https://github.com/jam-cc/AnomalyClaw/blob/1a0c4b4e8e25d8a231f694b264a6669925b6d0b6/LICENSE)：CC BY-NC 4.0，覆盖仓库代码、脚本、文档等；第三方专家和数据保留各自许可。后续若复用代码必须记录来源、修改与非商业边界；不得默认纳入可商用产品。
- [requirements.txt](https://github.com/jam-cc/AnomalyClaw/blob/1a0c4b4e8e25d8a231f694b264a6669925b6d0b6/requirements.txt)：大多未锁版本，包括 torch、torchvision、transformers、NumPy、OpenCV、openai>=1.40 等。vLLM 是可选服务端依赖；研究仓库不是完整锁定的部署环境。
- [真实入口](https://github.com/jam-cc/AnomalyClaw/blob/1a0c4b4e8e25d8a231f694b264a6669925b6d0b6/benchmark/scripts/agent_v12.py)：依赖 agent_v9、agent_v11、agent_prompt_v10、agent_tools_v8、infer，已核实这些文件存在。D1 可用 manifest 单域运行，需原图、参考图与 VLM endpoint；复算仓库自带结果的聚合脚本不等于实际运行 Agent。
- [本地启动脚本](https://github.com/jam-cc/AnomalyClaw/blob/1a0c4b4e8e25d8a231f694b264a6669925b6d0b6/benchmark/scripts/launch_qwen35_replicas.sh) 实际以 GPU 0–3 各启动一个 FP8 27B 实例，单实例 tensor-parallel=1，并非四卡自动切分一个模型。依赖 Bash、`.venv_qwen35/bin/python`、已有 `MODELS_DIR` 和模型目录。脚本日志默认 `/tmp`，与本项目指定缓存目录要求不同，不能原样执行。
- 27B 参数按 FP8 估算仅参数就约 27 GB（25.1 GiB），BF16 约 54 GB（50.3 GiB），尚不含视觉编码器、KV cache、工作区；四个副本再放大需求。P620 不具备这条路径的资源条件。托管端点可减少本地显存需求，但费用、数据外发、token 预算和授权仍未确定；当前不调用。
- Phase 1 最小成本选择：完全不运行 AnomalyClaw / VLM，付费模型调用数为 0；仅保留研究参考定位。

### VELM：视觉专家与语义分类衔接参考

- 固定 commit `baad0b93d0ee93239fe34205f9986bbee90855fd`。
- [LICENSE](https://github.com/Sassanmtr/VELM/blob/baad0b93d0ee93239fe34205f9986bbee90855fd/LICENSE)：MIT；版权行仍写 `2024 Your Name`，属于归属信息不完整，应保留原文并在后续复用前记录这一问题，不能自行补成其他作者。
- README 建议 Python 3.9；[requirements.txt](https://github.com/Sassanmtr/VELM/blob/baad0b93d0ee93239fe34205f9986bbee90855fd/requirements.txt) 未锁版本，含 transformers、accelerate、gem_torch、qwen-vl-utils、openai 和未固定 commit 的 OpenAI CLIP Git 依赖，还未显式列全源码导入的 torch / Pillow / tqdm 等。不能把该依赖文件直接并入主项目。
- 真实流程：先准备外部视觉专家热图，`create_contour.py` 生成轮廓，`generate_prompts.py` 生成输入，`run_llm.py` 调 GPT 或 Qwen，`eval.py` / `anomaly_vs_defect.py` 评估；不是一条命令自动训练检测器的完整平台。
- [run_llm.py](https://github.com/Sassanmtr/VELM/blob/baad0b93d0ee93239fe34205f9986bbee90855fd/run_llm.py) 实际加载 `Qwen/Qwen2-VL-7B-Instruct`，使用 `torch_dtype='auto', device_map='auto'`；后续函数又将模型 `.to(device)`，与自动分配/卸载可能有冲突。按 7B BF16 仅参数约 14 GB，不能放入 P620。README 目录树写的 `run_llm_hm.py` 与实际 `run_llm.py` 也存在差异。
- MVTec-AC / VisA-AC 衍生标注需单独检查来源与许可，不能把代码的 MIT 自动套用到这些数据。

### EfficientAD 参考实现：延后至对照阶段

- 固定 commit `3e2bda221cafd4e34354bbf27cd23a3b5c0d67e6`，默认分支为 `master`。
- [LICENSE](https://github.com/ZhuJD-China/EfficientAD/blob/3e2bda221cafd4e34354bbf27cd23a3b5c0d67e6/LICENSE)：Apache-2.0。数据、预训练教师权重及评估工具的独立来源仍需记录。
- [requirements.txt](https://github.com/ZhuJD-China/EfficientAD/blob/3e2bda221cafd4e34354bbf27cd23a3b5c0d67e6/requirements.txt) 固定 torch 2.1.2、torchvision 0.16.2、NumPy 1.26.4 等，并配置 cu118 extra index；environment.yml 使用 Python 3.10、pytorch-cuda=11.8。与当前 Anomalib 2.7.0 的 torch 最低要求不兼容，不应合装到基线环境。
- 真正入口包括 `efficientad.py`（训练）、`test.py`、`inference.py`（热图）、`conversion.py`（ONNX）；训练命令为 `python efficientad.py --dataset mvtec_ad --subdataset bottle --mvtec_ad_path ... --model_size small --weights models/teacher_small.pth`。
- 远程目录元数据显示普通克隆即带教师权重：small 10,779,715 字节，medium 32,110,923 字节。本阶段没有克隆或获取这两个二进制。
- 论文完整配置还涉及 ImageNet penalty 数据；默认关闭该项不能冒称完整论文复现。后续优先评估 Anomalib 内 EfficientAD，避免为一个对照引入另一套旧依赖。

### VisA 来源修正与 LOCO 范围

- 交接文档的 `wxs56837/visa_dataset` 是 fork，上游是 [amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)，上游固定 commit `2a692ab575001cbde74d402d897a7286086c6199`。
- 上游代码 Apache-2.0；数据依据 README / `LICENSE-DATASET` 为 CC BY 4.0。代码和数据许可不同，需分别署名；当前未下载。
- 原始 VisA 并非直接就是 MVTec 目录格式；需按 `split_csv/1cls.csv` 与 `utils/prepare_data.py` 转换，像素掩膜会从多类重编码为二值。首阶段不纳入。
- LOCO 含逻辑异常，按项目定义延后；本阶段不下载、不引入实验。

## 3. 依赖与运行风险表

| 风险 | 证据 | Phase 1 处理 |
|---|---|---|
| base / 系统环境污染 | python 与 pip 路径不同 | 新建唯一 `.venv`；所有命令显式指定项目解释器 |
| 现有 test 环境不可直接复用 | torch CPU 运算通过，但 torchvision 的 Pillow `_imaging` DLL 导入失败；pip check 却通过 | 不修补其他环境；新环境独立做 Pillow、torchvision、torch 导入和模型验证 |
| GPU 架构与显存限制 | CC 6.1、2 GB，约 710 MiB 空闲 | 第一条路径用 CPU；不升级驱动/CUDA；后续如需 GPU，单独核验 wheel 架构再决策 |
| CUDA 版本误判 | nvidia-smi 13.0 不代表安装 toolkit，也不代表 torch 包含 runtime | 记录 driver / toolkit / torch.version.cuda 三项，CPU 构建无需 CUDA toolkit |
| CUDA 13 不覆盖旧架构 | NVIDIA 发布说明移除 Maxwell/Pascal/Volta 的离线编译及库支持 | 不依据驱动显示 13.0 就安装 cu130；CPU 路径绕开此问题 |
| 新旧仓库依赖冲突 | Anomalib >=torch 2.10；EfficientAD 固定 torch 2.1.2 | 首阶段仅用 Anomalib，不复制旧 requirements |
| 传递依赖与 Windows DLL | 未运行完整 solver；现有环境已发生 DLL 错误 | 安装前生成解析报告与锁文件，优先 wheel；禁止失败后擅自改系统 DLL/驱动 |
| 默认测试集泄漏 | MVTecAD `val_split_mode='same_as_test'`，PostProcessor 默认按验证 F1 求阈值 | 显式 `from_train`，关闭默认后处理，正常验证集校准阈值 |
| 意外下载全集 | MVTecAD.prepare_data 在类别目录不存在时使用全集下载配置 | 在调用 Engine 前预检全部路径/文件，缺失即退出；数据必须事先人工或受控准备 |
| 网络链路不一致 | PyPI/HF 本机可访问；MVTec 本机 SSL 握手失败 | 保留证书校验；下一阶段先验证正式单类别链接；无法获取时使用用户合法已有副本，不切换不明镜像 |
| VLM 费用与外发 | 上游要求 endpoint，当前无调用授权 | Phase 1 不设置 VLM，不读取或创建 API key |
| 数据/代码商业限制 | MVTec AD NC-SA、AnomalyClaw NC | 在数据说明/README 中分别记录，不重新打包分发原始数据 |

NVIDIA 依据：[CUDA 13.0.1 发布说明](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html)。torch 配对依据：[官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。

## 4. MVTec AD bottle 数据与许可方案

数据只用于非商业研究原型。采用 [MVTec AD 官方来源](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)及官方提供的下载方式，保留 CC BY-NC-SA 4.0 许可、来源、获取日期、压缩包 SHA256 和解压文件清单。当前官方页面显示下载表单，本阶段未提交表单，未取得可验证的单类别归档 URL / Content-Length / 官方归档校验值。

[官方原始论文 Table 1](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf) 的 bottle 数据是 209 张正常训练图、20 张正常测试图、63 张异常测试图，900×900，3 个缺陷组；实际到盘后必须逐文件重新核对。这些数字是公开资料，不是本地已经下载的样本数量。异常目录预期为 `broken_large`、`broken_small`、`contamination`；不能宣称 bottle 覆盖划痕等全部目标缺陷。

```text
data/mvtec_ad/bottle/
  train/good/*.png
  test/good/*.png
  test/broken_large/*.png
  test/broken_small/*.png
  test/contamination/*.png
  ground_truth/broken_large/*_mask.png
  ground_truth/broken_small/*_mask.png
  ground_truth/contamination/*_mask.png
```

检查图像可解码、尺寸/通道、正常/异常标签、每张异常图与 mask 一一对应、mask 二值语义、无训练/验证/测试重叠。没有 normal 的实体 mask 时按全零处理。文件缺失或类别结构错误立即停止，不能触发库的自动全集下载。若官方暂只提供全集，不能把全集当作默认回退，须重新报告下载规模并取得用户确认。

## 5. 当前硬件下的具体单类别 PatchCore 方案（尚未执行）

### 环境与目录

唯一环境：项目根目录 `.venv`，使用现有 base 解释器仅创建 venv，不向 base 安装。候选核心版本：

```text
Python       3.12.12（现有解释器创建隔离环境）
torch        2.12.0+cpu
torchvision  0.27.0+cpu
anomalib     2.7.0
lightning    2.6.1
timm         1.0.22
```

上述版本的包存在性与 Python 声明已核对，但完整组合尚未通过 solver / 安装 / 导入；传递依赖必须在安装前解析并锁定。对 `pytorch-lightning` 同时约束到 2.6.1，防止 lightning 的传递依赖漂移到不同版本。若有冲突，先汇报，不无约束升级。

下载、HF/torch/uv 缓存、安装临时目录、数据、权重、输出均放在 E 盘本项目下。先创建 `.gitignore` 排除 `.venv/`、`.cache/`、`.tmp/`、`data/`、`weights/`、`outputs/`、`logs/`、`.env`、模型文件等，再考虑 Git 初始化；不能提交密钥/原始数据/权重。

以下是 **Phase 1 候选命令，当前没有执行**：

```powershell
Set-Location -LiteralPath 'E:\秋招\项目\VisionInspect'
$env:UV_CACHE_DIR = "$PWD\.cache\uv"
$env:PIP_CACHE_DIR = "$PWD\.cache\pip"
$env:HF_HOME = "$PWD\.cache\huggingface"
$env:TORCH_HOME = "$PWD\.cache\torch"
$env:TEMP = "$PWD\.tmp"
$env:TMP = "$PWD\.tmp"
# 先建立上述项目目录；这些变量只在当前进程设置，不写入系统环境。
uv venv --python 'D:\SOFTWARE\miniconda\python.exe' '.venv'
# 先以 requirements.in + 精确 torch/vision 来源生成锁文件并检查解析报告。
# 锁文件通过后才执行 uv pip sync；不运行裸 pip install。
# 随后执行 uv pip check --python '.venv\Scripts\python.exe'。
# 再用 .venv\Scripts\python.exe 做导入 / CPU 张量 / 实际模型验证。
```

requirements 输入应显式固定 CPU wheel 来源（cp312/win_amd64 已核验 HEAD 200），避免安装器默默选 CUDA 构建。候选锁定/安装入口为 `uv pip compile requirements.in --python '.venv\Scripts\python.exe' --generate-hashes -o requirements.lock`，检查锁定结果后执行 `uv pip sync requirements.lock --python '.venv\Scripts\python.exe' --require-hashes`；输入文件当前尚未创建，这两条命令也未执行。实际完整安装命令和锁文件将在 Phase 1 形成并留存，当前不伪造一份已验证 requirements。

### 执行顺序与配置

1. 安装前完成依赖解析/资源预算，创建隔离环境并按锁文件安装。验证 Pillow 图片解码、torch CPU 矩阵运算、torchvision 算子、anomalib/timm 导入，记录 Python / 完整依赖版本。任何 DLL 错误先停在项目环境内诊断。
2. 只受控获取 bottle 和一个骨干权重。选 `timm/resnet18.a1_in1k`，model.safetensors 的 HEAD 大小为 46,807,446 字节（44.64 MiB）；[模型卡](https://huggingface.co/timm/resnet18.a1_in1k) 标为 Apache-2.0。固定模型 revision 与本地 SHA256，不额外下载同一模型的 bin 副本。
3. 从正常训练集固定抽出 20% 作为验证集，`seed=42`，显式 `val_split_mode='from_train'`。按上游分割舍入规则，若本地确认 209 张，则预计训练 168 张 / 验证 41 张；保存真实文件名和 hash 清单。测试集完整保留 83 张，训练/校准阶段不得读取测试标签做选择。
4. 先以正常训练/验证数据中极少量图片做 **128×128 冒烟检查**，验证特征提取、coreset、预测图与保存加载；不报告正式指标。通过后按预先冻结的 **256×256、ResNet18、layer2/layer3、coreset=0.01、num_neighbors=9、batch=1、num_workers=0、CPU、FP32、1 epoch** 运行完整正常训练划分。设 4 个 torch 计算线程、单任务；不是同时启动多个训练。
5. 使用 `Patchcore.configure_pre_processor(image_size=(256,256))`，保留完整图像、ImageNet 标准化，不额外中心裁剪；模型设置 `post_processor=False`、`evaluator=False`，避免默认 F1 阈值/验证指标隐含异常标签假设。Engine 显式 `accelerator='cpu', devices=1`；模型最小包装与配置文件在 Phase 1 才创建。
6. 仅依据正常验证集的原始图像分数，以事先声明的 **95% 经验分位数（higher 插值）** 设图像阈值；这只是正常分数校准，不能声称优化了异常召回，也不保证总体误报率为 5%。41 张验证样本的尾部分位数很不稳定，应在报告中写明。像素层先报告无阈值指标，暂不为了 Dice/IoU 在测试 mask 上调阈值。
7. 保存配置/阈值后只对完整测试集做冻结评估：图像 AUROC、Average Precision、固定阈值 F1 / 异常召回率 / 混淆矩阵；像素 AUROC、AUPRO（FPR 上限 0.3）。本次像素指标在 256×256 网格计算，mask 最近邻缩放；热图用连续插值，保存原始浮点 anomaly map，明确分辨率协议，不混同 900×900 原图评估。
8. 保存每图分数/标签/路径、原始热图、叠加 PNG、模型 checkpoint、split 清单、配置、模型 hash、依赖锁文件、运行日志和 metrics.json。测量 fit / 校准 / test 分段墙钟时间、进程峰值内存、预热后推理延迟 p50/p95；CPU 运行显存项为“不适用”，不能填 0 冒充 GPU 峰值。

这是一条适应本机资源的 **PatchCore-ResNet18 基线**，区别于上游默认 Wide-ResNet50-2、0.1 coreset 的配置；不得写成论文默认配置复现或预填准确率。后续对照必须固定同一 split、预处理和指标协议。未经新的阶段汇报，不扩展全类别、EfficientAD、VLM、Agent 或前端。

### 资源预算（估算，非实测模型消耗）

| 项目 | 预算或计算依据 |
|---|---|
| 环境安装网络 | 约 0.3–1.5 GB 预算，含传递依赖；精确总量等依赖解析后求和。已核验的两份 CPU wheel 合计 126,921,936 字节，不是数 GB CUDA wheel |
| 骨干权重网络 | 46,807,446 字节，一份 safetensors；已测 HEAD，未下载正文 |
| bottle 网络 | 尚无实际归档长度；暂按不超过 1 GB 的单类别下载预算，拿到正式链接后先 HEAD 再确认，不能把此预算称为实际数据大小 |
| bottle 磁盘 | 预留 1–2 GiB（归档、解压、完整性检查）；292 张 RGB 900×900 原始像素约 676.7 MiB，63 张单通道 mask 约 48.7 MiB，仅用于未压缩量级估算 |
| 全部项目增量磁盘 | 建议预留 15 GiB，覆盖环境/缓存/数据/输出/临时文件；E 盘当前空间充足，避免使用 C 盘大缓存 |
| 正常训练特征 | 168×32×32×384×4 = 264,241,152 字节，约 252 MiB；256 分辨率、ResNet18 layer2/3 合并的理论原始张量 |
| coreset | 约 1,720×384×4 = 2.52 MiB；不是整个模型的 checkpoint 大小 |
| 单图距离矩阵 | 1,024×1,720×4 约 6.72 MiB；实际内存还含投影、复制、骨干、框架与指标缓冲 |
| 运行内存 | 为完整实验预留 4–8 GiB；峰值未知，先测小样本；超过 12 GiB 或异常增长就停止诊断 |
| GPU 显存 | CPU 路径不分配模型显存；不要求退出用户桌面进程释放显存 |
| 运行时长 | 未测，不能给已完成耗时。先设冒烟检查 10 分钟、完整 fit 60 分钟的观察/超时预算；coreset 是主要不确定项，超时停止并汇报，不自动扩大算力或更改实验协议 |
| 下载时间 | 未测速；仅作算术示例，2.5 GB 在 10/50 Mbps 下理论约 33/7 分钟，忽略 TLS、重试和解压开销，不是本机网络实测 |

若小样本显示 1% coreset 超时，先报告实际耗时并评估 0.1% coreset 或更小分辨率的独立配置，保留不同实验身份；不使用测试集效果决定降配方案。

## 6. 本阶段环境变化、问题和确认事项

新增内容只在 `docs/`：本审计报告、环境记录、只读检查脚本、实际检查日志、参考版本快照。原交接文档 SHA256 保持不变；无应用代码、模型、数据、环境安装或系统修改。

仍未知或未通过：完整依赖组合能否安装/导入、MVTec 正式单类别归档可达性及大小、本地数据清单、PatchCore 实际效果/耗时/内存、GPU torch 运行、容器部署和前端功能。这些均未被描述为已经完成。

需要用户确认后才进入下一阶段：

- 是否按本报告的 **E 盘项目 `.venv` + CPU + bottle + PatchCore-ResNet18** 路径开展 Phase 1，允许在项目目录内下载固定依赖、单类别合法数据和这一份约 46.8 MB 权重，资源预算按上表执行。
- 数据来源若使用官方表单，由用户提供已有合法 bottle 路径或正式下载入口也可；若正式入口只能取全集、需提交个人信息或费用，则停下来重新报告，不自动接受这些额外事项。

后续每个阶段继续报告实际命令、验证输出、环境变化、资源消耗与已知问题。当前在 Phase 0 停止，等待确认。
