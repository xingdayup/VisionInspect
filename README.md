# VisionInspect-Agent

基于 PatchCore、工具调用和证据复核 Agent 的中文质检工作台，支持单图、批量任务、可追溯证据、逐项人工复核及报告导出。

已验证的视觉基线是 **MVTec AD bottle → PatchCore-ResNet18 CPU → FastAPI → Web 工作台**。新工作流使用 **LangGraph + 本地 vLLM / Qwen3-VL-8B** 进行多步复核；本机工程实现与测试已完成，PAI 上的真实 GPU 加载和生成仍待验收。项目面向非商业研究，候选结论不构成工业放行授权。

## v0.4：业务闭环与迁移入口

- [业务功能](docs/BUSINESS_V1.md)：每批最多 100 张，检测、不确定样本多步复核、逐项人工处理、JSON/CSV 导出。
- [迁移到另一台 PAI 电脑](docs/MIGRATION_PAI.md)：Git bundle 离线 clone、独立校验资产包、隔离应用环境、显式准备 GPU 模型。
- [本轮验证与待办](docs/UPGRADE_STATUS.md)：区分实现、单元测试、真实 CPU 验证和待 GPU 验证。

工作台新增“批量质检”。“Agent 多步复核”需要完成 PAI GPU 配置；Windows 可继续选择“单模型”或明确标记的旧版规则流程。原始检测、Agent 复核和人工记录分别保存，人工记录不改写机器结果。

代码已纳入本地 Git。`cpu-baseline-v0.3.0` 保留改造前版本；数据、权重、`.venv`、node_modules、结果和密钥不进入提交。生成迁移包：

```powershell
.\.venv\Scripts\python.exe -B scripts/package_migration.py
```

完整新安装使用 `requirements-agent.txt`（Windows）或 `requirements-agent-linux.txt`（Linux）；原 requirements.txt 仍保留 CPU 阶段锁。PAI 详细步骤和资源核验见迁移说明。

## 已有 CPU 阶段与其他实验状态

更新日期：2026-10-22。以下区分实现、实测与待办，完整索引见 [项目文档](docs/README.md)。

| 内容 | 状态与依据 |
|---|---|
| bottle 视觉基线 | 已完成拟合、正常验证集校准、83 张完整测试集冻结评估及独立离线推理 |
| 六个工具与条件 Agent | 已实现输入输出契约、调用预算、超时、有限重试、失败转人工复核和完整轨迹 |
| A / C / D 对照 | 已完成 PatchCore 单模型、固定本地 VLM 复核、动态工具调用的真实 bottle 实验 |
| 中文 Web 工作台 | 上传、进度、热图、参考图、区域裁剪、轨迹、历史、研究复核及 JSON 导出已实际验证 |
| metal_nut | 已下载并校验 430 个文件、165,862,477 字节，图像解码通过；尚未生成实验 split、拟合或评估，未注册到工作台 |
| capsule / pill | 已核验元数据与许可，尚未下载或实验 |
| EfficientAD | 本地 teacher 权重与 CPU 架构冒烟已验证；完整 B 组未训练、未评估，未注册为可用模型 |
| Docker | Dockerfile、Linux CPU 哈希锁与基础镜像 digest 已准备；本机 WSL / Docker 引擎阻塞，未完成构建或运行验收 |

## 快速启动：使用当前电脑已有资产

本目录已有 `.venv`、bottle 模型与生产前端时，无需重新安装或训练。在 PowerShell 执行：

```powershell
Set-Location -LiteralPath 'E:\秋招\项目\VisionInspect'
. .\scripts\project_env.ps1
.\scripts\start_local.ps1
```

打开 <http://127.0.0.1:8000>，选择 **bottle / PatchCore**，上传 PNG、JPEG 或 WebP。单图上限 10 MiB、16,777,216 像素。CPU 旧版候选描述与新的 GPU 多步复核是独立工作流，旧 VLM 默认关闭。

项目 PowerShell 脚本使用 **UTF-8 with BOM**，兼容 Windows 自带 PowerShell 5.1。已修复无 BOM 中文被错误解码导致的 ParserError，并在 5.1 实际启动验证首页、health、ready 均返回 200。编辑 `.ps1` 时请保留 BOM；排查说明见 [运行文档](docs/RUNNING.md)。

可用输入示例：`data/mvtec_ad/bottle/test/broken_large/000.png`。服务按需启动一个常驻推理子进程，连续任务复用 PatchCore、正常参考索引及已加载的可选 VLM。空闲 **600 秒（10 分钟）**后退出子进程，释放内存；在 `configs/service.yaml` 修改 `idle_ttl_seconds`，重启服务生效。初始化上限 90 秒、工具 45 秒、单任务 180 秒 / 12 GiB 监管继续生效。正常验证图实测，含 VLM 的首次任务 **33.67 秒**、缓存命中的下一项 **3.85 秒**；此为单张工程验证，不是平均延迟或效果指标。详见 [常驻与 TTL](docs/RESIDENT_WORKER.md)。按 `Ctrl+C` 停止服务，也可通过 `-Port 8017` 指定端口。

如果提示前端未构建：

```powershell
. .\scripts\project_env.ps1
Push-Location frontend
npm ci --ignore-scripts --no-audit --no-fund
npm run build
Pop-Location
.\scripts\start_local.ps1
```

代码目录不包含可再分发的数据、权重和实验模型。复制代码到新电脑后，须按下文准备环境和资产，不能直接获得现有检测能力。

## 技术与工作流

- **视觉模型**：PyTorch 2.12.0+cpu、torchvision 0.27.0+cpu、Anomalib 2.7.0、timm 1.0.22；调用真实 PatchcoreModel。
- **Agent**：有界条件工作流与 Pydantic 契约，不依赖付费 API。最多 8 次工具调用、1 次 VLM；仅可重试错误允许额外尝试一次。
- **本地候选解释**：可选 SmolVLM-500M-Instruct，CPU、固定 revision、校验后离线加载；不决定最终状态。当前采用 separate-images-v2，待检图与正常参考分别输入并保存输入审计；[修复记录与验证边界](docs/VLM_INPUT.md)。
- **服务**：FastAPI、SQLite、单任务执行通道、常驻推理子进程与 10 分钟空闲 TTL；每项任务独立证据 / 报告目录，180 秒 / 12 GiB 监管上限。
- **前端**：React 19.3.1、TypeScript 7.0.2、Vite 8.3.5、Radix Toggle Group 1.2.1。

```text
上传图像 → 质量检查 → PatchCore 分数与热图
                         ↓ 按冻结规则触发
                  疑似区域 / 正常参考 / 可选 VLM
                         ↓
             确定性汇总 → 候选状态、证据、调用轨迹
                         ↓
                  人工研究复核 / JSON 导出
```

六个工具为 `inspect_image_quality`、`detect_anomaly`、`retrieve_normal_reference`、`analyze_suspicious_region`、`classify_defect_candidate`、`finalize_inspection`。质量异常、工具失败、边界分数或证据冲突进入 `manual_review`。VLM 不修改模型分数或阈值，人工复核与模型原报告分别保留。

## 已测结果：bottle

实验使用正常训练 168 张、正常验证 41 张，测试 20 张正常与 63 张异常，seed 42。配置为 ResNet18 layer2/3、完整图像缩放至 256×256、ImageNet 标准化、coreset 0.01、9 邻居、CPU FP32、4 线程、batch 1。此实验称为 **PatchCore-ResNet18 CPU 基线**，不等同于论文默认配置复现。

| 指标 | 实测值 |
|---|---:|
| Image AUROC | 1.000000 |
| Average Precision | 1.000000（四舍五入） |
| 固定阈值 F1 | 0.976744 |
| 异常召回率 | 1.000000 |
| Pixel AUROC | 0.978275 |
| AUPRO，FPR 上限 0.3 | 0.923896 |

图像阈值为 **8.742960929870605**，仅由正常验证分数的 95% 经验分位数（higher）确定，使用严格 `score > threshold`。混淆矩阵为 `[[17, 3], [0, 63]]`，行是真值、列是预测，顺序为正常 / 异常，即误报 3、漏报 0。小验证集阈值不保证总体误报率。

像素指标在 **256×256** 网格计算，包含正常测试图，mask 使用最近邻缩放；展示热图采用单图归一化，亮色不能跨图直接比较。指标读取原始浮点热图。

预热 5 次后，单图“预处理 + 模型”推理延迟 p50 / p95 为 **42.43 / 49.32 ms**，不含读写、服务和证据导出。完整指标及资源口径见 [基线实验记录](docs/EXPERIMENTS.md)。

固定 VLM 复核与动态 Agent 均处理同一完整测试集，结果同为 78 张自动候选、5 张人工暂缓；动态 VLM 调用 **83 → 5**，整组热执行 **332.206 → 38.576 秒**。这些暂缓样本未被计作预测正确，不能据此声称准确率提升。本地外部 API 费用为 0，不包含硬件与电费。详见 [冻结 A/C/D 对照](docs/COMPARISON.md)。

## 在新环境安装

已验证主机为 Windows 10、Python 3.12.12、i9-9900K、约 64 GiB 内存；当前方案全部使用 CPU。Node 24.22.0 / npm 11.19.1 沿用本机已有安装，未升级系统运行时。

使用项目独立环境，不向 Conda base、系统 Python 或其他项目环境安装依赖。下面的解释器路径是本机实际路径，新机器须替换为已有且兼容的 Python 3.12：

```powershell
. .\scripts\project_env.ps1
# 仅当 .venv 不存在时执行；不要覆盖已有环境。
uv venv --python 'D:\SOFTWARE\miniconda\python.exe' .venv --no-python-downloads
uv pip sync requirements.txt --python .venv\Scripts\python.exe --require-hashes --torch-backend cpu --no-python-downloads --strict
uv pip check --python .venv\Scripts\python.exe
```

`requirements.txt` 为 Windows CPython 3.12 CPU 哈希锁；`requirements-linux.txt` 为 Linux CPU 锁，目前仅验证解析，未验证容器安装。`requirements.in` 是直接依赖。更新依赖须先检查兼容性与下载预算，不能无约束升级。

`project_env.ps1` 将缓存和临时目录放在当前项目，仅影响当前进程及子进程。新电脑依赖、缓存、数据、权重与输出建议预留 15 GiB；完整拟合和服务均有 12 GiB 进程树上限。安装明细见 [环境记录](docs/ENVIRONMENT.md)。

## 数据与模型准备

- MVTec AD 来自用户授权的 `foersben/mvtec-ad` 第三方镜像，固定 revision `c75b39616f84db43677bcc8228caaafaf5096d7f`，许可 **CC BY-NC-SA 4.0**。非官方托管，未获得官方归档逐文件哈希，不声称与官方归档逐字节一致。
- bottle 为 357 个文件、156,570,806 字节；metal_nut 为 430 个文件、165,862,477 字节。每类 provenance 分开保存，不下载全集。
- ResNet18 权重约 46.8 MB；可选 SmolVLM 本地资产约 1.02 GB。模型使用固定 revision 和 SHA256，代码与资产各自遵守许可。
- 应用和实验命令不会自动下载缺失数据或权重，模型执行阶段禁用 Python 网络连接。数据、模型、密钥不得加入 Git 或烘焙进镜像。

资产准备是显式操作。先阅读 [数据与权重说明](docs/DATASETS.md) 和 [许可证](docs/THIRD_PARTY.md)，核验来源、许可、revision 与预算后，再按 [脚本索引](scripts/README.md) 执行相应准备脚本。`prepare_assets.py preflight` 与 `prepare_vlm.py --preflight` 只读取元数据；其余准备命令可能下载文件。

bottle 数据必须包含 `train/good`、`test/good`、`test/broken_large`、`test/broken_small`、`test/contamination` 及对应 `ground_truth`。损坏、缺失、掩膜错误或校验变化会明确失败。

## CLI、API 与开发

使用当前注册模型进行监督式单图检测：

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B -m visioninspect.inspection --image .\data\mvtec_ad\bottle\test\broken_large\000.png --category bottle --mode dynamic
```

可选 `--mode baseline` / `--mode fixed`，使用已验证本地 VLM 时加 `--vlm`。CLI 报告保存在 `outputs/cli/<任务编号>/report.json`；Web 服务状态与证据位于 `outputs/service/`。

重新执行独立基线实验：

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B -m visioninspect.baseline check-data
.\.venv\Scripts\python.exe -B -m visioninspect.baseline smoke
.\.venv\Scripts\python.exe -B -m visioninspect.baseline fit
# 将 <fit-run> 替换为上一步打印的正式 fit 目录，不得使用 smoke 目录。
.\.venv\Scripts\python.exe -B -m visioninspect.baseline calibrate --run '<fit-run>'
.\.venv\Scripts\python.exe -B -m visioninspect.baseline evaluate --run '<fit-run>'
.\.venv\Scripts\python.exe -B -m visioninspect.baseline predict --run '<fit-run>' --image '<image-path>'
```

六个子命令均接受 `--config`，默认 `configs/patchcore_cpu.yaml`。每次创建独立运行目录，记录配置、环境、命令、退出码、耗时及峰值内存；校准不可覆盖，同一 fit 不重复正式测试。`scripts/run_phase1.ps1` 可按上述协议运行一次完整 bottle 流水线。新 fit 不会自动替换服务模型；需更新 `configs/models.yaml`，再显式执行 `scripts/prepare_tools.py bottle` 生成匹配的正常参考索引与策略。

开发前端时，项目根目录启动一个后端进程，另一个终端运行 Vite：

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B -m uvicorn visioninspect.api:app --host 127.0.0.1 --port 8000 --workers 1
# 另一个项目终端执行：
. .\scripts\project_env.ps1
Set-Location frontend
npm run dev
```

开发地址见 Vite 输出，默认代理后端；生产构建由 FastAPI 同源提供。`VITE_API_BASE_URL` 仅用于构建时指定后端地址，跨域需另行配置。系统面向单机研究，默认 localhost；没有账户与公网部署验收。

常用接口：`GET /health`、`GET /ready`、`GET /api/models`、`GET /api/runtime`（常驻进程、缓存及 TTL 状态）、`POST /api/tasks`、`GET /api/tasks/{id}`、`GET /api/tasks/{id}/report`。工具 schema、历史、证据、复核等完整接口见 [架构文档](docs/ARCHITECTURE.md)。模型健康状态中的文件存在不代表运行时加载一定成功。

验证命令：

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
uv pip check --python .venv\Scripts\python.exe
```

最近一次完整验证（2026-10-22，常驻与 TTL 修改后）：**86 项测试通过**（44 条第三方弃用警告），正常验证图的真实缓存复用、TTL 到期重载与热图一致性检查通过。此前整理阶段的 98 个 Python 包依赖一致性检查通过；此后未安装或升级依赖。详细命令、结果和日志见 [常驻验证记录](docs/RESIDENT_WORKER.md) 与 [文档索引](docs/README.md)。工程故障注入测试与数据集效果指标分别记录，不将少量测试用例的通过率推广为真实环境可靠性。

## 目录

```text
VisionInspect/
├── README.md                    # 项目入口、状态与运行方法
├── AGENT.md                     # 仓库执行规范，原文件保留
├── VisionInspect-Agent_交接文档_含前端要求.md
├── configs/                     # 基线、Agent、模型注册与待验证配置
├── visioninspect/               # 算法适配、工具、Agent、API 与任务监管
├── frontend/                    # React 源码、锁文件和本地产物 dist/
├── tests/                       # 数据、协议、工具、API 与故障恢复测试
├── scripts/                     # 环境、准备、运行、评估和验证入口
├── docs/                        # 当前索引、运行说明、历史阶段证据
│   └── design/                  # 产品定义与界面设计记录
├── requirements.in              # 直接 Python 依赖
├── requirements*.txt            # Windows / Linux CPU 完整哈希锁
├── Dockerfile                   # 待实际验收的 CPU 容器配置
├── data/ · weights/ · outputs/   # 本地数据、模型与不可随意移动的实验资产
└── logs/ · .cache/ · .tmp/ · .venv/  # 日志、可重建缓存与项目环境
```

[文档索引](docs/README.md) / [脚本索引](scripts/README.md) 说明各文件用途。实验输出、模型注册及文档引用互相关联，整理时保留其路径。旧缓存移入 `.cache/workspace-archive/`，可恢复，没有删除数据、模型、日志或历史实验。项目尚未初始化 Git；`.gitignore` 与 `.dockerignore` 已排除本地资产、环境、日志与秘密。

## 待完成与限制

1. **多类别**：metal_nut 尚待严格数据检查、split、拟合、校准与评估；capsule / pill 还需逐类下载与验证。当前工作台只注册 bottle。
2. **EfficientAD B 组**：当前 Anomalib 2.7.0 完整训练的损失使用额外通用图像作正则化；预训练 teacher、MVTec 正常样本和正则化图像是三类不同输入。本地限定路径检查未确认授权 ImageNet / Imagenette 数据。原作者允许省略惩罚项的变体，但不能称为当前完整训练。架构吞吐冒烟使用明确标记的合成张量，仅验证工程流程；未替代正式实验数据。70,000 步 CPU 计算估算约 40 小时，超过现有 60 分钟预算，不自动降低训练配置。
3. **Docker**：本机上次验证为 `WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED` / Docker Desktop 无法启动。未更改系统组件。须由用户提供可用引擎，再实际验证构建、挂载、健康检查和真实推理；Linux 依赖解析和 registry 元数据检查不能替代部署验收。入口为 `scripts/run_docker.ps1`，状态见 [部署记录](docs/PHASE5_STATUS.md)。
4. **外部泛化**：VisA 尚待独立扩展验证，LOCO 为后续可选；当前结果不能外推到其他类别或产线。VLM 描述未经过缺陷类型准确性评估。

当前范围没有付费 API、服务器租赁或持续后台任务。数据和模型采用各自上游许可，详见 [第三方来源](docs/THIRD_PARTY.md)。本项目尚未单独确定代码对外发布许可证，不以依赖的 MIT / Apache-2.0 许可替代本项目许可。
