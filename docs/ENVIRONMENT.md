# 环境实测记录

当前安装状态见后文各阶段追加记录。原 Phase 0 快照保留，用于区分审计事实与后续实施；当前总状态见 [文档索引](README.md)。

审计日期：2026-10-21（Asia/Shanghai）。本文件记录环境现状，不代表已创建项目环境或已运行异常检测模型。

## 项目与主机

| 项目 | 实测 |
|---|---|
| 工作目录 | `E:\秋招\项目\VisionInspect` |
| 审计开始时内容 | 仅交接文档，24,569 字节；无现有代码、依赖清单、数据、权重、项目环境 |
| Git | `git status --short --branch`、`git remote -v` 均返回 `not a git repository`；未初始化 |
| 上级约束文件 | 检查 E 盘根目录、秋招、项目、当前目录，未发现 `AGENTS.md` |
| 系统 | Windows 10 专业版，64 位，10.0.19045 / build 19045 |
| CPU | Intel Core i9-9900K，8 核 / 16 线程 |
| 内存 | 68,648,992,768 字节物理内存，约 63.93 GiB；初次采样空闲物理内存约 42.72 GiB |
| GPU | NVIDIA Quadro P620，2,048 MiB，计算能力 6.1，WDDM，正在承担显示任务 |
| GPU 驱动 | 582.78；Windows 驱动版本 32.0.15.8278 |
| 显存初次采样 | 已用 1,253 MiB；`memory.free` 710 MiB；驱动保留部分使已用与空闲之和不等于总量 |
| CUDA | `nvidia-smi` 显示 13.0，指驱动支持的 CUDA 上限；`nvcc` 未在 PATH 找到，标准 CUDA 安装目录未发现；不能据此断言主机任意路径都没有 toolkit |
| 其他工具 | Conda 26.1.1；uv 0.11.7；Git 2.47.1.windows.1；Node 24.22.0；npm 11.19.1；找到 Docker 命令，但未启动或验证 Docker/WSL/GPU 容器 |

初次磁盘采样（可用空间随系统活动变化）：

| 磁盘 | 可用 GiB |
|---|---:|
| C | 40.44 |
| D | 327.80 |
| E（本项目） | 1,041.81 |
| F | 1,147.37 |
| H | 1,343.39 |

## Python 环境

| 路径 | Python | 已有关键包 | 审计判断 |
|---|---|---|---|
| `D:\SOFTWARE\miniconda\python.exe` | 3.12.12 | NumPy 2.5.0、pip 26.0.1；无 torch / torchvision / anomalib / lightning | 当前裸 `python` 指向 Conda base；禁止在此安装项目依赖 |
| `D:\SOFTWARE\miniconda\envs\test\python.exe` | 3.12.13 | torch 2.12.0+cpu、torchvision 元数据 0.27.0、NumPy 2.4.6；无 anomalib / lightning / timm | 已有通用环境，归属未知，且 Pillow DLL 导致 torchvision 导入失败；不修改、不复用安装 |
| `C:\Users\llj\AppData\Local\Programs\Python\Python310\python.exe` | 3.10.5 | torch 2.1.2+cpu、torchvision 0.24.1+cpu、NumPy 1.22.4、pip 25.3 | torch / torchvision 版本不成对，不能作为项目环境 |

`conda env list --json` 中 `test` 有 D/d 大小写重复登记；按 Windows 路径语义是同一路径，不算两套环境。`py -0p` 仅列出 Python 3.10。裸 `pip` 解析到 `...\Python312\Scripts\pip.exe`，其 Python312 解释器路径未发现，和裸 `python` 不一致。后续只允许使用项目解释器路径或 `uv --python` 明确目标。

实际运行结果：

```text
test 环境：
torch_version 2.12.0+cpu
cuda_runtime None
arch_list []
cuda_available False
cpu_tensor_sum 64.0

系统 Python 3.10：
torch_version 2.1.2+cpu
cuda_runtime None
arch_list []
cuda_available False
cpu_tensor_sum 64.0

test 环境 torchvision 导入：
ImportError: DLL load failed while importing _imaging: 操作系统无法运行 %1。

test 环境 python -B -m pip check：
No broken requirements found.
```

`pip check` 只能验证声明的依赖约束，不能证明 DLL 或模型运行正常。CUDA 不可用是当前两个 torch 安装均为 CPU 构建的实测结果，不代表已验证 GPU 无计算能力。未运行 GPU 张量、PatchCore 推理、训练或评估。

## 网络与安装前兼容性检查

使用 PyPI JSON / simple 索引 GET 和 wheel / 权重 HEAD 检查，未下载 wheel、数据集或模型正文。

| 资源 | 实测 |
|---|---|
| PyPI anomalib 索引 | HTTP 200 |
| anomalib 2.7.0 | PyPI 确实存在；Python >=3.10，通用 wheel |
| lightning 2.6.1、timm 1.0.22 | PyPI 元数据可访问，分别要求 Python >=3.10、>=3.8 |
| PyTorch CPU 索引 | HTTP 200 |
| torch 2.12.0+cpu / cp312 / win_amd64 | HEAD 200，122,900,744 字节 |
| torchvision 0.27.0+cpu / cp312 / win_amd64 | HEAD 200，4,021,192 字节 |
| timm/resnet18.a1_in1k 的 model.safetensors | HEAD 200，46,807,446 字节；模型 revision `491b427b45c94c7fb0e78b5474cc919aff584bbf` |
| MVTec 官方页面 | 网页检索工具可读；本机 PowerShell HEAD 发生 SSL 握手失败。未禁用证书校验；实际下载链路仍需下一阶段验证 |

未进行测速，不能把 HEAD 成功等同于大文件稳定下载。没有通过表单提交个人信息，没有授权付费 API，没有调用任何模型推理 API。

当前建议在项目 `.venv` 中使用 Python 3.12.12、torch 2.12.0+cpu、torchvision 0.27.0+cpu、anomalib 2.7.0、lightning 2.6.1、timm 1.0.22；这是已核验包存在性和声明要求的候选组合，尚未进行完整依赖求解、安装、导入和模型验证，不能称为已验证可运行环境。完整传递依赖需在下一阶段锁定；`pytorch-lightning` 与 `lightning` 同时约束到 2.6.1，避免传递依赖漂移。uv 创建的 venv 默认没有 pip，依赖一致性检查使用 `uv pip check --python '.venv\Scripts\python.exe'`，不假定 `python -m pip` 一定可用。

## 实际命令与证据

本阶段先执行了 `Get-Content -Raw -Encoding UTF8` 完整读取交接文档，随后用 `Get-ChildItem -Force`、`rg --files --hidden`、`git status --short --branch`、`git remote -v` 检查项目；检查了各级 `AGENTS.md`。

主机、解释器与 torch 检查的完整可重复命令保存于 [phase0_readonly.ps1](audit/phase0_readonly.ps1)，实际执行输出保存于 [phase0_local.log](audit/phase0_local.log)。执行命令为：

```powershell
& '.\docs\audit\phase0_readonly.ps1' *> '.\docs\audit\phase0_local.log'
```

网络检查实际使用的命令形式：

```powershell
Invoke-RestMethod -Uri 'https://pypi.org/pypi/anomalib/2.7.0/json' -TimeoutSec 20
Invoke-RestMethod -Uri 'https://pypi.org/pypi/lightning/2.6.1/json' -TimeoutSec 20
Invoke-RestMethod -Uri 'https://pypi.org/pypi/timm/1.0.22/json' -TimeoutSec 20
Invoke-WebRequest -Uri 'https://pypi.org/simple/anomalib/' -Method Head -TimeoutSec 20
Invoke-WebRequest -Uri 'https://download.pytorch.org/whl/cpu/' -Method Head -TimeoutSec 20
Invoke-WebRequest -Uri 'https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors' -Method Head -MaximumRedirection 5 -TimeoutSec 20
Invoke-WebRequest -Uri 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad' -Method Head -MaximumRedirection 5 -TimeoutSec 20
```

wheel 大小检查实际通过 base Python 的标准库 `urllib.request` 读取 CPU wheel 索引，选择精确 cp312/win_amd64 文件后发出 `Request(url, method='HEAD')`，没有在 base 中安装或导入项目依赖。GitHub 核验通过连接器 GET 仓库元数据、contents、commits、release 和文本文件；固定版本见 [reference_snapshot.json](audit/reference_snapshot.json)。

交接文档原始 SHA256：`A09E63C297808E0CE0D0B86223D9C0212B722AAD82E89DC657C973C9A021C79F`。

## 环境变化与未验证范围

只新增 `docs/` 内审计文档、命令、日志与参考快照。没有新建 Python 环境，没有安装/升级依赖，没有更改驱动/CUDA/系统 Python/其他项目环境，没有克隆仓库、初始化 Git、下载数据或权重，也没有启动服务或后台任务。

本阶段的 CPU 张量检查与导入检查均为真实执行；异常检测效果、运行时间、峰值内存、GPU 推理、Docker 能力、前端闭环全部尚未验证。项目的最终范围仍包括真实结果驱动的 Web 前端，当前按阶段顺序暂不实现。

## Phase 1 实际环境变化（2026-10-21）

- 已创建唯一项目环境：`E:\秋招\项目\VisionInspect\.venv`，Python 3.12.12。创建使用已有解释器，项目包全部装入 `.venv`，未安装到 base / 系统 Python / test。
- 已锁定并安装 Windows 依赖 91 个：torch 2.12.0+cpu、torchvision 0.27.0+cpu、Anomalib 2.7.0、lightning / pytorch-lightning 2.6.1、timm 1.0.22、torchmetrics 1.8.2；根据用户追加的骨架要求增加 FastAPI 0.115.6、uvicorn 0.34.0、Pydantic 2.12.3 等。
- `uv pip check` 通过；Pillow PNG 解码、CPU 矩阵运算、torchvision NMS、Anomalib 导入、本地骨干加载均通过。torch.version.cuda=None、cuda.is_available=False，符合 CPU 构建。真实模型冒烟、拟合、校准、测试和独立推理已完成。
- 完整依赖与哈希保存在根目录 requirements.txt。Linux CPU 依赖另生成 requirements-linux.txt，仅完成解析，未在 Linux 安装验证。
- 依赖目标发行文件预算 326,726,471 字节；FrEIA 0.2 的 34,270 字节纯 Python 源码包已审查并成功构建。安装器未提供精确网络流量计数，预算不能冒充实测网络字节总数。
- 前端独立安装在 frontend/node_modules，24 个包；React 19.3.1、Vite 8.3.5、TypeScript 7.0.2，与已有 Node 24.22.0 的 engines 兼容。npm ci 与 TypeScript/Vite 构建通过；没有升级 Node/npm，没有全局安装前端依赖。
- 缓存、临时文件、权重、数据和输出均在本项目 E 盘目录；进程环境变量只在项目脚本和子进程中设置，没有修改系统代理、证书、CUDA 或驱动。
- 本次逻辑文件大小：.venv 约 1.065 GB、.cache 约 1.306 GB、数据约 156.8 MB、权重约 46.8 MB、输出约 87.4 MB、前端依赖约 76.6 MB。缓存与环境可能共享硬链接，不能将逻辑大小之和当成独占物理占用。具体快照见 logs/final-verification.json。
- Docker CLI 检查返回退出码 1：dockerDesktopLinuxEngine 命名管道不存在。没有启动 Docker Desktop、拉取基础镜像或进行 Docker build。
- 现有弃用警告涉及 timm / Anomalib、torch.jit.script、Starlette / AnyIO，已保留日志；当前验证无失败。

主要实际命令：

```powershell
. '.\scripts\project_env.ps1'
uv venv --python 'D:\SOFTWARE\miniconda\python.exe' '.venv' --no-python-downloads
uv pip compile requirements.in --python '.venv\Scripts\python.exe' --generate-hashes --torch-backend cpu --no-python-downloads --output-file requirements.txt
# 增加服务骨架依赖时，以前一锁文件作为 constraints，重新解析、预算核验后同步。
uv pip sync requirements.txt --python '.venv\Scripts\python.exe' --require-hashes --torch-backend cpu --no-python-downloads --strict
uv pip check --python '.venv\Scripts\python.exe'
& '.\.venv\Scripts\python.exe' -B -m pytest -q
& '.\scripts\run_phase1.ps1'
# 在 frontend 工作目录运行 npm ci 与 npm run build。
```

引导时曾将 uv 的 --torch-backend 参数错误写入 requirements.in，解析退出码 2；已移到 CLI 后解析通过。失败日志保留在 logs/dependency-resolution-attempt1.log，没有隐藏或将其作为成功安装记录。

Windows venv 启动器会派生真正计算的 Python 子进程。初版仅观察启动器的约 3.8 MB 读数无效；资源控制现已改为整个进程树，并新增回归测试。正式 fit 的实际计算子进程 OS lifetime peak working set 已由外部监控补测，总计约 1.443 GiB，见对应 external-resource-monitor.json。初始无效读数仍保留，不能用于资源结论。修正后重新运行 smoke，结果通过，进程树峰值约 0.570 GiB。

## Phase 2–4 追加记录（2026-10-21 至 22）

核心视觉版本未变，使用上一 Windows 锁作为 constraints 新增 python-multipart 0.0.32、transformers 5.20.2、tokenizers 0.23.3、regex 2026.9.29、typer 0.27.3、shellingham 1.5.4、annotated-doc 0.0.5。现为 98 包，uv pip check 通过。完整分发预算 342,892,238 字节，不代表实际新增下载流量。Linux CPU 哈希锁重新解析成功，尚未在 Linux 安装验证。

新增本地 SmolVLM 文件 1,019,893,574 字节，主权重完整 SHA256 校验通过，CPU 断网推理成功。VLM + PatchCore 对照的进程树峰值约 2.67 GiB，GPU 不使用。实际记录见 PHASE3_REPORT.md、COMPARISON.md。

API 进程显式将 tempfile 固定到 E 盘项目 .tmp，缓存环境仅影响当前进程及子进程。前端没有增加依赖或改变 Node/npm。

曾为容器验收启动已安装 Docker Desktop；引擎报无法启动，WSL 缺少 Windows 可选组件，未擅自启用组件。没有 Docker build、镜像拉取、系统网络/驱动/CUDA 变更；随后使用 docker desktop stop 停止测试应用。此前 Phase 1“未启动 Docker Desktop”是该阶段历史状态，并非当前累计状态。

## 2026-10-22 后续验证与目录整理

- 前端追加固定 @radix-ui/react-toggle-group 1.2.1 及其依赖，用于证据切换的键盘和单选语义。16 个新增包均核验 MIT，registry 解包大小合计 555,272 字节；旧有 React / Vite / TypeScript 版本未改变。记录在 logs/frontend-dependency-budget.json，未升级 Node/npm 或全局安装。
- 上次生产构建退出 0：JS 275.61 kB / gzip 86.27 kB，CSS 19.99 kB / gzip 4.75 kB。本轮只整理目录与文档，没有改前端代码或重复构建。
- 新增 EfficientAD 官方 teacher 归档及 Small teacher 共约 50.74 MB，均校验 SHA；metal_nut 数据 165,862,477 字节，未训练。没有新增 Python 包，仍为独立 .venv 的 98 包；细节见 DATASETS.md。
- 本轮完整工程验证：56 passed、44 第三方弃用 warnings、33.27 秒，退出 0；uv pip check 检查 98 包通过。没有运行新正式数据集评估。
- 根目录 PRODUCT.md / DESIGN.md 移至 docs/design/，SHA 不变；字节码和测试 / 编译临时缓存移入 .cache/workspace-archive/，未永久删除。环境、下载缓存、数据、权重、实验和日志均保留原位置。操作及保护资产 SHA 见 logs/workspace-organization.json，最终核验见 logs/readme-validation.json。

## 2026-10-22 初始化计时与超时提示修复

没有安装或升级依赖，也没有修改 Python、驱动、系统网络、用户正在运行的服务或模型权重。生产前端重新构建成功：JS 276.94 kB / gzip 86.67 kB、CSS 19.99 kB / gzip 4.75 kB，退出 0；日志为 logs/cold-start-frontend-build.log。

完整测试 74 passed / 44 第三方弃用 warnings / 33.14 秒，退出 0。使用一张正常验证图、独立状态目录运行固定流程与本地 VLM：任务 33.41 秒、子进程树峰值 3,018,649,600 字节（约 2.81 GiB），外层验证程序及任务合计峰值 3,110,264,832 字节（约 2.90 GiB）。GPU 显存不适用，外部 API 费用 0。记录在 logs/cold-start-fulltests-execution.json 与 logs/cold-start-http-execution.json；未新增正式训练或测试集评估。

## 2026-10-22 常驻与 TTL

改为应用管理的单个常驻推理子进程，新增 configs/service.yaml，默认空闲 TTL 600 秒。使用已有 Python 标准库、psutil 与 YAML 依赖，没有新增或升级任何依赖、修改系统设置、下载数据 / 权重、构建前端或启动外部后台服务。测试生成的推理进程均已关闭；用户原有 localhost:8000 服务未被停止，需用户重启一次启用新调度器。

最终完整套件 86 passed / 44 弃用 warnings / 38.69 秒，命令退出 0，外层监督记录 40.917 秒。真实正常验证图缓存与 TTL 验收命令 67.469 秒、退出 0；含 VLM 首次任务 33.674 秒、连续下一项 3.852 秒；TTL 回收后单模型任务 25.140 秒。逐任务当前进程树 RSS 采样峰值最高 3,062,300,672 字节（约 2.85 GiB）。内存按任务窗口采样，外层跨多个世代累计的 OS 峰值不可当作同时驻留量。GPU 不使用，外部 API 费用 0。

实际命令、证据路径、缓存 / 失效行为与局限见 [常驻记录](RESIDENT_WORKER.md)。本轮不修改冻结 A/C/D 指标，不将单样本工程延迟写作性能分布或准确率提升。
