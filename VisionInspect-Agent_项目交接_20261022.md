# VisionInspect-Agent 当前项目交接文档

> **状态：历史文档。** 本文写于 2026-10-22 上午，其中"服务器尚无虚拟环境""权重仅有部分分片"等描述已于当日与次日被实际进展取代。当前有效的事实请看 [docs/UPGRADE_STATUS.md](docs/UPGRADE_STATUS.md)（工程与实验时间线）和 [docs/JUDGEABILITY.md](docs/JUDGEABILITY.md)（复核可判性的实测数字）。本文保留，用于回看当时的计划与判断。

交接日期：2026-10-22（Asia/Shanghai）  
适用版本：GPU 产品源码 `43934fe`，以及下文列出的本地未提交改动。  
本地目录：`E:\秋招\项目\VisionInspect`。

> 接手时最重要的事实：完整工作台与 GPU 改造源码已经存在，本机工程测试已有通过记录；当前 AutoDL 服务器尚无项目虚拟环境，Qwen 权重只有部分文件，PatchCore CUDA 与 Qwen/vLLM 的完整运行验收尚未完成。用户已要求由自己手动下载依赖和模型，接手者不要恢复自动下载。

## 1. 当前目标与用户已确认的范围

产品是中文桌面质检工作台：上传图像，由视觉模型提取异常，再由 Agent 针对不确定样本补充证据、复核，最后保留可追溯报告和人工记录。

- 保留完整前后端、单图、批次、证据、历史、复核和导出。
- 正式运行目标为 Linux NVIDIA GPU；PatchCore 与 VLM 均迁移到 GPU。Windows 本机用于开发和工程测试，不再作为当前产品的 CPU 推理部署目标。
- 当前只注册 `bottle`；没有承诺当前 GPU 产品已支持全部 MVTec 类别。
- 工作流只有 `baseline` 和 `agent_review`。旧 SmolVLM、fixed/dynamic 工作流、EfficientAD 分支、Windows 启动入口和 CPU Docker 配置已移出当前源码。
- 用户曾讨论自进化，随后决定暂不实施；当前没有自动学习、自动改阈值或自动训练机制。
- 用户最新决定：Qwen 优先使用 ModelScope，由用户自己执行下载和依赖安装。不要重新开始先前的远程中转下载。

根目录的 [原始项目交接要求](VisionInspect-Agent_交接文档_含前端要求.md) 是最初需求来源。后续用户明确确认的 GPU 产品范围及上述人工下载要求应一并传递，不能机械恢复原文所有历史阶段或已删除分支。执行规范见 [AGENT.md](AGENT.md)。

## 2. 完成情况：源码、测试与实验分开看

| 项目 | 实际状态 | 证据或限制 |
|---|---|---|
| 中文 React 工作台、FastAPI、SQLite、上传/批次/证据/复核/导出 | 已实现，有本机工程验证 | 不能据此认定新服务器已部署 |
| PatchCore 单类别 CPU 基线 | 历史实验已执行 | checkpoint、校准、测试指标保留；当前推理代码要求 CUDA |
| PatchCore CUDA 迁移 | 源码已实现 | 新服务器真实分数、热图、一致性与延迟待验证 |
| LangGraph 有界复核与证据校验 | 已实现，规则和流程测试通过 | 真实 Qwen 多步行为及业务收益待验证 |
| vLLM 运行时 | PAI 原生扩展检查曾由用户验证通过 | PAI 与当前 AutoDL 不是同一环境，结论不能直接复用 |
| 当前 AutoDL 审计与基础资产恢复 | 已执行 | 源码 HEAD 为 `43934fe`；367 个迁移资产文件已校验恢复 |
| AutoDL 应用 / vLLM 虚拟环境 | 未建立 | 交接时再次只读检查，两处 `.venv` 均不存在 |
| Qwen 完整权重、加载与生成 | 未完成 | 4 个权重分片为 `.part`，最终 `source.json` 不存在 |
| ModelScope 续传入口 | 本地草稿 | 尚未提交、尚未同步到服务器；真实下载和自动化回归尚未验证 |
| 双模型同卡显存、完整 GPU 产品验收 | 待完成 | 不能宣称已完成 GPU 部署或性能达标 |
| EfficientAD 完整对照、GPU Agent 收益实验 | 当前未完成且不在现版实现中 | 历史代码可从 Git 标签查看；没有授权正则化图像与正式结果 |

## 3. 产品运行流程与候选状态

```text
React 上传 / 批次草稿
          ↓ REST 上传、轮询
FastAPI → SQLite 队列（单 worker、串行执行）
          ↓
图像质量检查 → CUDA PatchCore → 原始分数、浮点异常图、疑似区域
                                    ↓
                    baseline：冻结规则生成候选
                    agent_review：仅不确定区间进入 LangGraph
                                    ↓
                       Qwen3-VL / 本机 vLLM
                     观察 → 补充局部/参考 → 结束
                                    ↓
                       程序校验证据与调用预算
                                    ↓
                 候选结果 + 原始检测 + 人工追加记录
```

### 3.1 PatchCore 与异常热图

PatchCore 从正常训练图建立特征库。推理时，ResNet18 提取待检特征，按与正常特征库的距离产生图像异常分数和异常图；热图是异常图的可视化，叠加图把它映射回图片。热图不由 VLM 生成，不代表经过标注验证的缺陷类别或原因。

当前模型配置：ResNet18 `layer2/layer3`、完整图缩放至 256×256、ImageNet 标准化、coreset 0.01、邻居数 9、FP32、CUDA、batch 1、workers 0、线程 4、seed 42。当前实现无 CPU 模型推理回退。

### 3.2 Agent 与 VLM 实际参与的位置

- `baseline` 不调用 VLM；`agent_review` 先做相同的检测和取证，分数落在冻结不确定区间才进入模型复核。
- LangGraph 管理 `baseline / decide / tool / finish` 节点；Qwen 同时承担视觉观察和下一步动作选择，没有另加载一个规划大模型。
- 模型动作只能是 `compare_region`、`compare_reference`、`finish`。模型调用最多 3 次，重试也计入。
- 整体视图是待检原图与正常训练参考；局部视图包含原图、待检裁剪及参考图对应坐标裁剪。参考不是另一件待检产品。前端可查看实际发送的图像证据。
- JSON schema、证据 ID、区域 ID、重复动作、引用范围及正常结论条件由程序校验；需要正常结论时，所有已提取区域必须完成对应局部检查并有一致依据。
- 模糊、矛盾、缺失证据、超时、格式错误或预算耗尽均转人工。原始分数和阈值不会被模型改写。
- 提示词及结构化协议在 `visioninspect/review_contracts.py`；循环与门控在 `review_agent.py`；请求及输入留档在 `vllm_client.py`。

`normal_candidate` 表示正常候选，`anomaly_candidate` 表示异常候选，`manual_review` 表示需要人工复核。这三者均是研究系统状态，没有连接产线放行设备或工业验收规范。Agent 可在程序约束内形成复核候选，不能将其描述为获得工业放行授权。

### 3.3 常驻进程与 TTL

PatchCore resident worker 与 vLLM 是两个独立模型进程，按需启动，默认空闲 TTL 均为 600 秒。空闲回收后，下次请求重新启动并加载 checkpoint/权重；首次、TTL 后首个请求的耗时包含冷启动，不能用热推理耗时解释整个工具步骤。

vLLM 处理期间有 lease 防止空闲回收。应用退出会清理所属子进程。当前不提供分布式调度，也未实现 LangGraph checkpointer 的逐轮断点续跑。

## 4. 代码导航

| 目录 / 文件 | 接手时关注的职责 |
|---|---|
| `frontend/src/main.tsx`、`BatchWorkspace.tsx` | 单图、证据、历史、批次与系统界面 |
| `visioninspect/api.py` | REST、上传限制、健康检查、证据文件访问与静态前端 |
| `visioninspect/service.py` | SQLite、任务/批次调度、重试、人工追加记录 |
| `visioninspect/model.py`、`model_cache.py` | CUDA PatchCore、checkpoint、缓存加载 |
| `visioninspect/resident.py`、`worker.py`、`resources.py` | 常驻检测进程、TTL、超时与资源监管 |
| `visioninspect/agent.py`、`tools.py` | 检测工具编排、质量检查、区域与参考证据 |
| `visioninspect/review_agent.py`、`review_contracts.py` | LangGraph 多步复核、schema、提示词和确定性门控 |
| `visioninspect/vllm_runtime.py`、`vllm_client.py` | vLLM 生命周期、本机接口和实际输入/输出留档 |
| `visioninspect/baseline.py`、`data.py`、`metrics.py` | 实验 CLI、数据清单、划分、校准与指标 |
| `configs/models.yaml` | 当前类别与 checkpoint/profile/参考索引注册 |
| `configs/patchcore.yaml`、`agent.yaml`、`gpu.yaml`、`service.yaml` | 模型、规则、GPU 运行时与 TTL |
| `requirements.txt`、`requirements-vllm.txt` | 应用与 vLLM 两套 Linux Python 哈希锁 |
| `scripts/`、`tests/`、`docs/` | 部署和验收入口、工程回归、业务架构与许可 |

主要接口：`GET /health`、`/ready`、`/api/models`、`/api/runtime`；`POST /api/tasks`；任务查询、`/report`、`/review`、`/retry`；批次创建、上传 `/items`、控制 `/actions/{action}`、JSON/CSV 报告；`GET /api/tools/schemas`；实验 `/api/runs`。完整参数以 `api.py` 和运行后的 `/docs` 为准。

单图上限 10 MiB、16,777,216 像素；批次最多 100 张，全局待执行上限 200。暂停/取消在当前图片完成后生效；重试创建关联新任务，原尝试保留。重启后执行中任务标记失败、批次暂停、排队项保留。

默认业务存储为 `outputs/service/history.sqlite3` 和同目录任务证据，可用 `VISIONINSPECT_STATE_DIR` 指定。原始检测、Agent 判断和人工记录分别保存。搬迁历史记录应停止本项目服务后做 SQLite 一致性备份，连同证据目录迁移。

## 5. Git 状态与未提交内容

交接核对时，本地与 AutoDL 已提交源码 HEAD 均为 **`43934fe`**，分支 `main`，本地没有配置 Git remote。服务器工作区干净，本地工作区有以下未提交改动：

| 文件 | 状态 / 意图 |
|---|---|
| `scripts/setup_gpu.sh` | 已修改：支持 HTTPS 包索引、wheelhouse 与在线补齐；显式开关才完全离线 |
| `scripts/setup_vllm.sh` | 同上；独立 vLLM 环境安装 |
| `configs/qwen3_vl_assets.json` | 未跟踪：固定 ModelScope revision、15 个文件大小与哈希、原来源身份 |
| `scripts/download_modelscope.py` | 未跟踪：标准库预检、Range 续传、SHA256 与磁盘余量检查草稿 |
| `scripts/env_autodl.sh` | 未跟踪：AutoDL 路径与索引环境变量草稿，不安装依赖 |

这些草稿不属于 `43934fe`，既有 133 项通过记录不覆盖它们。服务器尚无这些草稿；只交接 Git bundle 会漏掉它们。接手者需另存上述文件及 `git diff`，完成审查和测试后再提交/同步。本次新增交接文档及文档入口也应随工作区一起交接。

重要历史引用：

- `4440cfb`：完整产品保留、GPU 改造与旧分支清理；标签 `v0.5.0-gpu-source` 指向此提交，只代表源码版本。
- `43934fe`：新增独立 vLLM 环境、锁文件和预算预检。
- `pre-gpu-only-20261022` → `82f019b`：清理前完整历史，旧 CPU、EfficientAD、Docker 和阶段文档从这里查。
- `cpu-baseline-v0.3.0`：历史 CPU 基线。

不要用 `git reset --hard`、`git clean` 或覆盖工作区的方式“同步”草稿。可先只读检查 `git status --short`、`git diff --stat`、`git log -5 --oneline`。

## 6. 数据、冻结资产与历史实验

### 6.1 必须单独交接的资产

Git 不含 `data/`、`weights/`、`outputs/`、`logs/`、环境、缓存或 `frontend/dist/`。完整接手不能只复制源码。

- `data/mvtec_ad/bottle/`：209 张训练正常图、20 张测试正常图、63 张异常图、63 张掩膜及两份说明，共 357 文件、156,570,806 字节。
- `data/manifests/bottle.json`：固定文件校验和划分，训练 168、正常验证 41，测试 83；测试不能用于调提示词、阈值或策略。
- `data/mvtec_ad/source.json`：数据来源记录。
- `weights/resnet18.a1_in1k/`：46,807,446 字节的骨干 safetensors 及来源记录。
- ~~`outputs/20261021-232056-fit-ae72d8/`：历史 checkpoint 与校准；该目录被 `configs/models.yaml` 引用。~~
- ~~`weights/profiles/bottle.json`、`bottle-references.npz`：冻结策略与正常训练参考索引。~~
  > 以上两项已删除（2026-10-23）：注册在 `7617017` 已切到 GPU 资产，CPU 版本只作历史对照，现从本机移除。
- `frontend/dist/`：服务器没有 Node/npm，当前已恢复构建产物；若改前端，需在开发机重新构建后迁移。
- 历史实验和验收所需的 `outputs/`、`logs/`：当前迁移包未等价包含本机全部历史结果或用户检测历史，应按交接需要另行保留。

关键身份：

```text
checkpoint SHA256:
4a4d94f150f5b2021aa593c8b277bc942178f261eacc336f14c3e33cef76587a
calibration SHA256:
a1bf923b8612e9fd5e23ba66c78bd59ed4791195aa780d97d0d22362caea5cf4
image threshold: 8.742960929870605
review interval: [8.492477416992188, 9.452290534973145]
pixel threshold: 7.434370517730713
```

阈值来自 41 张正常验证图的 95% 经验分位数、`higher` 插值，分数严格大于图像阈值才为异常候选；小验证集不能保证总体误报率。以上冻结值不因换 GPU 或 Agent 解释而自动改变。

### 6.2 已有 CPU 结果，不能冒充 GPU 或 Agent 收益

历史测试目录：`outputs/20261021-232247-evaluate-5225f8/`。

| 指标 | 历史实测 |
|---|---|
| 图像 AUROC / AP | 1.0 / 约 1.0 |
| 固定阈值 F1 / 异常召回 | 0.976744 / 1.0 |
| 混淆矩阵（行真实、列预测；normal/anomaly） | `[[17, 3], [0, 63]]` |
| Pixel AUROC / AUPRO | 0.978275 / 0.923896 |
| 像素协议 | 256×256 网格，mask 最近邻缩放，AUPRO FPR 上限 0.3 |
| 热推理 p50 / p95 | 42.43 / 49.32 ms，预热 5 次，含预处理、不含文件读写 |

准确名称为 **PatchCore-ResNet18 CPU 基线**，不是论文默认配置复现。原始 `metrics.json`、`latency.json`、逐图分数与命令记录为准；这些数字不代表新 GPU、冷启动、整条产品链路或 VLM 的耗时和准确率。

## 7. 环境现状与硬性约束

### 7.1 Windows 开发机

路径为 `E:\秋招\项目\VisionInspect`，已有 `.venv\Scripts\python.exe`（Python 3.12.12，CPU 开发环境）。它可以执行工程测试，不能用来验证 CUDA。不要在本地尝试安装服务器 CUDA 锁来“修复”测试跳过。

### 7.2 当前 AutoDL（本次只读复核）

| 项目 | 实测 / 当前状态 |
|---|---|
| 项目目录 | `/root/autodl-tmp/VisionInspect` |
| 当前连接入口 | SSH 主机 `connect.westc.seetacloud.com`，端口 `36929`，用户 `root`；凭证另行交接，实例变更后重新确认 |
| GPU 资产目录 | `/root/autodl-tmp/VisionInspect-GPU` |
| 系统 | Ubuntu 22.04.5，glibc 2.35，x86_64 |
| 基础解释器 | `/root/miniconda3/bin/python`，Python 3.12.3，仅用于审计及创建虚拟环境 |
| 平台 PyTorch | torch 2.12.1+cu130、torchvision 0.27.1+cu130；不是项目目标版本 |
| GPU 报告 | `NVIDIA GeForce RTX 4080 SUPER, 32760 MiB`，SM 8.9、BF16 可用 |
| 驱动 | 580.105.08；基础矩阵运算和 SDPA 通过 |
| 容器配额 | CPU 16 核，内存约 62 GiB；主机核数不等于容器配额 |
| 数据盘 | 总量 50 GiB；交接复核空闲 43,510,460,416 字节，约 40.52 GiB |
| 项目 / vLLM 环境 | `/root/autodl-tmp/VisionInspect/.venv` 与 GPU 根目录 `.venv` 均不存在 |
| 基础资产 | 已恢复并校验 367 文件，包含 bottle、骨干、历史 checkpoint/profile/参考及前端构建 |

GPU 型号与显存按 `nvidia-smi` 实际返回记录，重新分配实例后应再次检查。基础算子通过不等于 vLLM 原生扩展、完整模型或应用通过。

以前的 PAI 是 A10 / 24 GiB、Python 3.12.13、torch 2.13.0、vLLM 0.27.1；用户返回原生检查 `passed: True`。不要把该结果写成当前 AutoDL 已安装或已通过。

不得向系统 Python 或 Conda base 安装项目依赖；不得改驱动、CUDA、系统 Python、系统网络或其他项目环境；不调用付费 API；不把凭证、数据、模型或用户结果提交 Git。不自动扩容或租新服务器。

### 7.3 目标依赖与预算

应用用项目 `.venv`，vLLM 用 GPU 根目录 `.venv`，互不共享系统包。锁定核心版本：torch 2.13.0、torchvision 0.28.0、Anomalib 2.7.0、timm 1.0.22、LangGraph 1.2.16、vLLM 0.27.1、Transformers 5.16.1。完整版本/哈希看两份 requirements 锁；不要无约束升级。

应用候选分发约 2.98 GB，vLLM 约 3.87 GB；两套冷下载合计约 6.85 GB，按包文件去重约 4.03 GB。FrEIA 0.2 是唯一获准的 34,270 字节纯 Python 源码例外，固定哈希和构建后端，其余包只允许 wheel。

本次用户批准的上限是**累计下载 26 GB**、沿用现有盘；模型下载前空闲至少 25 GiB，落盘后保留至少 8 GiB。这不是允许丢弃缓存后重新完整下载的额外预算。新服务器或扩大范围需重新核对。

安装脚本的依赖预算和 ModelScope 草稿的单次模型响应预算彼此独立，**没有自动保证跨脚本累计不超过 26 GB**。接手者必须先合并实际账本；旧 `autodl-approved-downloads.json` 的 relay reservation 是已取消传输的预约值，不能当作实耗，也不能直接忽略本地与服务器中转两段流量。

`logs/autodl-session.json` 仍保留早期“等待预算批准”的状态文字；预算后来已批准，之后用户又要求改为手动下载。该 JSON 不是最新执行状态，不能据此重复索要预算批准或恢复自动部署。云服务器本身的租用费用由用户管理，不因未调用付费 API 就记为零成本。

## 8. 下载中断点与 ModelScope 草稿

用户要求手动下载后，先前所属 SCP、安装等待和权重下载进程已停止，部分文件保留；停止记录在服务器 `logs/autodl-user-stopped-downloads.json`。本次交接只做只读核查，没有恢复下载或安装。

服务器已回收并校验的 `.cache/wheelhouse` 文件合计 **2,035,940,194 字节**。仍保留 `.cache/autodl-wheel-relay.tar` 等中转残件；不要把残件当完整包，也不要未经检查全部清空。

模型目录为 `/root/autodl-tmp/VisionInspect-GPU/weights/qwen3-vl-8b-instruct`，11 个小文件已准备；四个未完成权重的大小如下：

| `.part` 文件 | 已有字节 |
|---|---:|
| `model-00001-of-00004.safetensors.part` | 2,277,507,072 |
| `model-00002-of-00004.safetensors.part` | 1,337,982,976 |
| `model-00003-of-00004.safetensors.part` | 918,552,576 |
| `model-00004-of-00004.safetensors.part` | 1,193,279,488 |

这些只是前缀，不能改名充当完整权重。成功完成四分片、大小及 SHA256 校验并生成有效 `source.json` 后才能加载。

ModelScope 草稿使用标准库直接访问固定仓库接口，没有安装 ModelScope SDK。已有元数据核对记录表明，固定 ModelScope revision 下所选 15 个文件与此前 Hugging Face 资产身份一致；这使续传既有前缀有可核对依据，但实际 Range 响应、网络速度和完整下载仍未验证。

```text
model: Qwen/Qwen3-VL-8B-Instruct
Hugging Face revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
ModelScope revision: 5d854aab08710c16b980ec6d603d863b3821b915
15 个文件总量: 17,545,914,364 字节
其中 4 个 BF16 权重总量: 17,534,339,512 字节
```

续传草稿会在服务器忽略 Range 时拒绝直接追加，在完整 SHA256 不符时保留 `.part` 并失败。接手前应补充 Range、截断、错误哈希、预算及磁盘门槛回归；不能仅因脚本文件存在就称下载功能已经验收。

## 9. 接手后的执行顺序与命令

以下部署命令全部在 **AutoDL 的 SSH / JupyterLab 终端**运行，不在 Windows PowerShell 运行。凭证由用户另行提供，本文件不包含密码、私钥或令牌。

### 9.1 先接源码与资产，不下载

1. 保留本地未提交草稿，审查、补测试，再同步到服务器。确认服务器拥有 ModelScope 脚本和配置后才能执行下列对应命令。
2. 检查 `git status`、源码版本、现有数据和分片、磁盘与预算。若换实例，重新执行审计。
3. 资产迁移见 [MIGRATION_PAI](docs/MIGRATION_PAI.md)。其中 `/mnt/workspace`、`/usr/local/bin/python3` 是旧 PAI 示例，不能原样用于当前 AutoDL；当前使用本节路径与基础解释器。

```bash
cd /root/autodl-tmp/VisionInspect
git status --short
git log -1 --oneline
nvidia-smi
df -h /root/autodl-tmp
/root/miniconda3/bin/python -B scripts/audit_gpu_environment.py --output logs/gpu-audit-handoff.json
```

### 9.2 用户手动准备模型与依赖（尚未执行）

下面展示草稿经测试并同步后的操作。必须先核对剩余累计预算；不是要求接手 Agent 自动执行。

```bash
cd /root/autodl-tmp/VisionInspect
source scripts/env_autodl.sh

# 只读取元数据并预检；失败先排查，不扩大预算或重下全部资产
"$BOOTSTRAP_PYTHON" -B scripts/download_modelscope.py preflight \
  --root "$VISIONINSPECT_GPU_ROOT" --minimum-free-gib 25

# 用户确认预检、累计预算后，手动续传 ModelScope 权重
"$BOOTSTRAP_PYTHON" -u -B scripts/download_modelscope.py download \
  --root "$VISIONINSPECT_GPU_ROOT" --minimum-free-gib 25

# 按哈希锁建立两个独立环境；已有 wheelhouse 复用，缺包才经 HTTPS 索引补齐
bash scripts/setup_gpu.sh --download-budget-bytes 3200000000
bash scripts/setup_vllm.sh --download-budget-bytes 4100000000
```

先准备模型是为了利用当前空闲量完成 25 GiB 下载前门槛，之后仍须检查依赖展开与缓存占用。上述两个依赖预算只是各自预检额度，不可相加理解为额外批准。索引草稿默认指定清华 PyPI HTTPS 地址；预检仍访问官方 PyPI 元数据，之前存在元数据读超时，因此换 pip 索引不保证预检网络问题消失。

若不使用 `env_autodl.sh`，必须显式设置 `BOOTSTRAP_PYTHON`、`VISIONINSPECT_GPU_ROOT`、`VISIONINSPECT_VLLM_PYTHON`、`VISIONINSPECT_RELOCATED_ASSETS=1`。每个新终端重新设置。当前 `configs/gpu.yaml` 默认 PAI 路径不适用于 AutoDL。

### 9.3 安装完成后再验收与启动（尚未执行）

```bash
cd /root/autodl-tmp/VisionInspect
source scripts/env_autodl.sh

# 安装脚本已执行原生检查；出错时可独立重跑并保留完整输出
"$VISIONINSPECT_VLLM_PYTHON" -B scripts/pai_vllm.py check --root "$VISIONINSPECT_GPU_ROOT"

# 先检测模型，再验证 VLM 图像协议和独立 TTL
.venv/bin/python -B scripts/verify_gpu.py --ttl-smoke
.venv/bin/python -B scripts/verify_gpu.py --with-vlm --ttl-smoke

# 两项验收通过后，前台启动完整产品
bash scripts/start_gpu.sh
```

Web 默认端口 8000，vLLM 仅监听 `127.0.0.1:8001`。通过服务器受控访问入口访问工作台；不要将本机模型端口额外公开。该配置不能当作已完成公网账号鉴权和生产访问控制。Ctrl+C 停止前台服务，不创建未经请求的后台常驻任务。

`verify_gpu.py` 只用正常验证样本，结果在新的 `outputs/*-gpu-verification-*`，失败写 `failure.json`。需检查 `verification.json` 的 `full_agent_loop_exercised`：若为 false，不能声称真实多步 Agent 分支已覆盖。双图/三图协议通过也不等于完整业务收益验证。

Web 使用顺序：选择 bottle → 先用 `baseline` 检查原始分数与热图 → 再用 `agent_review` 查看是否进入复核、实际输入与调用轨迹 → 追加人工记录 → 导出报告。批次先上传成草稿，再手动开始；错误项和原始尝试应保留。

### 9.4 新的正式 GPU 实验（独立于工程验收）

```bash
.venv/bin/python -B -m visioninspect.baseline check-data --config configs/patchcore.yaml
.venv/bin/python -B -m visioninspect.baseline smoke --config configs/patchcore.yaml
.venv/bin/python -B -m visioninspect.baseline fit --config configs/patchcore.yaml
.venv/bin/python -B -m visioninspect.baseline calibrate --run <本次GPU拟合目录>
.venv/bin/python -B -m visioninspect.baseline evaluate --run <本次GPU拟合目录>
```

先停止本项目 Web，避免同卡模型竞争。每次新拟合创建新目录，正式测试冻结后只评估一次，不调已有实验来追求指标。若更换产品注册模型，更新 `configs/models.yaml` 的新目录及新 profile/参考路径，再用 `scripts/prepare_tools.py bottle` 准备，禁止覆盖冻结资产。

## 10. 已执行的验证与环境变化

既有本机命令及结果：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --junitxml=logs/autodl-source-tests.xml
```

结果为 **133 passed、1 skipped、53 warnings，55.94 秒，退出 0**，见 `logs/autodl-source-tests.log` 与 XML。跳过项需要 CUDA；替身覆盖的 API/流程不等于真实模型验证。该记录早于 ModelScope 草稿。

前一轮 GPU 清理时，TypeScript/Vite 构建、Shell 语法检查、Git 私有资产检查及真实浏览器导航检查通过，见 `logs/gpu-cleanup-audit.json`、`gpu-browser-qa.json`。本次交接只整理文档与只读复核，没有重新训练、重算阈值、下载、安装、修改 CUDA/驱动或启动产品服务。

本次核对命令包括 `git status --short`、`git log -5 --oneline`、文档/源码读取；服务器用基础 Python 只读检查源码 HEAD、两处环境目录、磁盘、wheelhouse 和权重分片。结果写入本交接文档。交接文档的最终检查命令、退出码另存 `logs/handoff-document-validation.json`。

## 11. 已知问题与故障定位

| 现象 / 风险 | 接手处理 |
|---|---|
| 不确定模型是不是已经部署 | 先检查两处 `.venv`、权重 `source.json`、验收 JSON 和 `/ready`，不能只看页面或 `/health` 的 status |
| 模型说“两个瓶子”或只输出 Rust 等片段 | 检查该任务实际输入、提示词、完整 JSON 和证据引用；当前 Qwen 协议已改进，但真实生成仍待验证，不将旧 SmolVLM 描述当有效结论 |
| 不开 VLM 仍首张很慢 | 查 resident 冷启动、checkpoint 读取、CUDA 初始化、证据导出与队列等待；报告中热推理计时不含全部步骤 |
| 超时、OOM、缺少有效分数 | 保留失败日志和已有证据，转人工；不悄悄回退 CPU、降低模型或增大预算 |
| 两模型是否能共存 | vLLM 利用率配置 0.85 只是配置，真实峰值与 PatchCore 余量待测 |
| 新环境 vLLM 导入失败 | 固定 0.27.1 检查的是 `vllm._C_stable_libtorch`，不是旧的 `vllm._C`；保留报错和版本，不改系统 DLL/驱动 |
| 官方 PyPI 元数据读超时 | 预检有界重试，保留实际异常；不能关闭 TLS 或把 pip 换源当作所有联网问题已解决 |
| 路径迁移或历史绝对路径 | 显式启用 relocated assets，仍核对原清单、checkpoint 与 profile，不直接改哈希绕过验证 |
| 旧 README 路径或状态与交接不一致 | README / MIGRATION / DATASETS 含 PAI 默认路径和旧阶段文字，以代码、实测和本交接的时间快照区分；同步新手动流程后统一更新 |

单图检测预算 180 秒、12 GiB RSS；初始化 90 秒；vLLM 启动 300 秒、请求 45 秒；服务/项目 RSS 上限 24/28 GiB。正式实验 smoke/fit/calibrate/evaluate 超时分别为 10/60/15/15 分钟。峰值进程内存与 GPU 显存是不同指标，都需实际记录。

## 12. 许可、保密与交接边界

许可检查来源与记录见 [DATASETS](docs/DATASETS.md) 和 [THIRD_PARTY](docs/THIRD_PARTY.md)：MVTec AD 记录为 CC BY-NC-SA 4.0，默认仅非商业研究；当前 bottle 来自用户授权核验的 foersben 镜像、固定 revision `c75b39616f84db43677bcc8228caaafaf5096d7f`。已核验镜像文件，但没有官方逐文件哈希，不能声称与官方归档逐字节相同。

Qwen3-VL、ResNet 权重和 Anomalib 的许可记录为 Apache-2.0；数据、程序、预训练权重分别受各自许可约束。项目本身尚未选择对外发布许可证。这里转录既有核验记录，不代表本次重新完成所有上游法律条款核验；扩大或商业使用前需复核适用条款。

服务器凭证在用户会话中曾提供，交接文档不复写。SSH 账号、密码或私钥应通过用户认可的单独渠道交接，不能放进 Git、文档、命令参数或示例配置。分发源码前运行 `scripts/check_git_content.py`；资产包只在获授权研究实例间迁移，不能因为 Git 不含数据就默认资产可公开分发。

## 13. 接手清单与近期优先级

1. 获取完整工作区、Git 历史、上述未提交草稿、必要资产与日志；确认新负责人能独立访问服务器。只读核对后再决定同步，保留部分下载与用户历史。
2. 审查 ModelScope/索引草稿，补上必要回归；明确修正旧路径文档，提交后仅同步源码。当前用户要求人工下载，继续遵守。
3. 用户按预检结果手动完成模型与两个环境；记录实际命令、退出码、完整版本、累计下载、磁盘余量和失败。
4. 完成真实 PatchCore CUDA、checkpoint 重载一致性、Qwen 多图结构化生成、实际 Agent 补充动作及两个 TTL 验收；测冷/热延迟与显存峰值。
5. 实测通过后才能更新“GPU 部署完成”的状态。Agent 相对 baseline 的误报/漏报收益需要独立标注数据与冻结协议，当前没有这项结论。

接手第一条消息可直接使用：

> 请先阅读《VisionInspect-Agent_项目交接_20261022.md》和 AGENT.md，核对 Git、未提交草稿、资产与服务器状态。当前用户要求自行手动下载，先完成 ModelScope 草稿审查/测试和可复制命令，不要自动下载安装。当前没有完整 GPU 模型验收，保留原 checkpoint、阈值、划分和下载分片，不恢复旧 CPU 分支，不把计划写成已完成结果。
