# 项目文档索引

更新：2026-10-22。根目录 [README](../README.md) 是当前入口；各 Phase 报告保留当时的事实，后续状态以下表及追加记录为准，不改写冻结实验。

## 当前可用范围

| 状态 | 内容 | 证据 |
|---|---|---|
| 已完成实测 | bottle PatchCore-ResNet18 CPU，168 train / 41 validation / 83 test | [基线实验](EXPERIMENTS.md)、[Phase 1](PHASE1_REPORT.md) |
| 已完成实测 | 六个契约工具、动态 Agent、固定 / 动态本地 VLM 对照 | [Phase 2](PHASE2_REPORT.md)、[Phase 3](PHASE3_REPORT.md)、[A/C/D 对照](COMPARISON.md) |
| 已完成实测 | 中文 Web 六个视图、真实上传推理、历史、证据、研究复核与 JSON 导出 | [Phase 4](PHASE4_REPORT.md)、[运行说明](RUNNING.md) |
| 部分实现，待实验 | EfficientAD teacher 校验、低层适配器与合成张量架构冒烟；未完成正式 B 组训练或评估 | [数据与权重](DATASETS.md)、[第三方来源](THIRD_PARTY.md) |
| 数据已准备，待实验 | metal_nut 430 文件 / 165,862,477 字节，700×700；尚无 split、模型或测试指标 | 本地 data/mvtec_ad/metal_nut-source.json、logs/metal_nut-download.log |
| 元数据已核验 | capsule / pill 尚未下载 | 本地 logs/category-extension-preflight.json |
| 配置已准备，待验收 | CPU Docker 镜像来源与 digest 已核验，未构建或运行 | [Phase 5 状态](PHASE5_STATUS.md) |

## 阅读入口

- [运行说明](RUNNING.md)：本地生产、开发前端、CLI 与 Docker 入口，以及冷启动误触工具超时的修复与实测。
- [常驻进程与 TTL](RESIDENT_WORKER.md)：模型缓存复用、空闲回收、任务隔离、资源与故障恢复验证；默认 TTL 10 分钟。
- [PAI GPU 环境准备](GPU_PAI_SETUP.md)：用户提供的 A10 / DSW 信息、只读审计脚本、候选版本与未完成的 GPU 验证；尚未部署 vLLM。
- [架构与接口](ARCHITECTURE.md)：队列、子进程监管、工具路由、REST 接口与数据保存。
- [VLM 输入核验](VLM_INPUT.md)：旧拼图压缩问题、两图独立输入修复、实际输入文件与语义验证边界。
- [数据与权重](DATASETS.md)：来源、许可、revision、文件校验、已下载与未准备资产。
- [环境记录](ENVIRONMENT.md)：主机审计、独立环境、完整锁文件与分阶段环境变化。
- [第三方来源](THIRD_PARTY.md)：代码、数据、权重分别适用的许可。
- [脚本索引](../scripts/README.md)：哪些命令只检查，哪些会下载或产生新实验。
- [产品定义](design/PRODUCT.md)、[界面设计](design/DESIGN.md)：中文桌面质检工作台的用途与交互选择。
- [原始交接文档](../VisionInspect-Agent_交接文档_含前端要求.md)、[执行规范](../AGENT.md)：保留原位置和原始内容。

## 实验与历史

| 文档 | 记录范围 |
|---|---|
| [Phase 0 审计](PHASE0_AUDIT.md) | 代码、环境、参考仓库、许可与资源；原始命令在 audit/ |
| [Phase 1](PHASE1_REPORT.md)、[实验记录](EXPERIMENTS.md) | bottle 视觉基线及冻结指标；报告中的“未实现”描述限于该阶段 |
| [Phase 2](PHASE2_REPORT.md) | 类型化工具与最小 Agent |
| [Phase 3](PHASE3_REPORT.md)、[对照实验](COMPARISON.md) | C/D 真实完整测试集、调用量、延迟、暂缓样本及局限 |
| [Phase 4](PHASE4_REPORT.md) | 前后端浏览器实测、真实任务、截图和本地交付 |
| [Phase 5](PHASE5_STATUS.md) | 容器准备情况、WSL / Docker 阻塞与后续验收条件 |

## 最近一次验证：常驻与 TTL（2026-10-22）

完整套件 **86 passed / 44 warnings / 38.69 秒**，退出 0；覆盖缓存失效、进程复用、忙碌时不回收、空闲 TTL、崩溃 / 超时 / 内存限制后的重建、Windows 启动器异常清理与服务关闭。正常验证图的真实含 VLM 任务为首次 33.67 秒、连续第二项 3.85 秒，复用同一进程和模型；验收用 2 秒 TTL 确认到期退出与冷重载，分数 / 热图通过一致性检查。配置默认仍为 600 秒，未运行新正式测试集评估。日志为 logs/resident-final-tests-execution.json / .child.log、logs/resident-real-execution.json / .child.log；完整记录见 [常驻进程](RESIDENT_WORKER.md)。没有安装依赖、下载资产或修改系统环境，用户现有服务未被自动重启。

## 初始化计时修复（历史记录）

模型初始化单独计时后，完整套件 74 passed / 44 warnings / 33.14 秒，退出 0。隔离状态目录的真实正常验证图 HTTP 检测通过：初始化 23.47 秒、检测工具 0.15 秒、含本地 VLM 的任务 33.41 秒，子进程树峰值 3,018,649,600 字节；未重跑正式测试集或更改阈值。浏览器核实旧失败任务显示“检测超时”且能打开执行轨迹。实际命令、日志和局限见 [运行说明](RUNNING.md)。下节保留此前整理阶段的结果。

## 目录整理验收（历史记录）

产品与设计文档从根目录归入 docs/design/，逐文件 SHA256 一致。源代码、脚本、测试、配置、数据、权重、outputs 和 logs 原位置保留，以免破坏模型注册、历史任务和冻结实验引用。Python 字节码、pytest 及编译临时缓存归档到 `.cache/workspace-archive/`，未永久删除；操作与保护文件哈希记录在本地 logs/workspace-organization.json。

本轮不安装依赖、不下载新资产、不启动新训练或正式测试集评估，也不修改系统配置。使用现有项目环境执行：

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp .tmp/pytest-readme-verification -o junit_family=xunit1 --junitxml=logs/readme-tests.xml
uv pip check --python .venv\Scripts\python.exe
```

两条验证命令均退出 0。完整测试 **56 passed、44 条第三方弃用 warnings、33.27 秒**；98 个 Python 包兼容检查通过。结果见本地 logs/readme-tests.log、logs/readme-tests.xml、logs/readme-dependency-check.log；文档链接与保护资产校验见 logs/readme-validation.json。统计来自本次完整套件，不将历史或针对性测试次数相加。本轮未改前端代码或重新构建，保留上次成功构建的 dist。

当前无长期运行的项目服务。启动服务须由用户执行 scripts/start_local.ps1；数据与权重缺失时不会触发隐式下载。

整理后校验：18 份文档内的 59 个本地链接均有效；10 个保护文件与两份移动文档的 SHA256 均保持一致。bottle 与 metal_nut 的 787 个数据文件重新核对 provenance 的字节数和 SHA256，通过。四个 PowerShell 运行入口的语法解析通过，没有因此启动服务或实验。

## v0.4 业务与 PAI 迁移

- [业务规格](BUSINESS_V1.md)
- [迁移到 PAI](MIGRATION_PAI.md)
- [实现、实测和待办](UPGRADE_STATUS.md)
