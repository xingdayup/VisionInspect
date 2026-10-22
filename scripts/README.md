# 脚本索引

除前端 npm 命令外，均在项目根目录执行。先运行 `. .\scripts\project_env.ps1`，Python 明确使用 `.\.venv\Scripts\python.exe -B`。脚本保留原路径，历史 command.json 与转录日志仍可对应。

## 常用入口

| 入口 | 用途与变化 |
|---|---|
| `project_env.ps1` | 当前进程的 E 盘项目缓存、临时目录与线程设置，不安装依赖 |
| `start_local.ps1 [-Port 8000]` | 启动 localhost 单 worker API 与已构建前端；Ctrl+C 停止 |
| `run_phase1.ps1` | 新建 bottle check-data / smoke / fit / calibrate / evaluate / predict 实验，记录实际命令；会占用 CPU、内存和输出磁盘 |
| `run_docker.ps1` | 引擎预检通过后才 build / run；会拉取镜像与安装锁定依赖，当前未验收，不会安装 WSL 或改系统组件 |
| `guard.py --seconds N --record <json> -- <command>` | 有限本地命令的超时、12 GiB 进程树监管，保存子进程日志 |

模型命令行为 `python -m visioninspect.baseline <check-data|smoke|fit|calibrate|evaluate|predict>`；工作台同协议的监督检测为 `python -m visioninspect.inspection --image <path> --mode dynamic`。这里的 python 指上述项目解释器。

## 资产准备：必须显式调用

| 脚本 | 获取内容与前置条件 |
|---|---|
| `prepare_assets.py preflight` | 只获取 MVTec 镜像数据卡、目录与许可；审查 revision 和预算后才能下载 |
| `prepare_assets.py weights` | 固定 revision 的 ResNet18 safetensors，约 46.8 MB |
| `prepare_assets.py bottle` | 使用已审核的镜像预检记录，逐文件获取 bottle，约 156.6 MB；不获取全集 |
| `preflight_categories.py` | metal_nut / capsule / pill 的目录元数据与许可检查，不下载图像 |
| `prepare_category.py <metal_nut|capsule|pill>` | 单次仅一类，固定 revision、大小 / 哈希 / 许可 / 路径核验，图像解码后写该类 provenance 和配置；须逐类复核预算 |
| `prepare_vlm.py --preflight` | 固定 revision 的 SmolVLM 元数据；不下载权重 |
| `prepare_vlm.py` | 约 1.02 GB 本地 VLM 资产；需先完成许可与预算检查 |
| `resume_vlm.py` | 已审核 SmolVLM 大文件的分段继续，最终完整 SHA256 通过才可用；不是通用下载器 |
| `prepare_efficientad.py` | 固定 SHA 的官方 teacher 归档约 39.96 MB，只提取 Small teacher；不下载正则化数据 |
| `prepare_tools.py [category]` | 使用注册的 fit、正常训练与验证数据构建参考索引和策略，写 weights/profiles；新 fit 后显式运行，不可用测试结果调参 |

当前只有 bottle 注册模型。metal_nut 数据已到盘，capsule / pill 未下载；本次整理没有继续下载或训练。许可与实际资源见 [DATASETS](../docs/DATASETS.md) / [THIRD_PARTY](../docs/THIRD_PARTY.md)。

## 评估与验证

| 脚本 | 范围 |
|---|---|
| `benchmark_agent.py --category bottle --mode <fixed|dynamic> [--vlm]` | 新建冻结 Agent 对照，读取完整测试集并保存逐图证据；必须先有正式 fit、calibration、evaluation 与训练 / 验证策略 |
| `verify_comparison.py <agent-run>` | 与对应冻结 A 结果逐图核对分数及原始热图，不拟合 |
| `snapshot_sources.py <agent-run> ...` | 仅当当前实现 SHA 与 frozen.json 完全相同才归档源文件 |
| `verify_vlm.py` | 已准备本地模型的离线推理检查；不是正式语义准确性评估 |
| `verify_http.py` | 连接已启动的 localhost:8000 服务，在该服务状态目录创建真实任务与证据；不是无写入的健康检查 |
| `verify_supervisor_http.py` | 监管改动后的真实模型 HTTP 回归，隔离输出；会运行真实 bottle 推理 |
| `verify_resident.py [--skip-vlm]` | 独立状态目录，正常验证图检查连续任务缓存复用、2 秒验收 TTL 退出、冷重载分数 / 热图一致及关闭清理；默认含本地 VLM，结束后清理自己的子进程，不修改 600 秒生产配置 |
| `smoke_efficientad.py` | 7 步合成张量架构 / 吞吐检查，**不产生正式 B 组数据集指标** |
| `summarize_controlled_tests.py` | 从 logs/completion-tests.xml 汇总受控路由 / 故障注入案例，写 controlled-agent-metrics.json；不能当作真实环境可靠性 |

## 审计与预算

| 脚本 | 范围 |
|---|---|
| `dependency_budget.py` | 依赖发行文件元数据与候选下载预算，不等于实测网络流量 |
| `frontend_dependency_budget.py` | 新前端依赖的许可 / 版本 / 解包大小与预算核验 |
| `check_container_sources.py` | 匿名读取官方 registry 元数据，验证 FROM digest 与 linux/amd64 manifest，不拉镜像层 |
| `audit_regularization_data.py` | 限定数据目录与已配置路径的只读检查，目录存在不等于取得使用许可 |
| `audit_gpu_environment.py [--output <json>]` | 由用户在 GPU 机器手动运行；审计 CUDA 版本、配额、磁盘及小规模真实 GPU 算子，不安装依赖或下载模型；正常退出不等于 GPU 通过，须看 cuda_smoke.passed |
| `final_audit.py` | 写当前环境、保护文件哈希、主要目录逻辑大小；可能双计硬链接，不是物理磁盘增量 |

原始主机只读命令位于 docs/audit/phase0_readonly.ps1。工具失败和超预算应保留日志并报告，不改系统环境或自动降低正式配置。
