# VisionInspect-Agent GPU 版执行规范

当前范围由用户于 2026-10-22 确认：保留完整 Web 产品，PatchCore 与 VLM 均使用 NVIDIA GPU。旧 CPU / SmolVLM / fixed / dynamic 实现保存在 Git 标签 `pre-gpu-only-20261022`，不再属于当前运行代码。

## 产品与架构

- React / TypeScript / Vite 工作台，FastAPI API，SQLite 持久队列与文件证据。
- PatchCore-ResNet18、256×256、FP32、CUDA；LangGraph 复核；vLLM / Qwen3-VL-8B-Instruct BF16。
- 只提供 `baseline`（GPU 单模型）和 `agent_review`（GPU 检测及多步复核）。无 CUDA 必须失败，不得回退 CPU。
- 保留单图、100 张批次、暂停继续取消、重试、证据、历史、人工复核、JSON/CSV 导出。
- 先检查图像质量、运行 PatchCore，再按冻结不确定区间复核。模型最多调用 3 次，包含重试；证据缺失、冲突、超时或预算耗尽转人工。
- 所有候选结论均为研究结果，不构成工业放行授权。原始检测、Agent 判断与人工记录分别保存。

## 环境与数据管理

- 先审计 Python、GPU、驱动、CUDA、包版本、磁盘和许可证，再安装或下载。
- 项目依赖仅安装在本项目 `.venv`。vLLM 可只读复用用户已验证的 GPU Python；不得向 base / 系统环境安装或升级依赖，不改驱动、CUDA、其他项目或系统网络设置。
- 使用 `requirements.txt` Linux GPU 哈希锁。除固定哈希的 FrEIA 0.2 纯 Python 源码包外仅允许 wheel；构建后端固定版本，不执行无约束升级。
- 缓存、临时文件、权重、数据与结果保存在项目或明确配置的 GPU 资产目录，不提交 Git。
- 安装与大文件下载必须先检查实际下载量、磁盘及显存预算。超预算停止并报告，不扩大下载范围或自动租服务器。
- 不调用付费 API，不提交密钥，不关闭证书校验，不隐式下载模型。不得把硬件云租金记为零成本。
- 原数据、checkpoint、split、校准与用户历史保留。历史 CPU 阈值和指标不能冒充 CUDA 验证结果。

## 验证与交付

- 开始修改前检查 Git 状态并阅读 README 和相关 docs。引用本文件时遵守最新用户要求。
- 修复增加相关回归测试；测试可使用明确标记的模拟数据，不能冒充实验结果。
- GPU 数值验证只能在真实 CUDA 设备上执行；无 GPU 的测试必须跳过或明确失败，不能用 CPU 代替。
- 测试集不得用于提示词、策略或阈值调优；GPU 迁移验收优先使用正常验证划分。
- 每阶段记录命令、退出码、版本、实际结果、环境变化、资源消耗、限制及下一步；不把计划描述为已完成。
- 保持一个 API worker 和串行检测通道。PatchCore 与 vLLM 独立常驻进程、空闲 TTL、超时及进程树内存监管。
- 仅对用户指定服务器和项目目录操作；不停止其他项目进程。SSH 私钥只从用户指定本地文件读取，不输出正文。
