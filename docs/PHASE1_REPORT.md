# Phase 1 交付报告

## 1. 完成内容

已完成 CPU bottle 单类别基线，六个命令、配置、依赖锁文件、数据/权重校验、进程树资源限制、校准、原始热图与指标导出均已实现并实际运行。按用户中途追加要求，增加只读 FastAPI、六个工具 schema、React/TypeScript/Vite 骨架、Windows/Linux 分别锁定的依赖以及 CPU Dockerfile。

未实现 Agent、VLM、在线检测任务和完整浏览器检测闭环；工具 schema 不等于工具服务已经实现。未初始化 Git、未创建提交或远程仓库。

## 2. 验证结果

- `uv pip check` 通过，实际 CPU / Pillow / torchvision / Anomalib 验证通过。
- `python -m pytest -q`：23 passed；原始日志 logs/unit-tests.log。
- `scripts/run_phase1.ps1`：check-data、smoke、fit、calibrate、evaluate、predict 全部退出码 0。每步命令/环境/配置/日志/状态在对应运行目录。
- Image AUROC 1.0000、AP 1.0000、F1 0.9767、异常召回 1.0000；Pixel AUROC 0.9783、AUPRO@0.3 0.9239。83 张测试图，3 张正常误报，0 张异常漏检。
- 预热后单图推理 p50 42.43 ms、p95 49.32 ms；不是 API 端到端延迟。
- 前端 `npm run build` 通过；Docker 检查失败原因为 daemon 未启动，未拉取镜像。
- 本机临时 HTTP 验证健康状态、真实运行记录与指标读取返回 200，任务提交明确返回 503；验证后服务已关闭。

完整配置、指标协议与原始输出链接见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 3. 环境变化

仅项目 `.venv` 安装 91 个 Python 包，前端目录安装 24 个包；核心版本按计划固定。新增 .gitignore、环境模板、文档及配置；缓存/临时文件均在项目 E 盘。未升级系统 Python、Node/npm、CUDA、驱动或其他环境；没有付费 API、服务器租赁、常驻后台服务或密钥。

环境完整记录见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 4. 资源消耗

数据 156.6 MB、骨干 46.8 MB，checkpoint 13.83 MB。依赖目标发行文件预算 326.7 MB，未将其伪装成实际网络流量计数。目录逻辑大小合计约 2.56 GiB，可能含共享硬链接，不代表独占物理增量。fit 命令耗时 84.19 s，算法阶段 74.33 s，进程树峰值工作集上界约 1.443 GiB。

## 5. 问题与风险

- 官方数据连接 TLS 握手失败，改用用户明确授权并核验许可的镜像，已固定 revision 与文件哈希；没有官方归档逐字节比对结论。
- Windows venv 启动器导致初版资源读数低估，已改为整棵进程树并增加回归测试；原错误读数保留，fit 使用外部 OS lifetime peak 补测，修正后 smoke 另行验证。
- 初次 requirements 输入误用了 uv 专用参数，修正后解析/安装通过；失败日志仍保留。
- 3/20 正常测试图误报，阈值保持验证集校准结果。不能把完美 AUROC 称为无误报系统或工业产线能力。
- 第三方弃用警告、内部 AUPRO 接口依赖、尚未验证的 Docker 和完整 UI，均保留为已知限制。

## 6. 下一步

停在 Phase 1 结束处。下一阶段按交接顺序封装实际视觉检测、参考图检索与区域复核工具，并验证工具输入输出和失败行为；随后再构建有限状态 Agent。前后端骨架将在服务阶段接入真实任务，不用演示数据补齐缺失功能。

## 7. 需要用户决策

本阶段无阻塞决策，也不需要租 GPU 服务器。下一阶段范围需按原有阶段要求另行确认；当前不自动进入 Phase 2。
