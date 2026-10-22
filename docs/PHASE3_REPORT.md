# Phase 3 Agent、任务服务与 A/D 对照

## 完成内容

有限状态条件 Agent、六个本地工具、完整输入输出校验、调用预算与一次可重试错误重试。FastAPI 接入任务队列、真实子进程推理、SQLite 持久历史、证据读取、人工研究复核、JSON 导出；CLI 与网页复用同一监督器。没有付费 API、密钥或远程模型执行。

## 实际验证

- `python -B -m pytest -q`：33 passed、44 warnings、26.08 秒，日志 `logs/phase4-tests.log`。新增超时结果作废与有界重试恢复测试。
- 兼容路径改动后，`python -B -m pytest -q tests/test_tools_agent.py tests/test_service.py`：10 passed，3.07 秒。
- `python -B scripts/verify_http.py`：退出码 0。真实 broken_large/000.png 任务 `383879c113164b7d9955357d770b8462`，202 创建、completed / anomaly_candidate；26 秒左右完成，原图、热图、叠加图和 .npy 均 HTTP 200，JSON 报告可读。轨迹依次为质量、模型、区域、规则汇总，数据见 `logs/integration-task.json`。
- 入库 SQL 占位符曾导致 1 failed / 30 passed，见 `logs/phase3-tests.log`；修正后服务测试通过，保留失败记录。一次临时内联命令的引号错误导致 SyntaxError，后改为可重跑的 verify_http.py，未把失败作为推理结果。

## 冻结 A / D 比较

命令：`python -B scripts/benchmark_agent.py --mode dynamic`，退出码 0。输出 `outputs/20261021-235303-agent-dynamic`，每张图报告与 frozen.json 可核验。83 张完整测试图，不使用测试结果调整策略。

| 项目 | A 单模型 | D 动态工具（VLM 关闭） |
|---|---:|---:|
| 完整测试图 | 83 | 83 |
| 自动产生候选 | 83 | 78 |
| 人工复核暂缓 | 0 | 5（均正常标签） |
| 自动候选误报 | 3 | 1 |
| 自动候选漏报 | 0 | 0 |
| 工具成功率 / 无效调用率 | 不适用 | 100% / 0% |
| 平均工具调用 | 不适用 | 3.8916 |
| 热进程端到端 p50 / p95 | 基线模型计时口径不同 | 0.2180 / 0.3663 秒 |

D 接受覆盖率 93.98%，平均 3.89 次调用，83 张总计约 18.63 秒，进程峰值 629,866,496 字节。人工复核属于拒绝自动判断，不能把暂缓样本当正确结果，也不能凭 accepted FP/FN 声称总体准确率提升。模型连续分数未被 Agent 改写，AUROC、AP、像素指标仍使用 Phase 1 原始结果。

工作台每次独立子进程冷启动约 25–26 秒，远大于热模型预测约 0.04 秒；这两个口径分别报告，不混为网页响应性能。

## 环境变化与可选 VLM

保留原有 91 个包版本，新增 python-multipart 0.0.32、transformers 5.20.2 及其 5 个新增依赖，现为 98 包。`uv pip check` 通过。解析使用上一完整锁作为 constraints；Windows 总分发预算 342,892,238 字节，Linux 哈希锁单独解析成功。未改变 base、test、系统 Python、CUDA 或驱动。

本地 SmolVLM 固定版本总文件 1,019,893,574 字节，完整主权重 SHA256 校验通过。初始流式下载缓慢，受控停止退出 15，保留前缀；同来源 HTTP Range 续传成功，没有替换来源或关闭证书。资产完整后才生成 source.json。许可证及哈希见 THIRD_PARTY.md。

`python scripts/guard.py --seconds 180 --record logs/vlm-smoke-execution.json -- .venv/Scripts/python.exe -B scripts/verify_vlm.py` 退出 0；验证图是正常 validation 的 train/good/002.png，参考是训练 train/good/000.png，未使用测试图片。模型加载 22.27 秒、生成 4.20 秒；进程树峰值 2,704,224,256 字节。Python socket 连接已禁用，未联网。输出保留为未经验证的 unknown + 原始描述，不作为已验证缺陷类别。

早期 guard 未显式绑定 Windows 子进程 stdout，冒烟 stdout 日志为空，但结构化 smoke JSON 和退出码/资源记录完整。后续 guard 已显式保存 child.log，避免漏掉失败信息。

固定 VLM 与动态 VLM 对照均已完成，两个命令和逐图分数/热图一致性核验均退出 0。C 调用 VLM 83 次，D 调用 5 次；同为 78 张自动候选、5 张正常样本暂缓复核。具体指标、计时口径及原始产物目录见 COMPARISON.md，不能据此宣称准确率提升。EfficientAD 标准训练需要额外 ImageNet 类惩罚图像，当前尚未准备，未以替代数据冒充标准对照。多类别、VisA、LOCO 未混入 bottle 实验。
