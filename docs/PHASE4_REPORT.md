# Phase 4 中文工作台与本地交付

本报告保留该阶段的构建与测试数字；后续 Radix、监管故障恢复和最新 56 项完整测试见 [环境追加记录](ENVIRONMENT.md) 与 [当前文档索引](README.md)。当前截图为本地 logs/screenshots/completion-evidence.png。本次目录整理未重新运行前端或正式模型实验。

## 完成内容

中文 React/TypeScript/Vite 六个页面：工作台上传、视觉证据、Agent 轨迹、历史筛选、报告导出与系统状态。数据全部来自实际 API；没有硬编码假分数或演示热图。支持 PNG/JPEG/WebP、10 MiB 大小限制、重复提交禁用、进度轮询、失败反馈、单图原始热图下载。使用原生语义控件、键盘焦点、桌面/窄屏布局。

视觉证据包含原图与叠加/热图切换、原始浮点 .npy、模型分数/阈值/耗时、疑似区域、正常训练参考、可选本地 VLM 原始描述。人工研究复核与模型原始结论分别保存。导航切换回到顶部；更换任务时清空上个任务的复核输入，防止误存至另一任务。

## 实际验证

- `npm run build`：TypeScript 检查和 Vite 生产构建退出码 0；最终 JS 约 250.8 kB（gzip 约 78.2 kB），CSS 约 20.0 kB。
- `python -B -m pytest -q`：36 passed、44 第三方弃用 warnings、27.54 秒，退出码 0，见 logs/final-tests.log。
- 随后补充 Windows / POSIX 权重来源路径兼容回归，针对性测试 `tests/test_tools_agent.py tests/test_assets.py` 为 12 passed，0.55 秒，见 logs/final-portability-tests.log。未把这次针对性运行冒称完整套件运行。冻结 C/D 实现的三个源文件已在修改前复制并核对 SHA，与 frozen.json 完全匹配，保存在各实验 source/visioninspect。
- `uv pip check --python .venv/Scripts/python.exe`：98 包兼容检查通过。
- 真实浏览器上传 broken_small/000.png，任务 `44175c6e7aad4497b9df83c7a8c3eb9f`，异常分数 17.123121，阈值 8.742961，4 次工具调用、端到端 26.11 秒；保存了带文字依据的研究复核记录，模型原报告未修改。
- 浏览器完成历史按任务编号检索、打开任务、报告 JSON 下载。浏览器默认曾将验收下载保存到用户下载目录，随后移至 `logs/browser-export-44175c6e7aad.json`，未修改浏览器设置。
- 浏览器上传正常 validation 图 train/good/002.png，固定复核并开启本地 VLM；任务 `82c060be9da649d1bd59e29aa65540ca`，六个工具全部 completed，分数 8.2612，正常候选，完整任务 33.484 秒；真实参考图及未经验证的原始候选描述可见。结构化快照 logs/browser-vlm-task.json。
- 390px 断点：document/body scrollWidth 375px，无横向溢出，所有证据图 complete 且 naturalWidth > 0。1440px 桌面证据视图已人工检查。
- 实际停止临时后端后点击系统“重新检查”，页面显示后端不可用与明确连接错误；重新启动后恢复历史和连接。
- 选择 README.md 验证前端非法文件提示，未提交检测，开始按钮保持禁用。API 测试覆盖损坏图片、超大请求、模型缺失 503、未知任务、报告前置条件和持久记录。
- `python scripts/guard.py --seconds 180 --record logs/final-cli-execution.json -- .venv/Scripts/python.exe -B -m visioninspect.inspection --image data/mvtec_ad/bottle/train/good/002.png --mode dynamic`：退出 0，CLI 任务 909c432e771d4812a62a10ebedd02095 为 completed / normal_candidate，报告与进程日志均保留。
- `python scripts/prepare_vlm.py --preflight`：退出 0，只读取固定 revision 元数据，13 个文件、1,019,893,574 字节、Apache-2.0，未额外下载权重。

最终真实截图为 logs/screenshots/workstation-final.png；离线反馈截图 logs/screenshots/backend-offline.png。截图属于本地验收证据，未提交或发布。

## 环境变化与资源

前端沿用项目内固定 React 19.3.1 / Vite 8.3.5 / TypeScript 7.0.2，无全局 Node 升级或额外前端依赖。Python 使用项目 .venv，完整 98 包版本见 logs/installed-phase4.json 与最终审计 JSON。缓存、数据、权重、项目临时文件和输出均在 E 盘项目内。

本地启动使用 `scripts/start_local.ps1`，单个 uvicorn worker、一个推理执行通道。API 启动前显式设置 Python tempfile 到项目 .tmp；multipart 解析前限制请求总大小，避免不受控写临时文件。SQLite 连接按操作提交并关闭。

资源快照由 `python -B scripts/final_audit.py` 生成 `logs/final-delivery-audit.json`，逻辑大小可能重复计算硬链接，不能当成物理磁盘增量或精确网络下载量。VLM 实验进程树峰值约 2.67 GiB；GPU/显存不适用。

实际主要目录合计约 4.006 GiB，E 盘剩余约 1039.3 GiB。CLI 独立任务监管记录为退出 0、25.821 秒、进程树峰值 679,428,096 字节。原始交接文档哈希未变化。

## 部署状态与已知限制

已提供包含前端构建阶段的 CPU Dockerfile、Linux 哈希锁、readiness 检查与受资源限制的启动脚本，数据/权重外部挂载，outputs 可写。**Docker 尚未构建或运行验收**：现有 Docker Desktop 启动失败，WSL 返回 WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED。仅尝试启动已安装应用，未启用系统组件、改 BIOS/网络、重启系统或拉取镜像；测试后已停止 Docker Desktop。

单机研究工作台没有账户体系、多人协作或产线发布能力；默认只绑定 localhost。VLM 为小型英文通用模型，描述未经缺陷语义验证；candidate 保留 unknown，不把可见描述当作已验证缺陷类型。EfficientAD、其他类别、VisA、LOCO 尚未评估。完整结果与限制见 COMPARISON.md。

## 下一步与用户决策

本地 CPU 版可按 RUNNING.md 启动。Docker 的剩余验证需要用户提供可用引擎，或明确接受暂保留容器验收待办；此决定已单独询问，不能用未回复代替启用 Windows 组件的授权。扩展数据或模型前须重新核验许可、兼容性与资源预算。
