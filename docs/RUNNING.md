# 运行工作台

PowerShell 脚本统一保存为 UTF-8 with BOM，兼容 Windows PowerShell 5.1 与 PowerShell 7；不用更改系统执行策略或控制台代码页。

```powershell
cd E:\秋招\项目\VisionInspect
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B -m pytest -q
cd frontend
npm ci --ignore-scripts --no-audit --no-fund
npm run build
cd ..
.\scripts\start_local.ps1
```

打开 `http://127.0.0.1:8000`，上传 bottle 图像，选择单模型 / 动态 Agent / 固定复核。默认不启用 VLM；本地资产完整后可勾选。按 Ctrl+C 停止服务，不安装系统服务或自动后台任务。

开发时在项目环境启动 `python -m uvicorn visioninspect.api:app --host 127.0.0.1 --port 8000 --workers 1`，另一个终端在 frontend 执行 `npm run dev`，打开 Vite 输出地址。默认同源代理，无需密钥。若前后端分开部署，`VITE_API_BASE_URL` 需在前端构建时配置，后端跨域访问策略需另行设置；推荐保留同源部署。

CLI 使用相同监督流程：

```powershell
.\.venv\Scripts\python.exe -B -m visioninspect.inspection --image .\data\mvtec_ad\bottle\test\broken_small\000.png --mode dynamic
```

任务文件为 outputs/service/{32位任务编号}，CLI 为 outputs/cli/{编号}；包含 request、progress、report、execution JSON、worker.log 与 evidence 文件。报告中的人工候选记录与原始模型状态分开保存。浏览器下载由浏览器默认下载位置管理；验收下载文件已移动至 E 盘 logs，未修改浏览器设置。

## 模型资产

新机器需先按 DATASETS.md、ENVIRONMENT.md 与 Phase 1 命令准备同一数据、权重、split、模型 checkpoint、校准文件，再执行 `python scripts/prepare_tools.py` 构建正常训练参考索引。不得把 weights、data、outputs 加入 Git。工作台不会隐式联网下载。

可选 VLM 准备先运行 `python scripts/prepare_vlm.py --preflight`，它只获取固定 revision 的官方元数据并打印许可/文件大小，不下载权重。核验 `logs/smolvlm-meta.json` 与预算后，再显式运行 `python scripts/prepare_vlm.py`。其断点分段继续脚本为 `scripts/resume_vlm.py`，最终须通过完整 SHA256，才写 source.json 并允许加载。不应直接使用未经校验的 `.part` 文件。

## Docker

`Dockerfile` 构建前端并装入同一 CPU 后端镜像，固定 Python、uv、Node 和 Linux 哈希锁。模型、数据不进入镜像。Windows：

```powershell
.\scripts\run_docker.ps1
```

脚本先检查引擎，失败即停止；构建成功才启动容器。限制 4 CPU、12 GiB，端口仅绑定本机，weights/data 只读，outputs 可写。Linux 宿主机需要给 uid 10001 提供 outputs 写权限。容器通过 `/ready` 检查模型资产可用性。

当前电脑 Docker Desktop 无法启动，`wsl --status` 返回 `WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED`。需要用户决定是否启用 WSL Windows 可选组件及重启。尚未构建或运行此镜像，不能宣称 Docker 已验收。CPU 本地工作台可独立运行。

## Windows PowerShell 5.1 中文编码修复

2026-10-22 用户启动时出现 MissingPropertyName、UnexpectedToken、TerminatorExpectedAtEndOfString，且中文显示乱码。已在 Windows PowerShell 5.1.19041.6456 复现：它将没有 BOM 的 UTF-8 脚本按旧系统代码页读取，乱码中的字符影响字符串解析。此前 PowerShell 7 的解析检查未覆盖该问题。

修复为 5 个项目 `.ps1` 文件添加 UTF-8 BOM，解码后的脚本内容逐字保持一致，原字节副本归档在 `.cache/powershell-encoding-backup/`。没有修改执行策略、系统语言、驱动或 Python 环境。

实际验证：

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_powershell.py
.\.venv\Scripts\python.exe -B logs/verify-powershell51-startup.py
```

两条命令退出 0。6 项编码 / 真实 5.1 解析测试通过；临时使用 8017 端口启动原 start_local.ps1，1.875 秒内启动成功，`/`、`/health`、`/ready` 均 200。该次未提交模型推理任务，验证后已停止临时服务。完整记录在 logs/powershell51-before-fix.json、logs/powershell-encoding-change.json、logs/powershell-encoding-tests.log、logs/powershell51-startup-verification.json。编码回归是针对性测试，不与此前 56 项完整套件数字相加。

当前正常使用仍在根目录运行 `.\scripts\start_local.ps1`，默认端口 8000。如果编辑器再次保存成无 BOM UTF-8，需恢复为 UTF-8 with BOM，再启动；重新安装 Python 依赖不能解决脚本读取编码问题。

## 检测超时与模型初始化（2026-10-22 修复时记录）

截图任务 b8ce3c61e5f54b7d969f3021b98fd96c 的质量检查通过，但异常检测工具耗时 47.018 秒，超过原有 45 秒上限，因此输出作废；尚未调用 VLM。该计时包含冷启动导入、checkpoint 校验与模型加载，不能当作单张图片的模型计算耗时。“当前输入无法可靠检测”的旧提示也没有准确区分服务故障与图像质量。

该次修复将检测模型初始化单独计时，默认预算 90 秒，记录各部分耗时；初始化失败或返回时超过预算便转人工复核。随后工具调用仍执行原有 45 秒超时检查，整个任务仍受独立监督器 180 秒 / 12 GiB 限制。VLM 加载仍计入其工具调用。当时每项任务仍重新加载模型；后续已实现常驻与空闲 TTL，见下一节。网页区分“检测超时”“模型初始化失败”“图像质量未通过”，提供“查看执行轨迹”入口。

原失败任务保留原始结果，不接受已作废的分数。该次计时修复只需浏览器刷新并重新提交；后续常驻机制修改了服务进程中的调度器，启用时必须重启服务。刷新旧任务仅更新提示，不会重新检测。

实际执行命令（均退出 0）：

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_initialization.py tests/test_tools_agent.py tests/test_vlm_input.py
.\.venv\Scripts\python.exe -B scripts/guard.py --seconds 120 --record logs/cold-start-fulltests-execution.json -- .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider -o junit_family=xunit1 --junitxml=logs/cold-start-tests.xml
.\.venv\Scripts\python.exe -B scripts/guard.py --seconds 190 --record logs/cold-start-http-execution.json -- .\.venv\Scripts\python.exe -B logs/verify-initialization-http.py
# 在 frontend 工作目录执行：
npm run build
```

最后一项验证程序的外层 190 秒只覆盖验证程序自身开销，服务内每项任务的硬上限仍是 180 秒。该脚本是本地验收记录，保存在被 Git 忽略的 logs 中；常规使用不需要运行它。

验证结果：

- 针对性测试 34 passed / 4.36 秒；完整套件 74 passed / 44 第三方弃用 warnings / 33.14 秒。模拟时钟用例覆盖 47 秒初始化与 45 秒工具预算分离、初始化超预算、总时限、真正工具超时、坏质量跳过加载及资产缺失，不是实测模型性能。
- 使用正常验证图 `train/good/002.png`，独立 HTTP 状态目录为 outputs/20261022-111546-initialization-http-8f4e45/service。初始化 23.471 秒（导入 8.235、校验 0.029、checkpoint 加载 15.201、设置 0.007），检测工具 0.147 秒，其中模型报告计算耗时 0.047 秒；含本地 VLM 的完整任务 33.411 秒。六项工具均成功，分数 8.26118278503418，阈值仍为 8.742960929870605，状态 normal_candidate。
- 原始热图 256×256 且全为有限数值；报告及四类主要证据 HTTP 返回 200；VLM 实际编码器输入为 [1,2,3,512,512]、128 个图像 token。候选解释内容没有因此获得语义准确性验证。
- 任务进程树峰值 3,018,649,600 字节（约 2.81 GiB），外层验收含任务峰值约 2.90 GiB；GPU 显存不适用。外层验证程序 35.576 秒，退出 0。未重跑正式测试集、调整模型 / 阈值或改写旧失败任务。
- 浏览器打开真实历史失败记录，确认“检测超时”提示与轨迹按钮；截图保存在 logs/screenshots/cold-start-timeout-fixed.png，浏览器检查记录在 logs/cold-start-browser-verification.json。

完整日志：logs/initialization-tests.log、logs/cold-start-fulltests-execution.child.log、logs/cold-start-tests.xml、logs/cold-start-http-execution.child.log、logs/cold-start-frontend-build.log；真实任务详细记录在上述独立 outputs 目录的 verification.json。没有新增依赖、数据下载、常驻任务或系统配置修改。

## 常驻推理进程与空闲 TTL

当前服务会复用已加载的模型，默认连续空闲 600 秒后退出推理子进程。`configs/service.yaml`：

```yaml
schema_version: 1
idle_ttl_seconds: 600
```

TTL 为正数，单位秒，表示任务结束后允许的连续空闲时间。任务运行时不会因空闲 TTL 被回收；180 秒任务超时及 12 GiB 内存限制继续生效。TTL 到期后再次检测，需要重新加载模型。

更新代码或 TTL 后，在原服务终端按 Ctrl+C，等进程退出，再运行 `.\scripts\start_local.ps1`；浏览器刷新。此次没有替用户停止正在运行的旧服务，仅刷新页面不能启用新的调度器。常驻进程只由应用持有，关闭应用时一并清理，不注册系统服务或定时任务。

运行状态可在本机打开 `http://127.0.0.1:8000/api/runtime`，或执行：

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/api/runtime' | ConvertTo-Json -Depth 5
```

`state=idle` 且 `cache.detector_loaded=true` 表示检测模型已保留；`expires_in_seconds` 为剩余空闲时间。任务报告的 `runtime.worker_pid` / `generation` 可证明是否为同一进程，`initialization.timings.cache_hit` 表示检测缓存命中。API 的进程状态与 `/health` 文件可用性是不同检查。

真实验证结果：含 VLM 的连续两项正常验证图检测为 33.67 秒与 3.85 秒，同一推理 PID，第二项检测与 VLM 均复用缓存；验收进程把 TTL 临时设为 2 秒，确认退出后下一项重新加载，分数及原始热图在 rtol=1e-5 / atol=1e-6 内一致。没有修改默认 600 秒或开展新的正式测试集评估。完整命令、输出与资源见 [常驻验证记录](RESIDENT_WORKER.md)。
