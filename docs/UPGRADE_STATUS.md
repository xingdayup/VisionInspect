# GPU 版改造与验收记录

更新：2026-10-22。用户确认保留完整产品，并将 PatchCore 迁移到 GPU。业务、架构及服务器操作分别见 [BUSINESS_V1](BUSINESS_V1.md)、[ARCHITECTURE](ARCHITECTURE.md)、[MIGRATION_PAI](MIGRATION_PAI.md)。

## 已完成的源码改造

- 保留 React 工作台、FastAPI、SQLite、单图与批次、证据、人工复核、历史、导出、LangGraph 与 vLLM；工作流仅有 `baseline` 和 `agent_review`。
- PatchCore 的参数、特征、memory bank、近邻计算和异常图计算迁移到 CUDA FP32。无 CUDA 明确失败；图像解码、文件校验与结果保存仍由 CPU 完成。
- 历史 checkpoint 先在主机内存安全读取，再严格加载到 CUDA；不改模型、预处理、原文件或冻结阈值。报告记录实际设备、checkpoint 原设备和 CUDA 分配器峰值。
- 保留两个模型进程的独立空闲 TTL、启动/任务超时、进程树内存监管和失败证据。数据库升级到 schema 3，删除旧批次 VLM 开关列前备份，保留任务与复核记录。
- 删除旧 CPU SmolVLM、fixed/dynamic 工作流、EfficientAD 实现、Windows 专用入口、CPU Docker 配置、重复锁与过期验证脚本。历史源码与实验文档保存在 Git 标签 `pre-gpu-only-20261022`。
- 统一 Linux GPU 依赖锁、部署入口 `setup_gpu.sh` / `start_gpu.sh`，增加有限时长的真实 GPU 验收入口 `verify_gpu.py`。原数据、权重、校准、划分及用户结果没有物理删除。

## 本机实际执行与结果

以下使用已有项目开发环境，没有向系统、Conda base 或其他环境安装依赖：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --junitxml=logs/gpu-cleanup-tests.xml
npm --prefix frontend run build
& 'D:\SOFTWARE\Git\Git\bin\bash.exe' -n scripts/setup_gpu.sh
& 'D:\SOFTWARE\Git\Git\bin\bash.exe' -n scripts/start_gpu.sh
.\.venv\Scripts\python.exe -B scripts/check_git_content.py
```

- 工程测试：**126 passed、1 skipped、53 warnings，57.94 秒，退出 0**。日志：`logs/gpu-cleanup-tests.log` / `.xml`。覆盖 API、队列、图像校验、划分与校准、证据门控、LangGraph、TTL、环境隔离、网络重试及数据库迁移。
- 唯一跳过项需要真实 CUDA，验证参数/输入/memory bank 的设备与 checkpoint 重载。API 和流程测试中的替身只验证工程行为，不构成 GPU 模型运行或效果证据。
- TypeScript 检查与 Vite 生产构建退出 0。真实浏览器检查工作台、批次及系统页：完整导航保留，仅两种工作流，无 CUDA 明确提示并禁止提交检测；记录在 `logs/gpu-browser-qa.json`，截图为 `logs/gpu-only-ui.png`。临时验收服务已停止。
- Shell 入口通过 Git Bash 语法检查；不代表已在 Ubuntu 安装或运行。Python 源码语法、文档链接和 Git 私有资产检查另行记录在 `logs/gpu-cleanup-audit.json`。
- 无重新训练、无重算阈值、无正式测试集评估，也未调用付费 API。

## 依赖解析与资源预算

实际解析命令使用现有 Python，禁止自动下载解释器：

```powershell
$env:UV_CACHE_DIR="$PWD\.cache\uv"
$env:UV_PYTHON_DOWNLOADS='never'
uv pip compile requirements.in --python .venv/Scripts/python.exe --python-platform x86_64-manylinux_2_35 --generate-hashes --no-emit-index-url --only-binary :all: --no-binary freia --output-file requirements.txt
```

解析成功，131 个锁定包，日志为 `logs/gpu-requirements-compile.log`。FrEIA 0.2 是唯一经过审查的纯 Python 源码例外（34,270 字节，固定 SHA256），构建后端 setuptools 固定版本；其余包要求兼容 wheel。

通过官方 PyPI 元数据、Linux CPython 3.12 / x86_64 wheel 标签估算，候选包合计 **2,983,835,068 字节，约 2.98 GB**，超过原 **1.5 GB** 上限。`logs/gpu-dependency-budget.json` 的 `passed: false` 表示预算停止。这个估算在 Windows 执行，不能代表服务器磁盘或原生依赖已验证。本轮没有下载或安装这些 CUDA 分发包。

安装脚本保留预算限制；需明确确认新预算后才允许传入更高的 `--download-budget-bytes`。不自动复用系统包、升级 CUDA 或驱动。Qwen 权重另需约 17.54 GB 下载、至少 50 GiB 可用空间，未包含在依赖预算内。两模型同时运行的 A10 显存峰值仍待实测。

## 已有远端证据与待验收

用户提供的 PAI 输出已核实：NVIDIA A10、CUDA 13、SM 8.6，torch 2.13.0、torchvision 0.28.0、vLLM 0.27.1、Transformers 5.16.1，原生扩展及模型架构检查 `passed: True`、退出 0。这不能证明应用独立环境、权重加载或生成已完成。

服务器仍需完成：应用环境安装；真实 PatchCore CUDA 分数、热图与重载一致性；Qwen 固定权重加载、双图/三图结构化生成；真实复核分支与两个进程的 TTL；冷启动、延迟、内存和显存峰值。新入口仅使用正常验证图，不调冻结区间，不报告准确率收益；若未触发多步动作，会明确记录 `full_agent_loop_exercised: false`。

历史 CPU 指标不能作为 GPU 验收。Agent 的误报/漏报收益仍需独立标注数据，不能用已见测试图调提示词、阈值或策略。当前交付是 GPU 源码改造与本机工程验证，完整 GPU 运行验收待服务器执行。

## AutoDL 部署准备与预算批准

用户随后提供了新服务器，并明确批准总下载上限 26 GB、模型下载前至少空闲 25 GiB、完成后至少保留 8 GiB。本次实测服务器报告 RTX 4080 SUPER / 32760 MiB 显存、16 核容器配额、约 62 GiB 内存、Python 3.12.3、平台 torch 2.12.1+cu130；CUDA 小算子通过。与此前 PAI 属于不同环境。

GPU 源码提交 4440cfb 与 367 个资产文件已迁移到 `/root/autodl-tmp/VisionInspect` 并核验。新服务器没有 vLLM，不能直接复用此前 PAI 的验证结论；已增加 `requirements-vllm.txt` 与隔离安装入口。两环境冷下载约 6.85 GB，按官方文件去重约 4.03 GB；中转使用剩余字节范围，既有分片保留，传输账本不将两段流量合并成一次。

服务器 PyPI 可达，Hugging Face 页面连接失败，但其官方元数据返回的 CDN 可达。没有修改系统 DNS、代理或证书，没有切换未知镜像。分发文件与权重均须匹配固定哈希后才进入安装或加载。

新增预算和锁选择回归后，本机执行 `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --junitxml=logs/autodl-source-tests.xml`，133 passed、1 skipped、53 warnings，55.94 秒，退出 0。真实服务器的安装、模型加载与验收结果另见后续记录，不能以这些本机测试代替。
