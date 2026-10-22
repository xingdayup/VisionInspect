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

- 工程测试：**126 passed、1 skipped、53 warnings，57.94 秒，退出 0**（当时基线，后续轮次的数字见文末记录）。日志：`logs/gpu-cleanup-tests.log` / `.xml`。覆盖 API、队列、图像校验、划分与校准、证据门控、LangGraph、TTL、环境隔离、网络重试及数据库迁移。
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

## 2026-10-22 工程化：格式化、静态检查、标注与全新服务器入口

本轮只改可读性、静态检查和部署入口，不修改任何推理逻辑、阈值、模型或实验结论。

- 格式化：`ruff format` 覆盖 53 个 Python 文件，`prettier 3.6.2` 覆盖前端 4 个文件（`main.tsx` 从 219 行展开为 1581 行，最长行从 1936 字符降到 153）。纯排版，无行为变化。
- 静态检查：ruff 规则集 `E4/E7/E9/F/I/UP`，修复 42 处未使用导入与单行多导入、2 处手工问题；`pyproject.toml` 固化配置，`scripts/*.py` 豁免 `E402`。
- 标注与文档：包内 152 个函数返回标注从 **8% → 100%**，docstring 从 **3% → 81%**。`service.py`、`resident.py`、`vllm_runtime.py` 等并发模块补上不变量说明（队列事务边界、generation token、lease 语义、失败降级）。
- CI：新增 `.github/workflows/ci.yml`，含 style（ruff + prettier + `tsc` + `vite build`）与 pytest 两个 job。本机等价命令全部通过；GitHub runner 上的依赖安装尚未验证。
- 部署：新增 `scripts/deploy_fresh_gpu.sh` 单一入口，固定「审计 → 资产检查 → 应用环境 → vLLM 环境 → 网络探测 → 权重 → 验收」顺序。`pai_vllm.py` 移除内置默认根目录，改为 `--root` → `VISIONINSPECT_GPU_ROOT` → 报错；`download` 在缺少 `runtime-check.json` 时提示先运行 `setup_vllm.sh`。
- 回归：新增 5 项测试覆盖根目录回退与下载前置条件，本机 `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider` 为 **138 passed、1 skipped，55.55 秒，退出 0**。

这些是本机工程验证，不构成 GPU 运行、精度或收益证据。新实例上仍需执行 `deploy_fresh_gpu.sh` 第 5 阶段的两次验收并记录峰值。

## 2026-10-22 真机验收：AutoDL RTX 4080 SUPER 32 GB

第一次在真实 GPU 上跑通完整链路，验收记录在服务器 `outputs/20261022-224407-gpu-verification-f294a5`。

```json
{"passed": true, "full_agent_loop_exercised": true, "vlm_requested": true}
```

| 项 | 实测 |
| --- | --- |
| 设备 / 运行时 | RTX 4080 SUPER（SM 8.9，BF16）/ CUDA 13.0 / torch 2.13.0+cu130 |
| PatchCore CUDA 峰值分配 | 57,566,208 字节（54.9 MiB） |
| checkpoint 重载后热图差异 | 0.0（逐位一致） |
| clear 样本（区间外） | `normal_candidate`，0 次模型调用 |
| uncertain 样本（区间内） | `normal_candidate`，2 次模型调用 |
| 单次结构化决策耗时 | 3.76 秒（957 prompt tokens / 123 completion tokens） |
| vLLM 峰值显存 | 27,654 MiB |
| vLLM 进程树峰值 RSS | 7,798,042,624 字节（7.26 GiB） |
| 空闲 TTL 回收 | 通过（冒烟用 2 秒 TTL） |

只使用正常验证样本，未重算阈值、未评估准确率。这些数字证明**工程链路可用**，不构成精度或收益证据。

### 本轮修掉的四个真实缺陷

| 缺陷 | 现象 | 提交 |
| --- | --- | --- |
| vLLM 子进程 PATH 缺少 venv 的 bin | flashinfer 首次推理 JIT 编译时 `FileNotFoundError: ninja`，引擎初始化失败 | `d72529a` |
| 上一修复用 `sys.executable.resolve()` 取目录 | venv 的 python 是软链，resolve 后指向 conda base，修复等于无效 | `145c97f` |
| 提示词证据 id 与实际不符 | 提示词写 `query_crop`/`reference_crop`，实际 id 是 `region:Rk`/`matched:Rk:N`，模型必然绑错证据 | `02377b7` |
| `compare_region` 省略 `reference_id` 被直接拒绝 | 契约中该字段可选，校验却要求必填，合理输出被判 schema_error | `16ece16` |

另外调整了失败策略：`schema_error` 属于模型格式失误而非证据矛盾，允许在调用预算内重试一次。

### 网络实测

| 路线 | 速度 |
| --- | --- |
| Hugging Face（经 AutoDL 学术加速代理） | 0.7 MB/s |
| Hugging Face（直连） | 0（12 秒 0 字节） |
| ModelScope（国内直连） | 17 MB/s |
| 清华 PyPI（经代理） | 285 kB/s |
| 清华 PyPI（不走代理） | 15.5 MB/s |

权重改用 ModelScope 固定 revision `5d854aab08710c16b980ec6d603d863b3821b915` 下载，15 个文件仍逐一通过 `pai_vllm.py` 的大小与 SHA256 校验后才写入 `source.json`，没有放宽任何验证。学术加速只对 Hugging Face 有用，对 pip 与 ModelScope 反而是负担。

### 仍未完成

`baseline` 与 `agent_review` 的对照实验（图像级 AUROC、AUPRO、复核触发比例）。历史 CPU 记录 `image_auroc 1.0 / aupro 0.924` 来自 CPU checkpoint，不能作为 GPU 结果引用；需要在新设备上重新 `fit` → `calibrate` → `evaluate` 才有可比数据。

## 2026-10-22 GPU 实验：baseline 指标与 agent_review 对照

### GPU 上的 baseline（新拟合，独立于冻结资产）

| 项 | GPU | CPU（历史） |
| --- | --- | --- |
| 拟合 168 张 | 18.9 s | 74.3 s |
| 校准 41 张 | 12.7 s | — |
| 评估 83 张 | 21.4 s | 22.0 s |
| 单图推理 p50 | 13.7 ms | 42.4 ms |
| image AUROC | 1.000 | 1.000 |
| image F1 | 0.9844 | 0.9767 |
| pixel AUROC | 0.9781 | 0.9783 |
| AUPRO | 0.9181 | 0.9239 |

GPU 把推理提速约 3 倍，精度与 CPU 处于同一水平；差异来自不同 checkpoint，不是设备。

### baseline 与 agent_review 同图对照（冻结资产，83 张测试图）

`scripts/experiment_review_comparison.py` 逐图运行两种工作流，不修改阈值与区间。

| 指标 | 不加 agent | agent（预算 3） | agent（预算 5） |
| --- | ---: | ---: | ---: |
| 误报落到用户手里 | 3 | 1 | 1 |
| 转人工 | 0 | 4 | 3 |
| 自动下结论 | 83 | 79 | 80 |
| 其中正确 | 80 | 76 | 79 |
| 模型调用 | 0 | 15 | 19 |
| 触发图单张耗时 | ~14 ms | 32.8 s | 49.2 s |

复盘：区间内 5 张全部是 `test/good`。预算 3 时 4 张转人工，原因是「尚有未完成正常局部核查的疑似区域」——检出 3 个区域需要 1 次整体 + 3 次局部 + 1 次结束共 5 次调用，而预算只有 3。改到 5 次后其中 1 张被正确判回 `normal`，人工降到 3 张，`configs/gpu.yaml` 的 `max_model_calls` 因此固定为 5。

### 单类方法的能力边界（这轮最重要的结论）

PatchCore 只在 `train/good` 上建立 memory bank，训练与校准阶段没有任何缺陷样本。由此得到两个结构性事实：

1. **缺陷侧的分布只存在于测试集**。任何用缺陷分位数推导阈值或复核上界的做法，都会消耗掉评估数据。当前上界公式 `threshold + (threshold - median)` 只用验证集正常图，因此是干净的。
2. **复核 Agent 只能减少误报，不能减少漏报**。它对缺陷没有先验，只能复核"像正常却被判异常"的样本；漏报来自模型没见过的缺陷形态，复核救不了。

测试集上的分数分布验证了这一点：正常图最高 9.380，最低缺陷（contamination）10.632，中间是 1.25 宽的空档；`broken_large` / `broken_small` 最低分都在 17 以上，从不参与边界问题。

### 仍未完成

冻结资产仍是 CPU 时代的 checkpoint（正常图最高 10.114，缺陷最低 10.421，空档仅 0.31）；新 GPU 拟合的分离度是它的 4 倍，但尚未替换注册。替换需要重新生成 profile 与参考索引，属于资产变更，需单独决定。
