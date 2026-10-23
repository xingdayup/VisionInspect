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

## 2026-10-23 复核可判性诊断

把 4 张落入复核区间的测试图逐轮拆开后，失败点全部落在确定性的门控或模型协议上，没有一张是"看不清"：

| 门控 | 现象 | 判断 |
| --- | --- | --- |
| 重复视图 | 查完 R1…R4 后不选 finish，又请求已看过的区域，循环立即终止 | 程序反应过刚：这是规划失误，不是证据矛盾 |
| 参考相似度下限 0.99939 | 实测 0.99923–0.99963，差 1e-4 即判不合格 | 分位数落在噪声量级，已改为绝对下限 0.99 |
| 区域截断 | 正常件纹理常产生 >4 个连通区，规则一律禁止 normal | 越"正常"越判不了，已改为可配置，默认放行并保留 truncated 标记 |
| 调用预算 | 1 次整体 + N 个区域 + 1 次结束 = N+2；N=4 需要 6 次 | 原预算 5 在数学上不可能完成，已改为 6 |

对应改动：`393a6cd`（相似度绝对下限、截断策略可配置、提示词）、`00ba961`（预算 6）、`7db8186`（重复视图给予一次纠正调用）。

**验证未完成**：修复后启动的对照实验在运行中因实例停机中断，需要重新开机后重跑 `experiment_review_comparison.py` 才能给出"是否可判"的结论。在此之前不得声称复核已能自动下结论。

### 2026-10-23 第二轮：把判断依据显式交给模型

参考 [arXiv:2502.09057](https://arxiv.org/abs/2502.09057)（VLM 少样本视觉质检）的做法——在上下文里提供"判据文本 + 对应样例"，而不是只给参考图让模型自行推断。据此做了两处补充：

- 提示词加入四条判据清单：形状与边缘、纹理、异物、以及"反光/亮度/色差本身不算缺陷，除非同时伴随几何或纹理变化"。后者直接针对模型把反光误判为缺陷的倾向。
- 复核上下文显式给出 `remaining_regions` 与 `coverage`（例如 `1/4`），模型不再需要从区域列表里自行推算还剩哪些没看。对应 `7db8186` 之后的上下文改动与测试。

本机测试 144 passed、1 skipped。**仍未在真机上验证**：实例停机，对照实验无法执行。

### 2026-10-23 第三轮：静默弃权与“真实失败序列回放”

- **静默弃权**：若模型把全部区域都判为 normal，却仍以 `unknown` 结束，循环会要求它给出一次明确结论（`silent_abstention`），而不是直接转人工。弃权只保留给证据冲突或确实无法辨读的情况。
- **回放验证**：把真机上实际出现的失败序列（R1→R2→R3→R4 全 normal，之后重复请求 R4）写成回归测试，喂给真实的 LangGraph 循环。现在是 **6 次调用、`normal_candidate`、accepted=true**，即同一组模型行为在修复后能够收敛。
- **结构化输出修复**：代码块或前后散文包裹的 JSON 现在会被提取并重新校验（`parse_decision`），失败不再直接判死一次复核。

**边界**：回放使用的是记录下来的模型输出，不是新的推理。它证明"循环不再自我阻塞"，不证明"模型在真实图像上会这样回答"。真机验证仍需实例恢复后执行。

### 弃权的处理方式（依据 ReCoVERR）

[arXiv:2402.15610](https://arxiv.org/abs/2402.15610) 针对"选择性预测对 VLM 过度保守"提出：低置信时不要直接弃权，而是围绕当前预测继续取证，证据足够就作答。据此把弃权改成一次**更窄的追问**：当全部区域均已判为 normal 而模型仍给 `unknown` 时，下一次调用会在上下文里带上

> 全部疑似区域均已比对且未见异常。请直接给出 verdict=normal 或 verdict=anomaly 的 finish 结论，不要再用 unknown。

该字段 `abstention_nudge` 只作用一次，测试固定了这一点。弃权仍然保留给证据冲突或确实无法辨读的情况，没有被取消。

本机测试 151 passed、1 skipped。真机验证仍待实例恢复。

### 证据分辨率下限

复核用的局部裁剪直接来自异常热图，尺寸常常只有 50–70 像素（单元测试里就出现过 64×72）。这样的图送进 VLM，"看不清"几乎是必然结果，与提示词无关。

现在发送前统一把最短边不足 256 像素的证据放大到 256（保持长宽比，LANCZOS）。**这不增加信息**，只是消除"给模型的图比模型能看的还小"这个人为劣势；实际发送尺寸仍记录在 `vlm_input/round_*/request.json` 的 audit 里，便于事后核对。

### 拒绝原因改为机器可读

报告新增 `agent_review.gates`，把中文拒绝原因映射成稳定代码（`region_unreviewed`、`regions_truncated`、`anomaly_conflict`、`no_normal_evidence`、`reference_similarity`、`no_final_decision`、`model_failure`）。理由文本面向人，代码面向统计。

实验脚本的汇总里新增 `rejection_gates`，下次真机跑完可以直接按原因分组，不必再靠通读中文原因去猜是哪条门控拦住的——这轮排查就是靠人工读原因，花了很久。

## 2026-10-23 修复后首次真机验证：从"全阻断"变为"部分可判"

实例恢复后重跑 `experiment_review_comparison.py`（运行目录 `outputs/20261023-133956-review-comparison-cd718c`）。配置：阈值 8.989、区间 [8.824, 9.610]、预算 8、相似度下限 0.99。

| 图 | 调用 | 局部观察 | 最终结论 | 拒绝原因 |
| --- | ---: | --- | --- | --- |
| good/001 | 3 | R1=anomaly, R2=normal | normal | `anomaly_conflict`, `region_unreviewed` |
| good/006 | 5 | R1/R2/R4=normal, R3=unclear | normal | `region_unreviewed` |
| good/015 | 5 | R1/R2/R3=anomaly, R4=normal | normal | `anomaly_conflict`, `region_unreviewed` |
| good/019 | 5 | R1–R4 全 normal | normal | 无，**accepted** |

汇总：复核 4 张，1 张给出 `normal_candidate`（正确），3 张转人工；误报落到用户手里的从 2 降到 0（对应两张被转人工）；平均单张复核 42.8 秒、最多 5 次调用。

**与修复前的区别**：此前 4/4 被程序自身阻断（重复视图触发终止、预算不足、相似度噪声）。现在阻断全部来自证据本身——模型在局部标了 anomaly 却给出 normal，或某个区域只能给 unclear。门控在做它该做的事，而不是替程序的缺陷背锅。

**仍未解决**：模型自相矛盾。3 张失败里 2 张属于此类。是否要再追问一次（可能把本已判对的正常图推成 anomaly），还是维持转人工，属于产品取舍，未擅自改动。

### 尝试"unclear 补查"及其结果（运行 `outputs/20261023-134905-review-comparison-4168fe`）

对 unclear 区域重开一次补查后重跑同一批：

| 图 | 局部观察 | 结论 |
| --- | --- | --- |
| good/001 | R1=anomaly, R2=unclear | manual（`model_failure`） |
| good/006 | R1/R2/R4 normal, R3 仍 unclear | manual |
| good/015 | **R1–R4 全部 anomaly** | `anomaly_candidate`（**假阳性**，该图分数 8.888、基线判对） |
| good/019 | R1–R3 anomaly, R4 normal | manual（`anomaly_conflict`） |

结论：补查没有解决 unclear（006 补到 6 次调用仍为 unclear），且这一轮模型把正常图的局部区域大面积标成 anomaly。上一轮同一批图里 019 被判回 normal，这一轮变成 manual——**同样的图、同样的模型，区域级判断在不同轮次间不稳定**。

因此该能力改为可配置并**默认关闭**（`regions.reopen_unclear_regions: false`），保留代码路径以便在更好的模型上重新评估。

**注意**：两次运行之间上下文结构有变化，不是受控 A/B，不能据此断言"补查导致变差"；能确定的只是它没有带来改善，而区域级判断的方差很大。

**当前瓶颈**：不再是循环或门控，而是 VLM 对正常图局部裁剪的假阳性与不一致。

### 2026-10-23 区域级测量：真正的问题是锚定，不是看不看得清

新增 `scripts/measure_region_judgement.py`：对 8 张已知正常图各取 2 个区域，用同一套提示词与证据视图做两次独立判断——一次无上下文，一次在前面注入"该区域存在纹理差异"的既有结论。

| 指标 | 结果 |
| --- | --- |
| 孤立判断的假阳性 | 1 / 11 = **9.1%** |
| 注入既有结论后仍判 anomaly | 6 / 11 |
| **被既有结论翻转** | **5 / 11 = 45.5%** |

结论明确：模型对正常区域的独立判断并不差（91% 判对），但**把上一轮的结论喂回去，它会跟着走**。而循环原本同时提供 `previous_observations`（全部历史观察）和带结论的 `reviewed_regions`，等于每轮都在强化首轮判断。这解释了同一批图在不同轮次出现"全 normal"与"全 anomaly"的漂移。

修复（`9d4e31d`）：上下文只提供已查看区域的 **id 列表**，不再给结论，也不再把历史观察回灌；提示词增加"每轮独立判断当前视图，不要为与之前一致而改变结论"。有测试固定该契约。

这是本轮最重要的发现：**此前多轮把问题归因于模型能力，实测表明一半以上是提示上下文的自强化。**

### 去锚定后的复测（运行 `outputs/20261023-141942-review-comparison-*`）

| 图 | 分数 | baseline | 复核后 | 调用 |
| --- | ---: | --- | --- | ---: |
| good/001 | 8.861 | normal（正确） | manual（`no_final_decision`） | 4 |
| good/006 | 9.320 | anomaly（**误报**） | **normal** | 6 |
| good/015 | 8.888 | normal（正确） | **normal** | 6 |
| good/019 | 9.380 | anomaly（**误报**） | **normal** | 6 |

汇总：**3/4 给出 `normal_candidate` 且全部正确**，1 张转人工。误报流到用户的数量从 2 降到 0；自动下结论从 83 降到 80（1 张被降级为人工）；平均单张复核约 6 次调用。

同一批图在去锚定前是 1/4 可判、且出现过把正常图判成 anomaly 的假阳性。这次没有出现假阳性。

**归因说明**：两次运行之间只有"上下文不再回灌历史结论"这一处改动（另修正了上一跑因残留 vLLM 进程占住 8001 导致的失败）。区域级测量给的 45% 翻转率支持这一机制，但两次运行仍是同一批 4 张图，样本很小，不能当作统计结论。

### 剩下的那一张：追问已发出，模型两次都看不清

`good/001` 的最后轨迹（运行 `outputs/20261023-143250-review-comparison-434aac`）：

```
round 1  global      compare_region  R1 normal
round 2  region R1   compare_region  R1 normal
round 3  region R2   compare_region  R2 unclear
round 4  region R2   finish unknown  R2 unclear   → 触发追问（schema_error 重试）
round 5  region R2   finish unknown  R2 unclear   → 不再追问，转人工
```

追问确实发出了，模型在拿到"请针对 R2 给出明确判断"之后仍然写 unclear。这说明该区域对它而言确实是不可判读的，转人工是正确结果，而不是循环缺陷。

当前这批图的最终状态：**3/4 自动给出正确结论，1/4 因真实不可判读转人工，零假阳性。**

### 未完成的测量

把区域级测量扩到 20 张图时，因与对照实验争用同一块 GPU 与 8001 端口，在 8 个区域处中止（此前 11 个区域的数字仍只有参考价值）。**扩大样本的测量仍未完成。**

### 2026-10-23 扩大评测：区间外复核会把缺陷判成正常

为测量可判性，新增仅用于实验的开关 `VISIONINSPECT_REVIEW_FORCE=1`（生产永不设置，有测试固定默认关闭），并给实验脚本加 `--force-review N`：强制复核区间外难度最高的 N 张（一半是分数最高的正常图，一半是分数最低的缺陷图）。

运行 `logs/review-hard2.log`，共复核 24 张（区间内 4 + 强制 20），结果：

| 真值 | 复核数 | 判 normal | 判 anomaly | 转人工 | 错误 |
| --- | ---: | ---: | ---: | ---: | --- |
| 缺陷 | 10 | **6** | 1 | 3 | **6 个漏检** |
| 正常 | 14 | 10 | 1 | 3 | 1 个误报 |

**这是目前最重要的负面结果**：缺陷图里分数最低的 10 张（contamination 系列，10.63–11 区间），复核后 6 张被判为 `normal_candidate`。也就是说，**复核在区间外不具备安全性**。

由此可以确认两件事：

1. 现行生产策略（只复核 `[8.824, 9.610]` 这 4 张）不是保守过度，而是必要的——一旦放宽区间，VLM 会把真实缺陷判成正常。
2. 之前"3/4 可判、零假阳性"的结论只在**狭窄区间内**成立，不能外推。

**待办**：若将来要放宽区间，必须先加一道"不得把基线高置信异常降级为正常"的保护（例如基线分数高于阈值一定幅度时禁止 normal 结论），而不是直接扩大复核范围。

### 复核循环的行为矩阵（本机可验证部分）

下列行为由回归测试固定，不依赖 GPU；它们是"循环不再自我阻塞"的证据，**不**代表模型在真实图像上的表现。

| 场景 | 期望结果 | 测试 |
| --- | --- | --- |
| 4 个区域全 normal，随后重复请求已看视图 | 纠正一次后 `normal_candidate` | `test_four_region_case_that_used_to_fail_now_concludes` |
| 全部区域 normal 仍以 `unknown` 结束 | 追问一次后 `normal_candidate` | `test_silent_abstention_gets_one_request_for_a_conclusion` |
| 连续两次重复视图 | 停止并转人工 | `test_repeated_view_gets_one_corrective_call_then_stops` |
| 纠正池共享（重复 + 解析失败） | 3 次调用内停止 | `test_corrections_share_one_bounded_pool` |
| JSON 被代码块或散文包裹 | 提取后重新校验通过 | `test_fenced_or_wrapped_json_is_repaired_once` |
| 响应中没有任何 JSON 对象 | `schema_error` 失败 | `test_text_without_an_object_still_fails` |
| 区域提取被截断 | 策略开关决定是否拦截 | `test_truncation_vetoes_normal_only_when_the_policy_says_so` |
| 可见异常 | `anomaly_candidate` | `test_visible_anomaly_can_finish_as_candidate` |
| 局部 normal 试图覆盖已记录的异常 | 判为冲突并转人工 | `test_local_normal_cannot_erase_previously_visible_anomaly` |
| 证据图只有 40×30 | 放大到最短边 256 | `test_small_evidence_crops_are_upscaled_to_the_minimum_side` |
