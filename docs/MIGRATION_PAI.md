# 迁移到 PAI DSW

本版使用 Git 管理代码，以独立资产包迁移 bottle、冻结 checkpoint、校准与参考索引、ResNet 权重和前端生产文件。虚拟环境、node_modules、用户检测历史、密钥和大型 VLM 不进入 Git，也不复制 Windows 环境到 Linux。

## 1. 在 Windows 生成迁移包

先完成源码提交，再运行：

```powershell
.\.venv\Scripts\python.exe -B scripts/check_git_content.py
.\.venv\Scripts\python.exe -B scripts/package_migration.py
```

输出在 `.cache/transfer/日期-提交号/`，包含：

- `visioninspect.bundle`：代码、提交历史与 tag，可离线 clone，无需 SSH。
- `visioninspect-assets.tar.gz`：必要本地资产，输入总量上限 500 MiB。
- `transfer-manifest.json`：归档及逐文件大小和 SHA256。
- `START-HERE.txt`：恢复入口。
- `visioninspect-pai-migration.zip`：上述文件的单文件传输包，旁边附 ZIP 的 SHA256。

仅私下迁移到自己的研究实例。bottle 为 CC-BY-NC-SA-4.0，归档内保留数据许可；不将数据包或权重上传到 Git 仓库。

## 2. 上传并恢复

通过 DSW 文件管理界面把三个迁移文件放到 `/mnt/workspace/visioninspect-transfer/`，然后在 DSW 终端运行：

也可只上传 `visioninspect-pai-migration.zip` 到 `/mnt/workspace/`，用平台 Python 解压，不需安装工具：

```bash
/usr/local/bin/python3 -m zipfile -e /mnt/workspace/visioninspect-pai-migration.zip /mnt/workspace/visioninspect-transfer
```

随后执行：

```bash
cd /mnt/workspace/visioninspect-transfer
git clone visioninspect.bundle /mnt/workspace/VisionInspect
cd /mnt/workspace/VisionInspect
git bundle verify /mnt/workspace/visioninspect-transfer/visioninspect.bundle
/usr/local/bin/python3 -B scripts/install_assets.py --package /mnt/workspace/visioninspect-transfer
bash scripts/setup_pai.sh
```

恢复器拒绝路径逃逸、非普通文件、校验失败及覆盖不同内容的现有文件。相同文件可复用。应用依赖仅安装到新目录的 `.venv`；现有平台 Python、CUDA、驱动与 vLLM 不变。预检先核对 Linux/Python wheel、1.5 GB 下载上限和 15 GiB 可用空间，失败即停止安装。

没有 Git 远程地址也能 clone bundle。以后使用远程仓库时由你指定自己的地址；不要把数据、权重或 `.env` 加入提交。

## 3. 先验证 CPU 迁移

```bash
export VISIONINSPECT_RELOCATED_ASSETS=1
VISIONINSPECT_GPU_ENABLED=0 .venv/bin/python -B scripts/verify_upgrade_cpu.py
VISIONINSPECT_GPU_ENABLED=0 bash scripts/start_pai.sh
```

通过 DSW 的受控 Web 服务入口访问应用的 8000 端口。启动脚本在前台运行，按 Ctrl+C 停止。不要额外开放公网模型端口。若 8000 已占用，可用 `PORT=8017 bash scripts/start_pai.sh`。

`VISIONINSPECT_RELOCATED_ASSETS=1` 明确允许数据物理路径改变；仍校验原 manifest、图片和 mask 哈希，保留原 split、digest、模型和阈值。无需重新训练或重新校准。`verify_upgrade_cpu.py` 只检查正常验证图，验证清晰路径和 GPU 不可用时的保留证据、转人工、重试，不生成准确率指标。

## 4. 显式准备 GPU 模型

当前 DSW 审计为 A10 24 GB，平台 Python 3.12.13 / torch 2.13.0+cu130 / vLLM 0.27.1。FP16/BF16 小矩阵通过只说明基础 CUDA 可用，尚不证明 vLLM 原生内核和模型已运行。

应用继续使用独立 CPU 环境；以下准备命令只读复用平台已有 GPU 包：

```bash
/usr/local/bin/python3 -B scripts/pai_vllm.py check
/usr/local/bin/python3 -B scripts/pai_vllm.py preflight
```

如果旧迁移包的 `check` 报 `No module named 'vllm._C'`，先更新检查脚本，不要据此重装平台依赖。固定 vLLM 0.27.1 的原生扩展检查应为 `vllm._C_stable_libtorch`，与[该版本官方 CUDA 入口](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/platforms/cuda.py)一致。该扩展不存在或加载失败时仍须停止；不能跳过检查。更新脚本通过导入只证明这一检查项，实际模型加载和生成仍需后续验证。

检查 `logs/runtime-check.json` 与 `logs/model-preflight.json`，这里的 logs 在 `/mnt/systemDisk/VisionInspect-GPU/` 内。预检固定：

- Qwen/Qwen3-VL-8B-Instruct，Apache-2.0。
- revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。
- 四份 BF16 权重共 17,534,339,512 字节，另有少量 tokenizer/配置；显式下载前至少留出 50 GiB。
- 首轮配置：BF16、上下文 4096、单并发、最多三张图片、图像面积上限 512×512、显存利用率目标 0.88、eager 模式。
- 24 GB 显存是否足够及实际网络速度仍以真实加载为准；OOM 时不自动量化、换模型或修改系统环境。

确认预检无冲突后，显式运行这一条下载命令：

```bash
/usr/local/bin/python3 -B scripts/pai_vllm.py download
```

仅下载固定文件，逐一校验，保存在 `/mnt/systemDisk/VisionInspect-GPU/weights/qwen3-vl-8b-instruct/`。不下载其他模型或数据集。

## 5. GPU 工程验证和启动

```bash
export VISIONINSPECT_RELOCATED_ASSETS=1
VISIONINSPECT_GPU_ENABLED=1 .venv/bin/python -B scripts/verify_pai_review.py \
  --image data/mvtec_ad/bottle/train/good/114.png --ttl-smoke
VISIONINSPECT_GPU_ENABLED=1 bash scripts/start_pai.sh
```

验证脚本使用真实图片检查两图、三图 JSON 接口，比较直接 PatchCore 与工作台的分数和热图，并临时缩短 GPU TTL 到 2 秒验证退出。114.png 是已记录分数 8.778727531433105 的正常验证图，处于原冻结不确定区间；没有使用测试标签选择提示词。中心裁剪只用于输入协议测试，不是已识别缺陷。脚本保存 `full_agent_loop_exercised`，不得把跳过复核或一次结束当作多步复核已实测；若仍无法核查充分则保留人工状态，不修改区间强行通过。

启动后选择“Agent 多步复核”。vLLM 只监听 127.0.0.1:8001，由应用按需启动；空闲 600 秒退出。CPU worker 独立 TTL 600 秒。GPU 冷启动最多 300 秒，准备后单图执行最多 180 秒、模型请求最多 45 秒、最多 3 次模型尝试。服务或项目进程树 RSS 超过 24/28 GiB 时终止本项目模型服务并记录错误。

GPU 服务设置 HF 离线及继承式 Python socket 外联限制，保留本地 IPC。它不是系统防火墙；若原生内核缺失或请求外部下载，保留日志并诊断，不关闭限制、不修改系统网络。

## 6. 使用顺序与回传

1. 单图：选择 bottle / PatchCore 和工作流，提交后看视觉证据、结论来源和执行轨迹。
2. 批量：填写名称，选择最多 100 张，上传生成草稿，确认成功与失败项后点击“开始批次”。
3. 复核：进入“仅显示待人工复核”，逐项填写状态和说明；原机器结论保持不变。
4. 导出：单图 JSON、批次 JSON 和 CSV。批次汇总按每张图最新尝试计算，旧尝试仍保留。
5. 故障：失败项创建重试后处于等待状态，需要继续批次。暂停/取消在当前图片结束后生效。

请回传命令终端输出，以及 GPU 根目录的 `runtime-check.json`、应用输出目录的 `verification.json` 或 `failure.json` 和 `vllm-server.log` 中的错误段。不要回传密钥。当前没有 SSH，GPU 命令必须由你在 DSW 执行；本机测试不能代替该验收。
