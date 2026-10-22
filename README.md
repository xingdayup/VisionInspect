# VisionInspect-Agent · GPU

中文视觉质检工作台：CUDA PatchCore 定位异常，LangGraph 驱动 Qwen3-VL 复核不确定样本，保留图像证据、决策轨迹和人工记录。

本版仅支持 **Linux NVIDIA GPU**。当前注册类别为 MVTec AD `bottle`，用途为非商业研究。候选结论不构成工业放行授权。

## 功能与架构

- 单图上传、最多 100 张批次、暂停/继续/取消、失败重试。
- 原图、浮点异常图、热图叠加、局部裁剪、正常训练参考和实际 VLM 输入。
- 原始模型、Agent、人工三层记录；历史查询、JSON/CSV 导出。
- PatchCore-ResNet18：CUDA、FP32、256×256、layer2/layer3；没有 CPU 推理回退。
- LangGraph：不确定区间触发复核，选择局部对比、切换参考或结束，最多 3 次模型调用（含重试）。
- vLLM：本地 Qwen3-VL-8B-Instruct BF16，模型端口仅监听 127.0.0.1:8001。
- FastAPI / SQLite / React / TypeScript / Vite；单 API worker，串行任务，两个模型进程独立空闲 TTL 600 秒。

```text
网页上传 → 质量检查 → CUDA PatchCore 分数与热图
                         ↓ 不确定区间
              LangGraph ↔ vLLM / Qwen3-VL
                         ↓ 程序校验证据
              候选结果 / 人工复核 → 报告导出
```

只有 `baseline`（GPU 单模型）和 `agent_review`（GPU 多步复核）两种工作流。正常复核必须完成所有提取区域的对应参考比较；失败、矛盾、模糊和预算不足转人工。模型分数与冻结阈值不被 VLM 改写。

## 服务器准备与运行

目标是 Linux x86_64 / Python 3.12 / CUDA 13 PyTorch 2.13.0 + torchvision 0.28.0；vLLM 运行时固定 0.27.1。已有 A10 24 GiB 作为候选硬件，完整两模型显存占用待实测。

**全新服务器**（两个环境都不存在、vLLM 未安装）请直接使用单一入口，它按正确顺序执行审计、建环境、下载权重和验收：

```bash
git clone https://github.com/xingdayup/VisionInspect.git /root/autodl-tmp/VisionInspect
cd /root/autodl-tmp/VisionInspect
bash scripts/deploy_fresh_gpu.sh          # 自动加载 scripts/env_autodl.sh
PORT=6006 bash scripts/start_gpu.sh
```

资产（数据、weight、checkpoint、前端构建）不在 Git 内，需先上传；脚本第 1 阶段会逐项检查并在缺失时停止。以下分步说明适用于已有 vLLM 的旧服务器，或需要手工控制每一步的场景。

1. 将代码放入服务器项目目录，按 [迁移说明](docs/MIGRATION_PAI.md) 恢复校验资产。
2. 运行只读审计及依赖预检：

```bash
/usr/local/bin/python3 -B scripts/audit_gpu_environment.py --output logs/gpu-audit.json
/usr/local/bin/python3 -B scripts/preflight_pai_app.py
```

预检默认保留原有 **1.5 GB 下载上限、15 GiB 空闲空间**。本次 Linux 目标元数据估算依赖为 **2.98 GB**，已超过旧上限；脚本会停止。需明确确认新预算后再使用 `--download-budget-bytes`，不会自动扩大或安装。

3. 预检通过后，在独立项目环境安装并验证 CUDA：

```bash
bash scripts/setup_gpu.sh
export VISIONINSPECT_RELOCATED_ASSETS=1
.venv/bin/python -B scripts/verify_gpu.py --ttl-smoke
```

应用环境使用 `.venv`。vLLM 只读复用已验证的 GPU Python，`configs/gpu.yaml` 中默认 `/usr/local/bin/python3`（PAI 路径）；AutoDL 实例由 `scripts/env_autodl.sh` 覆盖为 GPU 根目录下的独立 `.venv`。两者都不修改平台包。

新服务器没有 vLLM 时，可先核验 `requirements-vllm.txt`：其 Linux 目标包约 3.87 GB，应用与 vLLM 冷下载合计约 6.85 GB，按文件去重约 4.03 GB。预算获批准后建立第二个独立环境：

```bash
# 指向服务器已审计的现有 Python；仅用它创建隔离环境
export BOOTSTRAP_PYTHON=/root/miniconda3/bin/python
export VISIONINSPECT_GPU_ROOT=/root/autodl-tmp/VisionInspect-GPU
bash scripts/setup_gpu.sh --download-budget-bytes 3200000000
bash scripts/setup_vllm.sh --download-budget-bytes 4100000000
export VISIONINSPECT_VLLM_PYTHON="$VISIONINSPECT_GPU_ROOT/.venv/bin/python"
```

上述路径适用于本次 AutoDL 实例，其他服务器先审计再指定。已有校验下载目录时，两个安装脚本均支持 `VISIONINSPECT_WHEELHOUSE`，通过哈希锁从本地文件安装。环境不共享系统包，不修改 Conda base。

4. 通过显式预检准备 Qwen，并验证完整 GPU 工程协议。**`check` 必须使用已安装 vLLM 的解释器**：`download` 会先读它写出的 `logs/runtime-check.json`，缺少该文件时直接报错并提示先运行 `setup_vllm.sh`。

```bash
export VISIONINSPECT_VLLM_PYTHON="$VISIONINSPECT_GPU_ROOT/.venv/bin/python"
"$VISIONINSPECT_VLLM_PYTHON" -B scripts/pai_vllm.py check --root "$VISIONINSPECT_GPU_ROOT"
# 核对许可、约 17.54 GB 权重及空闲空间后再执行：
"$VISIONINSPECT_VLLM_PYTHON" -B scripts/pai_vllm.py preflight --root "$VISIONINSPECT_GPU_ROOT" --minimum-free-gib 25
"$VISIONINSPECT_VLLM_PYTHON" -B scripts/pai_vllm.py download --root "$VISIONINSPECT_GPU_ROOT" --minimum-free-gib 25
.venv/bin/python -B scripts/verify_gpu.py --with-vlm --ttl-smoke
PORT=6006 bash scripts/start_gpu.sh
```

GPU 资产根目录**没有内置默认值**：优先 `--root`，其次 `VISIONINSPECT_GPU_ROOT`，两者都没有时立即报错，不会在系统盘上建出旧 PAI 路径。Web 端口由 `PORT` 决定（默认 8000），AutoDL 实例只映射 6006/6008。

模型预检默认要求 50 GiB 空闲。本次用户另行批准总下载上限 26 GB、沿用现有盘、下载前至少 25 GiB、完成后至少留 8 GiB；可显式传入 `--minimum-free-gib 25`，脚本拒绝低于全部资产加 8 GiB 余量的预算。网络不通时不改系统 DNS、代理或证书，仅使用核验后的官方来源。

## 实验与数据

`python` 均指项目 `.venv/bin/python`：

```bash
python -B -m visioninspect.baseline check-data --config configs/patchcore.yaml
python -B -m visioninspect.baseline smoke --config configs/patchcore.yaml
python -B -m visioninspect.baseline fit --config configs/patchcore.yaml
python -B -m visioninspect.baseline calibrate --run <新的GPU拟合目录>
python -B -m visioninspect.baseline evaluate --run <新的GPU拟合目录>
```

正式 GPU 拟合、校准和测试创建新运行目录。当前历史 checkpoint 可以只读加载到 CUDA，但其原始校准来自 CPU；旧准确率与延迟不代表 GPU 结果。`verify_gpu.py` 仅进行真实正常验证图、CUDA 放置、checkpoint 重载、一致性、可选多图接口及 TTL 的工程验收，不生成准确率收益。

新实验完成后，在 `configs/models.yaml` 指向新的 `fit_run`、`profile` 与 `references` 路径，再执行 `python -B scripts/prepare_tools.py bottle` 生成新策略和参考索引；已有策略/索引拒绝覆盖。训练和验收前先停止本项目 Web 服务，避免同卡模型争用。

数据和模型许可见 [DATASETS](docs/DATASETS.md) / [THIRD_PARTY](docs/THIRD_PARTY.md)。不下载全集或未授权数据，不把数据、权重、环境、密钥及用户结果提交 Git。

## 验证与当前边界

提交前必须全部通过；CI（`.github/workflows/ci.yml`）执行同一组命令。

```bash
ruff format --check visioninspect tests scripts     # ruff 0.14.4
ruff check visioninspect tests scripts
.venv/bin/python -B -m pytest -q -p no:cacheprovider
npm --prefix frontend ci --ignore-scripts --no-audit --no-fund
npm --prefix frontend run format:check              # prettier 3.6.2
npm --prefix frontend run build
.venv/bin/python -B scripts/check_git_content.py
```

Python 侧不引入与运行时无关的依赖：`ruff` 通过 `uvx ruff@0.14.4` 调用，不装进项目 `.venv`。代码风格约定是 ruff 默认行宽 88、isort 分组、pyupgrade 现代写法；lint 规则集为 `E4/E7/E9/F/I/UP`，脚本目录豁免 `E402`。

无 CUDA 的开发机器只验证 API、队列、规则和进程协议，真实 GPU 测试会明确跳过。PAI 已由用户核实 A10、CUDA 13、vLLM 原生扩展和架构注册；应用 CUDA 安装、PatchCore GPU 数值、Qwen 权重加载、多步复核和显存峰值仍须服务器验证。最新记录见 [UPGRADE_STATUS](docs/UPGRADE_STATUS.md)。

本次按用户要求移除了 CPU SmolVLM、旧 fixed/dynamic 流程、EfficientAD 分支、Windows 运行脚本、CPU 容器配置和重复依赖锁。历史代码及实验说明由 Git 标签 `pre-gpu-only-20261022` 保留；本地数据、旧权重与结果没有物理删除。

目录：`visioninspect/` 核心运行代码，`frontend/` 工作台，`configs/` 产品配置，`scripts/` GPU 部署与验证，`tests/` 工程回归，`docs/` 业务、架构、迁移与许可。
