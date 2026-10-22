# 同机部署（应用 + 两个模型同一台 GPU 服务器）

本文回答三件事：**这个拓扑在哪条边界上成立、部署完成怎么判定、出问题该查哪里。**
逐条安装命令见 [迁移说明](MIGRATION_PAI.md) 与根 [README](../README.md) 的服务器准备一节的对应步骤。

## 0. 全新服务器：单一入口

两个环境都不存在时，用脚本而不是手工按顺序敲命令，顺序错了会在第 4 步才暴露：

```bash
bash scripts/deploy_fresh_gpu.sh
```

它依次执行审计、检查已上传资产、建立应用 `.venv`、建立 vLLM 环境并跑原生检查、探测 Hugging Face 可达性、预检与下载权重、两项验收，最后打印启动命令。可覆盖的变量：`BOOTSTRAP_PYTHON`、`APP_DOWNLOAD_BUDGET_BYTES`、`VLLM_DOWNLOAD_BUDGET_BYTES`、`MINIMUM_FREE_GIB`、`SKIP_MODEL_DOWNLOAD=1`、`SKIP_VERIFY=1`。

`pai_vllm.py` 不再有内置默认根目录：它读 `--root`，其次读 `VISIONINSPECT_GPU_ROOT`，两者都没有时立刻报错，不会在系统盘上建出空的旧 PAI 路径。

## 1. 拓扑边界

当前代码把"本机"写死在四处，因此**应用、PatchCore、vLLM 必须同机**才能开箱运行：

- `visioninspect/vllm_client.py`：endpoint 必须是 `http://127.0.0.1:8001`，否则构造即失败
- `visioninspect/vllm_runtime.py`：校验同一 endpoint，并 `bind('127.0.0.1', 8001)` 探测端口占用
- `visioninspect/worker.py`：`offline()` 只放行 127.0.0.1 的 8001，其余连接一律 RuntimeError
- `visioninspect/api.py`：本机 `torch.cuda.is_available()` 为假时，上传直接 503

同机部署不需要改这些；拆分部署必须改这四处，另加 PatchCore 的远程化。**在完成一次同机验收前不要动这条路。**

## 2. 前置条件

| 条件 | 要求 |
| --- | --- |
| 应用隔离环境 `.venv` | 由 `setup_gpu.sh` 在本仓库内创建，不得复用平台或系统环境 |
| vLLM 隔离环境 | 由 `setup_vllm.sh` 在 GPU 资产根目录创建，与 `.venv` 互不共享包 |
| Qwen 四分片权重 | 逐文件校验 SHA256 与 `source.json`，缺一片都不能加载 |
| 基础资产 | 数据、骨干权重、checkpoint、profile、参考索引、前端构建 |
| 磁盘余量 | 下载前满足门槛，落盘后保留余量 |
| 平台 PyTorch | **不可复用**：应用必须跑自己 `.venv` 里的目标版本 |
| 显存 | 两模型同卡共处的峰值必须实测，不能只按分配比例推断 |

平台解释器只用于审计和创建虚拟环境；运行一律用 `.venv/bin/python`。

## 3. 路径与环境变量

`configs/gpu.yaml` 里的 `root` 是示例路径，实例不同必须覆盖，且**每个新终端重新设置**：

```bash
source scripts/env_autodl.sh
```

不使用该文件时，至少显式导出 `BOOTSTRAP_PYTHON`、`VISIONINSPECT_GPU_ROOT`、
`VISIONINSPECT_VLLM_PYTHON`、`VISIONINSPECT_RELOCATED_ASSETS=1`。
`VISIONINSPECT_PYPI_INDEX` 必须使用 HTTPS；只有显式设置
`VISIONINSPECT_OFFLINE_INSTALL=1` 才会禁用在线补齐。

## 4. 同卡两模型的显存共处

vLLM 启动参数固定 `--gpu-memory-utilization 0.85`（`scripts/pai_vllm.py`），PatchCore 再占一份。
0.85 只是分配比例，**不保证两张模型装得下**。

1. 先跑验收，读 `verification.json` 与 `nvidia-smi` 实测峰值；
2. 溢出时下调 `--gpu-memory-utilization`（0.75 → 0.7），不降精度、不换模型、不回退 CPU；
3. Web 运行期间不做拟合、校准或正式评估。

## 5. 部署完成判据

- [ ] `audit_gpu_environment.py` 退出 0
- [ ] 两个 `setup_*.sh` 都通过结尾的原生检查
- [ ] 权重四分片大小与 SHA256 全对，`source.json` 有效
- [ ] `verify_gpu.py --ttl-smoke` 退出 0
- [ ] `verify_gpu.py --with-vlm --ttl-smoke` 退出 0，峰值写入 `outputs/*-gpu-verification-*`
- [ ] 页面显示"GPU 检测与复核"，上传一张真图跑通全链路
- [ ] 空闲 600 秒后 PatchCore 与 vLLM 子进程确实退出
- [ ] 检查 `full_agent_loop_exercised`：为 `false` 时**不得**声称多步复核分支已覆盖

## 6. 故障对照

| 现象 | 实际原因 |
| --- | --- |
| 上传 503「CUDA 不可用」 | 启动用了平台解释器，或装成 CPU 版 torch |
| 选复核 503「GPU 服务未配置或模型资产缺失」 | `configs/gpu.yaml` 的 `root` 与 `VISIONINSPECT_GPU_ROOT` 不一致；或 `source.json` 与实际权重不符 |
| vLLM 一直 loading 后失败 | 8001 被别的进程占用（脚本不抢占）；或权重哈希校验失败 |
| `start_gpu.sh` 报 Frontend build missing | `frontend/dist` 未构建，也未随资产恢复 |
| 双模型同时跑 OOM | 第 4 节，下调显存占用比例 |
| 预检停在预算 | 应用锁与 vLLM 锁各自独立预算，不能相加当作额外批准 |

## 7. 不要做

- 不向 base/系统环境装依赖，不改驱动、CUDA、系统 Python 与网络设置
- 不把 `.part` 前缀改名冒充完整权重
- 不把云服务器租金记为"零成本"
- 不把"基础算子通过"写成"模型已验收"
