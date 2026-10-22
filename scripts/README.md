# GPU 工具入口

所有部署命令在 Linux GPU 服务器执行，应用 Python 为 `.venv/bin/python`。不会向系统/base 安装依赖，不自动下载数据或模型。

| 文件 | 用途 |
|---|---|
| `audit_gpu_environment.py` | 只读版本、GPU、驱动、CUDA、磁盘与小算子审计 |
| `preflight_pai_app.py` | 固定依赖的兼容性与下载预算，TLS 校验、有界重试、进度 |
| `setup_gpu.sh` | 项目独立 CUDA 环境、哈希锁、CUDA 运算验证 |
| `start_gpu.sh` | 校验项目环境和 CUDA 后启动单 worker Web 服务 |
| `verify_gpu.py` | 真实正常验证图、CUDA 放置/重载/一致性，可选多图 VLM 和 TTL |
| `pai_vllm.py` | 已有 GPU 运行时检查、权重预检、显式下载与有限调试 |
| `vllm_managed.py` | vLLM 生命周期监管，由应用调用 |
| `check_project_environment.py` | 拒绝 base、系统环境、链接环境及系统包共享 |
| `check_git_content.py` | 防止资产、环境、密钥及大文件进入 Git |
| `package_migration.py` / `install_assets.py` | 离线 Git 与独立校验资产迁移 |
| `prepare_assets.py` | 显式核验并准备已授权 bottle / ResNet 资产 |
| `prepare_tools.py` | 新实验的正常训练参考与正常验证策略；拒绝覆盖已冻结资产 |

实验入口：`python -m visioninspect.baseline <check-data|smoke|fit|calibrate|evaluate|predict>`；监督推理入口：`python -m visioninspect.inspection --image <path> --mode <baseline|agent_review>`。两种模型推理均要求 CUDA。
