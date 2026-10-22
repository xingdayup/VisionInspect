# Linux GPU 服务器迁移

代码由 Git 管理，数据、checkpoint、profile、参考索引、ResNet 权重与前端构建由独立资产包迁移。虚拟环境、用户检测历史、密钥和大型 VLM 不进入 Git。仅私下迁移自己的非商业研究实例。

## 导出与恢复

在有既有资产的源项目，先提交源码，再用当前项目解释器执行：

```bash
python -B scripts/package_migration.py
```

结果为 `.cache/transfer/日期-提交号/visioninspect-pai-migration.zip` 及 SHA256，内含 Git bundle、资产 tar、逐文件校验清单与恢复入口。源电脑打包可使用现有解释器，部署运行仅支持 Linux GPU。

上传 ZIP 至服务器 `/mnt/workspace/`，确认目标目录没有需要保留的另一项目后执行：

```bash
/usr/local/bin/python3 -m zipfile -e /mnt/workspace/visioninspect-pai-migration.zip /mnt/workspace/visioninspect-transfer
git clone /mnt/workspace/visioninspect-transfer/visioninspect.bundle /mnt/workspace/VisionInspect
cd /mnt/workspace/VisionInspect
git bundle verify /mnt/workspace/visioninspect-transfer/visioninspect.bundle
/usr/local/bin/python3 -B scripts/install_assets.py --package /mnt/workspace/visioninspect-transfer
```

恢复器拒绝路径逃逸、链接、大小/哈希不符及覆盖不同文件；相同内容可复用。不要删除已有项目或虚拟环境来掩盖冲突。旧服务器有手工修改时先保留 Git 差异，在新目录恢复 GPU 版。

## 审计、安装与启动

按根 README 的步骤运行 GPU 审计、依赖预检、`setup_gpu.sh`、`verify_gpu.py` 和 `start_gpu.sh`。GPU 依赖体积超过旧 1.5 GB 预算时会停止，需要确认新预算。仅创建项目 `.venv`，不替换平台 torch、CUDA 或驱动。

固定 vLLM 0.27.1 检查模块为 `vllm._C_stable_libtorch`。已有平台原生导入与模型架构检查通过，只证明这一阶段；不代表 Qwen 权重和生成已验证。可配置 `VISIONINSPECT_VLLM_PYTHON` 指向已审计的 GPU Python。

Qwen3-VL-8B-Instruct 固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，Apache-2.0，四份 BF16 权重 17,534,339,512 字节；下载前至少 50 GiB 空间。仅显式 `pai_vllm.py download` 下载，源记录与权重逐一校验。

`VISIONINSPECT_RELOCATED_ASSETS=1` 允许物理路径改变，仍核对原清单、图像、mask、checkpoint 与 profile。历史 CPU checkpoint 加载到 CUDA 时不改文件。原 CPU 指标保留，GPU 数值与阈值适用性待新设备验证；新的拟合/校准/正式评估创建新目录。

Web 服务使用受控入口访问 8000，vLLM 仅监听 127.0.0.1:8001。不额外开放公网模型端口。单服务进程，Ctrl+C 停止。实际命令日志在 logs，GPU 验收结果在新 outputs 目录，模型服务日志在配置的 GPU 根目录。
