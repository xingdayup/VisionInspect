# Phase 1 实际实验记录

日期：2026-10-21，Asia/Shanghai。以下来自本机实际输出，不是目标指标。

本文保留 Phase 1 实际范围，末尾的前端 / Agent“尚未实现”指该阶段。后续已完成的 A/C/D 对照与浏览器集成见 [COMPARISON](COMPARISON.md)、[Phase 4](PHASE4_REPORT.md)；当前总状态见 [文档索引](README.md)。本次整理没有重跑正式测试集或修改冻结模型。

## 数据与冻结协议

MVTec AD bottle，用户授权的第三方镜像 `foersben/mvtec-ad`，revision `c75b39616f84db43677bcc8228caaafaf5096d7f`。357 个文件全部核验大小及仓库 LFS SHA256 / Git blob SHA1；正常训练 209 张分为 168 train + 41 validation，测试 20 normal + 63 anomaly。来源与许可见 DATASETS.md。

模型：Anomalib 2.7.0 PatchcoreModel，ResNet18 a1_in1k，layer2/3，输入 256×256，无裁剪，ImageNet 标准化，coreset=0.01，9 邻居，batch=1，CPU/FP32，4 线程，seed=42。特征提取和 k-center 直接调用上游算法，没有 Trainer，也没有默认验证集、后处理器或自动下载器。

正常验证分数 95% 分位数（higher）得到阈值 **8.742960929870605**。只将 score > threshold 作为 anomaly_candidate。测试集只执行一次正式冻结评估；指标未用于修改阈值或配置。

## 运行位置与命令

完整命令和退出码见本地 `logs/phase1-pipeline-20261021-232019.log`；流水线索引见 `logs/phase1-runs.json`。每个运行目录另有 command.json、config.json、environment.json、execution.log 和 status.json。

| 步骤 | outputs 下运行编号 | 命令总耗时 | 结果 |
|---|---|---:|---|
| check-data | 20261021-232020-check-data-8e9d06 | 13.54 s | 168 / 41 / 83，校验通过 |
| smoke | 20261021-232034-smoke-3780f7 | 22.30 s | 16 train / 4 validation，128×128，重载一致 |
| fit | 20261021-232056-fit-ae72d8 | 84.19 s | 特征库 1720×384，算法阶段 74.33 s |
| calibrate | 20261021-232221-calibrate-cd7a38 | 24.95 s | 41 张正常验证图，校准阶段 15.20 s |
| evaluate | 20261021-232247-evaluate-5225f8 | 32.17 s | 83 张测试图，评估阶段 21.98 s |
| predict | 20261021-232319-predict-fc635c | 22.63 s | 独立进程、禁用网络、单图推理成功 |
| smoke 资源监控回归 | 20261021-232408-smoke-ffe99b | 23.68 s | 修正 Windows 子进程监控后通过 |

总耗时包含 Python 启动及依赖导入，算法阶段计时与其不同。实际执行顺序：

```text
python -B -m visioninspect.baseline check-data
python -B -m visioninspect.baseline smoke
python -B -m visioninspect.baseline fit
python -B -m visioninspect.baseline calibrate --run <上述 fit 目录>
python -B -m visioninspect.baseline evaluate --run <上述 fit 目录>
python -B -m visioninspect.baseline predict --run <上述 fit 目录> --image <bottle/test/broken_large/000.png>
```

上述 python 实际为项目 `.venv/Scripts/python.exe`，完整绝对路径记录在运行 command.json 和转录日志中。

## 正式测试指标

原始文件：[metrics.json](../outputs/20261021-232247-evaluate-5225f8/metrics.json)。评估分辨率 256×256，包含正常测试图，mask 最近邻缩放；不是原图分辨率评估。

| 指标 | 实测 |
|---|---:|
| Image AUROC | 1.000000 |
| Image Average Precision | 1.000000（浮点原值 0.9999999999999998） |
| 固定阈值 F1 | 0.976744 |
| 异常召回率 | 1.000000 |
| Pixel AUROC | 0.978275 |
| AUPRO@FPR≤0.3 | 0.923896 |

混淆矩阵按 `[normal, anomaly]` 排列，行是真值、列是预测：`[[17,3],[0,63]]`。20 张正常图中 3 张误报，63 张异常图均检出。图像排序指标完美并不代表阈值下无误报，也不能推广到其他类别或真实产线。

单图推理延迟：验证样本预热 5 次后，83 张测试图 p50 **42.43 ms**、p95 **49.32 ms**。计时包含预处理和模型，不含文件读取、热图导出或服务往返。

## checkpoint 与资源

- checkpoint：13,832,719 字节；SHA256 `4a4d94f150f5b2021aa593c8b277bc942178f261eacc336f14c3e33cef76587a`。
- split manifest digest：`6c75ed400f11c737ecef43dc5b6c55a79b07a5e782366efb528096e23a05c49e`。
- 校准文件 SHA256：`a1bf923b8612e9fd5e23ba66c78bd59ed4791195aa780d97d0d22362caea5cf4`。
- fit 计算子进程 + 启动器的 OS 峰值工作集之和：1,549,111,296 字节，约 **1.443 GiB**。原启动器读数无效，使用 external-resource-monitor.json；这是进程树保守上界，非系统整体内存。
- calibrate 进程树上界：610,287,616 字节；evaluate：1,338,671,104 字节；predict：597,299,200 字节。
- GPU 显存：不适用，所有模型计算均在 CPU。
- 骨干权重 46,807,446 字节，数据文件 156,570,806 字节；安装器未提供精确网络流量，依赖发行文件预算 326,726,471 字节。

## 独立单图推理

输入 bottle/test/broken_large/000.png；分数 **19.891008377075195**，阈值 **8.742960929870605**，状态 `anomaly_candidate`。输出原始浮点热图、热图 PNG、叠加图及 prediction.json。该状态仅为研究原型候选状态。

![真实模型叠加图](../outputs/20261021-232319-predict-fc635c/evidence/input_overlay.png)

## 验证与限制

- 单元/回归测试：23 passed；包括坏数据、掩膜、独立划分、测试泄漏防护、离线连接阻止、指标完美检测器参照、API 骨架、schema、Windows 子进程资源监控，以及下载资产复用/不覆盖已有文件。
- 真正的 checkpoint 重载一致性由 bottle smoke 验证（rtol=1e-5、atol=1e-6），独立 predict 进程也禁止网络后成功。
- 前端 TypeScript + Vite 构建通过；尚未进行浏览器完整检测闭环验收。
- 临时本机 HTTP 服务验证 health、实际运行列表和正式指标均返回 200，未实现的任务提交返回 503；验证后服务已关闭，无常驻后台进程。
- Linux 锁文件解析通过；Docker daemon 不可用，镜像未构建/运行。不能宣称部署通过。
- 仅一个类别、一套固定配置；正常验证集少，不保证误报率。未进行多类别、EfficientAD、VLM 或 Agent 对照。
- Anomalib 的 AUPRO 底层类为固定版本内部接口；升级依赖必须重新验证。现有依赖的弃用警告保留于日志，未修改第三方源码。
