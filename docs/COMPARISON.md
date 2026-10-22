# bottle 冻结 A / C / D 对照

版本说明：本文冻结实验使用旧单合图 VLM 输入。后续核验发现该合图被处理器压缩为正方形，已修复为两图独立输入 separate-images-v2；原始实验与源文件保持不变，本文延迟和调用量不代表新版本，也没有据此声称描述准确。详见 [输入审计与修复](VLM_INPUT.md)。

数据为完整 83 张 bottle 测试图（20 正常、63 异常），训练 168、正常验证 41，seed 42，256×256，CPU FP32。模型、图像阈值、像素阈值和 Agent 复核规则均来源于训练/正常验证，测试结果未用于调参。C 与 D 使用同一策略 digest `33c754a76cd75e61b9ba112658f898c0d1d10c3566ca5efc565f4dbe6cf473b1`，仅路由方式不同；frozen.json 记录模型、数据、策略与实现源文件 SHA。

| 指标 | A PatchCore | C 固定复核 + 本地 VLM | D 条件 Agent + 本地 VLM |
|---|---:|---:|---:|
| 测试图 | 83 | 83 | 83 |
| 自动候选 / 人工复核暂缓 | 83 / 0 | 78 / 5 | 78 / 5 |
| 自动候选误报 / 漏报 | 3 / 0 | 1 / 0 | 1 / 0 |
| 接受覆盖率 | 100% | 93.98% | 93.98% |
| VLM 调用次数 | 0 | 83 | 5 |
| 平均工具调用 | 不适用 | 6.0000 | 3.9518 |
| 工具成功率 | 不适用 | 100% | 100% |
| 无效调用率 | 不适用 | 0% | 0% |
| 任务完成率 | 100% | 100% | 100% |
| 热进程端到端 p50 | 模型计时口径不同 | 4.0778 s | 0.2159 s |
| 热进程端到端 p95 | 模型计时口径不同 | 5.2366 s | 3.3720 s |
| 整组热执行耗时 | 不作跨口径比较 | 332.206 s | 38.576 s |
| 包含进程/模型初始化的命令耗时 | 见 Phase 1 | 361.508 s | 68.461 s |
| 监管进程树峰值 | 见 Phase 1 | 2,849,452,032 B | 2,865,430,528 B |
| 外部 API 费用 | 0 | 0 | 0 |

5 张暂缓样本均为正常标签。它们没有被记作正确结果，accepted FP/FN 排除了这些样本。A 的 3 个误报中，有 2 个进入复核；另有 3 个原本正常候选也进入复核。因此不能把 3→1 误报直接解读为整体准确率提升，也没有人工复核后的金标准准确率。

C 与 D 的候选状态相同。动态路由减少 78 次 VLM 调用（93.98%），本次顺序执行的整组热处理耗时减少约 88.39%。这是在同一台 CPU、同一批图像和当前冻结策略下的实测，单次顺序实验未提供置信区间。VLM 只输出未经验证的候选描述，当前没有证据证明这些描述提升检测精度。两个实验为公平计时均预先加载 VLM，故不能据此声称内存下降。

`verify_comparison.py` 对两组所有 83 张逐一核对：图像分数最大绝对差 0，原始 256×256 热图最大绝对差 0，均通过 rtol=1e-5 / atol=1e-6。连续视觉指标因此沿用同一冻结 A 结果：图像 AUROC 1.0、AP 约 1.0、固定阈值 F1 0.976744、召回率 1.0；像素 AUROC 0.978275、AUPRO@0.3 0.923896。F1/召回是 A 的全样本二分类指标，不是把人工暂缓移除后的 C/D 分类指标。

## 实际运行与产物

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B .\scripts\guard.py --seconds 900 --record .\logs\phase3-fixed-vlm-execution.json -- .\.venv\Scripts\python.exe -B .\scripts\benchmark_agent.py --mode fixed --vlm
.\.venv\Scripts\python.exe -B .\scripts\guard.py --seconds 900 --record .\logs\phase3-dynamic-vlm-execution.json -- .\.venv\Scripts\python.exe -B .\scripts\benchmark_agent.py --mode dynamic --vlm
.\.venv\Scripts\python.exe -B .\scripts\verify_comparison.py .\outputs\20261022-000114-agent-fixed
.\.venv\Scripts\python.exe -B .\scripts\verify_comparison.py .\outputs\20261022-000731-agent-dynamic
```

四个命令退出码均 0。C：outputs/20261022-000114-agent-fixed；D：outputs/20261022-000731-agent-dynamic。目录内保存 frozen.json、逐图 report.json、原始/显示热图、参考与裁剪图、predictions.json、metrics.json、score-parity.json。D 的完整进程 stdout/stderr 见 logs/phase3-dynamic-vlm-execution.child.log；C 的完整逐工具轨迹见逐图报告。早期 guard stdout 继承问题详见 Phase 3 报告。

额外的 D（VLM 关闭）为 outputs/20261021-235303-agent-dynamic，平均 3.8916 调用，VLM 调用 0，同样暂缓 5 张；它属于可选消融，不能冒称 C。

C/D 冻结时记录的 agent.py、tools.py、vlm.py 已复制到各自 source/visioninspect，复制前逐个验证 SHA 与 frozen.json 一致。之后仅对 VLM 来源文件的 Windows 分隔符作跨平台规范化，未改变数值模型或路由策略；Windows/POSIX 路径回归通过。容器实际运行仍待验证。

## 尚未包含

B EfficientAD 未正式训练和评估：teacher 与合成张量架构冒烟已经验证，但当前 Anomalib 完整训练路径所需的授权通用正则化图像尚未准备，CPU 估算也超过预算，见 DATASETS.md。metal_nut 后续仅完成数据准备；VisA、LOCO、其他 MVTec 类别均未混入这里的 bottle 结果。此实验为 PatchCore-ResNet18 CPU 研究基线与有限状态条件 Agent，不是论文默认配置复现，也不证明工业产线可靠性。
