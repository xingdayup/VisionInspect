# Phase 2 工具与有限状态 Agent

六个工具均有 Pydantic 输入输出约束：质量检查、真实 PatchCore 推理、正常训练参考检索、热图连通区域裁剪、可选本地 VLM、确定性结果汇总。VLM 缺失时明确返回不可用与 unknown，不生成假描述。

`scripts/prepare_tools.py` 仅读取 168 张正常训练图与 41 张正常验证图，生成特征参考索引和冻结策略。参考图来自正常训练集。图像阈值 8.742960929870605；复核区间 [8.492477416992188, 9.452290534973145]；像素阈值 7.434370517730713。校准策略 digest 为 `08c1f22d4b52fd10c5266aa65fb63745c8a4ffc4890ac051cfbd6c0660def632`。这些界限是研究原型的正常验证样本经验规则，不保证总体误报率。

实际命令：

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B .\scripts\prepare_tools.py
.\.venv\Scripts\python.exe -B -m pytest -q
```

退出码均为 0。`logs/phase2-prepare-tools.log` 保存生成记录；`logs/phase2-tests.log` 为 29 passed、44 warnings、20.13 秒。测试包含坏图片、平坦图质量、未知工具、区域裁剪路径、低质量降级、工具失败降级、条件分支、调用预算耗尽。单元测试替身只用于路由验证，不用于实验或网页结果。

参考索引约 0.24 MB；本阶段未安装依赖、未修改系统环境。冻结策略与模型 SHA 校验，工具不自动联网。硬超时与内存终止需要由下一阶段任务子进程监管器执行；当前 Agent 自身记录每次工具耗时并将超限结果作废。

下一步：本地 VLM 资产验证、真实 A/C/D 对照与任务服务。EfficientAD 标准训练需要额外 ImageNet 类图像惩罚数据，尚未准备，不能将替代数据训练冒充标准对照。
