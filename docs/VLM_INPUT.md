# VLM 输入核验与协议修复

日期：2026-10-22。此记录针对图像输入处理，不代表缺陷语义准确性验收。

## 已核实的旧输入

用户任务 57ad58a1ea1b4568891b3cd50476c916 使用固定复核与本地 VLM，原图与 MVTec bottle/test/contamination/007.png 的 SHA256 一致。送入的是该待检原图和本任务 reference_1.png，没有取成两张正常参考图；也没有把 R1 / R2 裁剪送入 VLM。

旧代码将两张 900×900 图先各缩放至 512×512，拼成一张 1024×512 图片。调用处理器时关闭 image splitting，安装的 Transformers 5.20.2 Idefics3ImageProcessor 随后将整张合图强制缩放为 512×512。实际 pixel_values shape 为 [1,1,3,512,512]，只有 64 个视觉占位 token；两个正方形来源均被横向压缩。提示词明确要求描述 Left / Right 的瓶口差异，因此“左瓶 / 右瓶”的措辞来自对照任务。

只复现处理器、未生成新文本的输入审计：

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B scripts/guard.py --seconds 90 --record logs/vlm-input-audit-execution.json -- .\.venv\Scripts\python.exe -B logs/audit-vlm-input.py
```

退出 0，结果在 outputs/diagnostics-vlm-input-57ad58a1/input-audit.json；actual-composite-input.png 为传给处理器的合图，actual-vision-encoder-input.png 为根据处理器张量反归一化得到的实际编码器输入。该审计只核验现有测试图片的传递链路，没有据此搜索提示词、拟合阈值或重跑正式测试指标。源码修改前副本保存在该目录 source-before/。

## separate-images-v2 修复

- 待检图与正常参考图作为两个独立 image 占位符传入。文本角色改为 Image 1 / Image 2，原有类别、描述范围、unknown 与禁止推断原因的语义要求保持一致；未加入根据这张测试图推断出的异物名称。
- 每张图保留比例、缩放并留边至 512×512，不中心裁剪，不再创建横向合图。没有参考时仅传一张图。
- 校验处理器实际输出图像数、尺寸和视觉 token 数。两图模式 shape 必须为 [1,2,3,512,512]，128 个视觉 token；不一致则明确失败。
- 新 VLM 任务在 outputs/service/<task-id>/vlm_input/ 保存 input_1.png、input_2.png、encoder_1.png、encoder_2.png 与 input.json，记录角色、原始 / 准备尺寸、prompt、模板、模型 revision、处理器形状、文件 SHA256。encoder 图片来自实际输入张量的反归一化，仅供人工核验。
- 工具输出 backend 标识附加 separate-images-v2。CandidateResult 仍为 unknown / verified=false；PatchCore、阈值和最终规则未修改。

当前仍以完整待检图和第一张正常参考图作为输入，**R1 / R2 尚未接入 VLM**。本修复解决了已确认的输入几何和图片分隔问题，不能据此认定它是此前描述失败的唯一原因，也不能保证修复后识别所有异物。

## 实际验证与边界

针对性测试命令：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_vlm_input.py tests/test_tools_agent.py tests/test_assets.py
.\.venv\Scripts\python.exe -B scripts/guard.py --seconds 180 --record logs/vlm-input-v2-smoke-execution.json -- .\.venv\Scripts\python.exe -B logs/verify-vlm-input-v2.py
```

针对性测试为 30 passed、4.30 秒、退出 0，包含输入图片顺序、图片占位符、比例保留、编码器输入审计和工具调用边界；记录在 logs/vlm-input-v2-tests.log，不冒称完整套件重跑。

离线实际生成使用正常 validation 的 train/good/002.png，与正常 training 的 train/good/000.png 对照；未使用测试图生成或迭代提示词。输出在 outputs/20261022-110407-vlm-input-v2-9b5231，实际 shape [1,2,3,512,512]、128 个视觉 token，生成成功。模型加载 20.697 秒，输入审计加生成 3.259 秒；监管命令 26.119 秒退出 0，进程树峰值 2,871,877,632 字节。全部为 CPU FP32，本地资产，Python socket 禁用，没有安装依赖或下载新模型。

单次原始输出为“The rim is slightly worn.”，同样未经人工确认；这条输出仅证明完整调用正常，**不能算缺陷描述准确率，也不能因为正常样本返回了磨损描述就宣称语义验证通过**。正式语义效果仍待独立数据和标注评估。

已有 C/D 83 张冻结对照及各任务报告全部保留。它们使用旧单合图协议，相关调用量与延迟只适用于旧版本，不能移用于 v2。原始实现已在各实验 source/visioninspect 归档并匹配 frozen.json；不覆盖或用新版本输出替换旧结果。新任务使用新进程加载 v2；旧任务页面继续显示原结果。
