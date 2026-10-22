# 当前实现架构

React / TypeScript / Vite 工作台通过 REST 上传到 FastAPI。SQLite 保存任务元数据和复核记录；文件系统保存输入、逐工具进度、模型证据、原始报告和执行日志。生产前端由 FastAPI 同源提供；开发阶段 Vite 代理 `/api`、`/health`。

## 状态与隔离

任务状态为 queued → running → completed / failed。单个服务进程、单个排队执行线程，共用一个按需启动的常驻 Python 推理子进程。仅缓存检测模型、正常训练参考索引及可选 VLM；每项任务重新创建 ToolKit / Agent，输入、进度、报告和证据目录独立。监管器每 0.2 秒检查当前进程树 RSS：单任务 180 秒或 12 GiB 上限触发终止，任务标记 failed + manual_review，下一项任务重建进程。空闲期也检查内存及进程存活。

`configs/service.yaml` 的 `idle_ttl_seconds` 默认为 600。最近一项任务完成后开始计时；新任务执行时暂停空闲回收，完成后重置计时。到期退出整个推理进程，从 OS 层释放模型和运行库内存；下次使用冷启动。服务停止时清理进程，服务重启时把未结束任务标记为中断。CLI 使用相同调度器，但单次命令完成后会关闭自己的调度器，跨 CLI 调用不共享缓存。完整行为与验证见 [常驻进程](RESIDENT_WORKER.md)。

子进程的每项模型任务禁用 Python socket 连接；父子进程通过 stdin 命令与原子写入的完成文件通信，不新增监听端口。完成文件带请求随机标识与报告 SHA256，匹配后监督器才接受报告。模型使用本地权重，`local_files_only=True`，不执行远程模型代码。注册表为 `configs/models.yaml`；策略为 `configs/agent.yaml`；正常参考索引和校准策略的 SHA 与模型、数据清单绑定。输入路径来自服务分配的任务目录，不能从 API 指定任意文件路径。

## 条件工作流

1. 图像质量不合格或检查失败：直接转人工复核。
2. 输入有效：先独立初始化检测模型，再运行真实 PatchCore，保存原图、raw float `.npy`、显示热图和叠加图。
3. 动态流程：分数超过图像阈值或进入复核区间时提取连通区域；仅复核区间内检索正常训练参考，可选调用本地 VLM。
4. 固定流程：质量通过后总是提取区域、检索参考；启用 VLM 时固定调用候选解释。
5. 确定性汇总：失败、质量问题、边界分数或证据冲突为 manual_review；其余严格按 `score > threshold` 生成候选。VLM 不能修改分数、阈值或直接授权放行。

最多 8 次调用，最多 1 次 VLM，仅 retryable 失败允许 1 次重试。每条轨迹保存名称、原因、输入、输出或错误、状态、耗时和尝试次数。单工具执行返回后若超过 45 秒即丢弃结果，任务子进程监管提供硬终止边界。

检测模型初始化从工具计时中分离，单独保存 `initialization` 的状态、耗时与导入 / 校验 / checkpoint 加载 / 运行设置耗时，预算由 `model_initialization_timeout_seconds: 90` 控制。缓存命中时 `initialization.timings.cache_hit=true`，各加载分项为 0，无需重复恢复权重；首次加载或失效时为 false。初始化返回后若超过该预算或任务总时限，则不执行检测并转人工复核；独立的 180 秒监督器仍会终止整个任务。初始化不占额外工具调用次数，且仍计入端到端任务耗时；VLM 首次加载仍包含在候选解释工具的 45 秒计时内。网页进度与轨迹继续显示初始化状态 / 耗时，JSON 记录实际缓存命中。冻结模型与阈值未更改。

检测缓存只有一个槽位，键包含模型注册信息、校准策略及 checkpoint / profile / 参考索引的文件身份、大小和时间戳；类别、模型或资产变化时失效。VLM 有独立的一个槽位，配置或源记录 / 资产文件变化时失效。首次实际加载仍执行完整 SHA256 校验；命中缓存不重复读取所有权重做哈希。原始参考图使用前仍校验哈希。工具错误后回收进程，避免后续任务复用可疑状态。

报告 monetary_cost 表示外部 API 费用，本地后端固定为 0；不包括本机硬件或电费。没有付费提供方、凭证或云资源启动路径。

VLM 当前使用 separate-images-v2：完整待检图与第一张正常参考图独立传入，每图保留比例并留边至 512×512，关闭额外切片；尚未传入区域裁剪。实际处理器图像数、token 数和编码器输入图保存到每个任务的 vlm_input/。旧 C/D 使用单合图版本，其延迟不代表 v2，详见 [输入核验](VLM_INPUT.md)。

## 工具与契约

`GET /api/tools/schemas` 返回六个工具的完整 Pydantic schema。质量工具检查解码、大小、对比度、拉普拉斯清晰度和曝光范围；范围由正常验证集经验校准。参考检索使用 ResNet18 layer2/layer3 全局均值特征的余弦相似度，检索源严格限于正常训练划分。区域工具在 256×256 原始热图上以冻结像素阈值提取连通区域，再映射回原图裁剪。

## HTTP 接口

| 接口 | 行为 |
|---|---|
| `GET /health` | 服务存活、CPU 模式、模型文件与本地 VLM 文件可用性；文件存在不等于权重已成功加载 |
| `GET /ready` | 至少一个视觉模型资产完整时 200，否则 503 |
| `GET /api/models` | 已注册类别与模型，可用性和缺失原因 |
| `GET /api/runtime` | 推理进程状态、generation、PID、已加载缓存、空闲秒数与 TTL 剩余时间；状态为 not_started / running / idle / stopped |
| `POST /api/tasks` | multipart file、category、model、mode、vlm；返回 202 与任务编号 |
| `GET /api/tasks` | 支持 status、state、q、limit、offset，包含总数 |
| `GET /api/tasks/{id}` | 元数据、进度、真实轨迹和完成后的报告 |
| `GET /api/tasks/{id}/artifacts/{name}` | 只允许原图、热图、叠加、参考、区域 PNG 和原始 `.npy` |
| `GET /api/tasks/{id}/report` | 导出 JSON，含独立的最近人工复核字段 |
| `POST /api/tasks/{id}/review` | 记录研究复核状态、非空说明和时间，保留模型原报告；复核事件追加保存 |
| `GET /api/runs` / `.../{id}/metrics` | 真实 CLI 基线实验与指标 |

上传仅支持 PNG/JPEG/WebP，最大 10 MiB、16,777,216 像素；坏文件返回 422，过大 413，未注册类别 422，资产缺失 503，排队超过 8 条返回 429。任务失败时提供可读原因并转人工复核。

## 本地部署范围

当前服务面向单机研究，无账户体系、产线接口或公网部署授权。默认绑定 127.0.0.1。使用一个 uvicorn worker，不能用多个服务进程共同写同一状态目录。SQLite 与文件保持在项目 outputs 下。Docker 为 CPU 镜像，数据、模型和实验目录通过 volume 提供；GPU 支持未验证，不自动切换 CUDA 构建。

## v0.4 新工作流

`agent_review` 使用 LangGraph 管理 baseline → decision → evidence tool → decision → finalize 循环。Qwen3-VL 经独立 vLLM 服务返回结构化观察与白名单动作；Python 绑定证据、验证区域覆盖并决定候选状态。旧 baseline/fixed/dynamic 保留兼容。SQLite 持久调度单图与批次，分别保留机器、Agent、人工结果。CPU worker 与 GPU 服务独立 TTL，GPU 失败保留 CPU 结果。业务规格与部署以 BUSINESS_V1.md、MIGRATION_PAI.md 为准，GPU 实机验证状态见 UPGRADE_STATUS.md。
