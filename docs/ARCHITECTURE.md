# GPU 产品架构

React / TypeScript / Vite 通过 REST 上传和轮询 FastAPI。SQLite WAL 保存任务、批次、关联重试和追加人工记录；文件系统保存图片、浮点热图、裁剪、模型输入、原始响应及报告。仅启动一个 API worker，执行通道串行。

## 运行链路

- `baseline`：质量检查 → CUDA PatchCore → 高分区域（如有）→ 冻结规则汇总。
- `agent_review`：检测与取证 → 不确定区间进入 LangGraph → 观察/决策 → 局部比较或切换参考 → 确定性校验 → 候选或转人工。
- PatchCore 模型、输入、特征与 memory bank 使用 cuda:0，FP32；参考相似度和图像可视化等轻量后处理在主存执行。CPU checkpoint 暂存反序列化、图片解码与 NumPy 输出不构成 CPU 模型推理。
- Qwen3-VL-8B 由独立 vLLM 进程提供本地接口，BF16。不另加载规划模型。

## 约束与证据

整体观察引用待检原图，正常整体观察同时引用本轮正常参考。局部观察引用待检裁剪与参考对应坐标裁剪。模型只能选择 `compare_region`、`compare_reference`、`finish`，最多 5 次尝试，含重试。不接受未提供的证据、重复视图、遗漏区域或冲突结论。

LangGraph 使用 baseline / decide / tool / finish 节点。运行中的状态由对象与图管理；SQLite 持久化任务和报告，不宣称已经实现 LangGraph checkpointer 的轮次断点续跑。重启后执行中任务标记失败，批次暂停，队列保留。

`initial_result`、`agent_review` 与人工追加记录独立。原始检测证据先保存，VLM 故障时保留有效检测并转人工；不改分数、阈值或测试标签。

## 资源与隔离

PatchCore resident worker 与 vLLM 各自按需启动、空闲 TTL 600 秒。vLLM lease 防止处理时被空闲回收；父进程退出清理子进程。单图 worker 180 秒/12 GiB RSS，模型初始化 90 秒，vLLM 启动 300 秒、单请求 45 秒、服务/项目 RSS 24/28 GiB。

vLLM 显存利用率配置 0.85，为同卡 PatchCore 预留空间；它是分配配置，实际峰值和能否容纳两模型需实测。CUDA 同步计时、allocator 峰值与进程 VRAM 采样分别记录，不混为同一指标。显存或内存不足明确失败，不降低精度、换模型或回退 CPU。

应用仅连接固定本机模型端口；模型从本地校验资产加载，禁止隐式下载、远程模型代码和付费 API。Python socket 限制不是系统防火墙，不声称约束所有原生代码网络行为。

运行配置在 `configs/patchcore.yaml`、`agent.yaml`、`service.yaml`、`gpu.yaml`，模型注册表为 `models.yaml`。同服务器两个 GPU Python 环境分别管理应用与已有 vLLM，平台包只读复用。
