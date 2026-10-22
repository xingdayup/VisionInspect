# 常驻推理进程与空闲 TTL

2026-10-22 用户要求将每项任务新建进程改为常驻推理进程与空闲 TTL。本次改动只涉及本地服务调度、模型缓存、状态接口、测试及文档；没有安装依赖、下载资产或修改冻结模型、阈值、提示词和正式实验。

## 使用与生效

默认配置在 [configs/service.yaml](../configs/service.yaml)：

```yaml
schema_version: 1
idle_ttl_seconds: 600
```

只接受大于 0 的有限数值，单位秒。空闲时间从最近一项任务完成后开始计算；执行下一项任务时暂停回收，完成后重新计时。超过 TTL 则终止整个推理子进程，释放模型及导入库占用的内存。下次任务重新启动，首张仍需冷启动。

由于修改了服务进程中的调度器，正在运行的旧服务必须重启：在原终端按 Ctrl+C，等待退出后，在项目根目录运行 `.\scripts\start_local.ps1`，再刷新页面。此次保留了用户自己启动的服务，没有自动结束它。修改 TTL 或推理代码后同样需要重启。

常驻进程仅由运行中的应用持有，不注册开机启动、定时任务或独立系统服务。关闭 API 时终止推理进程；父进程异常退出时，EOF / 父进程存活监控负责退出子进程。CLI 结束后也关闭自己的进程，跨 CLI 命令不共享缓存。

## 缓存与任务隔离

- 一个检测模型槽位：缓存 PatchCore（含 ResNet18 与正常特征库）和正常参考图索引。切换模型 / 类别时替换该槽位，避免不断累积模型。
- 一个可选 VLM 槽位：只在启用且流程真正调用候选解释时加载权重；后续任务可以复用。未勾选 VLM 的任务不会调用它，之前已加载的 VLM 可以保留到进程 TTL 到期。
- 每项任务创建新的 ToolKit / Agent，使用新的输入、进度、输出、证据和 VLM 输入审计目录。分数、图像、候选文本与人工记录不作为跨任务模型缓存。
- 模型注册信息、校准策略、checkpoint / profile / 参考索引文件的身份、大小及时间戳参与检测缓存键；VLM 源记录、权重等文件状态及生成配置参与其缓存键。发生变化即失效，下一次加载重新校验完整 SHA256。暖缓存不重复读取全部权重做哈希；参考原图仍在读取前校验。

主要代码为 [resident.py](../visioninspect/resident.py)、[model_cache.py](../visioninspect/model_cache.py)、[worker.py](../visioninspect/worker.py)、[service.py](../visioninspect/service.py) 与 [tools.py](../visioninspect/tools.py)。原始六个工具契约与候选决策规则保持原义。

## 超时、故障与资源

单通道串行处理任务，每项任务仍有独立的 **180 秒**监督期限和 **12 GiB**进程树内存限制；冷启动计入任务总时限。检测模型初始化仍有 90 秒返回后检查，单工具仍有 45 秒返回后检查。VLM 的首次加载与生成计入该工具预算。

任务运行及空闲期均检查进程状态与内存。内存使用当前进程树 RSS，每 0.2 秒采样；每任务 peak 字段是该时间窗口内的采样峰值，包含已有模型，不能当作只计算本任务新分配内存。此前模型进程的 OS lifetime peak 不用于当前任务限额。

崩溃、任务超时、内存超限或协议异常时销毁进程，将当前任务降级为 manual_review，下一项任务重新启动。Agent 返回工具错误时也回收缓存进程。完成文件必须携带匹配的随机请求标识和报告 SHA256，监督器才能接受报告；提前写出的报告不代表任务成功。Windows venv 启动器与实际解释器分别追踪，启动器异常退出时也会清理已追踪的子进程。

## 查看状态与日志

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/api/runtime' | ConvertTo-Json -Depth 5
```

该接口返回 state、generation、pid（启动器）、inference_pid（实际推理解释器）、已加载缓存、idle_seconds、idle_ttl_seconds 与 expires_in_seconds。not_started 表示尚未提交任务，running 表示任务执行中，idle 表示进程等待后续任务，stopped 表示已被回收。

每项报告 runtime 中记录 worker_pid、generation、task_sequence、model_cache 和 vlm_loaded_before_task；初始化分项的 cache_hit 表示检测缓存命中。`execution.json` 的 process_reused 表示使用了同一进程，不等于所有模型都已加载。

常驻进程正常完成一项任务时仍在运行，所以 `execution.exit_code` 可以为 null；这不是失败。`task_exit_code` 与 worker_report_accepted 记录本任务协议 / 报告是否成功，进程最终退出码记录在 `outputs/service/runtime/events.jsonl`。完整进程日志保存在同目录 worker-{generation}.log；逐任务 worker.log 仍在各任务目录。

## 实际执行与验证

```powershell
. .\scripts\project_env.ps1
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_resident.py tests/test_service.py tests/test_initialization.py
.\.venv\Scripts\python.exe -B scripts/guard.py --seconds 180 --record logs/resident-real-execution.json -- .\.venv\Scripts\python.exe -B scripts/verify_resident.py
.\.venv\Scripts\python.exe -B scripts/guard.py --seconds 120 --record logs/resident-final-tests-execution.json -- .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider -o junit_family=xunit1 --junitxml=logs/resident-final-tests.xml
```

首次针对性验证为 23 passed / 14.01 秒，退出 0。后续增加了 RSS 口径与启动器异常清理回归，最终完整套件 **86 passed / 44 条第三方弃用 warnings / 38.69 秒**，退出 0；监督命令总耗时 40.917 秒。详细记录见 logs/resident-final-tests-execution.child.log；不同轮次的测试数不相加。模拟子进程、计时与内存故障案例仅用于工程验证，不是模型效果指标。

真实模型验证使用正常验证图 `train/good/002.png`，独立状态目录为 `outputs/20261022-115831-resident-verification-80a4d2/service`，未使用正式测试集：

| 项目 | 首次，固定流程 + VLM | 连续第二项，固定流程 + VLM | TTL 到期后，单模型 |
|---|---:|---:|---:|
| 任务总耗时 | 33.674 秒 | 3.852 秒 | 25.140 秒 |
| 检测初始化 | 24.117 秒 | 0.000008 秒，命中缓存 | 23.631 秒 |
| 实际推理 PID | 32392 | 32392 | 35424 |
| VLM 已在任务前加载 | false | true | false，未启用 |
| 图像分数 | 8.26118278503418 | 8.26118278503418 | 8.26118278503418 |
| 任务窗口进程树采样峰值 | 3,038,818,304 B | 3,062,300,672 B | 628,834,304 B |

两项 VLM 任务均使用 separate-images-v2，编码器输入 [1,2,3,512,512]；模型与参考索引缓存复用，逐任务证据目录保持独立。三次输出的原始热图均为有限的 256×256 浮点数组，并通过 rtol=1e-5 / atol=1e-6 一致性检查；状态均为 normal_candidate。此验证没有评估候选解释的语义准确性。

验收进程在前两项完成后临时设 TTL 为 2 秒，确认旧进程退出，缓存状态清空；随后恢复该验收对象的 600 秒设置，第三项得到新 generation 和进程。没有改写生产配置。退出 TestClient 后，第三项推理进程也已释放。

真实验证命令耗时 67.469 秒，退出 0；每项任务采样峰值最高约 2.85 GiB，GPU 显存不适用，外部 API 费用 0。外层 guard 字段为 3,791,298,560 字节，是多个进程 / 世代各自峰值的累计口径，不等于同时驻留内存，不能与逐任务 RSS 峰值混用。没有新增数据或权重下载。

逐任务记录和输入审计位于上述独立输出目录，汇总为同目录 verification.json；该次输出共 56 个文件、9,295,480 字节逻辑大小。补充核对两次热图最大绝对差均为 0，10 个保护文件与 C/D 归档源代码哈希均未改变，记录在 logs/resident-preservation.json。命令 / 完整日志为 logs/resident-real-execution.json、logs/resident-real-execution.child.log、logs/resident-final-tests-execution.json、logs/resident-final-tests-execution.child.log。前端未修改，无需重建；Docker 镜像仍未完成构建或运行验收。

## 局限

缓存适用于当前单机、单 API worker、单推理通道。第一次检测、TTL 到期或故障后的下一次仍需冷启动。10 分钟默认 TTL 已由同一路径的短 TTL 工程验证覆盖，本轮没有实际等待整整 10 分钟进行计时验收。单张连续检测耗时不是 p50 / p95，也不代表吞吐或准确率提升。旧冻结 A/C/D 的延迟口径保持历史原值，不替换为本次数字。
