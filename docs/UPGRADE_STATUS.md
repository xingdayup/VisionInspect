# v0.4 业务与框架改造记录

更新：2026-10-22。业务定义见 BUSINESS_V1.md，迁移步骤见 MIGRATION_PAI.md。

## 已实现

- LangGraph `agent_review` 工作流、结构化多图 VLM 客户端、证据绑定、局部/参考选择、重复动作与调用预算控制、最终确定性校验。
- 独立 vLLM 子进程、父进程存活监管、lease 与空闲 TTL、内存限制、启动和推理分别计时、原始检测证据留存。
- SQLite 持久批次、逐文件上传错误记录、100 张限制、200 项队列、暂停继续取消、失败重试及原尝试保留、迁移前数据库备份。
- 工作台批次页面、三层结论、实际 VLM 输入图查看、逐项人工复核、JSON/CSV 导出、CPU/GPU 独立状态。
- Windows/Linux Agent 哈希锁、PAI 环境预检与隔离安装脚本、Git bundle、独立资产包与安全恢复工具。

## 已验证与实际命令

```powershell
.\.venv\Scripts\python.exe -B scripts/agent_dependency_budget.py
.\.venv\Scripts\python.exe -B scripts/run_upgrade_checks.py
.\.venv\Scripts\python.exe -B scripts/verify_upgrade_cpu.py
```

- 改造前 86 项测试通过。新增依赖锁解析不改原有版本，新增 22 包，候选 wheel 合计 3,584,414 字节；仅安装到项目 `.venv`。实际安装及前后版本在 logs/agent-install.log、packages-before-agent.txt、packages-after-agent.txt。
- 最新全量测试结果以 logs/upgrade-checks.json、upgrade-tests.xml 为准；一次完整执行为 118 passed，现有 PyTorch/Starlette 弃用及 pytest XML 属性警告如实保留。
- 前端 TypeScript 检查及 Vite 生产构建成功。真实浏览器验证包含批次创建、两张成功上传、一张损坏文件拒绝、真实正常图推理、合成平灰图质量失败、人工记录、汇总更新和 CSV 下载；合成图仅用于工程流程，未作为模型效果数据。
- 真实正常验证图 002.png 分数 8.26118278503418，114.png 分数 8.778727531433105；迁移前 CPU 分数与热图对比的最大热图差为 0。GPU 不可用时 114.png 转人工并保留证据，重试复用同一 CPU worker；记录在 outputs/20261022-144917-upgrade-cpu-verification-9039ef/verification.json。Windows 短暂文件共享冲突已增加有界读写重试，原失败日志保存在 logs/upgrade-cpu-share-violation.log，重新执行通过。没有重新训练、修改阈值或重跑正式测试集指标。
- Bash 脚本通过 Git Bash `bash -n` 语法检查；这不等于已在 Ubuntu 安装或运行。
- 原始交接文档、AGENT.md、checkpoint、校准与原划分不改写。Windows 路径迁移显式启用，并继续核验图像哈希和原清单身份。

## 待验证，不作为已完成成果

- PAI 的应用环境安装、vLLM 原生内核与 8B 权重实际加载、多图生成、真实复核分支、GPU TTL 和显存峰值，均需执行 MIGRATION_PAI.md 中的命令并回传日志。
- 新 Agent 误报/漏报收益需要独立标注数据。已见 bottle 测试集不能用来改提示词或策略，也不能用旧 CPU SmolVLM 对照结果代表新 Qwen 工作流。
- EfficientAD 完整对照仍缺授权额外图像及批准的计算预算；Docker 仍未完成实际构建运行，PAI 原生运行不依赖 Docker。

模型、数据、环境、缓存、用户结果均排除在 Git 外。Git 记录源码和规范，迁移包单独校验资产；没有创建公共仓库或调用付费 API。
