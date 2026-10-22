# Phase 5 部署状态：准备完成，Docker 验收阻塞

## 已准备

CPU Dockerfile：固定 Node 24.22.0、Python 3.12.12、uv 0.11.7，前端构建与后端同镜像，Linux CPU 哈希锁，非 root 用户，`/ready` 健康检查。数据和模型由 volume 挂载，不进入镜像。Windows 启动脚本限制 4 CPU、12 GiB，端口仅绑定 localhost；outputs 可写，weights/data 只读。没有声明 GPU 支持。

后续还通过 scripts/check_container_sources.py 核验三个基础镜像的 registry index digest 和 linux/amd64 manifest，Dockerfile 已按 digest 固定。压缩层元数据约 151.7 MB，不含依赖与构建缓存，不代表实际落盘大小；没有拉取镜像层。结果在 logs/container-source-preflight.json。这项来源检查不能替代 Docker build / run 验收。

本地 `scripts/start_local.ps1` 已实际启动并完成浏览器闭环；测试结束后停止临时服务，没有安装系统服务、计划任务或持续后台进程。

## 实际阻塞与命令

- `docker version` / `docker info`：引擎最初不存在，启动已安装 Docker Desktop 后仍报 `Docker Desktop is unable to start`。
- `wsl --status`：`WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED`。输出建议启用 Windows 子系统可选组件并可能重启，详见 logs/docker-blocker.json。
- `docker desktop stop`：退出 0，停止本次测试启动的应用，见 logs/docker-stop.log。
- `scripts/run_docker.ps1`：最终预检退出 1，明确提示引擎不可用；没有执行 build、run 或镜像拉取，见 logs/docker-preflight-final.log。

未擅自启用 Windows 组件、更新驱动/CUDA、改网络或重启电脑。因此 **Docker 构建、容器健康、挂载后的真实推理尚未验证**。不能以文件已存在或 Linux 依赖解析成功代替部署验收。

## 资源与环境

当时的快照 logs/final-delivery-audit.json：主要项目目录逻辑大小合计约 4.006 GiB（可能双计硬链接），E 盘剩余约 1039.3 GiB；后续新增 teacher 与 metal_nut 数据，因此这个历史数值不代表当前磁盘占用。项目 .venv 为 Python 3.12.12，98 个包，整理时 uv pip check 再次通过。原始交接文档 SHA256 保持 a09e63c297808e0ce0d0b86223d9c0212b722aad82e89dc657c973c9a021c79f。未初始化 Git、创建远程仓库或提交数据/模型。

## 需要用户决定

已询问是否先交付本地 CPU 版并保留 Docker 待办，或由用户手动准备可用引擎后继续。未收到答复前不把沉默当作启用系统组件的授权。取得可用引擎后，还须重新核对镜像/缓存磁盘位置与预算，再实际 build → 挂载资产 → ready → 上传真实图像 → 报告/历史 → 停止容器，才可宣布 Docker 验收完成。
