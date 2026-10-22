# PAI DSW GPU 环境准备记录

2026-10-22。此阶段只做环境与版本核验；尚未安装 GPU 环境、下载 Qwen 权重、启动 vLLM 或验证 GPU 复核效果。远端由用户手动运行命令，没有 SSH 访问。

## 用户提供的实际信息

- 平台：PAI DSW；系统 / 镜像名称为 ubuntu22.04-cuda13.0.3-py312-torch2.13.0-1.40.1，尾部字段的含义未确认。
- 配额：用户报告 8 核、32GB 内存。
- nvidia-smi：NVIDIA A10，24564 MiB 显存，空闲时使用 1 MiB；驱动 550.163.01，显示 CUDA 13.0。
- 默认 Python：/usr/local/bin/python3，site-packages 为 /usr/local/lib/python3.12/site-packages，平台提示这些包会提交到镜像。
- 已执行的 torch 探测：2.13.0+cu130、torch.version.cuda=13.0、cuda.is_available()=True。
- 挂载路径：/mnt/workspace、/mnt/systemDisk、/mnt/data。用户提供的信息显示 /mnt/workspace 与 /mnt/data 指向同一 NAS 资源；不可把两处可用空间相加。

这些结果确认设备可枚举，尚未证明 CUDA 算子、JIT 或 vLLM 自定义内核可运行。NVIDIA 常规兼容要求中 CUDA 13.x 对应驱动 R580 或更新；本实例显示为 R550，因此需核实平台实际加载的 libcuda / 兼容机制，不据此擅自升级或替换驱动、系统 CUDA、共享库。PAI 是否做了特定兼容配置尚未确认。

来源：[NVIDIA CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[vLLM GPU 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)。

## 下一条审计命令

把 [audit_gpu_environment.py](../scripts/audit_gpu_environment.py) 上传到 DSW 的项目目录，在该目录运行：

```bash
python3 audit_gpu_environment.py --output gpu-environment-audit.json
```

若已上传完整项目，在项目根目录运行：

```bash
python3 scripts/audit_gpu_environment.py --output logs/gpu-environment-audit.json
```

脚本不安装依赖、不下载模型、不修改驱动或系统环境；读取包版本、cgroup 配额、磁盘 / 共享内存空间、nvidia-smi 和实际 libcuda 路径，并在现有 torch 环境执行小规模 FP16 / BF16 矩阵乘法与 SDPA。生成一个 JSON 审计文件，进程结束即释放测试张量。脚本正常退出只表示审计完成，是否通过须看 cuda_smoke.passed。即使小算子通过，vLLM 内核与模型加载仍待验证。

本机 Windows 项目环境仅用于校验脚本语法与 CPU 分支；不能把它的结果当作 DSW GPU 已通过。

## 独立环境与资产预算

计划项目路径为 /mnt/workspace/VisionInspect-GPU，推理环境为其 .venv-vllm；模型、缓存、临时文件、日志均放该项目下。不得向默认 /usr/local Python、Conda base 或其他项目环境直接装依赖，也不卸载现有 torch 2.13。安装前先核对磁盘、依赖冲突、发行文件预算并生成版本锁定记录。

2026-10-22 通过 PyPI 元数据核实的候选（未安装，不代表已完成兼容验收）：

| vLLM 版本 | torch 约束 | transformers 约束 | Linux x86_64 wheel 大小 |
|---|---|---|---:|
| 0.32.0 | 2.13.0 | >=5.16.1,<5.20.0 | 326,518,482 B |
| 0.11.2 | 2.9.0 | >=4.56.0,<5 | 370,306,629 B |
| 0.11.0 | 2.8.0 | >=4.55.2 | 438,217,982 B |

较早版本只用于比较依赖关系，不因版本旧就假定更适合此机器。vLLM 主版本号与 torch 表面一致也不能证明二进制兼容；实际选择须结合 CUDA build、驱动 / 兼容库与模型内核探测。不要直接把 Windows CPU 项目的 transformers 5.20.2 锁复制进 vLLM 环境；它不满足上述 0.32.0 的约束。

Qwen/Qwen3-VL-8B-Instruct 预检 revision 为 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b，Apache-2.0，四份 BF16 safetensors 共 17,534,339,512 B（约 17.53 GB / 16.33 GiB）。模型卡：[官方来源](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。尚未下载或批准自动下载。建议为环境、权重、缓存和产物预留至少 50 GiB；完整传递依赖的下载量尚未解析，不能把 vLLM 单个 wheel 大小当成总下载量。

拟采用单并发、短上下文、限定图片总像素的 GPU 调试配置。显存还需容纳视觉中间结果、KV cache 和引擎开销，必须以真实模型启动峰值为准。当前 CPU Agent 的 12 GiB 主存监管与 600 秒 TTL 不会自动管理独立 vLLM 服务，接入前需要独立 GPU 运行配置与生命周期管理；这些尚未实现。
