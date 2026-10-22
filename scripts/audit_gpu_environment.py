"""Portable read-only GPU audit. Installs nothing and downloads no model assets."""

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def read_optional(path):
    try:
        return Path(path).read_text(encoding="utf8").strip()
    except (OSError, UnicodeError):
        return None


def audit():
    result = dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        working_directory=str(Path.cwd()),
        host_cpu_count=os.cpu_count(),
        dependencies_installed=False,
        model_weights_downloaded=False,
    )
    result["packages"] = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "vllm",
        "triton",
        "flashinfer-python",
    ):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    result["cgroup_limits"] = {
        path: read_optional(path)
        for path in (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/cpu.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        )
    }
    result["disk"] = {}
    for path in (
        str(Path.cwd()),
        "/mnt/workspace",
        "/mnt/systemDisk",
        "/mnt/data",
        "/dev/shm",
    ):
        if Path(path).is_dir():
            usage = shutil.disk_usage(path)
            result["disk"][path] = dict(total_bytes=usage.total, free_bytes=usage.free)
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=15)
        result["nvidia_smi"] = dict(
            command=command,
            exit_code=process.returncode,
            stdout=process.stdout.strip(),
            stderr=process.stderr.strip(),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        result["nvidia_smi"] = dict(error=str(error))
    result["cuda_smoke"] = dict(passed=False)
    try:
        import torch

        result["torch"] = dict(
            version=torch.__version__,
            cuda_build=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
        )
        if torch.cuda.is_available():
            result["torch"].update(
                device=torch.cuda.get_device_name(0),
                compute_capability=list(torch.cuda.get_device_capability(0)),
                bf16_supported=torch.cuda.is_bf16_supported(),
            )
            checks = []
            torch.set_num_threads(4)
            for dtype in (torch.float16, torch.bfloat16):
                if dtype == torch.bfloat16 and not result["torch"]["bf16_supported"]:
                    continue
                started = time.perf_counter()
                tensor = torch.ones((1024, 1024), device="cuda", dtype=dtype)
                product = tensor @ tensor
                torch.cuda.synchronize()
                if not bool(torch.isfinite(product).all()) or not bool(
                    (product == 1024).all()
                ):
                    raise ValueError(f"{dtype}: matrix multiplication result failed")
                query = torch.ones((1, 8, 128, 64), device="cuda", dtype=dtype)
                attention = torch.nn.functional.scaled_dot_product_attention(
                    query, query, query
                )
                torch.cuda.synchronize()
                if not bool(torch.isfinite(attention).all()):
                    raise ValueError(f"{dtype}: attention failed")
                checks.append(
                    dict(
                        dtype=str(dtype),
                        matrix_shape=[1024, 1024],
                        sdpa_shape=list(query.shape),
                        seconds=time.perf_counter() - started,
                        finite=True,
                    )
                )
                del tensor, product, query, attention
            result["cuda_smoke"] = dict(
                passed=True,
                checks=checks,
                max_memory_allocated_bytes=torch.cuda.max_memory_allocated(0),
                max_memory_reserved_bytes=torch.cuda.max_memory_reserved(0),
                scope="Small matrix multiplication and SDPA only; vLLM kernels and model loading remain unverified",
            )
        else:
            result["cuda_smoke"]["reason"] = (
                "CUDA unavailable in this Python environment"
            )
    except Exception as error:
        result["cuda_smoke"]["error"] = f"{type(error).__name__}: {error}"
    if platform.system() == "Linux":
        try:
            library = ctypes.CDLL("libcuda.so.1")
            version = ctypes.c_int()
            status = library.cuDriverGetVersion(ctypes.byref(version))
            result["cuda_driver_api"] = dict(status=status, version=version.value)
        except OSError as error:
            result["cuda_driver_api"] = dict(error=str(error))
        mappings = read_optional("/proc/self/maps") or ""
        result["loaded_libcuda_paths"] = sorted(
            {line.split()[-1] for line in mappings.splitlines() if "libcuda.so" in line}
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf8")
    print(text)
    # A completed audit may reveal an unsuitable environment. Exit status is not a GPU-pass claim.
