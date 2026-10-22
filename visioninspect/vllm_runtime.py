"""Own only this application's vLLM process tree, with a lease-based idle TTL."""

from __future__ import annotations

import contextlib
import math
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import psutil
import yaml

from .common import ROOT, project_path, read_json, write_json
from .resources import kill_tree, resident_tree_memory


def gpu_config():
    path = project_path(os.environ.get("VISIONINSPECT_GPU_CONFIG", "configs/gpu.yaml"))
    value = yaml.safe_load(path.read_text(encoding="utf8"))
    if "VISIONINSPECT_VLLM_PYTHON" in os.environ:
        value["python"] = os.environ["VISIONINSPECT_VLLM_PYTHON"]
    if "VISIONINSPECT_GPU_ROOT" in os.environ:
        value["root"] = os.environ["VISIONINSPECT_GPU_ROOT"]
    if "VISIONINSPECT_GPU_ENABLED" in os.environ:
        value["enabled"] = os.environ["VISIONINSPECT_GPU_ENABLED"] == "1"
    if (
        value.get("schema_version") != 1
        or value["endpoint"] != "http://127.0.0.1:8001"
        or type(value["enabled"]) is not bool
    ):
        raise ValueError("Invalid GPU configuration or non-loopback endpoint")
    for key in (
        "idle_ttl_seconds",
        "startup_timeout_seconds",
        "request_timeout_seconds",
        "server_memory_limit_gib",
        "project_memory_limit_gib",
    ):
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], (int, float))
            or not math.isfinite(value[key])
            or value[key] <= 0
        ):
            raise ValueError(f"Invalid GPU limit: {key}")
    if type(value["max_model_calls"]) is not int or value["max_model_calls"] not in (
        1,
        2,
        3,
    ):
        raise ValueError("GPU model call budget must be 1–3")
    return value


def gpu_assets_available(config):
    root = Path(config["root"])
    if not config["enabled"] or not Path(config["python"]).is_file():
        return False
    try:
        from scripts.pai_vllm import MODEL, NAMES, REVISION, WEIGHTS

        directory = root / "weights/qwen3-vl-8b-instruct"
        source = read_json(directory / "source.json")
        files = source.get("files", [])
        if (
            source.get("model") != MODEL
            or source.get("revision") != REVISION
            or source.get("license") != "Apache-2.0"
        ):
            return False
        if len(files) != len(NAMES) or {f["rfilename"] for f in files} != NAMES:
            return False
        for item in files:
            name = item["rfilename"]
            path = directory / name
            if (
                name in WEIGHTS
                and (item["size"], item.get("lfs", {}).get("sha256")) != WEIGHTS[name]
            ):
                return False
            if not path.is_file() or path.stat().st_size != item["size"]:
                return False
        return True  # Full SHA verification is performed by the startup child, not every health request.
    except (OSError, ValueError, KeyError, TypeError):
        return False


class VLLMRuntime:
    def __init__(self, directory, config=None):
        self.config = config or gpu_config()
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.process = None
        self.log = None
        self.last_idle = time.monotonic()
        self.leases = 0
        self.ready = False
        self.loading = False
        self.closed = False
        self.generation = 0
        self.failure = None
        self.peak_rss = 0
        self.peak_gpu_mib = None
        self.tracked = {}
        self.last_gpu_sample = 0
        self.stop_event = threading.Event()
        self.monitor = threading.Thread(
            target=self._monitor, name="vllm-idle-watch", daemon=True
        )
        self.monitor.start()

    def status(self):
        with self.lock:
            return dict(
                enabled=self.config["enabled"],
                assets_available=gpu_assets_available(self.config),
                ready=self.ready,
                loading=self.loading,
                pid=self.process.pid
                if self.process and self.process.poll() is None
                else None,
                generation=self.generation,
                active_leases=self.leases,
                idle_ttl_seconds=self.config["idle_ttl_seconds"],
                idle_seconds=0
                if self.leases
                else max(0, time.monotonic() - self.last_idle),
                failure=self.failure,
                backend="vllm",
                model=self.config["model"],
                device="NVIDIA GPU / BF16",
                peak_process_tree_rss_bytes=self.peak_rss,
                peak_gpu_memory_mib=self.peak_gpu_mib,
            )

    def _record(self, reason):
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "status.json", dict(self.status(), event=reason))

    def _stop(self, reason):
        process = self.process
        if process:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        kill_tree(psutil.Process(process.pid))
                    except psutil.NoSuchProcess:
                        pass
            for member in self.tracked.values():
                try:
                    if member.is_running():
                        kill_tree(member)
                except psutil.NoSuchProcess:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.ready = False
        self.loading = False
        self.process = None
        self.tracked = {}
        if self.log:
            self.log.close()
            self.log = None
        self._record(reason)

    def _spawn(self):
        if not gpu_assets_available(self.config):
            raise RuntimeError("GPU 服务未配置或本地模型资产缺失；不自动下载")
        if self.closed:
            raise RuntimeError("GPU 服务正在关闭")
        # Do not adopt, replace or terminate a server owned by somebody else.
        with socket.socket() as check:
            check.bind(("127.0.0.1", 8001))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / "bootstrap.log").open("a", encoding="utf8")
        command = [
            self.config["python"],
            "-B",
            str(ROOT / "scripts/vllm_managed.py"),
            "--root",
            self.config["root"],
            "--parent-pid",
            str(os.getpid()),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf8",
            env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1"),
        )
        self.generation += 1
        self.loading = True
        self.ready = False
        self.failure = None
        self.tracked = {}
        write_json(
            self.directory / "command.json",
            dict(command=command, pid=self.process.pid, generation=self.generation),
        )

    @contextlib.contextmanager
    def acquire(self, progress=None):
        started = time.monotonic()
        with self.lock:
            self.leases += 1
        try:
            with self.lock:
                if self.process is None or self.process.poll() is not None:
                    if self.process is not None:
                        self._stop("previous_exit")
                    self._spawn()
            if progress:
                progress(dict(stage="waiting_vlm", trace=[], loading=True))
            with httpx.Client(
                timeout=0.75, trust_env=False, follow_redirects=False
            ) as client:
                while True:
                    with self.lock:
                        if (
                            self.closed
                            or self.process is None
                            or self.process.poll() is not None
                        ):
                            raise RuntimeError(
                                self.failure or "vLLM 启动或运行失败，请检查 GPU 日志"
                            )
                        ready = self.ready
                    if ready:
                        break
                    if (
                        time.monotonic() - started
                        > self.config["startup_timeout_seconds"]
                    ):
                        raise RuntimeError("vLLM 冷启动超过配置时限")
                    try:
                        response = client.get(self.config["endpoint"] + "/v1/models")
                        if response.status_code == 200 and any(
                            m.get("id") == self.config["model"]
                            for m in response.json().get("data", [])
                        ):
                            with self.lock:
                                self.ready = True
                                self.loading = False
                                self._record("ready")
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    self.stop_event.wait(0.25)
            yield dict(
                startup_seconds=time.monotonic() - started, generation=self.generation
            )
        except Exception as error:
            with self.lock:
                self.failure = f"{type(error).__name__}: {error}"
                self._stop("startup_or_task_failure")
            raise
        finally:
            with self.lock:
                self.leases -= 1
                self.last_idle = time.monotonic()

    def _monitor(self):
        while not self.stop_event.wait(0.5):
            with self.lock:
                if self.process is None:
                    continue
                if self.process.poll() is not None:
                    self.failure = f"vLLM 进程退出：{self.process.returncode}"
                    self._stop("unexpected_exit")
                    continue
                try:
                    memory = resident_tree_memory(
                        psutil.Process(self.process.pid), self.tracked
                    )
                    self.peak_rss = max(memory, self.peak_rss)
                    project = resident_tree_memory(psutil.Process(os.getpid()))
                    if (
                        memory > self.config["server_memory_limit_gib"] * 1024**3
                        or project > self.config["project_memory_limit_gib"] * 1024**3
                    ):
                        self.failure = "GPU 服务或项目进程树内存超过配置预算"
                        self._stop("memory_limit")
                        continue
                except psutil.NoSuchProcess:
                    continue
                if time.monotonic() - self.last_gpu_sample > 5:
                    self.last_gpu_sample = time.monotonic()
                    try:
                        sample = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-compute-apps=pid,used_memory",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=2,
                        )
                        owned = set(self.tracked) | {self.process.pid}
                        total = 0
                        observed = False
                        for line in sample.stdout.splitlines():
                            pid, amount = [part.strip() for part in line.split(",")]
                            if int(pid) in owned:
                                total += int(amount)
                                observed = True
                        if observed:
                            self.peak_gpu_mib = max(self.peak_gpu_mib or 0, total)
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        pass
                if (
                    not self.leases
                    and time.monotonic() - self.last_idle
                    > self.config["idle_ttl_seconds"]
                ):
                    self._stop("idle_ttl")

    def close(self):
        with self.lock:
            self.closed = True
            if self.process:
                self._stop("application_closed")
        self.stop_event.set()
        self.monitor.join(timeout=2)

    def recycle(self, reason):
        with self.lock:
            if self.process:
                self._stop(reason)
