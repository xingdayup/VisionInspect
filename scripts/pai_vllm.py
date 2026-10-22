"""PAI A10 bootstrap: reuse preinstalled packages, prepare pinned assets, serve locally."""

import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener, urlopen

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
WEIGHTS = {
    "model-00001-of-00004.safetensors": (
        4902275944,
        "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa",
    ),
    "model-00002-of-00004.safetensors": (
        4915962496,
        "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5",
    ),
    "model-00003-of-00004.safetensors": (
        4999831048,
        "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192",
    ),
    "model-00004-of-00004.safetensors": (
        2716270024,
        "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5",
    ),
}
NAMES = set(WEIGHTS) | {
    "README.md",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}
EXPECTED = {
    "vllm": "0.27.1",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.16.1",
    "flashinfer-python": "0.6.16.post3",
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf8",
    )
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text("utf8"))


def configure(root, offline=False):
    for name, relative in {
        "HF_HOME": ".cache/huggingface",
        "HF_HUB_CACHE": ".cache/huggingface/hub",
        "TORCH_HOME": ".cache/torch",
        "XDG_CACHE_HOME": ".cache",
        "TRITON_CACHE_DIR": ".cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": ".cache/inductor",
        "CUDA_CACHE_PATH": ".cache/cuda",
        "FLASHINFER_WORKSPACE_BASE": ".cache/flashinfer",
        "TMPDIR": ".tmp",
    }.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(
        HF_ENDPOINT="https://huggingface.co",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_XET="1",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1" if offline else "0",
    )


def check(root):
    if sys.platform != "linux":
        raise RuntimeError("Run check on the PAI Linux machine, not Windows")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if any(versions[n].split("+")[0] != v for n, v in EXPECTED.items()):
        raise RuntimeError(
            f"Preinstalled versions changed; review before continuing: {versions}"
        )
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA/BF16 unavailable")
    importlib.import_module("vllm._C_stable_libtorch")
    from vllm.model_executor.models import ModelRegistry

    if "Qwen3VLForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        raise RuntimeError("Qwen3-VL architecture not registered")
    packages = sorted(
        (d.metadata["Name"], d.version)
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    )
    report = dict(
        passed=True,
        versions=versions,
        python=sys.version,
        executable=sys.executable,
        cuda_build=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        compute_capability=list(torch.cuda.get_device_capability(0)),
        packages=packages,
        scope="Preinstalled package/native import and architecture registration only; model kernels still unverified",
        packages_installed=False,
    )
    write(root / "logs/runtime-check.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def inspect_metadata(data):
    if (
        data["id"] != MODEL
        or data["sha"] != REVISION
        or data.get("gated")
        or data["cardData"]["license"] != "apache-2.0"
    ):
        raise ValueError("Model revision, license or access changed")
    files = [f for f in data["siblings"] if f["rfilename"] in NAMES]
    if {f["rfilename"] for f in files} != NAMES:
        raise ValueError("Required asset list differs")
    for f in files:
        if f["rfilename"] in WEIGHTS:
            size, digest = WEIGHTS[f["rfilename"]]
            if f["size"] != size or f["lfs"]["sha256"] != digest:
                raise ValueError("Weight size/hash differs")
    if sum(f["size"] for f in files) > 18_000_000_000:
        raise ValueError("18 GB asset budget exceeded")
    return files


def preflight(root, minimum_free_gib=50):
    with urlopen(
        f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true",
        timeout=30,
    ) as response:
        body = response.read(1024**2 + 1)
    if len(body) > 1024**2:
        raise ValueError("Metadata too large")
    data = json.loads(body)
    files = inspect_metadata(data)
    usage = shutil.disk_usage(root)
    required = int(minimum_free_gib * 1024**3)
    if required < sum(f["size"] for f in files) + 8 * 1024**3:
        raise ValueError(
            "Model disk budget must cover all assets plus an 8 GiB reserve"
        )
    report = dict(
        model=MODEL,
        revision=REVISION,
        license="Apache-2.0",
        files=files,
        artifact_bytes=sum(f["size"] for f in files),
        free_disk_bytes=usage.free,
        minimum_free_disk_bytes=required,
        minimum_reserve_bytes=8 * 1024**3,
        model_weights_downloaded=False,
        budget_note="Artifact bytes, not measured network traffic; retain room for cache, environment and outputs",
    )
    write(root / "logs/model-preflight.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2),
        flush=True,
    )
    if usage.free < required:
        raise RuntimeError("Model disk budget exceeded; stop before model download")


def verified(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    lfs = item.get("lfs")
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest() == (lfs["sha256"] if lfs else item["blobId"])


def model_dir(root):
    return root / "weights/qwen3-vl-8b-instruct"


def validate_source(root):
    directory = model_dir(root)
    source = read(directory / "source.json")
    if (
        source.get("model") != MODEL
        or source.get("revision") != REVISION
        or source.get("license") != "Apache-2.0"
    ):
        raise ValueError("Asset source mismatch")
    files = source.get("files", [])
    if len(files) != len(NAMES) or {f.get("rfilename") for f in files} != NAMES:
        raise ValueError("Required asset list differs")
    for item in files:
        name = item["rfilename"]
        if (
            name in WEIGHTS
            and (item["size"], item.get("lfs", {}).get("sha256")) != WEIGHTS[name]
        ):
            raise ValueError("Pinned weight metadata differs")
        if not verified(directory / name, item):
            raise ValueError(f"Asset missing/corrupt: {name}")
    return source


def resolve_root(value):
    """Resolve the GPU asset root from --root, then VISIONINSPECT_GPU_ROOT.

    There is deliberately no built-in default path: a stale PAI path would
    create an empty directory on a fresh machine's system disk and only fail
    much later, after environments and weights had been written elsewhere.
    """
    if value is not None:
        return Path(value).resolve()
    configured = os.environ.get("VISIONINSPECT_GPU_ROOT")
    if not configured:
        raise ValueError(
            "Set VISIONINSPECT_GPU_ROOT or pass --root; on AutoDL run: "
            "source scripts/env_autodl.sh"
        )
    return Path(configured).resolve()


def download(root, minimum_free_gib=50):
    marker = root / "logs/runtime-check.json"
    if not marker.is_file():
        raise RuntimeError(
            "runtime-check.json is missing: install the vLLM environment first "
            "(bash scripts/setup_vllm.sh runs the native check for you)"
        )
    if not read(marker).get("passed"):
        raise RuntimeError(
            "Runtime check did not pass; re-run bash scripts/setup_vllm.sh"
        )
    preflight(root, minimum_free_gib)
    report = read(root / "logs/model-preflight.json")
    from huggingface_hub import hf_hub_download

    directory = model_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    for item in report["files"]:
        path = directory / item["rfilename"]
        if not verified(path, item):
            hf_hub_download(
                repo_id=MODEL,
                filename=item["rfilename"],
                revision=REVISION,
                local_dir=directory,
                endpoint="https://huggingface.co",
            )
        if not verified(path, item):
            raise ValueError(
                f"Checksum failed: {item['rfilename']}; retained for diagnosis"
            )
        if shutil.disk_usage(root).free < 8 * 1024**3:
            raise RuntimeError("Model download reached the 8 GiB disk reserve")
        print("VERIFIED", item["rfilename"], item["size"], flush=True)
    write(
        directory / "source.json",
        dict(
            model=MODEL,
            revision=REVISION,
            license="Apache-2.0",
            artifact_bytes=report["artifact_bytes"],
            files=report["files"],
        ),
    )
    print("ASSETS_READY", directory, flush=True)


def server_command(root):
    return [
        sys.executable,
        "-B",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_dir(root)),
        "--served-model-name",
        "visioninspect-vlm",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--dtype",
        "bfloat16",
        "--seed",
        "42",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "2048",
        "--gpu-memory-utilization",
        "0.85",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"image": 3, "video": 0}),
        "--mm-processor-kwargs",
        json.dumps({"size": {"shortest_edge": 65536, "longest_edge": 262144}}),
        "--mm-processor-cache-gb",
        "0.25",
        "--no-trust-remote-code",
    ]


def cuda_toolkit_bin():
    """Return the CUDA toolkit bin directory, or None when none is installed."""
    for candidate in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        if candidate and (Path(candidate) / "bin/nvcc").is_file():
            return str(Path(candidate) / "bin")
    return None


def server_environment():
    """Build the child environment for the vLLM server.

    The venv is never activated, so its bin directory must be placed on PATH
    explicitly: flashinfer JIT-compiles its sampling kernels and calls "ninja"
    by name, which otherwise raises FileNotFoundError at first inference. The
    CUDA toolkit bin directory is included for the same reason.
    """
    guard = str(Path(__file__).resolve().parent / "runtime_guard")
    # sys.executable is a symlink to the base interpreter, so resolving it would
    # put the base bin directory on PATH instead of the venv that owns ninja.
    venv_bin = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    parts = [str(venv_bin)]
    toolkit = cuda_toolkit_bin()
    if toolkit:
        parts.append(toolkit)
    inherited = os.environ.get("PATH", "")
    if inherited:
        parts.append(inherited)
    return dict(
        os.environ,
        VISIONINSPECT_VLLM_OFFLINE="1",
        VLLM_HOST_IP="127.0.0.1",
        PATH=os.pathsep.join(parts),
        PYTHONPATH=guard + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )


def serve(root, seconds):
    if sys.platform != "linux":
        raise RuntimeError("Serve must run on the PAI Linux GPU machine")
    check(root)
    validate_source(root)
    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", 8001))
    import psutil

    configure(root, offline=True)
    command = server_command(root)
    started = time.monotonic()
    write(
        root / "logs/serve-command.json",
        dict(
            command=command,
            maximum_seconds=seconds,
            startup_seconds=300,
            host_memory_limit_bytes=24 * 1024**3,
            debug_service=True,
            idle_ttl_implemented=False,
        ),
    )
    log_path = root / "logs/vllm-server.log"
    failure = None
    peak = 0
    ready = False
    process = None
    reader = None
    tracked = {}
    local_http = build_opener(ProxyHandler({}))
    log = log_path.open("w", encoding="utf8")
    try:
        if True:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf8",
                errors="replace",
                start_new_session=True,
                env=server_environment(),
            )

            def relay():
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)

            reader = threading.Thread(target=relay, daemon=True)
            reader.start()
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                memory = 0
                try:
                    for member in [monitored] + monitored.children(recursive=True):
                        tracked[member.pid] = member
                        try:
                            memory += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                peak = max(peak, memory)
                elapsed = time.monotonic() - started
                if memory > 24 * 1024**3:
                    failure = "Host process-tree RSS exceeds 24 GiB"
                    break
                if elapsed > seconds:
                    failure = "Finite debug session ended"
                    break
                if not ready:
                    try:
                        with local_http.open(
                            "http://127.0.0.1:8001/health", timeout=0.5
                        ) as response:
                            ready = response.status == 200
                    except OSError:
                        pass
                    if not ready and elapsed > 300:
                        failure = "Startup exceeded 300 seconds"
                        break
                time.sleep(0.5)
            if process.poll() is not None and process.returncode != 0:
                failure = f"vLLM exit code {process.returncode}"
    except KeyboardInterrupt:
        failure = "Stopped by user"
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            for member in tracked.values():
                try:
                    if member.is_running():
                        member.kill()
                except psutil.NoSuchProcess:
                    pass
        if reader:
            reader.join(timeout=5)
        log.close()
        write(
            root / "logs/serve-execution.json",
            dict(
                command=command,
                ready_observed=ready,
                seconds=time.monotonic() - started,
                peak_process_tree_rss_bytes=peak,
                process_exit_code=process.returncode if process else None,
                stop_reason=failure,
                gpu_memory="Inspect vLLM log or nvidia-smi; not measured by this supervisor",
                model_quality_evaluated=False,
                idle_ttl_implemented=False,
            ),
        )
    if not ready or failure not in (
        None,
        "Stopped by user",
        "Finite debug session ended",
    ):
        raise RuntimeError(failure or "Server never became ready")


def probe(root, image):
    if not image.is_file() or image.stat().st_size > 10 * 1024**2:
        raise ValueError("Provide a real PNG/JPEG/WebP up to 10 MiB")
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(image.suffix.lower())
    if mime is None:
        raise ValueError("Unsupported image extension")
    content = base64.b64encode(image.read_bytes()).decode("ascii")
    payload = dict(
        model="visioninspect-vlm",
        temperature=0,
        max_tokens=128,
        messages=[
            dict(
                role="user",
                content=[
                    dict(
                        type="image_url",
                        image_url=dict(url=f"data:{mime};base64,{content}"),
                    ),
                    dict(
                        type="text",
                        text="Describe only visible irregularities at the bottle rim. Do not infer causes or product safety. If unclear, say unknown.",
                    ),
                ],
            )
        ],
    )
    request = Request(
        "http://127.0.0.1:8001/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with build_opener(ProxyHandler({})).open(request, timeout=90) as response:
        result = json.load(response)
    if not result.get("choices", [{}])[0].get("message", {}).get("content"):
        raise ValueError("No generated text")
    report = dict(
        passed=True,
        image=str(image.resolve()),
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        seconds=time.monotonic() - started,
        response=result,
        scope="Image API engineering smoke only",
        semantic_accuracy_evaluated=False,
        release_authorized=False,
    )
    write(root / "logs/probe-result.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["check", "preflight", "download", "command", "serve", "probe"],
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="GPU asset root; falls back to VISIONINSPECT_GPU_ROOT",
    )
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--minimum-free-gib",
        type=int,
        default=50,
        help="Explicitly approved model disk budget; retains at least 8 GiB",
    )
    args = parser.parse_args()
    try:
        root = resolve_root(args.root)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.seconds <= 3600:
        parser.error("--seconds must be between 1 and 3600")
    if args.command == "probe" and args.image is None:
        parser.error("probe requires --image")
    if sys.platform != "linux":
        parser.error("Run GPU preparation commands on the PAI Linux machine")
    root.mkdir(parents=True, exist_ok=True)
    configure(root, offline=args.command in {"check", "serve", "probe"})
    started = time.monotonic()
    error = None
    try:
        if args.command == "check":
            check(root)
        elif args.command == "preflight":
            preflight(root, args.minimum_free_gib)
        elif args.command == "download":
            download(root, args.minimum_free_gib)
        elif args.command == "command":
            print(json.dumps(server_command(root), ensure_ascii=False, indent=2))
        elif args.command == "serve":
            serve(root, args.seconds)
        elif args.command == "probe":
            probe(root, args.image)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        write(
            root / f"logs/{args.command}-command-execution.json",
            dict(
                command=sys.argv,
                error=error,
                seconds=time.monotonic() - started,
                exit_code=1 if error else 0,
                packages_installed=False,
                model_download_requested=args.command == "download",
            ),
        )
