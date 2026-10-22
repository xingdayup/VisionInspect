"""Platform-runtime child; exits and reaps vLLM if the owning API disappears."""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
import psutil
from pai_vllm import (
    configure,
    server_command,
    server_environment,
    validate_source,
    write,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Managed GPU runtime requires Linux")
    root = args.root.resolve()
    configure(root, offline=True)
    owner = psutil.Process(args.parent_pid)
    owner_created = owner.create_time()
    stop = threading.Event()

    def eof():
        for _ in sys.stdin:
            pass
        stop.set()

    threading.Thread(target=eof, daemon=True).start()
    # The finite probe exits before serving, releasing its temporary CUDA context.
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve().parent / "pai_vllm.py"),
            "check",
            "--root",
            str(root),
        ],
        check=True,
        timeout=90,
    )
    validate_source(root)
    if not owner.is_running() or owner.create_time() != owner_created or stop.is_set():
        return
    process = None
    tracked = {}
    started = time.monotonic()
    failure = None
    log = (root / "logs/vllm-server.log").open("a", encoding="utf8")
    try:
        command = server_command(root)
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=server_environment(),
        )
        write(
            root / "logs/managed-command.json",
            dict(command=command, pid=process.pid, parent_pid=args.parent_pid),
        )
        while process.poll() is None and not stop.wait(0.5):
            try:
                if not owner.is_running() or owner.create_time() != owner_created:
                    break
                for member in psutil.Process(process.pid).children(recursive=True):
                    tracked[member.pid] = member
            except psutil.NoSuchProcess:
                break
        if process.poll() is not None and process.returncode:
            failure = f"vLLM exit {process.returncode}"
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            for member in tracked.values():
                try:
                    if member.is_running():
                        member.kill()
                except psutil.NoSuchProcess:
                    pass
        log.close()
        write(
            root / "logs/managed-execution.json",
            dict(
                seconds=time.monotonic() - started,
                process_exit_code=process.returncode if process else None,
                error=failure,
            ),
        )
    if failure:
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
