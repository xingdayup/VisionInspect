"""Opt-in Python socket guard inherited by vLLM's spawned Python workers."""

import os

if os.environ.get("VISIONINSPECT_VLLM_OFFLINE") == "1":
    import socket

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def allowed(address):
        # Local Unix-domain IPC and loopback TCP are required by the engine.
        return isinstance(address, (str, bytes)) or (
            isinstance(address, tuple)
            and address[0] in {"127.0.0.1", "::1", "localhost"}
        )

    def connect(instance, address):
        if not allowed(address):
            raise PermissionError(
                "VisionInspect vLLM runtime denies external network; prepare local assets explicitly"
            )
        return original_connect(instance, address)

    def connect_ex(instance, address):
        if not allowed(address):
            raise PermissionError(
                "VisionInspect vLLM runtime denies external network; prepare local assets explicitly"
            )
        return original_connect_ex(instance, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
