"""Protocol and process supervision only; no Qwen weights or GPU inference here."""

import json
import os
import socket
import subprocess
import sys
import time

import httpx
import psutil
import pytest
from PIL import Image

from scripts.pai_vllm import MODEL, REVISION, validate_source
from visioninspect.common import offline, read_json, write_json
from visioninspect.tools import ToolFailure
from visioninspect.vllm_client import VLLMClient
from visioninspect.vllm_runtime import VLLMRuntime, gpu_assets_available, gpu_config


def test_multimodal_json_request_and_actual_input_audit(tmp_path, monkeypatch):
    received = []
    value = dict(
        observations=[
            dict(
                text="协议测试观察",
                evidence_ids=["original", "reference:1"],
                region_id=None,
                finding="unclear",
            )
        ],
        action="finish",
        verdict="unknown",
        region_id=None,
        reference_id=None,
        summary="协议测试，不是模型结论",
    )

    def response(request):
        received.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=dict(
                choices=[dict(message=dict(content=json.dumps(value)))],
                usage=dict(completion_tokens=20),
            ),
        )

    original = httpx.Client

    def client(*args, **kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return original(*args, **kwargs, transport=httpx.MockTransport(response))

    monkeypatch.setattr(httpx, "Client", client)
    inputs = []
    for name, color in [("original", (200, 20, 20)), ("reference:1", (20, 200, 20))]:
        path = tmp_path / (name.replace(":", "_") + ".png")
        Image.new("RGB", (900, 600), color).save(path)
        inputs.append(dict(id=name, path=path))
    decision, metrics = VLLMClient(gpu_config()).decide(
        {"fixture": "unit-test-only"}, inputs, tmp_path / "round_1"
    )
    assert decision.verdict == "unknown"
    content = received[0]["messages"][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 2
    assert received[0]["response_format"]["type"] == "json_schema"
    audit = read_json(tmp_path / "round_1/request.json")
    assert (
        audit["images"][0]["id"] == "original"
        and audit["images"][1]["id"] == "reference:1"
    )
    assert audit["images"][0]["size"] == [512, 341]
    assert metrics["input_images"][0]["path"] == "vlm_input/round_1/image_1.png"


def test_vllm_client_rejects_remote_url_and_bad_response(tmp_path, monkeypatch):
    config = gpu_config()
    config["endpoint"] = "https://example.com"
    with pytest.raises(ValueError):
        VLLMClient(config)
    path = tmp_path / "fixture.png"
    Image.new("RGB", (128, 128)).save(path)
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *a, **k: original(
            *a,
            **k,
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"choices": []})
            ),
        ),
    )
    with pytest.raises(ToolFailure, match="结构化"):
        VLLMClient(gpu_config()).decide(
            {}, [dict(id="original", path=path)], tmp_path / "round_1"
        )


def test_offline_guard_only_allows_explicit_loopback_port():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    try:
        with offline(allow_loopback_ports=(port,)):
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                pass
            with pytest.raises(RuntimeError):
                socket.create_connection(("example.com", 443))
            with pytest.raises(RuntimeError):
                socket.create_connection(("127.0.0.1", port + 1))
        with offline():
            with pytest.raises(RuntimeError):
                socket.create_connection(("127.0.0.1", port))
    finally:
        listener.close()


def test_unprepared_runtime_fails_without_spawning_or_downloading(tmp_path):
    config = gpu_config()
    config.update(enabled=False)
    runtime = VLLMRuntime(tmp_path, config)
    try:
        with pytest.raises(RuntimeError, match="不自动下载"):
            with runtime.acquire():
                pass
        assert (
            runtime.status()["pid"] is None and runtime.status()["active_leases"] == 0
        )
    finally:
        runtime.close()


def test_source_json_alone_cannot_advertise_model_availability(tmp_path):
    config = gpu_config()
    config.update(enabled=True, root=str(tmp_path), python=sys.executable)
    write_json(
        tmp_path / "weights/qwen3-vl-8b-instruct/source.json",
        dict(model=MODEL, revision=REVISION, license="Apache-2.0", files=[]),
    )
    assert not gpu_assets_available(config)


def test_real_child_lease_prevents_ttl_then_reaps_and_reloads(tmp_path, monkeypatch):
    # A tiny HTTP fixture proves lifetime mechanics, not vLLM native compatibility.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    code = """from http.server import HTTPServer,BaseHTTPRequestHandler
import json,sys
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  data=json.dumps({'data':[{'id':'visioninspect-vlm'}]}).encode();self.send_response(200);self.end_headers();self.wfile.write(data)
 def log_message(self,*args):pass
HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
"""
    config = gpu_config()
    config.update(
        endpoint=f"http://127.0.0.1:{port}",
        idle_ttl_seconds=0.2,
        startup_timeout_seconds=10,
    )
    monkeypatch.setattr(
        "visioninspect.vllm_runtime.gpu_assets_available", lambda c: True
    )

    class TestRuntime(VLLMRuntime):
        def _spawn(self):
            self.process = subprocess.Popen(
                [sys.executable, "-B", "-c", code, str(port)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.generation += 1
            self.loading = True
            self.ready = False

    runtime = TestRuntime(tmp_path, config)
    try:
        with runtime.acquire():
            first = runtime.process.pid
            time.sleep(0.7)
            assert (
                runtime.process.poll() is None
                and runtime.status()["active_leases"] == 1
            )
        deadline = time.monotonic() + 5
        while runtime.process is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert runtime.process is None and not psutil.pid_exists(first)
        with runtime.acquire():
            assert runtime.generation == 2 and runtime.process.pid != first
    finally:
        runtime.close()


def test_pinned_gpu_asset_manifest_rejects_missing_and_path_escape(tmp_path):
    directory = tmp_path / "weights/qwen3-vl-8b-instruct"
    directory.mkdir(parents=True)
    write_json(
        directory / "source.json",
        dict(
            model=MODEL,
            revision=REVISION,
            license="Apache-2.0",
            files=[{"rfilename": "../escape"}],
        ),
    )
    with pytest.raises(ValueError, match="asset list"):
        validate_source(tmp_path)


def test_spawned_runtime_guard_blocks_external_python_connections():
    from visioninspect.common import ROOT

    code = "import socket; socket.socket().connect(('203.0.113.1',443))"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=dict(
            os.environ,
            VISIONINSPECT_VLLM_OFFLINE="1",
            PYTHONPATH=str(ROOT / "scripts/runtime_guard"),
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0 and "denies external network" in result.stderr
