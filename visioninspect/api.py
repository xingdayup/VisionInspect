"""REST API for the local research workstation. Start one server worker only.

Route handlers return plain mappings; FastAPI serializes them as-is. Every
endpoint that mutates state goes through TaskStore, which owns the SQLite
queue, so the API layer stays free of scheduling logic.
"""

from __future__ import annotations

import csv
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from io import BytesIO, StringIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import ROOT, configure_environment, read_json
from .review_contracts import ReviewDecision
from .schemas import TOOL_CONTRACTS
from .service import TaskStore, cuda_available, model_health, registry
from .vllm_runtime import gpu_assets_available, gpu_config

MAX_IMAGE_BYTES = 10 * 1024**2
MAX_IMAGE_PIXELS = 16_777_216
ACCEPTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}


class BatchRequest(BaseModel):
    """Batch creation payload; unknown fields are rejected rather than ignored."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    category: str = "bottle"
    model: str = "patchcore"
    mode: str = "agent_review"

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("批次名称不能为空")
        return value.strip()


class ReviewRequest(BaseModel):
    """Human review payload; the note must carry a real explanation."""

    model_config = ConfigDict(extra="forbid")
    decision: str = Field(
        pattern=r"^(normal_candidate|anomaly_candidate|manual_review)$"
    )
    note: str = Field(min_length=1, max_length=2000)

    @field_validator("note")
    @classmethod
    def meaningful_note(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("复核说明不能为空")
        return value.strip()


class UploadBodyLimit:
    """Cap multipart bodies before parser spooling; all temp files remain in the project."""

    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or not (
                scope["path"] == "/api/tasks"
                or re.fullmatch(r"/api/batches/[0-9a-f]{32}/items", scope["path"])
            )
        ):
            return await self.app(scope, receive, send)
        limit = MAX_IMAGE_BYTES + 64 * 1024
        chunks: list[bytes] = []
        total = 0
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = 0
        if length > limit:
            return await JSONResponse(
                {"detail": "上传请求超过 10 MiB 图像上限"}, status_code=413
            )(scope, receive, send)
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > limit:
                return await JSONResponse(
                    {"detail": "上传请求超过 10 MiB 图像上限"}, status_code=413
                )(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        del chunks
        delivered = False

        async def replay() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return dict(type="http.request", body=body, more_body=False)
            return await receive()

        return await self.app(scope, replay, send)


def create_app(state_directory: Path | None = None) -> FastAPI:
    """Build the FastAPI application, optionally pinned to a test state directory."""
    configure_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.store = TaskStore(state_directory)
        try:
            yield
        finally:
            application.state.store.close()

    application = FastAPI(
        title="VisionInspect-Agent", version="0.5.0", lifespan=lifespan
    )
    application.add_middleware(UploadBodyLimit)

    def store(request: Request) -> TaskStore:
        """Return the lazily created store, so importing the app costs nothing."""
        if not hasattr(request.app.state, "store"):
            request.app.state.store = TaskStore(state_directory)
        return request.app.state.store

    def get_task(request: Request, task_id: str) -> dict:
        """Load one task, mapping storage errors onto HTTP semantics."""
        try:
            return store(request).get(task_id)
        except KeyError:
            raise HTTPException(404, detail="任务不存在")
        except (OSError, ValueError):
            raise HTTPException(
                409, detail="任务记录损坏或无法读取，请检查日志或重新提交"
            )

    def get_batch(request: Request, batch_id: str) -> dict:
        try:
            return store(request).get_batch(batch_id)
        except KeyError:
            raise HTTPException(404, detail="批次不存在")

    def validate_selection(category: str, model: str, mode: str) -> None:
        """Reject unknown workflows, unregistered models and missing assets."""
        if mode not in {"baseline", "agent_review"}:
            raise HTTPException(422, detail="只支持 GPU 单模型检测与 Agent 多步复核")
        if (
            category not in registry()
            or model not in registry()[category]
            or model == "label"
        ):
            raise HTTPException(422, detail="未注册的产品类别或模型")
        selected = next(
            m
            for m in model_health()
            if m["category"] == category and m["model"] == model
        )
        if not selected["available"]:
            raise HTTPException(503, detail=selected["reason"])
        if mode == "agent_review" and not gpu_assets_available(gpu_config()):
            raise HTTPException(
                503,
                detail="Agent GPU 服务未配置或模型资产缺失，请准备 PAI 服务或选择单模型检测",
            )

    async def image_body(file: UploadFile) -> tuple[bytes, str]:
        """Read an upload under an explicit cap and reject unusable images early."""
        content = bytearray()
        while block := await file.read(256 * 1024):
            content.extend(block)
            if len(content) > MAX_IMAGE_BYTES:
                raise HTTPException(413, detail="图像超过 10 MiB 上限")
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format not in ACCEPTED_IMAGE_FORMATS:
                    raise ValueError("只支持 PNG、JPEG、WebP")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("图像超过 1600 万像素上限")
                image.verify()
        except UnidentifiedImageError:
            raise HTTPException(
                422, detail="图像无法识别，请检查文件内容是否损坏或格式是否正确"
            )
        except OSError:
            raise HTTPException(422, detail="图像无法完整解码，请重新导出图片后上传")
        except (ValueError, Image.DecompressionBombError) as error:
            raise HTTPException(422, detail=f"图像不可用：{error}")
        return bytes(content), (file.filename or "image").replace("\\", "/").split("/")[
            -1
        ][:160]

    @application.get("/health")
    def health() -> dict:
        """Report capability and asset presence; never starts a model."""
        models = model_health()
        gpu = gpu_config()
        available = cuda_available()
        return dict(
            status="ok",
            phase="gpu-only",
            inference_service=available and any(m["available"] for m in models),
            cuda_available=available,
            models=models,
            device="CUDA",
            precision="FP32",
            worker_concurrency=1,
            paid_api=False,
            agent_review_available=available and gpu_assets_available(gpu),
            vlm_backend="vllm" if gpu["enabled"] else "disabled",
            weights_present=(
                ROOT / "weights/resnet18.a1_in1k/model.safetensors"
            ).is_file(),
            dataset_manifest_present=(ROOT / "data/manifests/bottle.json").is_file(),
        )

    @application.get("/ready")
    def ready() -> Response:
        """Return 200 only when inference is actually usable, 503 otherwise."""
        value = health()
        return JSONResponse(
            value, status_code=200 if value["inference_service"] else 503
        )

    @application.get("/api/models")
    def models() -> list[dict]:
        return model_health()

    @application.get("/api/runtime")
    def runtime(request: Request) -> dict:
        """Expose supervision state for the inference worker and the vLLM owner."""
        return dict(store(request).runtime.status(), gpu=store(request).gpu.status())

    @application.get("/api/tools/schemas")
    def schemas() -> dict:
        """Publish the tool contracts the review loop is allowed to call."""
        result = {
            name: dict(
                input=a.model_json_schema(),
                output=b.model_json_schema(),
                service_implemented=True,
            )
            for name, (a, b) in TOOL_CONTRACTS.items()
        }
        result["review_evidence"] = dict(
            input=dict(type="object", description="当前图片证据与有界工作流上下文"),
            output=ReviewDecision.model_json_schema(),
            service_implemented=True,
        )
        return result

    @application.post("/api/tasks", status_code=202)
    async def create_task(
        request: Request,
        file: UploadFile = File(...),
        category: str = Form("bottle"),
        model: str = Form("patchcore"),
        mode: str = Form("agent_review"),
    ) -> dict:
        """Validate an upload and enqueue one inspection; no client paths are trusted."""
        content, filename = await image_body(file)
        validate_selection(category, model, mode)
        if not cuda_available():
            raise HTTPException(503, detail="CUDA 不可用；本版本不支持 CPU 推理")
        try:
            return store(request).create(content, filename, category, model, mode)
        except OverflowError as error:
            raise HTTPException(429, detail=str(error))

    @application.get("/api/tasks")
    def tasks(
        request: Request,
        status: str | None = None,
        state: str | None = None,
        q: str = "",
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ) -> dict:
        return store(request).list(status, state, q[:160], limit, offset)

    @application.get("/api/tasks/{task_id}")
    def task(request: Request, task_id: str) -> dict:
        return get_task(request, task_id)

    @application.get("/api/tasks/{task_id}/report")
    def report(request: Request, task_id: str) -> Response:
        """Download the full report; unfinished or missing reports never stream."""
        value = get_task(request, task_id)
        if value["status"] not in {"completed", "failed"}:
            raise HTTPException(409, detail="报告尚未生成")
        if not isinstance(value["report"], dict):
            raise HTTPException(410, detail="报告文件已缺失，无法导出；请重新检测")
        result = dict(value["report"], human_review=value["review"])
        return JSONResponse(
            result,
            headers={
                "Content-Disposition": f'attachment; filename="inspection-{task_id}.json"'
            },
        )

    @application.post("/api/tasks/{task_id}/review")
    def review(request: Request, task_id: str, value: ReviewRequest) -> dict:
        """Append a human record; the original model report is never rewritten."""
        get_task(request, task_id)
        try:
            return store(request).review(task_id, value.decision, value.note.strip())
        except ValueError as error:
            raise HTTPException(409, detail=str(error))

    @application.post("/api/tasks/{task_id}/retry", status_code=202)
    def retry(request: Request, task_id: str) -> dict:
        """Create a linked attempt instead of mutating the failed one."""
        get_task(request, task_id)
        try:
            return store(request).retry(task_id)
        except ValueError as error:
            raise HTTPException(409, detail=str(error))
        except OverflowError as error:
            raise HTTPException(429, detail=str(error))

    @application.post("/api/batches", status_code=201)
    def create_batch(request: Request, value: BatchRequest) -> dict:
        validate_selection(value.category, value.model, value.mode)
        if not cuda_available():
            raise HTTPException(503, detail="CUDA 不可用；本版本不支持 CPU 推理")
        return store(request).create_batch(
            value.name, value.category, value.model, value.mode
        )

    @application.get("/api/batches")
    def batches(
        request: Request,
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ) -> dict:
        return store(request).list_batches(limit, offset)

    @application.get("/api/batches/{batch_id}")
    def batch(request: Request, batch_id: str) -> dict:
        return get_batch(request, batch_id)

    @application.post("/api/batches/{batch_id}/items", status_code=201)
    async def upload_batch_item(
        request: Request, batch_id: str, file: UploadFile = File(...)
    ) -> dict:
        """Stage one image into a draft batch and record per-file failures."""
        batch = get_batch(request, batch_id)
        try:
            content, filename = await image_body(file)
        except HTTPException as error:
            filename = (
                (file.filename or "image").replace("\\", "/").split("/")[-1][:160]
            )
            store(request).record_upload_error(batch_id, filename, str(error.detail))
            raise
        try:
            return store(request).create(
                content,
                filename,
                batch["category"],
                batch["model"],
                batch["mode"],
                batch_id,
            )
        except ValueError as error:
            raise HTTPException(409, detail=str(error))
        except OverflowError as error:
            raise HTTPException(429, detail=str(error))

    @application.post("/api/batches/{batch_id}/actions/{action}")
    def batch_action(request: Request, batch_id: str, action: str) -> dict:
        """Start, resume, pause or cancel a batch through its state machine."""
        get_batch(request, batch_id)
        try:
            return store(request).batch_action(batch_id, action)
        except ValueError as error:
            raise HTTPException(409, detail=str(error))

    @application.get("/api/batches/{batch_id}/report")
    def batch_report(
        request: Request,
        batch_id: str,
        format: str = Query("json", pattern="^(json|csv)$"),
    ) -> Response:
        """Export a batch as JSON or as an Excel-safe UTF-8 CSV."""
        batch = get_batch(request, batch_id)
        items = []
        for item in batch["items"]:
            try:
                items.append(store(request).get(item["id"]))
            except (OSError, ValueError):
                items.append(dict(item, artifact_error="报告损坏或缺失"))
        if format == "json":
            return JSONResponse(
                dict(schema_version=2, batch=batch, tasks=items),
                headers={
                    "Content-Disposition": f'attachment; filename="batch-{batch_id}.json"'
                },
            )
        stream = StringIO()
        writer = csv.writer(stream)
        writer.writerow(
            [
                "task_id",
                "filename",
                "status",
                "initial_state",
                "agent_state",
                "effective_state",
                "source",
                "score",
                "human_note",
                "retry_of",
            ]
        )
        for item in items:
            report = item.get("report") or {}
            effective = store(request).effective(item)
            cells = [
                item["id"],
                item["filename"],
                item["status"],
                (report.get("initial_result") or report.get("result") or {}).get(
                    "state", ""
                ),
                (report.get("result") or {}).get("state", ""),
                effective.get("state", ""),
                effective["source"],
                ((report.get("outputs") or {}).get("detection") or {}).get("score", ""),
                (item.get("review") or {}).get("note", ""),
                item.get("retry_of") or "",
            ]
            writer.writerow(
                [
                    "'" + cell
                    if isinstance(cell, str)
                    and cell
                    and cell.lstrip()[:1] in "=+-@\t\r\n"
                    else cell
                    for cell in cells
                ]
            )
        return Response(
            ("\ufeff" + stream.getvalue()).encode("utf8"),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="batch-{batch_id}.csv"'
            },
        )

    @application.get("/api/tasks/{task_id}/artifacts/{filename}")
    def artifact(request: Request, task_id: str, filename: str) -> Response:
        """Serve one whitelisted evidence file; paths are never taken from input."""
        get_task(request, task_id)
        if not re.fullmatch(
            r"(original|inspection_heatmap|inspection_overlay|reference_[12]|region_[1-4]|review_R[1-4]_ref_[12]|vlm_round_[1-3]_image_[1-3])\.png|inspection\.npy",
            filename,
        ):
            raise HTTPException(404, detail="证据文件不存在")
        path = store(request).directory(task_id) / "evidence" / filename
        vlm_image = re.fullmatch(r"vlm_round_([1-3])_image_([1-3])\.png", filename)
        if vlm_image:
            path = (
                store(request).directory(task_id)
                / "vlm_input"
                / f"round_{vlm_image[1]}"
                / f"image_{vlm_image[2]}.png"
            )
        if not path.is_file():
            raise HTTPException(404, detail="证据文件尚未生成或已缺失")
        return FileResponse(
            path,
            media_type="image/png"
            if filename.endswith(".png")
            else "application/octet-stream",
        )

    @application.get("/api/runs")
    def runs() -> list[dict]:
        """List offline experiment runs that produced a status record."""
        result = []
        for directory in sorted((ROOT / "outputs").iterdir(), reverse=True):
            if not (directory / "status.json").is_file():
                continue
            status = read_json(directory / "status.json")
            result.append(
                dict(
                    id=directory.name,
                    command=status["command"],
                    exit_code=status["exit_code"],
                    seconds=status["seconds"],
                    metrics_available=(directory / "metrics.json").is_file(),
                )
            )
        return result

    @application.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str) -> dict:
        if not re.fullmatch(r"\d{8}-\d{6}-[a-z-]+-[a-f0-9]{6}", run_id):
            raise HTTPException(400, detail="Invalid run ID")
        path = ROOT / "outputs" / run_id / "metrics.json"
        if not path.is_file():
            raise HTTPException(404, detail="No completed metrics for this run")
        return read_json(path)

    dist = ROOT / "frontend/dist"
    if dist.is_dir():
        application.mount(
            "/", StaticFiles(directory=dist, html=True), name="workstation"
        )
    return application


app = create_app()
