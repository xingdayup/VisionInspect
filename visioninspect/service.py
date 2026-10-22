"""Local queue, SQLite history and one supervised resident inference worker.

TaskStore owns everything that outlives a single request: the SQLite queue, the
batch state machine, the resident inference process and the vLLM lease. The API
layer never schedules work itself; it only enqueues and reads back state.

Concurrency model: one dispatcher thread runs tasks strictly in order, while
mutations take a single re-entrant lock. SQLite runs in WAL mode so readers are
never blocked by the writer.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .common import ROOT, project_path, read_json, write_json
from .resident import ResidentWorker
from .schemas import FinalResult
from .vllm_runtime import VLLMRuntime, gpu_config


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp used for every stored record."""
    return datetime.now(UTC).isoformat()


def cuda_available() -> bool:
    """Report whether this interpreter can see a CUDA device.

    Import errors are treated as 'no GPU' because a CPU-only torch build is a
    configuration error here, not a reason to fall back to CPU inference.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, OSError):
        return False


def registry() -> dict:
    """Read the product-to-model registry; it changes only when assets change."""
    return yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf8"))[
        "models"
    ]


def model_health() -> list[dict]:
    """Report per-model asset presence without loading any weights."""
    result = []
    for category, item in registry().items():
        for model, config in item.items():
            if model == "label":
                continue
            missing = [
                str(path)
                for path in (
                    project_path(config["fit_run"]) / "model.ckpt",
                    project_path(config["profile"]),
                    project_path(config["references"]),
                )
                if not path.is_file()
            ]
            result.append(
                dict(
                    category=category,
                    label=item["label"],
                    model=model,
                    available=not missing,
                    reason="模型资产完整"
                    if not missing
                    else "模型、校准策略或参考索引缺失",
                )
            )
    return result


class TaskStore:
    """Own the inspection queue, its SQLite history and both model processes."""

    def __init__(self, directory: Path | None = None) -> None:
        """Open the store, migrate the schema and start the dispatcher thread."""
        self.root = directory or project_path(
            os.environ.get("VISIONINSPECT_STATE_DIR", "outputs/service")
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "history.sqlite3"
        self.closed = False
        self.mutation_lock = threading.RLock()
        self.wake = threading.Event()
        if self.database.is_file():
            with sqlite3.connect(self.database) as old:
                if old.execute("PRAGMA user_version").fetchone()[0] < 3:
                    backup = self.root / "backups"
                    backup.mkdir(exist_ok=True)
                    with sqlite3.connect(
                        backup / f"history-{time.time_ns()}.sqlite3"
                    ) as target:
                        old.backup(target)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT, status TEXT, category TEXT, model TEXT, mode TEXT, filename TEXT, state TEXT, error TEXT, seconds REAL, review TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, name TEXT, status TEXT, category TEXT, model TEXT, mode TEXT, created_at TEXT, updated_at TEXT, snapshot TEXT)"
            )
            if any(row[1] == "vlm" for row in db.execute("PRAGMA table_info(batches)")):
                db.execute("ALTER TABLE batches DROP COLUMN vlm")
            db.execute(
                "CREATE TABLE IF NOT EXISTS task_links (task_id TEXT PRIMARY KEY, batch_id TEXT, retry_of TEXT, root_task_id TEXT, attempt INTEGER)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS task_queue ON tasks(status,created_at)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS batch_members ON task_links(batch_id)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS batch_upload_errors (batch_id TEXT, filename TEXT, message TEXT, recorded_at TEXT)"
            )
            db.execute("PRAGMA user_version=3")
            interrupted = [
                row["id"]
                for row in db.execute("SELECT id FROM tasks WHERE status='running'")
            ]
            db.execute(
                "UPDATE batches SET status='paused',updated_at=? WHERE status IN ('queued','running')",
                (timestamp(),),
            )
        for task_id in interrupted:
            self.fail(task_id, "interrupted: 服务中断；请重新提交", None)
        self.runtime = ResidentWorker(self.root / "runtime", self.worker_command)
        self.gpu = VLLMRuntime(self.root / "gpu-runtime")
        self.dispatcher = threading.Thread(
            target=self.dispatch, name="persistent-inspection-queue", daemon=True
        )
        self.dispatcher.start()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a short-lived connection whose block commits or rolls back as one."""
        connection = sqlite3.connect(self.database, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def directory(self, task_id: str) -> Path:
        """Return a task directory, rejecting anything that is not a 32-char hex id."""
        if (
            not isinstance(task_id, str)
            or len(task_id) != 32
            or any(c not in "0123456789abcdef" for c in task_id)
        ):
            raise KeyError(task_id)
        return self.root / task_id

    def get(self, task_id: str, details: bool = True) -> dict:
        """Load one task, optionally attaching its progress and report files."""
        directory = self.directory(task_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            link = db.execute(
                "SELECT batch_id,retry_of,root_task_id,attempt FROM task_links WHERE task_id=?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        result = dict(row)
        result["review"] = json.loads(result["review"]) if result["review"] else None
        if link:
            result.update(dict(link))
        if details:
            for name in ("progress", "report"):
                path = directory / f"{name}.json"
                result[name] = read_json(path) if path.is_file() else None
            if isinstance(result.get("report"), dict):
                result["report"]["human_review"] = result["review"]
                result["report"]["effective_result"] = self.effective(result)
                execution = directory / "execution.json"
                if execution.is_file():
                    try:
                        result["report"]["timings"] = read_json(execution).get(
                            "timings"
                        )
                    except (OSError, ValueError):
                        pass
        return result

    @staticmethod
    def effective(task: dict) -> dict:
        """Resolve the result a reader should trust: human, failure, agent, baseline."""
        if task.get("review"):
            return dict(
                state=task["review"]["decision"],
                source="human",
                reasons=[task["review"]["note"]],
            )
        report = task.get("report") or {}
        return dict(
            report.get("result") or dict(state=task.get("state"), reasons=[]),
            source="failure"
            if task.get("status") == "failed"
            else "agent"
            if (report.get("agent_review") or {}).get("calls")
            else "baseline",
        )

    def list(
        self,
        status: str | None = None,
        state: str | None = None,
        query: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Return a filtered, paginated page of tasks for the history view."""
        conditions = []
        arguments = []
        for name, value in [("status", status), ("state", state)]:
            if value:
                conditions.append(f"{name}=?")
                arguments.append(value)
        if query:
            conditions.append("(filename LIKE ? OR id LIKE ?)")
            arguments.extend([f"%{query}%", f"%{query}%"])
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.connect() as db:
            total = db.execute(
                "SELECT COUNT(*) FROM tasks" + where, arguments
            ).fetchone()[0]
            rows = db.execute(
                "SELECT * FROM tasks"
                + where
                + " ORDER BY created_at DESC LIMIT ? OFFSET ?",
                arguments + [limit, offset],
            ).fetchall()
        items = [dict(row) for row in rows]
        for item in items:
            item["review"] = json.loads(item["review"]) if item["review"] else None
        return dict(items=items, total=total, limit=limit, offset=offset)

    def snapshot(self, category: str, model: str) -> dict:
        """Freeze the policy, registration and profile digest at submission time.

        Storing the snapshot with the task means a later config edit cannot
        change how an already queued image is judged.
        """
        policy = yaml.safe_load(
            (ROOT / "configs/agent.yaml").read_text(encoding="utf8")
        )
        registration = registry()[category][model]
        profile = read_json(project_path(registration["profile"]))
        return dict(
            policy=policy,
            registration=registration,
            gpu=gpu_config(),
            profile_digest=profile["digest"],
        )

    def create(
        self,
        body: bytes,
        filename: str,
        category: str,
        model: str,
        mode: str,
        batch_id: str | None = None,
        retry_of: str | None = None,
    ) -> dict:
        """Persist one uploaded image and enqueue it, or stage it into a batch.

        Queue depth and per-batch limits are checked inside the same transaction
        that inserts the row, so two uploads cannot both pass a full queue.
        """
        if mode not in {"baseline", "agent_review"}:
            raise ValueError("Only GPU baseline and agent_review are supported")
        if self.closed:
            raise RuntimeError("服务正在关闭")
        with self.mutation_lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            pending = db.execute(
                "SELECT COUNT(*) FROM tasks WHERE status IN ('staged','queued','running')"
            ).fetchone()[0]
            if pending >= 200:
                raise OverflowError("任务队列已满（200 项），请稍后再试")
            snapshot = None
            status = "queued"
            if batch_id:
                batch = db.execute(
                    "SELECT * FROM batches WHERE id=?", (batch_id,)
                ).fetchone()
                if batch is None:
                    raise KeyError(batch_id)
                if batch["status"] not in ("draft", "paused", "completed"):
                    raise ValueError("批次当前状态不接受上传或重试")
                if not retry_of and batch["status"] != "draft":
                    raise ValueError("启动后的批次不能追加图片")
                count = db.execute(
                    "SELECT COUNT(*) FROM task_links WHERE batch_id=? AND retry_of IS NULL",
                    (batch_id,),
                ).fetchone()[0]
                if not retry_of and count >= 100:
                    raise OverflowError("每批最多 100 张图片")
                snapshot = json.loads(batch["snapshot"])
                status = "staged"
            task_id = uuid.uuid4().hex
            directory = self.directory(task_id)
            directory.mkdir()
            (directory / "input.image").write_bytes(body)
            request = dict(
                id=task_id,
                image_path=str(directory / "input.image"),
                category=category,
                model=model,
                mode=mode,
                filename=filename,
                created_at=timestamp(),
                queued_at=timestamp() if status == "queued" else None,
                snapshot=snapshot or self.snapshot(category, model),
            )
            write_json(directory / "request.json", request)
            write_json(directory / "progress.json", dict(stage=status, trace=[]))
            db.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    request["created_at"],
                    timestamp(),
                    status,
                    category,
                    model,
                    mode,
                    filename,
                    None,
                    None,
                    None,
                    None,
                ),
            )
            root = task_id
            attempt = 1
            if retry_of:
                previous = db.execute(
                    "SELECT root_task_id,attempt FROM task_links WHERE task_id=?",
                    (retry_of,),
                ).fetchone()
                if previous:
                    root = previous["root_task_id"]
                    attempt = previous["attempt"] + 1
            db.execute(
                "INSERT INTO task_links VALUES (?,?,?,?,?)",
                (task_id, batch_id, retry_of, root, attempt),
            )
            if batch_id and retry_of:
                db.execute(
                    "UPDATE batches SET status='paused',updated_at=? WHERE id=?",
                    (timestamp(), batch_id),
                )
        self.wake.set()
        return self.get(task_id)

    def dispatch(self) -> None:
        """Take queued tasks one at a time until shutdown.

        Only one task may run at once: the CUDA detector and the vLLM server are
        single-tenant, and serial execution keeps memory measurements meaningful.
        """
        while not self.closed:
            selected = None
            try:
                with self.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute(
                        "SELECT t.id,l.batch_id FROM tasks t LEFT JOIN task_links l ON t.id=l.task_id LEFT JOIN batches b ON l.batch_id=b.id WHERE t.status='queued' AND (l.batch_id IS NULL OR b.status IN ('queued','running')) ORDER BY t.created_at LIMIT 1"
                    ).fetchone()
                    if row:
                        selected = row["id"]
                        db.execute(
                            "UPDATE tasks SET status='running',updated_at=? WHERE id=?",
                            (timestamp(), selected),
                        )
                        if row["batch_id"]:
                            db.execute(
                                "UPDATE batches SET status='running',updated_at=? WHERE id=?",
                                (timestamp(), row["batch_id"]),
                            )
                if selected:
                    self.execute(selected)
                    self.settle_batches()
                    continue
            except Exception:
                (self.root / "dispatcher-error.log").write_text(
                    traceback.format_exc(), encoding="utf8"
                )
                if selected:
                    try:
                        self.fail(
                            selected,
                            "dispatcher_failure: 调度执行失败，请查看日志",
                            None,
                        )
                    except Exception:
                        pass
            self.wake.wait(0.2)
            self.wake.clear()

    def create_batch(self, name: str, category: str, model: str, mode: str) -> dict:
        """Create an empty draft batch; images are uploaded into it afterwards."""
        if mode not in {"baseline", "agent_review"}:
            raise ValueError("Only GPU baseline and agent_review are supported")
        batch_id = uuid.uuid4().hex
        now = timestamp()
        snapshot = self.snapshot(category, model)
        with self.connect() as db:
            db.execute(
                "INSERT INTO batches VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    batch_id,
                    name,
                    "draft",
                    category,
                    model,
                    mode,
                    now,
                    now,
                    json.dumps(snapshot, ensure_ascii=False),
                ),
            )
        return self.get_batch(batch_id)

    def get_batch(self, batch_id: str, details: bool = True) -> dict:
        """Load a batch with per-image counts computed from the latest attempt.

        Retries are linked rows, so the summary counts one image once even when
        it was attempted several times.
        """
        self.directory(batch_id)  # same strict identifier validation
        with self.connect() as db:
            row = db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
            if row is None:
                raise KeyError(batch_id)
            rows = db.execute(
                "SELECT t.*,l.root_task_id,l.retry_of,l.attempt FROM tasks t JOIN task_links l ON t.id=l.task_id WHERE l.batch_id=? ORDER BY t.created_at",
                (batch_id,),
            ).fetchall()
        batch = dict(row)
        batch["snapshot"] = json.loads(batch["snapshot"])
        with self.connect() as db:
            batch["upload_errors"] = [
                dict(item)
                for item in db.execute(
                    "SELECT filename,message,recorded_at FROM batch_upload_errors WHERE batch_id=? ORDER BY recorded_at",
                    (batch_id,),
                )
            ]
        items = [dict(item) for item in rows]
        latest = {}
        for item in items:
            item["review"] = json.loads(item["review"]) if item["review"] else None
            latest[item["root_task_id"]] = item
        counts = {
            name: 0
            for name in (
                "staged",
                "queued",
                "running",
                "completed",
                "failed",
                "cancelled",
                "normal_candidate",
                "anomaly_candidate",
                "manual_review",
                "human_reviewed",
            )
        }
        for item in latest.values():
            counts[item["status"]] = counts.get(item["status"], 0) + 1
            state = item["review"]["decision"] if item["review"] else item["state"]
            if state:
                counts[state] = counts.get(state, 0) + 1
            if item["review"]:
                counts["human_reviewed"] += 1
        batch.update(total=len(latest), attempts=len(items), counts=counts)
        if details:
            batch.update(items=list(latest.values()), all_attempts=items)
        return batch

    def list_batches(self, limit: int = 50, offset: int = 0) -> dict:
        """Return recent batches without their full item lists."""
        with self.connect() as db:
            ids = [
                r["id"]
                for r in db.execute(
                    "SELECT id FROM batches ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            ]
            total = db.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        return dict(
            items=[self.get_batch(i, False) for i in ids],
            total=total,
            limit=limit,
            offset=offset,
        )

    def batch_action(self, batch_id: str, action: str) -> dict:
        """Apply start/resume/pause/cancel, rejecting transitions that lost meaning.

        Pause and cancel only affect items that have not started; the item
        already running finishes so its evidence is not silently discarded.
        """
        with self.mutation_lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            batch = db.execute(
                "SELECT * FROM batches WHERE id=?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise KeyError(batch_id)
            current = batch["status"]
            now = timestamp()
            if action in ("start", "resume"):
                if current not in ("draft", "paused", "completed"):
                    raise ValueError("批次当前状态不能启动或继续")
                pending = db.execute(
                    "SELECT COUNT(*) FROM tasks WHERE id IN (SELECT task_id FROM task_links WHERE batch_id=?) AND status IN ('staged','queued')",
                    (batch_id,),
                ).fetchone()[0]
                running = db.execute(
                    "SELECT COUNT(*) FROM tasks WHERE id IN (SELECT task_id FROM task_links WHERE batch_id=?) AND status='running'",
                    (batch_id,),
                ).fetchone()[0]
                if not pending and not running:
                    raise ValueError("批次没有待执行图片")
                db.execute(
                    "UPDATE tasks SET status='queued',updated_at=? WHERE id IN (SELECT task_id FROM task_links WHERE batch_id=?) AND status='staged'",
                    (now, batch_id),
                )
                for row in db.execute(
                    "SELECT id FROM tasks WHERE id IN (SELECT task_id FROM task_links WHERE batch_id=?) AND status='queued'",
                    (batch_id,),
                ):
                    path = self.directory(row["id"]) / "request.json"
                    request = read_json(path)
                    if request.get("queued_at") is None:
                        request["queued_at"] = now
                        write_json(path, request)
                target = "queued"
            elif action == "pause":
                if current not in ("queued", "running"):
                    raise ValueError("批次当前状态不能暂停")
                target = "paused"
            elif action == "cancel":
                if current in ("completed", "cancelled"):
                    raise ValueError("批次已结束")
                db.execute(
                    "UPDATE tasks SET status='cancelled',updated_at=? WHERE id IN (SELECT task_id FROM task_links WHERE batch_id=?) AND status IN ('staged','queued')",
                    (now, batch_id),
                )
                target = "cancelled"
            else:
                raise ValueError("未知批次操作")
            db.execute(
                "UPDATE batches SET status=?,updated_at=? WHERE id=?",
                (target, now, batch_id),
            )
        self.wake.set()
        return self.get_batch(batch_id)

    def settle_batches(self) -> None:
        """Mark batches complete once no member task is pending or running."""
        with self.connect() as db:
            db.execute(
                "UPDATE batches SET status='completed',updated_at=? WHERE status IN ('queued','running','paused') AND NOT EXISTS (SELECT 1 FROM task_links l JOIN tasks t ON l.task_id=t.id WHERE l.batch_id=batches.id AND t.status IN ('queued','running','staged'))",
                (timestamp(),),
            )

    def record_upload_error(self, batch_id: str, filename: str, message: str) -> None:
        """Keep a rejected upload visible instead of failing the whole batch."""
        with self.connect() as db:
            db.execute(
                "INSERT INTO batch_upload_errors VALUES (?,?,?,?)",
                (batch_id, filename[:160], message[:500], timestamp()),
            )

    def retry(self, task_id: str) -> dict:
        """Create a linked new attempt instead of mutating the failed one."""
        with self.mutation_lock:
            task = self.get(task_id)
            if task["status"] != "failed":
                raise ValueError("只能重试失败任务")
            with self.connect() as db:
                existing = db.execute(
                    "SELECT t.id FROM tasks t JOIN task_links l ON t.id=l.task_id WHERE l.retry_of=? AND t.status IN ('staged','queued','running')",
                    (task_id,),
                ).fetchone()
            if existing:
                raise ValueError("此任务已有未结束的重试")
            # Fail fast when the original request record is unreadable.
            read_json(self.directory(task_id) / "request.json")
            return self.create(
                (self.directory(task_id) / "input.image").read_bytes(),
                task["filename"],
                task["category"],
                task["model"],
                task["mode"],
                task.get("batch_id"),
                task_id,
            )

    def update(self, task_id: str, **values: Any) -> None:
        """Update named task columns and stamp updated_at in one statement."""
        values["updated_at"] = timestamp()
        with self.connect() as db:
            db.execute(
                "UPDATE tasks SET "
                + ",".join(f"{key}=?" for key in values)
                + " WHERE id=?",
                list(values.values()) + [task_id],
            )

    def worker_command(self, directory: Path) -> list[str]:
        """Return the resident worker command, bound to this process as parent."""
        return [
            sys.executable,
            "-B",
            "-m",
            "visioninspect.worker",
            str(self.root.resolve()),
            "--resident",
            "--parent-pid",
            str(os.getpid()),
        ]

    def fail(self, task_id: str, failure: str, seconds: float | None) -> None:
        """Publish an authoritative manual-review report before exposing terminal failure.

        Whatever the worker left behind is preserved as evidence, but the stored
        result is always manual_review: an unaccepted run must never read as a
        passing candidate.
        """
        directory = self.directory(task_id)
        report_path = directory / "report.json"
        original = {}
        if report_path.is_file():
            backup = directory / "worker-report-unaccepted.json"
            if not backup.exists():
                shutil.copy2(report_path, backup)
            try:
                original = read_json(report_path)
            except (OSError, ValueError):
                original = {}
            if not isinstance(original, dict):
                original = {}
        baseline_path = directory / "baseline-result.json"
        if baseline_path.is_file():
            try:
                baseline = read_json(baseline_path)
                request = read_json(directory / "request.json")
                if baseline.get("profile_digest") == (
                    request.get("snapshot") or {}
                ).get("profile_digest"):
                    from .schemas import DetectionResult

                    DetectionResult.model_validate(baseline["outputs"]["detection"])
                    original = dict(baseline, detection_stage_preserved=True)
            except (ValueError, KeyError, TypeError, OSError):
                pass
        try:
            progress = read_json(directory / "progress.json")
        except (OSError, ValueError):
            progress = {}
        if not isinstance(progress, dict):
            progress = {}
        with self.connect() as db:
            metadata = dict(
                db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            )
        paths = (
            original.get("result", {}).get("evidence_paths", [])
            if isinstance(original.get("result"), dict)
            else []
        )
        if not isinstance(paths, list):
            paths = []
        result = FinalResult(
            state="manual_review",
            evidence_paths=[p for p in paths if isinstance(p, str)],
            rule_version="supervisor-failure-v1",
            reasons=[failure],
        ).model_dump()
        trace = original.get("trace") or progress.get("trace", [])
        if not isinstance(trace, list):
            trace = []
        report = dict(
            original,
            schema_version=original.get("schema_version", 1),
            task_id=task_id,
            filename=metadata["filename"],
            category=metadata["category"],
            mode=metadata["mode"],
            created_at=metadata["created_at"],
            result=result,
            errors=[failure],
            outputs=original.get("outputs", {})
            if isinstance(original.get("outputs", {}), dict)
            else {},
            seconds=seconds,
            tool_calls=len(trace),
            vlm_calls=original.get("vlm_calls", 0),
            monetary_cost=0,
            trace=trace,
            stop_reason="supervisor_failure",
            supervisor=dict(worker_report_accepted=False, failure=failure),
            notice="监督器未接受本次推理结果，必须人工复核；原始工作进程输出仅供排查。",
        )
        write_json(report_path, report)
        self.update(
            task_id,
            status="failed",
            state="manual_review",
            error=failure,
            seconds=seconds,
        )

    def execute(self, task_id: str) -> None:
        """Run one task to completion and settle its stored status.

        The vLLM lease is held only while the review workflow runs, and the
        worker reply is verified before the report is accepted. On any failure
        the task is published as manual_review rather than being left running.
        """
        directory = self.directory(task_id)
        started = time.perf_counter()
        self.update(task_id, status="running")
        try:
            request = read_json(directory / "request.json")
            gpu_lease = None
            gpu_initialization = None
            if request["mode"] == "agent_review":
                gpu_lease = self.gpu.acquire(
                    lambda value: write_json(directory / "progress.json", value)
                )
                try:
                    gpu_initialization = gpu_lease.__enter__()
                except Exception as error:
                    request["gpu_startup_error"] = str(error)
                    gpu_lease = None
                    write_json(directory / "request.json", request)
            worker_started = time.perf_counter()
            try:
                reply = self.runtime.run(directory)
            finally:
                if gpu_lease:
                    gpu_lease.__exit__(None, None, None)
            report = read_json(directory / "report.json")
            result = FinalResult.model_validate(report["result"])
            if reply.get("recycle"):
                self.runtime.recycle("task_tool_failure")
            if (report.get("agent_review") or {}).get("status") == "failed":
                self.gpu.recycle("review_failure")
            execution = read_json(directory / "execution.json")
            execution["worker_report_accepted"] = True
            execution.update(
                gpu_initialization=gpu_initialization, gpu_runtime=self.gpu.status()
            )
            queued = request.get("queued_at") or request["created_at"]
            execution["timings"] = dict(
                queued_seconds=max(
                    0,
                    (datetime.now(UTC) - datetime.fromisoformat(queued)).total_seconds()
                    - (time.perf_counter() - started),
                ),
                gpu_prepare_seconds=(gpu_initialization or {}).get(
                    "startup_seconds", 0
                ),
                worker_seconds=time.perf_counter() - worker_started,
                detector_initialization_seconds=(
                    report.get("initialization") or {}
                ).get("seconds", 0),
                detection_seconds=(
                    (report.get("outputs") or {}).get("detection") or {}
                ).get("elapsed_seconds"),
                agent_seconds=sum(
                    row.get("seconds", 0)
                    for row in report.get("trace", [])
                    if row["tool"] == "review_evidence"
                ),
                total_processing_seconds=time.perf_counter() - started,
            )
            write_json(directory / "execution.json", execution)
            failed_review = request["mode"] == "agent_review" and bool(
                report.get("errors")
            )
            self.update(
                task_id,
                status="failed" if failed_review else "completed",
                state=result.state,
                error="；".join(report["errors"]) if failed_review else None,
                seconds=time.perf_counter() - started,
            )
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            (directory / "supervisor-error.log").write_text(
                traceback.format_exc(), encoding="utf8"
            )
            self.runtime.recycle("task_failure")
            execution_path = directory / "execution.json"
            if execution_path.is_file():
                execution = read_json(execution_path)
                execution.update(
                    failure=failure, worker_report_accepted=False, task_exit_code=1
                )
                write_json(execution_path, execution)
            self.fail(task_id, failure, time.perf_counter() - started)

    def review(self, task_id: str, decision: str, note: str) -> dict:
        """Append a human decision without touching the original model report."""
        task = self.get(task_id)
        if task["status"] not in ("completed", "failed"):
            raise ValueError("任务结束后才能保存复核")
        # Separate append-only review events; never rewrite the original model report.
        value = dict(
            decision=decision,
            note=note,
            recorded_at=timestamp(),
            scope="研究复核记录，不构成产线放行授权",
        )
        directory = self.directory(task_id)
        with self.mutation_lock:
            with (directory / "reviews.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            self.update(task_id, review=json.dumps(value, ensure_ascii=False))
        return value

    def close(self) -> None:
        """Stop both model processes and fail anything still marked running."""
        self.closed = True
        self.wake.set()
        self.gpu.close()
        self.runtime.close()
        self.dispatcher.join(timeout=10)
        with self.connect() as db:
            db.execute(
                "UPDATE batches SET status='paused',updated_at=? WHERE status IN ('queued','running')",
                (timestamp(),),
            )
            pending = [
                row["id"]
                for row in db.execute("SELECT id FROM tasks WHERE status='running'")
            ]
        for task_id in pending:
            self.fail(task_id, "interrupted: 服务关闭；请重新提交", None)
