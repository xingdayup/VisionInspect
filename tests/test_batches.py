"""Queue/API behavior, using explicit synthetic flat images (not experiments)."""

import sqlite3
import time
from io import BytesIO
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from visioninspect.api import create_app
from visioninspect.service import TaskStore, timestamp

pytestmark = pytest.mark.usefixtures("gpu_service_fixture")


def png():
    stream = BytesIO()
    Image.new("RGB", (256, 256), "gray").save(stream, format="PNG")
    return stream.getvalue()


def batch(client):
    response = client.post(
        "/api/batches", json=dict(name="协议测试批次", mode="baseline")
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_real_batch_partial_upload_run_human_review_and_exports(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        identifier = batch(client)
        assert (
            client.post(
                f"/api/batches/{identifier}/items",
                files={"file": ("bad.png", b"broken")},
            ).status_code
            == 422
        )
        uploaded = client.post(
            f"/api/batches/{identifier}/items", files={"file": ("=fixture.png", png())}
        )
        assert uploaded.status_code == 201 and uploaded.json()["status"] == "staged"
        task_id = uploaded.json()["id"]
        assert (
            client.post(f"/api/batches/{identifier}/actions/start").status_code == 200
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            value = client.get(f"/api/batches/{identifier}").json()
            if value["status"] == "completed":
                break
            time.sleep(0.05)
        assert (
            value["counts"]["completed"] == 1 and value["counts"]["manual_review"] == 1
        )
        before = client.get(f"/api/tasks/{task_id}").json()["report"]["result"]
        assert (
            client.post(
                f"/api/tasks/{task_id}/review",
                json=dict(
                    decision="normal_candidate", note="明确标记的单元测试人工记录"
                ),
            ).status_code
            == 200
        )
        report = client.get(f"/api/batches/{identifier}/report").json()
        assert report["tasks"][0]["report"]["result"] == before
        assert report["tasks"][0]["report"]["effective_result"]["source"] == "human"
        assert report["batch"]["counts"]["human_reviewed"] == 1
        csv = client.get(f"/api/batches/{identifier}/report?format=csv")
        assert csv.status_code == 200 and "'=fixture.png" in csv.text
        assert client.get("/api/batches/not-an-id").status_code == 404
    with TestClient(create_app(tmp_path)) as client:
        assert (
            client.get(f"/api/batches/{identifier}").json()["counts"]["human_reviewed"]
            == 1
        )


def test_pause_resume_cancel_and_no_append_after_start(tmp_path):
    store = TaskStore(tmp_path)
    try:
        b = store.create_batch("单元测试", "bottle", "patchcore", "baseline")
        first = store.create(
            png(), "fixture.png", "bottle", "patchcore", "baseline", batch_id=b["id"]
        )
        # Pause before dispatch by holding a SQLite write transaction across the two state changes.
        store.batch_action(b["id"], "start")
        store.batch_action(b["id"], "pause")
        assert store.get_batch(b["id"])["status"] == "paused"
        with pytest.raises(ValueError):
            store.create(
                png(), "extra.png", "bottle", "patchcore", "baseline", batch_id=b["id"]
            )
        store.batch_action(b["id"], "resume")
        store.batch_action(b["id"], "cancel")
        assert store.get_batch(b["id"])["status"] == "cancelled"
        assert store.get(first["id"])["status"] in {"cancelled", "running", "completed"}
    finally:
        store.close()


def test_staged_capacity_and_per_batch_limit(tmp_path):
    store = TaskStore(tmp_path)
    try:
        for _ in range(2):
            b = store.create_batch("容量测试", "bottle", "patchcore", "baseline")
            for index in range(100):
                store.create(
                    png(),
                    f"fixture-{index}.png",
                    "bottle",
                    "patchcore",
                    "baseline",
                    batch_id=b["id"],
                )
            with pytest.raises(OverflowError):
                store.create(
                    png(),
                    "101.png",
                    "bottle",
                    "patchcore",
                    "baseline",
                    batch_id=b["id"],
                )
        with pytest.raises(OverflowError):
            store.create(png(), "201.png", "bottle", "patchcore", "baseline")
    finally:
        store.close()


def test_retry_preserves_failure_and_logical_batch_count(tmp_path):
    store = TaskStore(tmp_path)
    try:
        b = store.create_batch("重试测试", "bottle", "patchcore", "baseline")
        task = store.create(
            png(), "fixture.png", "bottle", "patchcore", "baseline", batch_id=b["id"]
        )
        store.fail(task["id"], "unit_test_injected_failure", 0)
        with store.connect() as db:
            db.execute("UPDATE batches SET status='completed' WHERE id=?", (b["id"],))
        retried = store.retry(task["id"])
        assert retried["retry_of"] == task["id"] and retried["attempt"] == 2
        assert store.get(task["id"])["status"] == "failed"
        assert (
            store.get_batch(b["id"])["total"] == 1
            and store.get_batch(b["id"])["attempts"] == 2
        )
        with pytest.raises(ValueError):
            store.retry(task["id"])
    finally:
        store.close()


def test_wal_safe_migration_backup_and_restart_pause(tmp_path):
    store = TaskStore(tmp_path)
    b = store.create_batch("迁移测试", "bottle", "patchcore", "baseline")
    store.create(
        png(), "fixture.png", "bottle", "patchcore", "baseline", batch_id=b["id"]
    )
    store.close()
    with sqlite3.connect(tmp_path / "history.sqlite3") as db:
        db.execute("UPDATE batches SET status='running'")
        db.execute("PRAGMA user_version=1")
    restarted = TaskStore(tmp_path)
    try:
        assert restarted.get_batch(b["id"])["status"] == "paused"
        assert restarted.get_batch(b["id"])["counts"]["staged"] == 1
        backups = list((tmp_path / "backups").glob("*.sqlite3"))
        assert len(backups) == 1
        with sqlite3.connect(backups[0]) as db:
            assert db.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 1
    finally:
        restarted.close()


def test_gpu_schema_removes_legacy_vlm_column_after_backup(tmp_path):
    with sqlite3.connect(tmp_path / "history.sqlite3") as db:
        db.execute(
            "CREATE TABLE batches (id TEXT PRIMARY KEY,name TEXT,status TEXT,category TEXT,model TEXT,mode TEXT,vlm INTEGER,created_at TEXT,updated_at TEXT,snapshot TEXT)"
        )
        db.execute(
            "INSERT INTO batches VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "a" * 32,
                "历史批次",
                "draft",
                "bottle",
                "patchcore",
                "baseline",
                0,
                timestamp(),
                timestamp(),
                "{}",
            ),
        )
        db.execute("PRAGMA user_version=2")
    store = TaskStore(tmp_path)
    try:
        with store.connect() as db:
            assert "vlm" not in {
                row[1] for row in db.execute("PRAGMA table_info(batches)")
            }
            assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert store.get_batch("a" * 32)["name"] == "历史批次"
        backup = next((tmp_path / "backups").glob("*.sqlite3"))
        with sqlite3.connect(backup) as db:
            assert "vlm" in {row[1] for row in db.execute("PRAGMA table_info(batches)")}
    finally:
        store.close()
