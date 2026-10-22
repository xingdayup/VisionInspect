"""Finite CUDA deployment acceptance using real normal validation images only."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visioninspect.common import (
    configure_environment,
    load_config,
    offline,
    project_path,
    read_json,
    sha256,
    write_json,
)


def detector_probe(output):
    import numpy as np
    import torch
    from PIL import Image

    from visioninspect.data import load_manifest
    from visioninspect.model import Detector, setup
    from visioninspect.service import registry

    registration = registry()["bottle"]["patchcore"]
    config = load_config(project_path(registration["config"]))
    setup(config)
    profile = read_json(project_path(registration["profile"]))
    checkpoint = project_path(registration["fit_run"]) / "model.ckpt"
    if sha256(checkpoint) != profile["checkpoint_sha256"]:
        raise ValueError("Checkpoint/profile identity mismatch")
    _, records = load_manifest(config, ("validation",))
    with offline():
        detector, _ = Detector.load(checkpoint, config)
        cases = {}
        examined = []
        for record in records:
            path = project_path(config["dataset"]["root"]) / record["path"]
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"))
            score, heat, _ = detector.predict(array)
            examined.append(dict(path=record["path"], score=score))
            key = (
                "uncertain"
                if profile["review_lower"] <= score <= profile["review_upper"]
                else "clear"
            )
            if key not in cases:
                np.save(output / f"{key}.npy", heat)
                cases[key] = dict(path=str(path), score=score, heatmap=f"{key}.npy")
            if len(cases) == 2:
                break
        detector.save(
            output / "roundtrip.ckpt",
            dict(purpose="GPU engineering roundtrip, no training"),
        )
        del detector
        gc.collect()
        torch.cuda.empty_cache()
        reloaded, _ = Detector.load(output / "roundtrip.ckpt")
        example = next(iter(cases.values()))
        with Image.open(example["path"]) as image:
            score, heat, _ = reloaded.predict(np.asarray(image.convert("RGB")))
        np.testing.assert_allclose(score, example["score"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            heat, np.load(output / example["heatmap"]), rtol=1e-5, atol=1e-6
        )
        write_json(
            output / "direct.json",
            dict(
                passed=True,
                device=str(reloaded.device),
                gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,
                cuda_build=torch.version.cuda,
                checkpoint_sha256=profile["checkpoint_sha256"],
                profile_digest=profile["digest"],
                cases=cases,
                examined=examined,
                checkpoint_roundtrip=True,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                calibration_recomputed=False,
                accuracy_evaluated=False,
            ),
        )


def wait_task(store, task):
    deadline = time.monotonic() + store.gpu.config["startup_timeout_seconds"] + 210
    while task["status"] in {"queued", "running"}:
        if time.monotonic() > deadline:
            raise TimeoutError("GPU task acceptance timed out")
        time.sleep(0.2)
        task = store.get(task["id"])
    return task


def verify(output, with_vlm, ttl_smoke):
    import numpy as np

    from visioninspect.service import TaskStore

    # The CUDA probe exits before the service starts; its context/weights are released.
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--probe-output",
        str(output),
    ]
    started = time.monotonic()
    with (output / "detector-probe.log").open("w", encoding="utf8") as log:
        process = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300
        )
    write_json(
        output / "probe-command.json",
        dict(
            argv=command,
            exit_code=process.returncode,
            seconds=time.monotonic() - started,
        ),
    )
    if process.returncode:
        raise RuntimeError("CUDA probe failed; inspect detector-probe.log")
    direct = read_json(output / "direct.json")
    store = TaskStore(output / "service")
    results = []
    try:
        if with_vlm and not store.gpu.status()["assets_available"]:
            raise RuntimeError(
                "Prepare the pinned local Qwen assets first; downloads are disabled here"
            )
        if with_vlm and "uncertain" not in direct["cases"]:
            raise RuntimeError(
                "No eligible normal validation sample for Agent acceptance; do not change the frozen review band"
            )
        for key, expected in direct["cases"].items():
            mode = "agent_review" if with_vlm else "baseline"
            path = Path(expected["path"])
            task = wait_task(
                store,
                store.create(path.read_bytes(), path.name, "bottle", "patchcore", mode),
            )
            detection = (task.get("report") or {}).get("outputs", {}).get("detection")
            if not detection:
                raise RuntimeError(
                    f"{key}: no valid CUDA detection; {task.get('error')}"
                )
            if detection["device"] != "cuda:0":
                raise RuntimeError("Unexpected inference device")
            actual = np.load(
                store.directory(task["id"]) / detection["anomaly_map_path"],
                allow_pickle=False,
            )
            np.testing.assert_allclose(
                detection["score"], expected["score"], rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                actual, np.load(output / expected["heatmap"]), rtol=1e-5, atol=1e-6
            )
            if task["status"] != "completed":
                raise RuntimeError(f"{key}: {task.get('error')}")
            results.append(
                dict(
                    case=key,
                    task_id=task["id"],
                    score=detection["score"],
                    state=task["state"],
                    model_calls=task["report"]["vlm_calls"],
                    heatmap_max_abs_delta=float(
                        np.max(np.abs(actual - np.load(output / expected["heatmap"])))
                    ),
                )
            )
        protocol = {}
        multistep = False
        if with_vlm and "uncertain" in direct["cases"]:
            from PIL import Image

            from visioninspect.vllm_client import VLLMClient

            uncertain = next(item for item in results if item["case"] == "uncertain")
            task = store.get(uncertain["task_id"])
            report = task["report"]
            directory = store.directory(task["id"])
            multistep = report["vlm_calls"] >= 2 and any(
                row["tool"] in {"compare_region", "compare_reference"}
                for row in report["trace"]
            )
            reference = (
                directory / report["outputs"]["references"]["reference_paths"][0]
            )
            crop = output / "protocol-query-crop.png"
            matched = output / "protocol-reference-crop.png"
            # A fixed corresponding crop tests the 3-image transport, never a defect label.
            for source, target in [
                (Path(direct["cases"]["uncertain"]["path"]), crop),
                (reference, matched),
            ]:
                with Image.open(source) as image:
                    w, h = image.size
                    image.convert("RGB").crop(
                        (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
                    ).save(target)
            query = dict(id="original", path=Path(direct["cases"]["uncertain"]["path"]))
            views = [
                [query, dict(id="reference:1", path=reference)],
                [
                    query,
                    dict(id="region:R1", path=crop),
                    dict(id="matched:R1:1", path=matched),
                ],
            ]
            with store.gpu.acquire():
                for images in views:
                    decision, metrics = VLLMClient(store.gpu.config).decide(
                        dict(
                            protocol_test=True,
                            notice="Engineering image protocol only, no defect label",
                        ),
                        images,
                        output / "vlm_input" / f"protocol-{len(images)}",
                    )
                    protocol[str(len(images))] = dict(
                        structured_response=True,
                        decision=decision.model_dump(),
                        metrics=metrics,
                    )
        ttl_result = None
        if ttl_smoke:
            store.runtime.ttl = 2.0
            store.gpu.config["idle_ttl_seconds"] = 2.0
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if (
                    store.runtime.status()["state"] == "stopped"
                    and not store.gpu.status()["pid"]
                ):
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError("Idle TTL did not release model processes")
            ttl_result = dict(passed=True, temporary_idle_ttl_seconds=2)
        write_json(
            output / "verification.json",
            dict(
                passed=True,
                scope="Real CUDA engineering acceptance, normal validation only",
                direct=direct,
                results=results,
                vlm_requested=with_vlm,
                multimodal_protocol=protocol,
                full_agent_loop_exercised=multistep,
                gpu_runtime=store.gpu.status(),
                ttl=ttl_result,
                accuracy_evaluated=False,
                calibration_recomputed=False,
            ),
        )
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-vlm", action="store_true")
    parser.add_argument("--ttl-smoke", action="store_true")
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("GPU acceptance must run on the Linux NVIDIA server")
    configure_environment()
    os.environ["VISIONINSPECT_RELOCATED_ASSETS"] = "1"
    if args.probe_output:
        detector_probe(args.probe_output)
        return
    output = (
        ROOT
        / "outputs"
        / f"{datetime.now():%Y%m%d-%H%M%S}-gpu-verification-{uuid4().hex[:6]}"
    )
    output.mkdir(parents=True)
    try:
        verify(output, args.with_vlm, args.ttl_smoke)
    except Exception as error:
        import traceback

        write_json(
            output / "failure.json",
            dict(
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
                accuracy_evaluated=False,
            ),
        )
        raise
    print(json.dumps(dict(passed=True, output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
