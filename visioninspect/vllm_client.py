"""Local-only multimodal JSON client. Never loads a second language model."""

from __future__ import annotations
import base64
import time
from pathlib import Path
import httpx
from PIL import Image, ImageOps
from .common import sha256, write_json
from .review_contracts import ReviewDecision, REVIEW_PROMPT
from .tools import ToolFailure


class VLLMClient:
    backend = "vllm-qwen3-vl-8b"

    def __init__(self, config, startup_error=None):
        if config["endpoint"] != "http://127.0.0.1:8001":
            raise ValueError("Only the configured loopback vLLM endpoint is allowed")
        self.config = config
        self.startup_error = startup_error

    def decide(self, context, images, directory: Path):
        if self.startup_error:
            raise ToolFailure("vlm_unavailable", self.startup_error)
        if not 1 <= len(images) <= 3:
            raise ToolFailure("schema_error", "VLM 每次需要 1–3 张独立证据图片")
        directory.mkdir(parents=True, exist_ok=True)
        content = [
            dict(
                type="text", text=__import__("json").dumps(context, ensure_ascii=False)
            )
        ]
        audit = []
        for index, item in enumerate(images, 1):
            with Image.open(item["path"]) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                saved = directory / f"image_{index}.png"
                image.save(saved)
            content.append(dict(type="text", text=f"证据身份：{item['id']}"))
            content.append(
                dict(
                    type="image_url",
                    image_url=dict(
                        url="data:image/png;base64,"
                        + base64.b64encode(saved.read_bytes()).decode("ascii")
                    ),
                )
            )
            audit.append(
                dict(
                    id=item["id"],
                    path=saved.name,
                    sha256=sha256(saved),
                    size=list(image.size),
                    source_sha256=sha256(Path(item["path"])),
                )
            )
        payload = dict(
            model=self.config["model"],
            temperature=0,
            seed=42,
            max_tokens=768,
            messages=[
                dict(role="system", content=REVIEW_PROMPT),
                dict(role="user", content=content),
            ],
            response_format=dict(
                type="json_schema",
                json_schema=dict(
                    name="inspection-review", schema=ReviewDecision.model_json_schema()
                ),
            ),
        )
        write_json(
            directory / "request.json",
            dict(
                context=context,
                images=audit,
                prompt=REVIEW_PROMPT,
                response_schema=ReviewDecision.model_json_schema(),
                model=self.config["model"],
                temperature=0,
                seed=42,
                max_tokens=768,
            ),
        )
        started = time.monotonic()
        try:
            with httpx.Client(
                timeout=self.config["request_timeout_seconds"],
                trust_env=False,
                follow_redirects=False,
            ) as client:
                with client.stream(
                    "POST",
                    self.config["endpoint"] + "/v1/chat/completions",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        raise ToolFailure(
                            "vlm_unavailable",
                            f"本地 VLM HTTP {response.status_code}",
                            retryable=response.status_code in {429, 502, 503, 504},
                        )
                    body = bytearray()
                    for block in response.iter_bytes():
                        body.extend(block)
                        if len(body) > 1024**2:
                            raise ToolFailure("schema_error", "VLM 响应超过 1 MiB")
            import json

            raw = json.loads(body)
            write_json(directory / "response.json", raw)
            decision = ReviewDecision.model_validate_json(
                raw["choices"][0]["message"]["content"]
            )
            return decision, dict(
                seconds=time.monotonic() - started,
                usage=raw.get("usage", {}),
                backend=self.backend,
                input_images=[
                    dict(item, path=f"vlm_input/{directory.name}/{item['path']}")
                    for item in audit
                ],
            )
        except httpx.TimeoutException as error:
            raise ToolFailure("timeout", "本地 VLM 请求超时", retryable=True) from error
        except httpx.HTTPError as error:
            raise ToolFailure(
                "vlm_unavailable",
                f"本地 VLM 连接失败：{type(error).__name__}",
                retryable=True,
            ) from error
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ToolFailure("schema_error", "VLM 响应不符合结构化复核协议") from error
