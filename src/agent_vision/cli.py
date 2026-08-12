#!/usr/bin/env python3
"""agent-vision: give any AI agent vision capability.

Two modes:

  see    -- CLI: describe/analyze local images, image URLs, or the latest
            image pasted into Codex via an OpenAI-compatible vision API.
            Defaults to Zhipu GLM-4V-Flash (free).

  proxy  -- local HTTP proxy that rewrites image content to text before
            forwarding the request to a text-only upstream (DeepSeek etc.).
            Agents point their base_url at this proxy; the proxy passes the
            original Authorization header through, so the main model key
            stays in the agent's existing config.

All vision calls use only Python standard library (urllib).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import mimetypes
import os
import platform
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .adapters import ADAPTERS, AgentAdapter, CodexAdapter, get_adapter
from . import config_home
from .runtime import DEFAULT_LISTEN, RuntimeManager
from .version import VERSION


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


ENV_FILE = config_home.env_file()
DEFAULT_VISION_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_VISION_MODEL = "glm-4v-flash"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES_PER_REQUEST = 100
VISION_BATCH_SIZE = 10
REALTIME_UNSUPPORTED_PATHS = ("/v1/live", "/v1/realtime")
REALTIME_UNSUPPORTED_MESSAGE = (
    "Realtime voice (/v1/live) is not supported by the configured upstream. "
    "DeepSeek does not provide a realtime voice API, and Codex voice uses "
    "OpenAI's GPT-Live channel. 请改用文字输入；如需语音，请切换到支持实时语音的服务商。"
)
TEST_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
DEFAULT_DESCRIBE_PROMPT = (
    "Describe this image with exact facts: all visible text, UI elements, "
    "layout, colors, error messages, and any data you can read. "
    "Do not guess. Reply in Chinese unless the user asks otherwise."
)

TASK_PROMPTS: dict[str, str] = {
    "describe": DEFAULT_DESCRIBE_PROMPT,
    "ocr": (
        "Extract every visible text exactly as it appears, including error "
        "codes, numbers, field labels, timestamps and file names. Keep line "
        "order and formatting. Reply in Chinese unless the user asks otherwise."
    ),
    "ui": (
        "Describe this UI or screenshot precisely: layout, controls, states, "
        "visible text, colors, spacing and any visual problems or inconsistencies. "
        "Do not guess. Reply in Chinese unless the user asks otherwise."
    ),
    "chart": (
        "Read this chart, plot or table: title, axes, units, categories, key "
        "values, trends and anomalies. Do not invent numbers. Reply in Chinese "
        "unless the user asks otherwise."
    ),
}

PROVIDERS: dict[str, dict[str, str]] = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-flash",
        "cost": "free",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "cost": "paid",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "cost": "paid",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "cost": "free-tier",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3.6-27b",
        "cost": "free-plan",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "cost": "free-quota",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen3-vl:free",
        "cost": "free-or-paid",
    },
}

CUSTOM_PROVIDERS_FILE = config_home.providers_file()


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE file; skip comments and blank lines."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            result[key] = value
    return result


_DOTENV = load_dotenv(ENV_FILE)
_ENV = {**_DOTENV, **os.environ}


def cfg(name: str, default: str = "") -> str:
    value = _ENV.get(name)
    return value if value not in (None, "") else default


def reload_env() -> None:
    """Reload .env into the process cache after writing config files."""
    _DOTENV.clear()
    _DOTENV.update(load_dotenv(ENV_FILE))
    _ENV.clear()
    _ENV.update(_DOTENV)
    _ENV.update(os.environ)


def load_custom_providers() -> dict[str, dict[str, str]]:
    """Load user-defined provider presets from providers.json."""
    result: dict[str, dict[str, str]] = {}
    if not CUSTOM_PROVIDERS_FILE.exists():
        return result
    try:
        data = json.loads(CUSTOM_PROVIDERS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"warning: cannot load {CUSTOM_PROVIDERS_FILE}: {error}", file=sys.stderr)
        return result
    items = data if isinstance(data, list) else data.get("providers", [])
    for item in items:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        result[str(item["id"])] = {
            "base_url": str(item.get("base_url", "")).strip(),
            "model": str(item.get("model", "")).strip(),
            "cost": str(item.get("cost", "")).strip(),
            "note": str(item.get("note", "")).strip(),
        }
    return result


def all_providers() -> dict[str, dict[str, str]]:
    merged = dict(PROVIDERS)
    merged.update(load_custom_providers())
    return merged


def resolve_provider(
    provider: str | None,
    base_url: str | None,
    model: str | None,
) -> tuple[str, str]:
    """Resolve base_url and model from --provider, explicit flags, then .env."""
    if provider:
        preset = all_providers().get(provider)
        if not preset:
            raise ValueError(f"unknown provider: {provider}")
        resolved_base = base_url or preset.get("base_url") or cfg("VISION_BASE_URL", DEFAULT_VISION_BASE_URL)
        resolved_model = model or preset.get("model") or cfg("VISION_MODEL", DEFAULT_VISION_MODEL)
    else:
        resolved_base = base_url or cfg("VISION_BASE_URL", DEFAULT_VISION_BASE_URL)
        resolved_model = model or cfg("VISION_MODEL", DEFAULT_VISION_MODEL)
    return resolved_base, resolved_model


def guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("image/"):
        return mime
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(Path(path).suffix.lower(), "image/png")


def encode_image(path: str) -> tuple[str, str]:
    mime = guess_mime(path)
    with open(path, "rb") as handle:
        data = handle.read()
    if not data:
        raise ValueError(f"empty image file: {path}")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image too large: {len(data) / 1024 / 1024:.1f} MB")
    return mime, base64.b64encode(data).decode("ascii")


def data_url_from_bytes(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def call_vision_model(
    *,
    mime: str,
    b64: str,
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """Call an OpenAI-compatible chat completions vision endpoint."""
    if not api_key:
        raise RuntimeError(
            "VISION_API_KEY is not configured; set it in .env or the environment"
        )
    base = base_url.rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if "glm" in model.lower():
        payload["thinking"] = {"type": "enabled"}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    retryable = {429, 500, 502, 503, 504}
    last_error = ""
    result = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {error.code}: {detail}"
            if error.code not in retryable:
                break
        except urllib.error.URLError as error:
            last_error = f"network error: {error.reason}"
        else:
            if isinstance(result, dict) and result.get("error"):
                last_error = "api error: " + json.dumps(result["error"], ensure_ascii=False)[:500]
                if not str(result["error"]).strip():
                    break
            else:
                break
        if attempt < 3:
            wait = 3 * (attempt + 1)
            print(f"vision retry {attempt + 1} after {wait}s: {last_error}", file=sys.stderr)
            time.sleep(wait)
            continue
        raise RuntimeError(last_error)
    if result is None:
        raise RuntimeError(last_error or "vision call failed")
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("vision model returned an unparsable result")
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content or "")


def _parse_batch_descriptions(text: str, count: int) -> list[str | None]:
    """Split a batch vision response by [IMG<n>] markers."""
    out: list[str | None] = [None] * count
    marker_re = re.compile(r"\[IMG(\d+)\]")
    matches = list(marker_re.finditer(text or ""))
    for index, match in enumerate(matches):
        try:
            image_index = int(match.group(1)) - 1
        except ValueError:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        description = text[match.end():end].strip()
        if 0 <= image_index < count and description and out[image_index] is None:
            out[image_index] = description
    return out


def call_vision_model_batch(
    images: list[tuple[str, str]],
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> list[str | None]:
    """Send multiple images in one chat completion and return per-image descriptions."""
    if not api_key:
        raise RuntimeError(
            "VISION_API_KEY is not configured; set it in .env or the environment"
        )
    base = base_url.rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    content: list[dict[str, object]] = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
            },
        }
        for mime, b64 in images
    ]
    content.append(
        {
            "type": "text",
            "text": (
                f"{prompt}\n\nThere are {len(images)} images in the exact order given. "
                "Describe every image separately. Start the description of each image with "
                "the marker [IMG1], [IMG2], [IMG3], ... in order. Do not skip any image."
            ),
        }
    )
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    retryable = {429, 500, 502, 503, 504}
    last_error = ""
    result = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {error.code}: {detail}"
            if error.code not in retryable:
                break
        except urllib.error.URLError as error:
            last_error = f"network error: {error.reason}"
        else:
            if isinstance(result, dict) and result.get("error"):
                last_error = "api error: " + json.dumps(result["error"], ensure_ascii=False)[:500]
                if not str(result["error"]).strip():
                    break
            else:
                break
        if attempt < 3:
            wait = 3 * (attempt + 1)
            print(f"vision batch retry {attempt + 1} after {wait}s: {last_error}", file=sys.stderr)
            time.sleep(wait)
            continue
        raise RuntimeError(last_error)
    if result is None:
        raise RuntimeError(last_error or "vision batch call failed")
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("vision model returned an unparsable batch result")
    if isinstance(content, list):
        text = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        text = str(content or "")
    return _parse_batch_descriptions(text, len(images))


_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_MAX = 256
_MEMORY_CACHE_LOCK = threading.Lock()
_DISK_CACHE_DIR = config_home.agent_vision_home() / "cache"


def _disk_cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _DISK_CACHE_DIR / f"{digest}.json"


def _cache_key(data: bytes, prompt: str, model: str) -> str:
    digest = hashlib.sha256(data).hexdigest()
    return f"{digest}|{model}|{prompt}"


def _cache_get(key: str) -> str | None:
    with _MEMORY_CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    path = _disk_cache_path(key)
    try:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, str):
                with _MEMORY_CACHE_LOCK:
                    _CACHE[key] = value
                    _CACHE.move_to_end(key)
                    while len(_CACHE) > _CACHE_MAX:
                        _CACHE.popitem(last=False)
                return value
    except (OSError, ValueError):
        pass
    return None


def _cache_set(key: str, value: str) -> None:
    with _MEMORY_CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    try:
        _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _disk_cache_path(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def describe_bytes(
    data: bytes,
    mime: str,
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    use_cache: bool = True,
) -> str:
    model = model or cfg("VISION_MODEL", DEFAULT_VISION_MODEL)
    api_key = api_key or cfg("VISION_API_KEY")
    base_url = base_url or cfg("VISION_BASE_URL", DEFAULT_VISION_BASE_URL)
    key = _cache_key(data, prompt, model)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached
    b64 = base64.b64encode(data).decode("ascii")
    description = call_vision_model(
        mime=mime,
        b64=b64,
        prompt=prompt,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
    if use_cache:
        _cache_set(key, description)
    return description


def describe_file(
    path: str,
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    use_cache: bool = True,
) -> str:
    mime, b64 = encode_image(path)
    data = base64.b64decode(b64)
    return describe_bytes(
        data,
        mime,
        prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        use_cache=use_cache,
    )


DATA_URL_RE = re.compile(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.+)$", re.DOTALL)


def image_url_from_part(part: object) -> str | None:
    if not isinstance(part, dict):
        return None
    ptype = part.get("type")
    url = part.get("image_url") or part.get("url") or ""
    if isinstance(url, dict):
        url = url.get("url") or ""
    if not isinstance(url, str) or not url.startswith("data:image/"):
        return None
    if ptype in ("image_url", "input_image") or isinstance(part.get("image_url"), str):
        return url
    return None



def load_url_image(url: str) -> tuple[str, bytes]:
    """Download an image URL and return (mime, bytes)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"image URL must use http:// or https://: {url}")
    try:
        response = urllib.request.urlopen(url, timeout=60)
    except urllib.error.URLError as error:
        raise ValueError(f"cannot download image URL: {error.reason}")
    try:
        length = response.headers.get("Content-Length")
        if length:
            try:
                if int(length) > MAX_IMAGE_BYTES:
                    raise ValueError(f"image too large: {int(length) / 1024 / 1024:.1f} MB")
            except ValueError:
                pass
        data = bytearray()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError(f"image too large: {len(data) / 1024 / 1024:.1f} MB")
    finally:
        response.close()
    mime = response.headers.get_content_type() or guess_mime(url)
    if not mime.startswith("image/"):
        mime = guess_mime(url)
    if not data:
        raise ValueError(f"empty image from URL: {url}")
    return mime, bytes(data)


def describe_source(
    source: str,
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    use_cache: bool = True,
) -> str:
    """Describe a local image path or an http(s) image URL."""
    if source.startswith(("http://", "https://")):
        mime, data = load_url_image(source)
        return describe_bytes(
            data,
            mime,
            prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_cache=use_cache,
        )
    return describe_file(
        source,
        prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        use_cache=use_cache,
    )


def _find_input_image(obj: object) -> tuple[str, bytes] | None:
    """Find an input_image part with a data URL in a decoded JSON value."""
    if isinstance(obj, dict):
        if obj.get("type") == "input_image":
            url = obj.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            match = DATA_URL_RE.match(url) if isinstance(url, str) else None
            if match:
                mime, b64 = match.group(1), match.group(2)
                try:
                    data = base64.b64decode(b64)
                except Exception:
                    data = b""
                if data and len(data) <= MAX_IMAGE_BYTES:
                    return mime, data
        for value in obj.values():
            found = _find_input_image(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_input_image(value)
            if found:
                return found
    return None


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _iter_reversed_lines(handle: object, chunk_size: int = 64 * 1024) -> str:
    """Yield text lines from a binary file from the end, without loading it all."""
    handle.seek(0, os.SEEK_END)
    tail = b""
    pos = handle.tell()
    while pos > 0:
        read_size = min(chunk_size, pos)
        pos -= read_size
        handle.seek(pos)
        tail = handle.read(read_size) + tail
        lines = tail.split(b"\n")
        tail = lines[0]
        for line in reversed(lines[1:]):
            yield line.decode("utf-8", errors="replace")
    if tail:
        yield tail.decode("utf-8", errors="replace")


def find_latest_pasted_image(
    session_dir: Path | None = None,
    max_files: int = 20,
) -> tuple[str, bytes]:
    """Return (mime, bytes) of the most recent image pasted into Codex.

    Codex stores pasted images as ``input_image`` parts with base64 data URLs
    in ``~/.codex/sessions/**/*.jsonl`` (or ``$CODEX_HOME/sessions``). Only the
    newest ``max_files`` session files are scanned; each file is read backwards
    so the first hit is the latest pasted image.
    """
    codex_home = os.environ.get("CODEX_HOME")
    root = Path(session_dir) if session_dir is not None else Path(codex_home or Path.home() / ".codex") / "sessions"
    if not root.is_dir():
        raise ValueError(f"Codex sessions directory not found: {root}")
    try:
        candidates = list(root.rglob("*.jsonl"))
    except OSError:
        candidates = []
    files = sorted(candidates, key=_safe_mtime, reverse=True)[: max_files]
    for file in files:
        try:
            with file.open("rb") as handle:
                for line in _iter_reversed_lines(handle):
                    if "input_image" not in line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    found = _find_input_image(obj)
                    if found:
                        return found
        except OSError:
            continue
    raise ValueError(f"no pasted image found in recent Codex sessions under {root}")


def _focus_text(content: list[object]) -> str:
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return " ".join(parts).strip()


class _Rewrite:
    def __init__(
        self,
        max_images: int = MAX_IMAGES_PER_REQUEST,
        model: str | None = None,
        base_url: str | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.replaced = 0
        self.max_images = max_images
        self.model = model
        self.base_url = base_url
        self.log = log
        self.last_error: str | None = None
        self.modified = False

    def describe_data_url(self, data_url: str, focus: str) -> str | None:
        description, error = self._describe_data_url(data_url, focus)
        self.last_error = error
        return description

    def _image_meta(self, data_url: str, focus: str) -> tuple[dict[str, object] | None, str]:
        match = DATA_URL_RE.match(data_url)
        if not match:
            return None, "unsupported image data URL format"
        mime, b64 = match.group(1), match.group(2)
        try:
            data = base64.b64decode(b64)
        except Exception as error:
            return None, f"invalid base64 image data: {error}"
        if not data:
            return None, "empty image data"
        if len(data) > MAX_IMAGE_BYTES:
            return None, f"image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
        prompt = DEFAULT_DESCRIBE_PROMPT
        if focus:
            prompt = (
                f"The user's request is: {focus[:500]}\n\n"
                + DEFAULT_DESCRIBE_PROMPT
            )
        key = _cache_key(data, prompt, self.model or cfg("VISION_MODEL", DEFAULT_VISION_MODEL))
        return {
            "mime": mime,
            "b64": b64,
            "data": data,
            "prompt": prompt,
            "key": key,
        }, ""

    def _describe_data_url(self, data_url: str, focus: str) -> tuple[str | None, str]:
        meta, error = self._image_meta(data_url, focus)
        if error:
            return None, error
        try:
            description = describe_bytes(
                meta["data"],
                meta["mime"],
                meta["prompt"],
                model=self.model,
                base_url=self.base_url,
            )
            return description, ""
        except Exception as error:
            message = str(error) or error.__class__.__name__
            print(f"vision rewrite failed: {message}", file=sys.stderr)
            if self.log:
                self.log(f"vision rewrite failed: {message}")
            return None, message

    def rewrite_content(self, content: object, chat: bool) -> object:
        if not isinstance(content, list):
            return content
        focus = _focus_text(content)
        todo: list[tuple[int, str]] = []
        for index, part in enumerate(content):
            url = image_url_from_part(part)
            if url and self.replaced < self.max_images:
                todo.append((index, url))
                self.replaced += 1
        results: dict[int, tuple[str | None, str]] = {}
        uncached: list[tuple[int, dict[str, object]]] = []
        for index, url in todo:
            meta, error = self._image_meta(url, focus)
            if error:
                results[index] = (None, error)
                continue
            cached = _cache_get(meta["key"])
            if cached is not None:
                results[index] = (cached, "")
            else:
                uncached.append((index, meta))

        if len(uncached) == 1:
            index, meta = uncached[0]
            try:
                description = describe_bytes(
                    meta["data"],
                    meta["mime"],
                    meta["prompt"],
                    model=self.model,
                    base_url=self.base_url,
                )
                _cache_set(meta["key"], description)
                results[index] = (description, "")
            except Exception as error:
                message = str(error) or error.__class__.__name__
                print(f"vision rewrite failed: {message}", file=sys.stderr)
                if self.log:
                    self.log(f"vision rewrite failed: {message}")
                results[index] = (None, message)
        elif uncached:
            for start in range(0, len(uncached), VISION_BATCH_SIZE):
                chunk = uncached[start : start + VISION_BATCH_SIZE]
                descriptions: list[str | None] | None = None
                try:
                    descriptions = call_vision_model_batch(
                        [(meta["mime"], meta["b64"]) for _, meta in chunk],
                        str(chunk[0][1]["prompt"]),
                        model=self.model,
                        base_url=self.base_url,
                    )
                except Exception as error:
                    message = str(error) or error.__class__.__name__
                    print(f"vision batch failed: {message}", file=sys.stderr)
                    if self.log:
                        self.log(f"vision batch failed: {message}")
                for offset, (index, meta) in enumerate(chunk):
                    description = None
                    if descriptions is not None and offset < len(descriptions):
                        description = descriptions[offset]
                    if description:
                        _cache_set(meta["key"], description)
                        results[index] = (description, "")
                        continue
                    try:
                        fallback = describe_bytes(
                            meta["data"],
                            meta["mime"],
                            meta["prompt"],
                            model=self.model,
                            base_url=self.base_url,
                        )
                        _cache_set(meta["key"], fallback)
                        results[index] = (fallback, "")
                    except Exception as error:
                        message = str(error) or error.__class__.__name__
                        print(f"vision rewrite failed: {message}", file=sys.stderr)
                        if self.log:
                            self.log(f"vision rewrite failed: {message}")
                        results[index] = (None, message)
        out: list[object] = []
        for index, part in enumerate(content):
            url = image_url_from_part(part)
            if not url:
                out.append(part)
                continue
            text_type = "text" if chat else "input_text"
            if index in results:
                self.modified = True
                description, error = results[index]
                if description:
                    out.append(
                        {
                            "type": text_type,
                            "text": "[image described by vision model] " + description.strip(),
                        }
                    )
                else:
                    reason = error or "unknown vision conversion error"
                    if self.log:
                        self.log(f"image conversion failed, replacing image with marker: {reason}")
                    out.append(
                        {
                            "type": text_type,
                            "text": (
                                "[image vision conversion failed: " + reason + "] "
                                "请用户重新粘贴图片或稍后重试。"
                            ),
                        }
                    )
                continue
            self.modified = True
            out.append(
                {
                    "type": text_type,
                    "text": (
                        f"[image omitted: too many images in one request "
                        f"(limit {self.max_images}); describe images separately]"
                    ),
                }
            )
        return out


def rewrite_body(
    body: bytes,
    max_images: int = MAX_IMAGES_PER_REQUEST,
    model: str | None = None,
    base_url: str | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[bytes, int]:
    """Replace image content parts with text. Returns (body, replaced_count)."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body, 0
    if not isinstance(payload, dict):
        return body, 0
    rewrite = _Rewrite(max_images=max_images, model=model, base_url=base_url, log=log)
    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("content"), list):
            item["content"] = rewrite.rewrite_content(item["content"], chat=False)
        if item.get("type") == "function_call_output" and isinstance(item.get("output"), list):
            item["output"] = rewrite.rewrite_content(item["output"], chat=False)
    for message in payload.get("messages") or []:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            message["content"] = rewrite.rewrite_content(message["content"], chat=True)
    if rewrite.replaced == 0 and not rewrite.modified:
        return body, 0
    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), rewrite.replaced


class _ProxyLog:
    """Append proxy diagnostics to a UTF-8 log file with a timestamp."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}\n"
        with self.lock:
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                pass


def _upstream_request(
    upstream: str,
    method: str,
    path: str,
    body: bytes,
    headers: dict[str, str],
    retries: int = 3,
) -> tuple[object, object]:
    """Connect to the upstream with a few retries to absorb transient DNS/network errors."""
    parsed = urlparse(upstream)
    if parsed.scheme == "https":
        connection_cls = http.client.HTTPSConnection
        default_port = 443
    else:
        connection_cls = http.client.HTTPConnection
        default_port = 80
    last_error: Exception | None = None
    for attempt in range(retries):
        conn = connection_cls(parsed.hostname or "127.0.0.1", parsed.port or default_port, timeout=300)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            return conn, response
        except Exception as error:
            last_error = error
            try:
                conn.close()
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"upstream unreachable after {retries} attempts: {last_error}")


def blocked_realtime_path(path: str) -> bool:
    """Return True when a Codex realtime voice request must not reach the upstream."""
    clean = path.split("?", 1)[0]
    return clean.startswith(REALTIME_UNSUPPORTED_PATHS)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if blocked_realtime_path(self.path):
            self._reject_realtime_voice()
            return
        new_body, _ = (
            rewrite_body(
                body,
                max_images=self.server.max_images,
                model=getattr(self.server, "model", None),
                base_url=getattr(self.server, "base_url", None),
                log=getattr(self.server, "logger", None),
            )
            if body
            else (body, 0)
        )
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in ("host", "content-length", "connection", "transfer-encoding"):
                continue
            headers[key] = value
        headers["Content-Length"] = str(len(new_body))
        try:
            conn, response = _upstream_request(
                self.server.upstream,
                self.command,
                self.path,
                new_body,
                headers,
            )
        except RuntimeError as error:
            message = (
                f"{error}. This is a transient upstream DNS/network error; "
                "your machine's network settings were not changed. "
                "Retry or run `agent-vision doctor`."
            )
            payload = message.encode("utf-8", errors="replace")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(response.status)
        for key, value in response.getheaders():
            lower = key.lower()
            if lower in ("content-length", "transfer-encoding", "connection"):
                continue
            self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            response.close()
            conn.close()

    def _reject_realtime_voice(self) -> None:
        logger = getattr(self.server, "logger", None)
        upstream = getattr(self.server, "upstream", "?")
        if logger:
            logger.write(
                f"realtime voice request blocked: {self.command} {self.path} "
                f"(upstream {upstream} has no realtime transport)"
            )
        payload = json.dumps(
            {
                "error": {
                    "message": REALTIME_UNSUPPORTED_MESSAGE,
                    "type": "unsupported_realtime_voice",
                    "code": "realtime_voice_unsupported",
                }
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(501)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_POST = _forward
    do_GET = _forward
    do_PUT = _forward
    do_DELETE = _forward
    do_OPTIONS = _forward

    def log_message(self, fmt: str, *args: object) -> None:
        line = "%s - %s" % (self.address_string(), fmt % args)
        sys.stderr.write(line + "\n")
        logger = getattr(self.server, "logger", None)
        if logger:
            logger.write(line)


def run_proxy(
    listen: str,
    upstream: str,
    max_images: int,
    provider: str | None = None,
    log_file: str | None = None,
) -> int:
    if log_file is None:
        log_file = str(config_home.runtime_log_file().parent / "proxy.log")
    logger = _ProxyLog(log_file)
    key_ready = bool(cfg("VISION_API_KEY"))
    if not key_ready:
        message = (
            "warning: VISION_API_KEY not configured; image conversion will fail "
            "and images will be replaced with failure markers"
        )
        print(message, file=sys.stderr)
        logger.write(message)
    try:
        base_url, model = resolve_provider(provider, None, None)
    except ValueError as error:
        raise SystemExit(str(error))
    host, _, port = listen.rpartition(":")
    parsed = urlparse(upstream)
    if parsed.scheme not in ("http", "https"):
        raise SystemExit("--upstream must be http:// or https:// origin, e.g. https://api.deepseek.com")
    server = ThreadingHTTPServer((host or "127.0.0.1", int(port or 19100)), ProxyHandler)
    server.upstream = upstream.rstrip("/")
    server.max_images = max_images
    server.model = model
    server.base_url = base_url
    server.logger = logger
    startup = (
        f"vision bridge proxy listening on {listen} -> {upstream}; "
        f"vision key configured: {key_ready}"
    )
    print(startup, flush=True)
    logger.write(startup)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_see(args: argparse.Namespace) -> int:
    exit_code = 0
    try:
        base_url, model = resolve_provider(args.provider, args.base_url, args.model)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if getattr(args, "latest", False) and args.images:
        print("error: --latest cannot be combined with image paths/URLs", file=sys.stderr)
        return 2
    prompt = args.question or TASK_PROMPTS.get(args.task or "describe", DEFAULT_DESCRIBE_PROMPT)
    if getattr(args, "latest", False):
        label = "[latest pasted image]"
        try:
            mime, data = find_latest_pasted_image()
        except (OSError, ValueError) as error:
            print(f"failed {label}: {error}", file=sys.stderr)
            return 1
        try:
            text = describe_bytes(
                data,
                mime,
                prompt,
                model=model,
                api_key=args.api_key,
                base_url=base_url,
                use_cache=not args.no_cache,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"failed {label}: {error}", file=sys.stderr)
            return 1
        print(f"===== {label} =====")
        print((text or "").strip())
        print()
        return 0
    if not args.images:
        print("error: specify image paths/URLs or use --latest", file=sys.stderr)
        return 2
    for index, image in enumerate(args.images, start=1):
        label = image if len(args.images) == 1 else f"[{index}/{len(args.images)}] {image}"
        try:
            text = describe_source(
                image,
                prompt,
                model=model,
                api_key=args.api_key,
                base_url=base_url,
                use_cache=not args.no_cache,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"failed {label}: {error}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"===== {label} =====")
        print((text or "").strip())
        print()
    return exit_code


def cmd_doctor(_args: argparse.Namespace) -> int:
    print("vision base url:", cfg("VISION_BASE_URL", DEFAULT_VISION_BASE_URL))
    print("vision model:   ", cfg("VISION_MODEL", DEFAULT_VISION_MODEL))
    print("api key set:    ", "yes" if cfg("VISION_API_KEY") else "no")
    checks: list[bool] = []

    def report(name: str, ok: bool, hint: str = "") -> None:
        mark = "✓" if ok else "✗"
        line = f"{mark} {name}"
        if hint:
            line += f"  ({hint})"
        print(line)
        checks.append(ok)

    entry_ok = bool(shutil.which("agent-vision"))
    try:
        launcher = ensure_launcher()
        launcher_ok = True
    except OSError:
        launcher = None
        launcher_ok = False
    report(
        "command entrypoint",
        entry_ok or launcher_ok,
        "use python -m agent_vision or " + str(launcher) if launcher else "use python -m agent_vision",
    )
    report(
        "config home writable",
        config_home_writable(),
        "grant access or run the generated finalize script",
    )
    try:
        runtime = make_runtime_manager()
        runtime_ok = bool(runtime.status().get("ready"))
    except Exception:
        runtime_ok = False
    report("proxy running", runtime_ok, "run `agent-vision start` (or `python -m agent_vision start`)")
    try:
        adapter = make_codex_adapter()
        detection = adapter.detect()
        base_url = str(detection.get("base_url") or "")
        codex_ok = "127.0.0.1" in base_url or "localhost" in base_url
        catalog_ok = bool(detection.get("catalog_patched"))
    except Exception:
        codex_ok = False
        catalog_ok = False
    report(
        "Codex points at local proxy",
        codex_ok,
        "run `agent-vision setup --agent codex --provider free --yes`",
    )
    report("model catalog image input", catalog_ok, "rerun setup to declare image input")
    try:
        autostart_ok = autostart_enabled()
    except Exception:
        autostart_ok = False
    report("autostart enabled", autostart_ok, "run `agent-vision autostart --enable`")
    if cfg("VISION_API_KEY"):
        vision = run_vision_test()
        report("vision available", bool(vision.get("ok")), str(vision.get("error") or ""))
    else:
        report("vision available", False, "VISION_API_KEY not set")
    return 0 if cfg("VISION_API_KEY") and all(checks) else 1


def cmd_providers(_args: argparse.Namespace) -> int:
    providers = all_providers()
    if not providers:
        print("no providers configured")
        return 0
    width = max(len(pid) for pid in providers)
    for pid in sorted(providers):
        preset = providers[pid]
        cost = preset.get("cost") or "n/a"
        note = preset.get("note") or ""
        print(f"{pid:<{width}}  model={preset.get('model') or '-'}  cost={cost}  {note}".rstrip())
    return 0


SETUP_MODES = {
    "free": ("1. Free（免费）", "智谱 GLM-4V-Flash，永久免费"),
    "quality": ("2. Quality（效果优先）", "通义千问 / GPT-4o / Gemini 中任选"),
    "custom": ("3. Custom（自定义服务商）", "填写自己的 OpenAI 兼容接口"),
}
SETUP_FREE_PRESET = "zhipu"
SETUP_QUALITY_OPTIONS = [
    ("dashscope", "阿里通义千问 Qwen-VL-Max（国内访问快，效果强）"),
    ("openai", "OpenAI GPT-4o-mini（海外，综合能力强）"),
    ("gemini", "Google Gemini 2.0 Flash（免费额度）"),
]
SETUP_KEY_HINTS = {
    "zhipu": "智谱 BigModel: https://open.bigmodel.cn/",
    "dashscope": "阿里云百炼: https://bailian.console.aliyun.com/",
    "openai": "OpenAI: https://platform.openai.com/api-keys",
    "gemini": "Google AI Studio: https://aistudio.google.com/apikey",
}


def detect_environment() -> dict[str, str]:
    """Collect environment facts for the setup report."""
    location = str(Path(__file__).resolve())
    install = "installed (site-packages)" if "site-packages" in location else "source tree (editable or PYTHONPATH)"
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "version": VERSION,
        "install": install,
    }


def agent_evidence(
    home: Path,
    appdata: str | None,
    which: Callable[[str], str | None],
) -> list[dict[str, object]]:
    """Detect installed agents from config paths and executables. Read-only."""
    candidates = [
        {
            "id": "codex",
            "name": "Codex",
            "paths": [home / ".codex" / "config.toml", home / ".codex" / "config.json"],
            "executables": ["codex"],
        },
        {
            "id": "claude",
            "name": "Claude Code",
            "paths": [home / ".claude.json", home / ".claude" / "settings.json"],
            "executables": ["claude"],
        },
        {
            "id": "cursor",
            "name": "Cursor",
            "paths": [
                Path(appdata) / "Cursor" if appdata else None,
                home / ".config" / "Cursor",
                home / "Library" / "Application Support" / "Cursor",
            ],
            "executables": ["cursor"],
        },
        {
            "id": "opencode",
            "name": "OpenCode",
            "paths": [
                Path(appdata) / "opencode" if appdata else None,
                home / ".config" / "opencode",
                home / ".local" / "share" / "opencode",
            ],
            "executables": ["opencode"],
        },
    ]
    result: list[dict[str, object]] = []
    for marker in candidates:
        evidence: list[str] = []
        for path in marker["paths"]:
            if path and path.exists():
                evidence.append(str(path))
        for executable in marker["executables"]:
            found = which(executable)
            if found:
                evidence.append(f"{executable} ({found})")
        result.append(
            {
                "id": marker["id"],
                "name": marker["name"],
                "found": bool(evidence),
                "evidence": evidence,
            }
        )
    return result


def detect_agents() -> list[dict[str, object]]:
    home = Path.home()
    appdata = os.environ.get("APPDATA")
    return agent_evidence(home, appdata, shutil.which)


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def choose_setup_mode(choice: str | None) -> str:
    if choice in SETUP_MODES:
        return choice
    print("选择视觉模式：")
    for _key, (label, description) in SETUP_MODES.items():
        print(f"  {label} - {description}")
    picked = _ask("输入 1/2/3（默认 1）: ") or "1"
    mapping = {"1": "free", "2": "quality", "3": "custom"}
    return mapping.get(picked, "free")


def choose_quality_provider() -> str:
    print("选择效果优先的服务商：")
    for index, (_provider_id, label) in enumerate(SETUP_QUALITY_OPTIONS, start=1):
        print(f"  {index}. {label}")
    picked = _ask("输入序号（默认 1）: ") or "1"
    try:
        return SETUP_QUALITY_OPTIONS[int(picked) - 1][0]
    except (ValueError, IndexError):
        return SETUP_QUALITY_OPTIONS[0][0]


def quality_preset(provider_id: str) -> dict[str, str]:
    preset = PROVIDERS.get(provider_id)
    if not preset:
        raise ValueError(f"unknown quality provider: {provider_id}")
    return {"id": provider_id, "base_url": preset["base_url"], "model": preset["model"], "cost": preset.get("cost", "")}


def custom_preset(base_url: str, model: str, cost: str = "") -> dict[str, str]:
    return {
        "id": "custom",
        "base_url": base_url.strip(),
        "model": model.strip(),
        "cost": cost.strip() or "custom",
    }


def collect_custom_provider() -> dict[str, str]:
    print("自定义 OpenAI 兼容服务商：")
    base_url = _ask("接口地址 base_url（例如 https://api.example.com/v1）: ")
    model = _ask("模型名 model（例如 your-vision-model）: ")
    cost = _ask("计费说明（例如 paid / free-tier，可留空）: ")
    if not base_url or not model:
        raise ValueError("base_url 和 model 不能为空")
    return custom_preset(base_url, model, cost)


def resolve_api_key(provided: str | None, existing: str) -> str:
    if provided:
        return provided.strip()
    if existing:
        return existing
    print("还没有配置 API Key。")
    return _ask("请粘贴 API Key 后回车（可留空，稍后自行填写）: ")


def render_dotenv(path: Path, updates: dict[str, str]) -> str:
    """Render updated .env content, preserving comments and unrelated keys."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        key = None
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
        if key in updates and key not in seen:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(raw)
    for key, value in updates.items():
        if key not in seen:
            if out and out[-1].strip() != "":
                out.append("")
            out.append(f"{key}={value}")
    return "\n".join(out).rstrip() + "\n"


def render_providers_json(path: Path, entry: dict[str, str]) -> str:
    """Render providers.json content, replacing an existing entry with the same id."""
    providers: list[dict[str, str]] = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data if isinstance(data, list) else data.get("providers", [])
            providers = [item for item in raw if isinstance(item, dict)]
        except (OSError, ValueError):
            providers = []
    provider_id = entry.get("id")
    providers = [item for item in providers if item.get("id") != provider_id]
    providers.append(entry)
    return json.dumps({"providers": providers}, ensure_ascii=False, indent=2) + "\n"


def cmd_setup(args: argparse.Namespace) -> int:
    agent_id = getattr(args, "agent", None)
    explicit_provider = getattr(args, "provider", None) or getattr(args, "base_url", None)
    if agent_id:
        return cmd_setup_full(args, agent_id=agent_id)
    if not explicit_provider:
        for candidate_id in ("codex", "opencode"):
            try:
                detection = make_adapter(candidate_id).detect()
            except Exception:
                detection = {}
            found = bool(detection.get("config_exists") or detection.get("installed"))
            if found:
                name = str(detection.get("name") or candidate_id)
                print(f"Detected {name} - running one-click setup.\n")
                return cmd_setup_full(args, agent_id=candidate_id)
    return cmd_setup_provider(args)


def make_codex_adapter() -> CodexAdapter:
    return CodexAdapter()


def make_adapter(agent_id: str) -> AgentAdapter:
    if agent_id == "codex":
        return make_codex_adapter()
    return get_adapter(agent_id)


def make_runtime_manager() -> RuntimeManager:
    return RuntimeManager()


def select_provider_config(args: argparse.Namespace) -> tuple[dict[str, str], str]:
    mode = choose_setup_mode(getattr(args, "provider", None))
    quality_choice: str | None = None
    custom_values: dict[str, str] | None = None
    if mode == "quality" and not getattr(args, "provider", None):
        quality_choice = choose_quality_provider()
    if mode == "custom":
        if getattr(args, "base_url", None) and getattr(args, "model", None):
            custom_values = custom_preset(args.base_url, args.model, getattr(args, "cost", "") or "")
        else:
            custom_values = collect_custom_provider()
    if mode == "free":
        values = quality_preset(SETUP_FREE_PRESET)
    elif mode == "quality":
        values = quality_preset(quality_choice or SETUP_QUALITY_OPTIONS[0][0])
    else:
        values = custom_values or quality_preset(SETUP_FREE_PRESET)
    api_key = resolve_api_key(getattr(args, "api_key", None), cfg("VISION_API_KEY"))
    return values, api_key


def provider_config_plan(values: dict[str, str], api_key: str) -> tuple[str, str | None]:
    dotenv_updates = {
        "VISION_API_KEY": api_key,
        "VISION_BASE_URL": values["base_url"],
        "VISION_MODEL": values["model"],
    }
    dotenv_content = render_dotenv(ENV_FILE, dotenv_updates)
    providers_content = render_providers_json(CUSTOM_PROVIDERS_FILE, values) if values.get("id") == "custom" else None
    return dotenv_content, providers_content


def cmd_setup_provider(args: argparse.Namespace) -> int:
    env = detect_environment()
    print("== agent-vision setup ==")
    print(f"Python {env['python']} | {env['system']} ({env['machine']}) | agent-vision {env['version']} ({env['install']})")

    agents = detect_agents()
    print("\nDetected Agents:")
    for agent in agents:
        mark = "✓" if agent["found"] else "✗"
        print(f"{mark} {agent['name']}")

    values, api_key = select_provider_config(args)
    dotenv_content, providers_content = provider_config_plan(values, api_key)

    print("\nPlan:")
    print(f"  {ENV_FILE}")
    print(f"    VISION_API_KEY = {'<provided>' if api_key else '<empty>'}")
    print(f"    VISION_BASE_URL = {values['base_url']}")
    print(f"    VISION_MODEL = {values['model']}")
    if providers_content is None:
        print("  providers.json: no change (uses built-in presets)")
    else:
        print(f"  {CUSTOM_PROVIDERS_FILE}")
        print(f"    update provider id={values['id']}")

    if args.dry_run:
        print("\nDry run: no files were modified.")
        return 0


    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(dotenv_content, encoding="utf-8")
    if providers_content is not None:
        CUSTOM_PROVIDERS_FILE.write_text(providers_content, encoding="utf-8")
    reload_env()

    hint = SETUP_KEY_HINTS.get(values["id"], "")
    print("\nDone. Config written.")
    if not api_key:
        print("warning: VISION_API_KEY is empty; edit .env or rerun setup to add it")
    elif hint:
        print(f"API key source: {hint}")
    return 0


def resolve_proxy_upstream(
    explicit: str | None,
    adapter: AgentAdapter | None = None,
    runtime: RuntimeManager | None = None,
) -> str:
    if explicit:
        return explicit.strip()
    env_upstream = cfg("VISION_PROXY_UPSTREAM")
    if env_upstream:
        return env_upstream
    runtime = runtime or make_runtime_manager()
    state = runtime.state()
    if state.get("upstream"):
        return str(state["upstream"])
    if adapter is not None:
        base_url = str(adapter.detect().get("base_url") or "")
        if base_url and "127.0.0.1" not in base_url and "localhost" not in base_url:
            return base_url
    return ""


def cmd_setup_full(args: argparse.Namespace, agent_id: str) -> int:
    env = detect_environment()
    print("== agent-vision setup ==")
    print(f"Python {env['python']} | {env['system']} ({env['machine']}) | agent-vision {env['version']} ({env['install']})")

    agents = detect_agents()
    print("\nDetected Agents:")
    for agent in agents:
        mark = "✓" if agent["found"] else "✗"
        print(f"{mark} {agent['name']}")

    values, api_key = select_provider_config(args)
    dotenv_content, providers_content = provider_config_plan(values, api_key)
    runtime = make_runtime_manager()
    adapter = make_adapter(agent_id)
    upstream = resolve_proxy_upstream(getattr(args, "proxy_upstream", None), adapter, runtime)
    listen = getattr(args, "listen", None) or cfg("VISION_PROXY_LISTEN") or DEFAULT_LISTEN
    try:
        agent_plan = adapter.plan(upstream=upstream or None)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    detection = agent_plan["detection"]
    manual_steps = agent_plan.get("manual_steps") or []

    print("\nPlan:")
    print(f"  {ENV_FILE}")
    print(f"    VISION_API_KEY = {'<provided>' if api_key else '<empty>'}")
    print(f"    VISION_BASE_URL = {values['base_url']}")
    print(f"    VISION_MODEL = {values['model']}")
    if providers_content is None:
        print("  providers.json: no change (uses built-in presets)")
    else:
        print(f"  {CUSTOM_PROVIDERS_FILE}: update provider id={values['id']}")
    if manual_steps:
        print(f"  {detection['name']}: manual configuration required")
        for step in manual_steps:
            print(f"    - {step}")
    else:
        print(f"  {detection['name']} config: {detection['config_path']}")
        for file_entry in agent_plan.get("files", []):
            print(f"    {file_entry.get('summary') or file_entry.get('action')}")
        print(f"  Runtime: start proxy on {listen}" + (f" -> {upstream}" if upstream else " (upstream missing)"))
        print("  Verify: vision API connectivity test")

    if args.dry_run:
        print("\nDry run: no files were modified and no runtime was started.")
        return 0

    if manual_steps:
        print("\nThis agent is not auto-patched yet. Follow the manual steps above.")
        return 1

    if not getattr(args, "yes", False):
        answer = _ask("Apply changes and start the runtime? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("cancelled")
            return 1

    if not config_home_writable():
        scripts = write_finalize_scripts()
        print("\nWarning: cannot write the user config directory from this session.", file=sys.stderr)
        print("A finalize script was generated. Run it in a normal terminal:", file=sys.stderr)
        for script in scripts:
            print(f"  {script}", file=sys.stderr)
        return 0

    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(dotenv_content, encoding="utf-8")
    if providers_content is not None:
        CUSTOM_PROVIDERS_FILE.write_text(providers_content, encoding="utf-8")
    reload_env()

    if not upstream:
        print("error: cannot determine proxy upstream; pass --proxy-upstream", file=sys.stderr)
        return 1

    start_result = runtime.start(upstream=upstream, listen=listen)
    if not start_result.get("ready"):
        print(f"error: runtime is not ready; see {runtime.log_file}", file=sys.stderr)
        return 1

    try:
        result = adapter.apply(upstream=upstream)
    except (OSError, ValueError, NotImplementedError) as error:
        scripts = write_finalize_scripts()
        print(f"error: {error}", file=sys.stderr)
        print("A finalize script was generated. Run it in a normal terminal:", file=sys.stderr)
        for script in scripts:
            print(f"  {script}", file=sys.stderr)
        return 1

    print(f"\nDone. {detection['name']} config updated.")
    print(f"  backup: {result['backup_path']}")
    vision = run_vision_test()
    health = collect_health(runtime=runtime, adapter=adapter, vision_available=bool(vision.get("ok")))
    print()
    print_health(health)
    if not vision.get("ok"):
        print(f"\nwarning: vision test failed: {vision.get('error')}", file=sys.stderr)
        return 1
    try:
        launcher = ensure_launcher()
        print(f"\nLauncher: {launcher}")
    except OSError as error:
        print(f"warning: cannot create launcher: {error}", file=sys.stderr)
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    try:
        adapter = get_adapter(args.agent)
        result = adapter.rollback()
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Rolled back {args.agent}:")
    print(f"  config: {result['config_path']}")
    print(f"  restored from: {result['restored_from']}")
    print(f"  state removed: {result['state_path']}")
    if result.get("catalog_restored_from"):
        print(f"  catalog restored from: {result['catalog_restored_from']}")
    return 0


def vision_ready() -> dict[str, str]:
    return {
        "key_set": bool(cfg("VISION_API_KEY")),
        "base_url": cfg("VISION_BASE_URL", DEFAULT_VISION_BASE_URL),
        "model": cfg("VISION_MODEL", DEFAULT_VISION_MODEL),
    }


def run_vision_test() -> dict[str, object]:
    ready = vision_ready()
    if not ready["key_set"]:
        return {"ok": False, "error": "VISION_API_KEY not set"}
    try:
        text = describe_bytes(
            base64.b64decode(TEST_PNG_B64),
            "image/png",
            "Reply with exactly OK",
            model=ready["model"],
            base_url=ready["base_url"],
            use_cache=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return {"ok": False, "error": str(error)}
    return {"ok": True, "text": (text or "").strip()}


def collect_health(
    runtime: RuntimeManager | None = None,
    adapter: AgentAdapter | None = None,
    vision_available: bool | None = None,
) -> dict[str, object]:
    runtime = runtime or make_runtime_manager()
    adapter = adapter or make_codex_adapter()
    rt = runtime.status()
    ready = vision_ready()
    codex = adapter.detect()
    codex_detected = bool(codex.get("config_exists") or codex.get("installed"))
    agent_states: dict[str, object] = {}
    for agent_id, adapter_cls in ADAPTERS.items():
        try:
            detection = adapter_cls().detect()
        except Exception:
            detection = {}
        agent_states[agent_id] = {
            "detected": bool(detection.get("config_exists") or detection.get("installed")),
            "patched": bool(detection.get("patched")),
        }
    return {
        "installed": True,
        "runtime_running": bool(rt["running"]),
        "runtime_ready": bool(rt.get("ready")),
        "runtime_pid": rt.get("pid"),
        "runtime_listen": rt.get("listen"),
        "runtime_upstream": rt.get("upstream"),
        "provider_configured": bool(ready["key_set"]),
        "codex_detected": codex_detected,
        "codex_patched": bool(codex.get("patched")),
        "codex_catalog_patched": bool(codex.get("catalog_patched")),
        "codex_connected": codex_detected and bool(codex.get("patched")) and bool(rt["running"]),
        "agents": agent_states,
        "vision_available": vision_available,
    }


def print_health(health: dict[str, object]) -> None:
    def mark(ok: bool) -> str:
        return "✓" if ok else "✗"

    print("Agent Vision:")
    print(f"{mark(bool(health['installed']))} Installed")
    print()
    print("Runtime:")
    running = bool(health["runtime_running"])
    line = f"{mark(running)} Running"
    if running and health.get("runtime_pid"):
        line += f" (pid {health['runtime_pid']})"
    print(line)
    print()
    print("Provider:")
    print(f"{mark(bool(health['provider_configured']))} Configured")
    print()
    print("Agent:")
    if health["codex_detected"]:
        print(f"{mark(bool(health['codex_connected']))} Codex connected")
        print(
            "    catalog: image input "
            + ("declared" if health.get("codex_catalog_patched") else "missing (will patch on setup)")
        )
    else:
        print(f"{mark(False)} Codex not detected")
    print()
    print("Vision:")
    vision_available = health.get("vision_available")
    if vision_available is None:
        print("  not tested")
    else:
        print(f"{mark(bool(vision_available))} Available")


def cmd_status(args: argparse.Namespace) -> int:
    runtime = make_runtime_manager()
    adapter = make_codex_adapter()
    vision = run_vision_test()
    health = collect_health(runtime=runtime, adapter=adapter, vision_available=bool(vision.get("ok")))
    print_health(health)
    print()

    env = detect_environment()
    print(f"agent-vision: {env['version']} ({env['install']})")
    print(f"python: {env['python']} | {env['system']} ({env['machine']})")
    if health["runtime_running"]:
        runtime_line = "runtime: running"
        if health["runtime_pid"]:
            runtime_line += f" (pid {health['runtime_pid']})"
        runtime_line += f" listen={health['runtime_listen']} upstream={health['runtime_upstream']}"
        print(runtime_line)
    else:
        print("runtime: not running")
    ready = vision_ready()
    print(f"vision provider: {ready['base_url']} / {ready['model']}")

    providers = all_providers()
    print(f"\nProviders: {len(providers)} presets")
    active_model = ready["model"]
    for provider_id in sorted(providers):
        preset = providers[provider_id]
        marker = "*" if preset.get("model") == active_model else " "
        print(f"  {marker} {provider_id}  model={preset.get('model') or '-'}  cost={preset.get('cost') or '-'}")

    print("\nAgents:")
    codex = adapter.detect()
    marks = ["installed" if codex.get("installed") else "not found"]
    if codex.get("patched"):
        marks.append("patched")
    print(f"  Codex: {', '.join(marks)}")
    if codex.get("model") or codex.get("model_provider"):
        print(f"    model={codex.get('model') or '-'} provider={codex.get('model_provider') or '-'}")
    if codex.get("backup_path"):
        print(f"    backup={codex.get('backup_path')}")
    if not health["runtime_running"] and (
        "127.0.0.1" in str(codex.get("base_url") or "")
        or "localhost" in str(codex.get("base_url") or "")
    ):
        print("\nHint: Codex base_url points at the local proxy, but the proxy is not running.")
        print("Run `agent-vision start` now, or `agent-vision autostart --enable` to start it at login.")
    for agent_id in ("claude", "cursor", "opencode"):
        try:
            other = get_adapter(agent_id).detect()
        except Exception:
            other = {}
        if not (other.get("config_exists") or other.get("installed")):
            continue
        other_marks = ["installed" if other.get("installed") else "not found"]
        if other.get("patched"):
            other_marks.append("patched")
        if agent_id in ("claude", "cursor"):
            other_marks.append("manual")
        print(f"  {other.get('name')}: {', '.join(other_marks)}")
        if other.get("base_url"):
            print(f"    base_url={other.get('base_url')}")

    if not vision.get("ok"):
        print(f"\nVision test failed: {vision.get('error')}", file=sys.stderr)
        return 1
    ok = bool(health["installed"]) and bool(health["runtime_running"]) and bool(health["provider_configured"])
    if health["codex_detected"]:
        ok = ok and bool(health["codex_connected"])
    return 0 if ok else 1


def cmd_start(args: argparse.Namespace) -> int:
    runtime = make_runtime_manager()
    adapter = make_codex_adapter()
    upstream = resolve_proxy_upstream(getattr(args, "upstream", None), adapter, runtime)
    listen = getattr(args, "listen", None) or cfg("VISION_PROXY_LISTEN") or DEFAULT_LISTEN
    if not upstream:
        print("error: no upstream configured; pass --upstream or set VISION_PROXY_UPSTREAM", file=sys.stderr)
        return 1
    result = runtime.start(upstream=upstream, listen=listen)
    if result.get("status") == "already_running":
        print(f"Runtime already running (pid {result.get('pid')}) on {result.get('listen')} -> {result.get('upstream')}")
        return 0
    if result.get("ready"):
        print(f"Runtime started (pid {result.get('pid')}) on {result.get('listen')} -> {result.get('upstream')}")
        return 0
    print(f"Runtime started (pid {result.get('pid')}) but port is not ready; see {runtime.log_file}", file=sys.stderr)
    return 1


def cmd_stop(_args: argparse.Namespace) -> int:
    runtime = make_runtime_manager()
    result = runtime.stop()
    if result.get("status") == "stop_failed":
        print(f"error: failed to stop runtime pid {result.get('pid')}", file=sys.stderr)
        return 1
    if result.get("pid"):
        print(f"Runtime stopped (pid {result.get('pid')})")
    else:
        print("Runtime is not running")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    runtime = make_runtime_manager()
    adapter = make_codex_adapter()
    upstream = resolve_proxy_upstream(getattr(args, "upstream", None), adapter, runtime)
    listen = getattr(args, "listen", None) or cfg("VISION_PROXY_LISTEN") or DEFAULT_LISTEN
    if not upstream:
        print("error: no upstream configured; pass --upstream or set VISION_PROXY_UPSTREAM", file=sys.stderr)
        return 1
    try:
        result = runtime.restart(upstream=upstream, listen=listen)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if result.get("ready"):
        print(f"Runtime restarted (pid {result.get('pid')}) on {result.get('listen')} -> {result.get('upstream')}")
        return 0
    print(f"Runtime restarted (pid {result.get('pid')}) but port is not ready; see {runtime.log_file}", file=sys.stderr)
    return 1


AUTOSTART_FILENAME = "agent-vision-start.vbs"


def autostart_dir(override: str | None = None) -> Path:
    """Return the Windows per-user Startup folder (overridable for tests)."""
    override = override or os.environ.get("AGENT_VISION_AUTOSTART_DIR")
    if override:
        return Path(override).expanduser()
    if os.name != "nt" or not os.environ.get("APPDATA"):
        raise NotImplementedError(
            "autostart is only supported in the Windows user Startup folder"
        )
    return (
        Path(os.environ["APPDATA"])
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )


def autostart_vbs_path(override: str | None = None) -> Path:
    return autostart_dir(override) / AUTOSTART_FILENAME


def autostart_enabled(override: str | None = None) -> bool:
    return autostart_vbs_path(override).exists()


def _vbs_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _vbs_bytes(content: str) -> bytes:
    """Encode VBS launcher without a BOM; use ANSI on Windows."""
    for encoding in ("mbcs", "ascii"):
        try:
            return content.encode(encoding)
        except (LookupError, UnicodeEncodeError):
            continue
    return content.encode("utf-8")


def config_home_writable() -> bool:
    """Return True when the user config directory accepts writes."""
    root = config_home.ensure_home()
    probe = root / ".write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def launcher_cmd_path() -> Path:
    return config_home.ensure_home() / "agent-vision.cmd"


def render_launcher_cmd(python: str, src_root: Path | None = None) -> str:
    """Render a stable .cmd that calls the package module without PATH help."""
    lines = ["@echo off"]
    if src_root is not None:
        src = (src_root / "src").resolve()
        lines.append(f'set "PYTHONPATH={src}"')
    lines.append(f'"{python}" -m agent_vision %*')
    return "\r\n".join(lines) + "\r\n"


def ensure_launcher() -> Path:
    target = launcher_cmd_path()
    if not target.exists():
        src_root = config_home.legacy_source_root()
        content = render_launcher_cmd(sys.executable, src_root=src_root)
        target.write_text(content, encoding="utf-8")
    return target


def render_finalize_cmd(python: str, src_root: Path | None = None) -> str:
    lines = ["@echo off"]
    if src_root is not None:
        src = (src_root / "src").resolve()
        lines.append(f'set "PYTHONPATH={src}"')
    lines.append(f'"{python}" -m agent_vision setup --agent codex --provider free --yes')
    lines.append(f'"{python}" -m agent_vision autostart --enable')
    lines.append(f'"{python}" -m agent_vision status')
    lines.append("pause")
    return "\r\n".join(lines) + "\r\n"


def render_finalize_ps1(python: str, src_root: Path | None = None) -> str:
    lines = ["$ErrorActionPreference = 'Stop'"]
    if src_root is not None:
        src = (src_root / "src").resolve()
        lines.append(f"$env:PYTHONPATH = '{src}'")
    lines.append(f'& "{python}" -m agent_vision setup --agent codex --provider free --yes')
    lines.append(f'& "{python}" -m agent_vision autostart --enable')
    lines.append(f'& "{python}" -m agent_vision status')
    return "\r\n".join(lines) + "\r\n"


def write_finalize_scripts(base_dir: Path | None = None) -> list[Path]:
    """Write finalize scripts next to the caller when the sandbox blocks user config."""
    base = base_dir or Path.cwd()
    src_root = config_home.legacy_source_root()
    cmd_path = base / "agent-vision-finalize.cmd"
    ps1_path = base / "agent-vision-finalize.ps1"
    cmd_path.write_text(render_finalize_cmd(sys.executable, src_root=src_root), encoding="utf-8")
    ps1_path.write_text(render_finalize_ps1(sys.executable, src_root=src_root), encoding="utf-8")
    return [cmd_path, ps1_path]


def watchdog_tick(
    runtime: RuntimeManager,
    upstream: str,
    listen: str,
    log: object | None = None,
) -> bool:
    """Run one watchdog check. Returns True when the proxy is ready after the tick."""
    rt = runtime.status()
    if rt.get("ready"):
        return False
    try:
        result = runtime.start(upstream=upstream, listen=listen)
    except (OSError, ValueError, RuntimeError) as error:
        if log:
            log(f"watchdog failed to start proxy: {error}")
        return False
    ready = bool(result.get("ready") or result.get("status") == "already_running")
    if log:
        log(
            f"watchdog start attempt: status={result.get('status')} "
            f"pid={result.get('pid')} ready={ready}"
        )
    return ready


def cmd_watchdog(args: argparse.Namespace) -> int:
    runtime = make_runtime_manager()
    adapter = make_codex_adapter()
    upstream = resolve_proxy_upstream(getattr(args, "upstream", None), adapter, runtime)
    listen = getattr(args, "listen", None) or cfg("VISION_PROXY_LISTEN") or DEFAULT_LISTEN
    if not upstream:
        print(
            "error: no upstream configured; pass --upstream or set VISION_PROXY_UPSTREAM",
            file=sys.stderr,
        )
        return 1
    try:
        interval = max(2, int(getattr(args, "interval", 10) or 10))
    except (TypeError, ValueError):
        interval = 10
    interval = min(interval, 30)
    log_file = getattr(args, "log_file", None) or str(
        config_home.runtime_log_file().parent / "watchdog.log"
    )
    logger = _ProxyLog(log_file)
    startup = f"watchdog starting: check every {interval}s for {listen} -> {upstream}"
    logger.write(startup)
    print(startup, flush=True)
    try:
        while True:
            watchdog_tick(runtime, upstream, listen, log=logger.write)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    return 0


def render_autostart_vbs(
    python: str,
    upstream: str,
    src_root: Path | None = None,
    watchdog_interval: int | None = None,
) -> str:
    """Render a hidden VBS launcher that starts (or guards) the proxy at login."""
    lines = ['Set sh = CreateObject("WScript.Shell")']
    if src_root is not None:
        lines.append(f"sh.CurrentDirectory = {_vbs_quote(str(src_root))}")
        src = (src_root / "src").resolve()
        lines.append(
            "sh.Environment("
            + _vbs_quote("Process")
            + ")("
            + _vbs_quote("PYTHONPATH")
            + ") = "
            + _vbs_quote(str(src))
        )
    if watchdog_interval and watchdog_interval > 0:
        command = (
            f"{_vbs_quote(python)} -m agent_vision watchdog "
            f"--interval {watchdog_interval} --upstream {_vbs_quote(upstream)}"
        )
    else:
        command = (
            f"{_vbs_quote(python)} -m agent_vision start --upstream {_vbs_quote(upstream)}"
        )
    lines.append(f"sh.Run {_vbs_quote(command)}, 0, False")
    return "\r\n".join(lines) + "\r\n"


def cmd_autostart(args: argparse.Namespace) -> int:
    enable = bool(getattr(args, "enable", False))
    disable = bool(getattr(args, "disable", False))
    status_only = bool(getattr(args, "status", False))
    override = getattr(args, "startup_dir", None)
    try:
        target = autostart_vbs_path(override)
        directory = autostart_dir(override)
    except NotImplementedError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if status_only:
        if target.exists():
            content = target.read_text(encoding="utf-8", errors="replace")
            print("enabled")
            if "watchdog" in content:
                match = re.search(r"--interval (\d+)", content)
                interval = match.group(1) if match else "?"
                print(f"mode: watchdog ({interval}s)")
            else:
                print("mode: start")
        else:
            print("disabled")
        print(f"startup file: {target}")
        return 0

    if disable:
        if target.exists():
            target.unlink()
            print(f"autostart disabled (removed {target})")
        else:
            print("autostart is not enabled")
        return 0

    if not enable:
        print("usage: agent-vision autostart --enable|--disable|--status", file=sys.stderr)
        return 2

    try:
        watchdog_interval = int(getattr(args, "watchdog_interval", 10) or 0)
    except (TypeError, ValueError):
        watchdog_interval = 10
    runtime = make_runtime_manager()
    adapter = make_codex_adapter()
    upstream = resolve_proxy_upstream(getattr(args, "upstream", None), adapter, runtime)
    if not upstream:
        print(
            "error: cannot resolve proxy upstream; pass --upstream or run setup first",
            file=sys.stderr,
        )
        return 1
    src_root = config_home.legacy_source_root()
    content = render_autostart_vbs(
        sys.executable,
        upstream,
        src_root=src_root,
        watchdog_interval=watchdog_interval,
    )
    directory.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_vbs_bytes(content))
    print(f"autostart enabled (startup file: {target})")
    if watchdog_interval and watchdog_interval > 0:
        print(
            f"  command: {sys.executable} -m agent_vision watchdog "
            f"--interval {watchdog_interval} --upstream {upstream}"
        )
    else:
        print(f"  command: {sys.executable} -m agent_vision start --upstream {upstream}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-vision", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    see = sub.add_parser("see", help="describe/analyze local images, image URLs, or the latest pasted image")
    see.add_argument("images", nargs="*", help="local image paths or http(s) image URLs")
    see.add_argument("--latest", action="store_true", help="recover and analyze the latest image pasted into Codex from session files")
    see.add_argument("--task", choices=list(TASK_PROMPTS), default="describe", help="task preset: describe, ocr, ui, chart")
    see.add_argument("-q", "--question", default=None, help="question for the vision model (overrides --task)")
    see.add_argument("--model", default=None)
    see.add_argument("--base-url", default=None)
    see.add_argument("--api-key", default=None)
    see.add_argument("--provider", default=None, help="provider preset id, see `providers`")
    see.add_argument("--no-cache", action="store_true")
    see.set_defaults(handler=cmd_see)

    proxy = sub.add_parser("proxy", help="run the local image-strip proxy")
    proxy.add_argument("--listen", default="127.0.0.1:19100")
    proxy.add_argument("--upstream", required=True, help="origin to forward to, e.g. https://api.deepseek.com")
    proxy.add_argument("--max-images", type=int, default=MAX_IMAGES_PER_REQUEST)
    proxy.add_argument("--provider", default=None, help="provider preset id, see `providers`")
    proxy.add_argument("--log-file", default=None, help="path to proxy log file (default: ~/.agent-vision/logs/proxy.log)")
    proxy.set_defaults(handler=run_proxy)

    start = sub.add_parser("start", help="start the local vision proxy in the background")
    start.add_argument("--listen", default=None, help="proxy listen address, default 127.0.0.1:19100")
    start.add_argument("--upstream", default=None, help="upstream URL the proxy forwards to")
    start.set_defaults(handler=cmd_start)

    stop = sub.add_parser("stop", help="stop the local vision proxy")
    stop.set_defaults(handler=cmd_stop)

    restart = sub.add_parser("restart", help="restart the local vision proxy")
    restart.add_argument("--listen", default=None, help="proxy listen address, default 127.0.0.1:19100")
    restart.add_argument("--upstream", default=None, help="upstream URL the proxy forwards to")
    restart.set_defaults(handler=cmd_restart)

    autostart = sub.add_parser("autostart", help="manage Windows login autostart for the local proxy")
    autostart.add_argument("--enable", action="store_true", help="install the login autostart entry")
    autostart.add_argument("--disable", action="store_true", help="remove the login autostart entry")
    autostart.add_argument("--status", action="store_true", help="show whether autostart is enabled")
    autostart.add_argument("--upstream", default=None, help="upstream URL used by the autostart proxy")
    autostart.add_argument("--startup-dir", default=None, help="override the Windows Startup folder (for testing)")
    autostart.add_argument("--watchdog-interval", type=int, default=10, help="health-check interval in seconds; 0 disables the watchdog and uses plain start")
    autostart.set_defaults(handler=cmd_autostart)

    watchdog = sub.add_parser("watchdog", help="keep the local proxy alive, restarting it if the port goes down")
    watchdog.add_argument("--interval", type=int, default=10, help="seconds between health checks (2-30, default 10)")
    watchdog.add_argument("--listen", default=None, help="proxy listen address, default 127.0.0.1:19100")
    watchdog.add_argument("--upstream", default=None, help="upstream URL the proxy forwards to")
    watchdog.add_argument("--log-file", default=None, help="watchdog log file (default: ~/.agent-vision/logs/watchdog.log)")
    watchdog.set_defaults(handler=cmd_watchdog)

    doctor = sub.add_parser("doctor", help="check vision config")
    doctor.set_defaults(handler=cmd_doctor)

    providers = sub.add_parser("providers", help="list available vision provider presets")
    providers.set_defaults(handler=cmd_providers)

    setup = sub.add_parser("setup", help="guided install wizard (detect env, choose provider, write config)")
    setup.add_argument("--provider", choices=list(SETUP_MODES), default=None, help="skip interactive mode selection")
    setup.add_argument("--api-key", default=None, help="vision API key (avoids interactive prompt)")
    setup.add_argument("--base-url", default=None, help="custom provider base url (with --provider custom)")
    setup.add_argument("--model", default=None, help="custom provider model (with --provider custom)")
    setup.add_argument("--cost", default=None, help="custom provider cost note")
    setup.add_argument("--agent", choices=list(ADAPTERS), default=None, help="configure an agent adapter (codex, opencode auto; claude, cursor manual)")
    setup.add_argument("--yes", action="store_true", help="skip confirmation when applying agent changes")
    setup.add_argument("--proxy-upstream", default=None, help="upstream URL the local proxy forwards to")
    setup.add_argument("--listen", default=None, help="proxy listen address, default 127.0.0.1:19100")
    setup.add_argument("--dry-run", action="store_true", help="show actions without modifying files")
    setup.set_defaults(handler=cmd_setup)

    rollback = sub.add_parser("rollback", help="restore an agent config from its agent-vision backup")
    rollback.add_argument("agent", choices=list(ADAPTERS), help="agent id to roll back")
    rollback.set_defaults(handler=cmd_rollback)

    status = sub.add_parser("status", help="show agent-vision, provider, agent and vision-test status")
    status.add_argument("--test", action="store_true", help="run a real vision API test")
    status.set_defaults(handler=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_home.initialize()
    except OSError as error:
        print(f"warning: cannot initialize agent-vision config home: {error}", file=sys.stderr)
    if args.command == "proxy":
        return args.handler(args.listen, args.upstream, args.max_images, args.provider, args.log_file)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
