"""Codex adapter: detect, backup, patch, apply and roll back config.toml.

The patch is intentionally minimal: it rewrites only the ``base_url`` of the
active model provider to the local agent-vision proxy. It never changes
``model_provider``, ``model``, ``wire_api``, auth keys or any other setting,
because modern Codex relies on ``wire_api = "responses"`` and an injected
provider table would break startup.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from ..version import VERSION
from .base import AgentAdapter

PROXY_BASE_URL = "http://127.0.0.1:19100/v1"

_KEY_RE = re.compile(r'^\s*([A-Za-z0-9_.-]+)\s*=\s*"([^"]*)"\s*$')
_SECTION_RE = re.compile(r"^\[(.+)]\s*$")


class CodexAdapter(AgentAdapter):
    id = "codex"
    name = "Codex"

    def __init__(self, codex_dir: Path | None = None, home: Path | None = None):
        self.home = Path(home) if home is not None else Path.home()
        env_home = os.environ.get("CODEX_HOME")
        self.codex_dir = Path(codex_dir) if codex_dir is not None else Path(env_home or self.home / ".codex")
        self.config_path = self.codex_dir / "config.toml"
        self.state_path = self.codex_dir / "agent-vision-state.json"

    def _codex_in_path(self) -> bool:
        return shutil.which("codex") is not None

    @staticmethod
    def _top_level(text: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                break
            if not stripped or stripped.startswith("#"):
                continue
            match = _KEY_RE.match(line)
            if match:
                result[match.group(1)] = match.group(2)
        return result

    @staticmethod
    def _provider_table(text: str, provider: str) -> dict[str, str]:
        target = f"model_providers.{provider}"
        result: dict[str, str] = {}
        inside = False
        for line in text.splitlines():
            stripped = line.strip()
            section = _SECTION_RE.match(stripped)
            if section:
                inside = section.group(1).strip() == target
                continue
            if inside:
                match = _KEY_RE.match(line)
                if match:
                    result[match.group(1)] = match.group(2)
        return result

    def _current_model_config(self) -> dict[str, str]:
        if not self.config_path.exists():
            return {}
        text = self.config_path.read_text(encoding="utf-8")
        top = self._top_level(text)
        provider = top.get("model_provider", "")
        table = self._provider_table(text, provider) if provider else {}
        return {
            "model": top.get("model", ""),
            "model_provider": provider,
            "base_url": table.get("base_url", ""),
            "wire_api": table.get("wire_api", ""),
        }

    def detect(self) -> dict[str, object]:
        config_exists = self.config_path.exists()
        model = self._current_model_config()
        state = self.read_json(self.state_path)
        catalog_path = self._catalog_path()
        catalog_patched = False
        if catalog_path and catalog_path.exists() and model.get("model"):
            catalog_patched = not self._catalog_needs_patch(catalog_path, model["model"])
        return {
            "agent": self.id,
            "name": self.name,
            "installed": config_exists or self._codex_in_path(),
            "config_path": str(self.config_path),
            "config_exists": config_exists,
            "model": model.get("model", ""),
            "model_provider": model.get("model_provider", ""),
            "base_url": model.get("base_url", ""),
            "wire_api": model.get("wire_api", ""),
            "patched": model.get("base_url") == PROXY_BASE_URL,
            "patch_mode": "base-url",
            "catalog_path": str(catalog_path) if catalog_path else "",
            "catalog_patched": catalog_patched,
            "backup_path": str(state.get("backup_path", "")) if state else "",
            "upstream": str(state.get("upstream", "")) if state else "",
        }

    def backup(self) -> Path:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Codex config not found: {self.config_path}")
        backup_path = self.unique_backup_path(self.config_path, "agent-vision")
        backup_path.write_text(self.config_path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup_path

    def _detected_upstream(self, explicit: str | None) -> str:
        if explicit:
            return explicit.strip()
        base_url = self._current_model_config().get("base_url", "")
        if base_url and "127.0.0.1" not in base_url and "localhost" not in base_url:
            return base_url
        return ""


    def _catalog_path(self) -> Path | None:
        """Resolve model_catalog_json from config.toml (relative to codex_dir)."""
        if not self.config_path.exists():
            return None
        top = self._top_level(self.config_path.read_text(encoding="utf-8"))
        raw = top.get("model_catalog_json", "")
        if not raw:
            return None
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self.codex_dir / candidate

    @classmethod
    def _catalog_needs_patch(cls, catalog_path: Path, model: str) -> bool:
        """True when the active model is not yet declared to accept image input."""
        if not model or not catalog_path.exists():
            return False
        try:
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        for entry in data.get("models", []):
            if entry.get("slug") == model:
                modalities = entry.get("input_modalities") or []
                if "image" not in modalities:
                    return True
                return not bool(entry.get("supports_image_detail_original"))
        return False

    @classmethod
    def render_catalog_patch(cls, data: dict[str, object], model: str) -> dict[str, object]:
        """Return a copy of the catalog with image input declared for the model."""
        import copy

        patched = copy.deepcopy(data)
        for entry in patched.get("models", []):
            if entry.get("slug") == model:
                modalities = entry.setdefault("input_modalities", [])
                if "image" not in modalities:
                    modalities.append("image")
                entry["supports_image_detail_original"] = True
                break
        return patched

    def _catalog_backup(self, catalog_path: Path) -> Path:
        backup_path = self.unique_backup_path(catalog_path, "agent-vision")
        backup_path.write_text(catalog_path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup_path

    @classmethod
    def render_patched_config(
        cls,
        text: str,
        *,
        base_url: str = PROXY_BASE_URL,
    ) -> str:
        """Return config.toml with only the active provider's base_url replaced."""
        top = cls._top_level(text)
        provider = top.get("model_provider", "")
        if not provider:
            raise ValueError("cannot find model_provider in Codex config")
        target = f"model_providers.{provider}"
        lines = text.splitlines()
        out: list[str] = []
        in_target = False
        replaced = False
        for line in lines:
            stripped = line.strip()
            section = _SECTION_RE.match(stripped)
            if section:
                in_target = section.group(1).strip() == target
            if in_target:
                match = _KEY_RE.match(line)
                if match and match.group(1) == "base_url":
                    out.append(f'base_url = "{base_url}"')
                    replaced = True
                    continue
            out.append(line)
        if not replaced:
            raise ValueError(f"cannot locate base_url under [{target}]")
        return "\n".join(out).rstrip() + "\n"

    def plan(self, upstream: str | None = None) -> dict[str, object]:
        detection = self.detect()
        if not detection["config_exists"]:
            raise FileNotFoundError(f"Codex config not found: {self.config_path}")
        resolved_upstream = self._detected_upstream(upstream)
        provider = str(detection["model_provider"])
        backup_path = self.unique_backup_path(self.config_path, "agent-vision")
        files = [
            {
                "file": str(self.config_path),
                "action": "modify",
                "backup": str(backup_path),
                "summary": (
                    f'rewrite base_url under [model_providers.{provider}] to {PROXY_BASE_URL}; '
                    "keep model_provider, model, wire_api and auth untouched"
                ),
            },
            {
                "file": str(self.state_path),
                "action": "write",
                "summary": "write agent-vision state: backup path, provider, upstream",
            },
        ]
        catalog_path = self._catalog_path()
        catalog_updated = False
        if catalog_path and self._catalog_needs_patch(catalog_path, str(detection["model"])):
            catalog_updated = True
            files.append(
                {
                    "file": str(catalog_path),
                    "action": "modify",
                    "backup": str(self.unique_backup_path(catalog_path, "agent-vision")),
                    "summary": (
                        f"declare image input for model '{detection['model']}' in model catalog; "
                        "keeps all other catalog entries untouched"
                    ),
                }
            )
        return {
            "agent": self.id,
            "detection": detection,
            "files": files,
            "upstream": resolved_upstream,
            "catalog_path": str(catalog_path) if catalog_path else "",
            "catalog_updated": catalog_updated,
        }

    def apply(self, upstream: str | None = None) -> dict[str, object]:
        detection = self.detect()
        if not detection["config_exists"]:
            raise FileNotFoundError(f"Codex config not found: {self.config_path}")
        resolved_upstream = self._detected_upstream(upstream)
        backup_path = self.backup()
        original = backup_path.read_text(encoding="utf-8")
        self.config_path.write_text(self.render_patched_config(original), encoding="utf-8")
        catalog_path = self._catalog_path()
        catalog_backup_path = ""
        catalog_updated = False
        if catalog_path and self._catalog_needs_patch(catalog_path, str(detection["model"])):
            catalog_backup_path = str(self._catalog_backup(catalog_path))
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
            patched = self.render_catalog_patch(data, str(detection["model"]))
            catalog_path.write_text(
                json.dumps(patched, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            catalog_updated = True
        self.write_json(
            self.state_path,
            {
                "agent": self.id,
                "version": VERSION,
                "config_path": str(self.config_path),
                "backup_path": str(backup_path),
                "model": str(detection["model"]),
                "model_provider": str(detection["model_provider"]),
                "wire_api": str(detection["wire_api"]),
                "base_url": PROXY_BASE_URL,
                "upstream": resolved_upstream,
                "patch_mode": "base-url",
                "catalog_path": str(catalog_path) if catalog_path else "",
                "catalog_backup_path": catalog_backup_path,
                "catalog_updated": catalog_updated,
                "patched_at": self.timestamp(),
            },
        )
        return {
            "agent": self.id,
            "config_path": str(self.config_path),
            "backup_path": str(backup_path),
            "state_path": str(self.state_path),
            "upstream": resolved_upstream,
            "model": str(detection["model"]),
            "model_provider": str(detection["model_provider"]),
            "catalog_path": str(catalog_path) if catalog_path else "",
            "catalog_backup_path": catalog_backup_path,
            "catalog_updated": catalog_updated,
        }

    def rollback(self) -> dict[str, object]:
        state = self.read_json(self.state_path)
        if not state:
            raise FileNotFoundError("no agent-vision patch found for Codex")
        backup_path = Path(str(state["backup_path"]))
        if not backup_path.exists():
            raise FileNotFoundError(f"backup missing: {backup_path}")
        self.config_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
        catalog_restored_from = ""
        catalog_backup = state.get("catalog_backup_path")
        if catalog_backup:
            catalog_backup_path = Path(str(catalog_backup))
            catalog_path = Path(str(state.get("catalog_path") or self._catalog_path() or ""))
            if catalog_backup_path.exists() and catalog_path and catalog_path.exists():
                catalog_path.write_text(catalog_backup_path.read_text(encoding="utf-8"), encoding="utf-8")
                catalog_restored_from = str(catalog_backup_path)
        self.state_path.unlink(missing_ok=True)
        return {
            "agent": self.id,
            "config_path": str(self.config_path),
            "restored_from": str(backup_path),
            "state_path": str(self.state_path),
            "catalog_restored_from": catalog_restored_from,
        }
