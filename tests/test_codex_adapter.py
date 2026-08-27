import argparse
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent_vision.cli as av
from agent_vision.adapters import CodexAdapter

ORIGINAL_CONFIG = (
    'model_provider = "deepseek"\n'
    'model = "deepseek-v4-flash"\n'
    "\n"
    "[model_providers.deepseek]\n"
    'name = "DeepSeek"\n'
    'base_url = "https://api.deepseek.com"\n'
    'wire_api = "responses"\n'
    'env_key = "DEEPSEEK_API_KEY"\n'
    'experimental_bearer_token = "sk-test-key"\n'
    "\n"
    "[features]\n"
    'enabled = true\n'
)


CATALOG_CONFIG = ORIGINAL_CONFIG.replace(
    'model = "deepseek-v4-flash"\n',
    'model = "deepseek-v4-flash"\nmodel_catalog_json = "cc-switch-model-catalog.json"\n',
    1,
)


def catalog_content(text_only: bool = True) -> str:
    flash_mods = ["text"] if text_only else ["text", "image"]
    return json.dumps(
        {
            "models": [
                {
                    "slug": "deepseek-v4-flash",
                    "display_name": "DeepSeek V4 Flash",
                    "input_modalities": list(flash_mods),
                    "supports_image_detail_original": not text_only,
                },
                {
                    "slug": "deepseek-v4-pro",
                    "display_name": "DeepSeek V4 Pro",
                    "input_modalities": ["text"],
                    "supports_image_detail_original": False,
                },
            ]
        },
        indent=2,
    )


def make_adapter_with_catalog(tmp: str, text_only: bool = True) -> CodexAdapter:
    codex_dir = Path(tmp)
    (codex_dir / "config.toml").write_text(CATALOG_CONFIG, encoding="utf-8")
    (codex_dir / "cc-switch-model-catalog.json").write_text(catalog_content(text_only), encoding="utf-8")
    return CodexAdapter(codex_dir=codex_dir)


class FakeRuntime:
    log_file = Path("runtime.log")

    def state(self):
        return {"upstream": "https://api.deepseek.com"}

    def status(self):
        return {
            "running": True,
            "ready": True,
            "pid": 1,
            "listen": "127.0.0.1:19100",
            "upstream": "https://api.deepseek.com",
        }

    def start(self, upstream, listen):
        return {"status": "started", "ready": True, "pid": 1, "listen": listen, "upstream": upstream}

    def stop(self):
        return {"status": "stopped", "pid": 1}


def make_adapter(tmp: str) -> CodexAdapter:
    codex_dir = Path(tmp)
    (codex_dir / "config.toml").write_text(ORIGINAL_CONFIG, encoding="utf-8")
    return CodexAdapter(codex_dir=codex_dir)


def setup_args(**overrides):
    defaults = {
        "agent": None,
        "dry_run": False,
        "yes": False,
        "proxy_upstream": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class CodexDetectionTests(unittest.TestCase):
    def test_detects_config_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            detection = adapter.detect()
            self.assertTrue(detection["config_exists"])
            self.assertEqual(detection["model_provider"], "deepseek")
            self.assertEqual(detection["model"], "deepseek-v4-flash")
            self.assertEqual(detection["base_url"], "https://api.deepseek.com")
            self.assertEqual(detection["wire_api"], "responses")
            self.assertFalse(detection["patched"])

    def test_detects_not_installed_when_no_config_or_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CodexAdapter(codex_dir=Path(tmp))
            with mock.patch("agent_vision.adapters.codex.shutil.which", return_value=None):
                detection = adapter.detect()
            self.assertFalse(detection["installed"])
            self.assertFalse(detection["config_exists"])


class BackupTests(unittest.TestCase):
    def test_backup_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            first = adapter.backup()
            second = adapter.backup()
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_text(encoding="utf-8"), ORIGINAL_CONFIG)
            self.assertEqual(second.read_text(encoding="utf-8"), ORIGINAL_CONFIG)


class PlanAndApplyTests(unittest.TestCase):
    def test_plan_is_dry_run_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            plan = adapter.plan(upstream="https://api.deepseek.com")
            self.assertEqual(plan["agent"], "codex")
            self.assertEqual(plan["upstream"], "https://api.deepseek.com")
            self.assertTrue(plan["detection"]["config_exists"])
            self.assertEqual(len(plan["files"]), 2)
            self.assertEqual(adapter.config_path.read_text(encoding="utf-8"), ORIGINAL_CONFIG)
            self.assertFalse(adapter.state_path.exists())
            backup = Path(plan["files"][0]["backup"])
            self.assertFalse(backup.exists())

    def test_apply_patches_and_preserves_original_in_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            result = adapter.apply(upstream="https://api.deepseek.com")
            patched = adapter.config_path.read_text(encoding="utf-8")
            self.assertIn('model_provider = "deepseek"', patched)
            self.assertNotIn('model_provider = "agent-vision"', patched)
            self.assertNotIn("[model_providers.agent-vision]", patched)
            self.assertIn('base_url = "http://127.0.0.1:19100/v1"', patched)
            self.assertIn('wire_api = "responses"', patched)
            self.assertIn('experimental_bearer_token = "sk-test-key"', patched)
            self.assertIn('env_key = "DEEPSEEK_API_KEY"', patched)
            self.assertTrue(Path(result["backup_path"]).exists())
            self.assertEqual(Path(result["backup_path"]).read_text(encoding="utf-8"), ORIGINAL_CONFIG)
            state = json.loads(adapter.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["agent"], "codex")
            self.assertEqual(state["model_provider"], "deepseek")
            self.assertEqual(state["wire_api"], "responses")
            self.assertEqual(state["patch_mode"], "base-url")
            self.assertEqual(state["upstream"], "https://api.deepseek.com")
            self.assertEqual(state["backup_path"], result["backup_path"])

    def test_render_patched_config_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            first = adapter.render_patched_config(ORIGINAL_CONFIG)
            second = adapter.render_patched_config(first)
            self.assertEqual(first, second)

    def test_render_patched_config_raises_without_model_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            with self.assertRaises(ValueError):
                adapter.render_patched_config('model = "deepseek-v4-flash"\n')

    def test_render_patched_config_raises_when_section_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            text = 'model_provider = "missing"\nmodel = "deepseek-v4-flash"\n'
            with self.assertRaises(ValueError):
                adapter.render_patched_config(text)


class RollbackTests(unittest.TestCase):
    def test_rollback_restores_original_and_removes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            adapter.apply(upstream="https://api.deepseek.com")
            result = adapter.rollback()
            self.assertEqual(adapter.config_path.read_text(encoding="utf-8"), ORIGINAL_CONFIG)
            self.assertFalse(adapter.state_path.exists())
            self.assertTrue(Path(result["restored_from"]).exists())

    def test_rollback_without_patch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            with self.assertRaises(FileNotFoundError):
                adapter.rollback()


class SetupAgentTests(unittest.TestCase):
    def test_setup_agent_dry_run_prints_plan_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            with mock.patch.object(av, "make_codex_adapter", return_value=adapter), mock.patch.object(
                av, "make_runtime_manager", return_value=FakeRuntime()
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = av.cmd_setup(setup_args(agent="codex", provider="free", dry_run=True))
            self.assertEqual(code, 0)
            output = buffer.getvalue()
            self.assertIn("Dry run", output)
            self.assertIn("rewrite base_url under [model_providers.deepseek]", output)
            self.assertEqual(adapter.config_path.read_text(encoding="utf-8"), ORIGINAL_CONFIG)
            self.assertFalse(adapter.state_path.exists())

    def test_setup_agent_yes_applies_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            env_file = Path(tmp) / "home" / ".env"
            providers_file = Path(tmp) / "home" / "providers.json"
            with mock.patch.object(av, "make_codex_adapter", return_value=adapter), mock.patch.object(
                av, "make_runtime_manager", return_value=FakeRuntime()
            ), mock.patch.object(av, "run_vision_test", return_value={"ok": True, "text": "OK"}), mock.patch.object(
                av, "config_home_writable", return_value=True
            ), mock.patch.object(av, "ENV_FILE", env_file
            ), mock.patch.object(av, "CUSTOM_PROVIDERS_FILE", providers_file), mock.patch.object(
                av, "ensure_launcher", return_value=Path("launcher")
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = av.cmd_setup(setup_args(agent="codex", provider="free", yes=True))
            self.assertEqual(code, 0)
            patched = adapter.config_path.read_text(encoding="utf-8")
            self.assertIn('base_url = "http://127.0.0.1:19100/v1"', patched)
            self.assertIn('wire_api = "responses"', patched)
            self.assertNotIn('model_provider = "agent-vision"', patched)
            self.assertTrue(adapter.state_path.exists())
            self.assertIn("Done. Codex config updated.", buffer.getvalue())
            self.assertIn("✓ Available", buffer.getvalue())


class StatusTests(unittest.TestCase):
    def test_status_prints_overview(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            adapter.apply()
            with mock.patch.object(av, "make_codex_adapter", return_value=adapter), mock.patch.object(
                av, "make_runtime_manager", return_value=FakeRuntime()
            ), mock.patch.object(av, "run_vision_test", return_value={"ok": True, "text": "OK"}), mock.patch.object(
                av,
                "cfg",
                side_effect=lambda name, default="": "test-key" if name == "VISION_API_KEY" else default,
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = av.cmd_status(argparse.Namespace(test=False))
            self.assertEqual(code, 0)
            output = buffer.getvalue()
            self.assertIn("✓ Running", output)
            self.assertIn("✓ Codex connected", output)
            self.assertIn("not tested", output)
            self.assertIn("Providers:", output)
            self.assertIn("Codex:", output)

    def test_status_with_test_runs_vision_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            adapter.apply()
            with mock.patch.object(av, "make_codex_adapter", return_value=adapter), mock.patch.object(
                av, "make_runtime_manager", return_value=FakeRuntime()
            ), mock.patch.object(av, "run_vision_test", return_value={"ok": True, "text": "OK"}), mock.patch.object(
                av,
                "cfg",
                side_effect=lambda name, default="": "test-key" if name == "VISION_API_KEY" else default,
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = av.cmd_status(argparse.Namespace(test=True))
            self.assertEqual(code, 0)
            self.assertIn("✓ Available", buffer.getvalue())

    def test_run_vision_test_missing_key(self):
        with mock.patch.object(
            av,
            "cfg",
            side_effect=lambda name, default="": "" if name == "VISION_API_KEY" else default,
        ):
            result = av.run_vision_test()
        self.assertFalse(result["ok"])
        self.assertIn("VISION_API_KEY", str(result["error"]))

    def test_run_vision_test_success(self):
        with mock.patch.object(av, "describe_bytes", return_value="OK"), mock.patch.object(
            av,
            "cfg",
            side_effect=lambda name, default="": "test-key" if name == "VISION_API_KEY" else default,
        ):
            result = av.run_vision_test()
        self.assertTrue(result["ok"])
        self.assertEqual(result["text"], "OK")



class CatalogPatchTests(unittest.TestCase):
    def test_detect_reports_catalog_not_patched(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_with_catalog(tmp, text_only=True)
            detection = adapter.detect()
            self.assertIn("cc-switch-model-catalog.json", str(detection["catalog_path"]))
            self.assertFalse(detection["catalog_patched"])

    def test_detect_reports_catalog_patched_when_image_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_with_catalog(tmp, text_only=False)
            detection = adapter.detect()
            self.assertTrue(detection["catalog_patched"])

    def test_catalog_needs_patch_for_image_without_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "cc-switch-model-catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "slug": "deepseek-v4-flash-vision-exp",
                                "input_modalities": ["text", "image"],
                                "supports_image_detail_original": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(CodexAdapter._catalog_needs_patch(catalog, "deepseek-v4-flash-vision-exp"))

    def test_render_catalog_patch_sets_detail_for_image_model(self):
        data = {
            "models": [
                {
                    "slug": "deepseek-v4-flash-vision-exp",
                    "input_modalities": ["text", "image"],
                    "supports_image_detail_original": False,
                }
            ]
        }
        patched = CodexAdapter.render_catalog_patch(data, "deepseek-v4-flash-vision-exp")
        entry = patched["models"][0]
        self.assertEqual(entry["input_modalities"], ["text", "image"])
        self.assertTrue(entry["supports_image_detail_original"])

    def test_detect_skips_catalog_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            detection = adapter.detect()
            self.assertEqual(detection["catalog_path"], "")
            self.assertFalse(detection["catalog_patched"])

    def test_plan_includes_catalog_entry_when_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_with_catalog(tmp, text_only=True)
            plan = adapter.plan(upstream="https://api.deepseek.com")
            self.assertTrue(plan["catalog_updated"])
            self.assertEqual(len(plan["files"]), 3)
            catalog_files = [f for f in plan["files"] if "catalog" in str(f.get("file", ""))]
            self.assertEqual(len(catalog_files), 1)
            self.assertIn("image input", catalog_files[0]["summary"])
            self.assertFalse(Path(catalog_files[0]["backup"]).exists())

    def test_plan_keeps_two_files_without_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter(tmp)
            plan = adapter.plan(upstream="https://api.deepseek.com")
            self.assertFalse(plan["catalog_updated"])
            self.assertEqual(len(plan["files"]), 2)

    def test_apply_patches_catalog_and_records_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_with_catalog(tmp, text_only=True)
            result = adapter.apply(upstream="https://api.deepseek.com")
            self.assertTrue(result["catalog_updated"])
            data = json.loads(adapter.config_path.parent.joinpath("cc-switch-model-catalog.json").read_text(encoding="utf-8"))
            flash = next(m for m in data["models"] if m["slug"] == "deepseek-v4-flash")
            self.assertIn("image", flash["input_modalities"])
            self.assertTrue(flash["supports_image_detail_original"])
            pro = next(m for m in data["models"] if m["slug"] == "deepseek-v4-pro")
            self.assertNotIn("image", pro["input_modalities"])
            state = json.loads(adapter.state_path.read_text(encoding="utf-8"))
            self.assertTrue(state["catalog_updated"])
            self.assertTrue(Path(str(state["catalog_backup_path"])).exists())
            self.assertTrue(Path(str(result["catalog_backup_path"])).exists())

    def test_rollback_restores_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = make_adapter_with_catalog(tmp, text_only=True)
            catalog_path = adapter.config_path.parent.joinpath("cc-switch-model-catalog.json")
            original = catalog_path.read_text(encoding="utf-8")
            adapter.apply(upstream="https://api.deepseek.com")
            self.assertNotEqual(catalog_path.read_text(encoding="utf-8"), original)
            result = adapter.rollback()
            self.assertEqual(catalog_path.read_text(encoding="utf-8"), original)
            self.assertTrue(result["catalog_restored_from"])

    def test_render_catalog_patch_preserves_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = json.loads(catalog_content(text_only=True))
            patched = CodexAdapter.render_catalog_patch(data, "deepseek-v4-flash")
            flash = next(m for m in patched["models"] if m["slug"] == "deepseek-v4-flash")
            self.assertIn("image", flash["input_modalities"])
            pro = next(m for m in patched["models"] if m["slug"] == "deepseek-v4-pro")
            self.assertNotIn("image", pro["input_modalities"])
            self.assertEqual(len(patched["models"]), 2)


if __name__ == "__main__":
    unittest.main()
