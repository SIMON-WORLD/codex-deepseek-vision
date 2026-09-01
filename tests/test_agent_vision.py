import base64
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent_vision.cli as vb


def png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


def data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes()).decode("ascii")


class RewriteTests(unittest.TestCase):
    def setUp(self):
        vb._CACHE.clear()
        self._cache_tmp = tempfile.TemporaryDirectory()
        self._cache_patch = mock.patch.object(vb, "_DISK_CACHE_DIR", Path(self._cache_tmp.name))
        self._cache_patch.start()

    def tearDown(self):
        vb._CACHE.clear()
        self._cache_patch.stop()
        self._cache_tmp.cleanup()

    def test_chat_completions_image_replaced(self):
        body = json.dumps(
            {
                "model": "deepseek-v4-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "看这张报错图"},
                            {"type": "image_url", "image_url": {"url": data_url()}},
                        ],
                    }
                ],
            }
        ).encode("utf-8")
        with mock.patch.object(vb, "describe_bytes", return_value="错误：TypeError at line 42"):
            new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 1)
        payload = json.loads(new_body.decode("utf-8"))
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[1]["type"], "text")
        self.assertIn("TypeError", parts[1]["text"])

    def test_responses_api_image_replaced(self):
        body = json.dumps(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": data_url()},
                        ],
                    }
                ]
            }
        ).encode("utf-8")
        with mock.patch.object(vb, "describe_bytes", return_value="一张示意图"):
            new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 1)
        payload = json.loads(new_body.decode("utf-8"))
        part = payload["input"][0]["content"][0]
        self.assertEqual(part["type"], "input_text")
        self.assertIn("一张示意图", part["text"])

    def test_fail_closed_when_vision_fails(self):
        body = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url()}}]}
                ]
            }
        ).encode("utf-8")
        with mock.patch.object(vb, "describe_bytes", side_effect=RuntimeError("boom")):
            new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 1)
        payload = json.loads(new_body.decode("utf-8"))
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertIn("[image vision conversion failed: boom]", parts[0]["text"])
        self.assertNotIn("image_url", json.dumps(payload))

    def test_invalid_body_passes_through(self):
        body = b"not json"
        new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 0)
        self.assertEqual(new_body, body)


class SanitizeToolsTests(unittest.TestCase):
    def test_null_parameters_replaced(self):
        payload = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "automation_update", "parameters": None},
                }
            ]
        }
        changed = vb.sanitize_tools(payload)
        self.assertTrue(changed)
        self.assertEqual(
            payload["tools"][0]["function"]["parameters"],
            {"type": "object", "properties": {}},
        )

    def test_missing_type_replaced(self):
        payload = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "automation_update",
                        "parameters": {"properties": {"id": {"type": "string"}}},
                    },
                }
            ]
        }
        changed = vb.sanitize_tools(payload)
        self.assertTrue(changed)
        self.assertEqual(
            payload["tools"][0]["function"]["parameters"],
            {"type": "object", "properties": {}},
        )

    def test_valid_schema_unchanged(self):
        tool = {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
        payload = {"tools": [tool]}
        changed = vb.sanitize_tools(payload)
        self.assertFalse(changed)
        self.assertEqual(payload["tools"][0], tool)

    def test_no_tools_unchanged(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        changed = vb.sanitize_tools(payload)
        self.assertFalse(changed)
        self.assertEqual(payload, {"messages": [{"role": "user", "content": "hi"}]})

    def test_rewrite_body_reserializes_when_tools_cleaned(self):
        body = json.dumps(
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "automation_update", "parameters": None},
                    }
                ],
            }
        ).encode("utf-8")
        new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 0)
        payload = json.loads(new_body.decode("utf-8"))
        self.assertEqual(
            payload["tools"][0]["function"]["parameters"],
            {"type": "object", "properties": {}},
        )
        self.assertNotEqual(new_body, body)


class NativeVisionPassthroughTests(unittest.TestCase):
    def test_native_vision_model_detected(self):
        self.assertTrue(vb._is_native_vision_model("deepseek-v4-flash-vision-exp"))

    def test_text_models_not_native(self):
        self.assertFalse(vb._is_native_vision_model("deepseek-v4-flash"))
        self.assertFalse(vb._is_native_vision_model("deepseek-v4-pro"))

    def test_native_vision_model_does_not_rewrite_image(self):
        body = json.dumps(
            {
                "model": "deepseek-v4-flash-vision-exp",
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url()}}]}
                ],
            }
        ).encode("utf-8")
        with mock.patch.object(vb, "describe_bytes", side_effect=AssertionError("must not convert")):
            new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 0)
        self.assertEqual(new_body, body)

    def test_text_model_rewrites_image(self):
        body = json.dumps(
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url()}}]}],
            }
        ).encode("utf-8")
        with mock.patch.object(vb, "describe_bytes", return_value="ok"):
            new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 1)

    def test_force_convert_overrides_passthrough(self):
        body = json.dumps(
            {
                "model": "deepseek-v4-flash-vision-exp",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url()}}]}],
            }
        ).encode("utf-8")
        with mock.patch.dict(os.environ, {"VISION_FORCE_CONVERT": "1"}):
            with mock.patch.object(vb, "describe_bytes", return_value="forced"):
                new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 1)

    def test_env_passthrough_models_extended(self):
        with mock.patch.dict(os.environ, {"VISION_PASSTHROUGH_MODELS": "rightapi-vision,gpt-5.6-sol"}):
            self.assertTrue(vb._is_native_vision_model("gpt-5.6-sol"))
            self.assertTrue(vb._is_native_vision_model("rightapi-vision"))
            self.assertFalse(vb._is_native_vision_model("deepseek-v4-flash"))

    def test_passthrough_still_sanitizes_tools(self):
        body = json.dumps(
            {
                "model": "deepseek-v4-flash-vision-exp",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"type": "function", "function": {"name": "automation_update", "parameters": None}}
                ],
            }
        ).encode("utf-8")
        with mock.patch.object(vb, "describe_bytes", side_effect=AssertionError("must not convert")):
            new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 0)
        payload = json.loads(new_body.decode("utf-8"))
        self.assertEqual(payload["tools"][0]["function"]["parameters"], {"type": "object", "properties": {}})


class CacheTests(unittest.TestCase):
    def test_same_image_and_prompt_cached(self):
        vb._CACHE.clear()
        data = png_bytes()
        calls = {"n": 0}

        def fake_call(**kwargs):
            calls["n"] += 1
            return "cached description"

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(vb, "_DISK_CACHE_DIR", Path(tmp)):
                with mock.patch.object(vb, "call_vision_model", side_effect=fake_call):
                    first = vb.describe_bytes(data, "image/png", "是什么？")
                    second = vb.describe_bytes(data, "image/png", "是什么？")
        self.assertEqual(first, "cached description")
        self.assertEqual(second, "cached description")
        self.assertEqual(calls["n"], 1)
        vb._CACHE.clear()

    def test_same_image_persists_across_memory_clear(self):
        vb._CACHE.clear()
        data = png_bytes()
        calls = {"n": 0}

        def fake_call(**kwargs):
            calls["n"] += 1
            return "disk cached description"

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(vb, "_DISK_CACHE_DIR", Path(tmp)):
                with mock.patch.object(vb, "call_vision_model", side_effect=fake_call):
                    first = vb.describe_bytes(data, "image/png", "是什么？")
                    vb._CACHE.clear()
                    second = vb.describe_bytes(data, "image/png", "是什么？")
        self.assertEqual(first, "disk cached description")
        self.assertEqual(second, "disk cached description")
        self.assertEqual(calls["n"], 1)
        vb._CACHE.clear()

    def test_different_prompt_not_cached(self):
        vb._CACHE.clear()
        data = png_bytes()
        calls = {"n": 0}

        def fake_call(**kwargs):
            calls["n"] += 1
            return "ok"

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(vb, "_DISK_CACHE_DIR", Path(tmp)):
                with mock.patch.object(vb, "call_vision_model", side_effect=fake_call):
                    vb.describe_bytes(data, "image/png", "问题一")
                    vb.describe_bytes(data, "image/png", "问题二")
        self.assertEqual(calls["n"], 2)
        vb._CACHE.clear()


class CallVisionTests(unittest.TestCase):
    def test_builds_payload_and_parses_text(self):
        captured = {}

        class FakeResponse:
            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "画面里有文字"}}]}
                ).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=180):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            text = vb.call_vision_model(
                mime="image/png",
                b64="x",
                prompt="描述",
                base_url="https://example.com/v1",
                api_key="sk-test",
                model="glm-4v-flash",
            )
        self.assertEqual(text, "画面里有文字")
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(captured["body"]["model"], "glm-4v-flash")


class CliTests(unittest.TestCase):
    def test_see_prints_description(self):
        path = str(Path(__file__).resolve().parent / "sample.png")
        Path(path).write_bytes(png_bytes())
        with mock.patch.object(vb, "describe_file", return_value="描述结果"):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                code = vb.main(["see", path, "-q", "什么"])
        self.assertEqual(code, 0)
        self.assertIn("描述结果", out.getvalue())

    def test_see_requires_source(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            code = vb.main(["see"])
        self.assertEqual(code, 2)
        self.assertIn("specify image paths/URLs or use --latest", err.getvalue())

    def test_see_latest_and_images_mutually_exclusive(self):
        path = str(Path(__file__).resolve().parent / "sample.png")
        Path(path).write_bytes(png_bytes())
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            code = vb.main(["see", path, "--latest"])
        self.assertEqual(code, 2)
        self.assertIn("--latest", err.getvalue())

    def test_see_latest_describes_pasted_image(self):
        captured = {}

        def fake_describe_bytes(data, mime, prompt, **kwargs):
            captured["data"] = data
            captured["mime"] = mime
            captured["prompt"] = prompt
            return "猫的图片"

        with mock.patch.object(vb, "find_latest_pasted_images", return_value=[("image/png", png_bytes())]):
            with mock.patch.object(vb, "describe_bytes", side_effect=fake_describe_bytes):
                with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                    code = vb.main(["see", "--latest", "-q", "什么"])
        self.assertEqual(code, 0)
        self.assertEqual(captured["data"], png_bytes())
        self.assertEqual(captured["prompt"], "什么")
        self.assertIn("猫的图片", out.getvalue())

    def test_see_task_preset_uses_ocr_prompt(self):
        path = str(Path(__file__).resolve().parent / "sample.png")
        Path(path).write_bytes(png_bytes())
        captured = {}

        def fake_describe_file(image, prompt, **kwargs):
            captured["prompt"] = prompt
            return "ok"

        with mock.patch.object(vb, "describe_file", side_effect=fake_describe_file):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                code = vb.main(["see", path, "--task", "ocr"])
        self.assertEqual(code, 0)
        self.assertIn("Extract every visible text", captured["prompt"])

    def test_see_question_overrides_task(self):
        path = str(Path(__file__).resolve().parent / "sample.png")
        Path(path).write_bytes(png_bytes())
        captured = {}

        def fake_describe_file(image, prompt, **kwargs):
            captured["prompt"] = prompt
            return "ok"

        with mock.patch.object(vb, "describe_file", side_effect=fake_describe_file):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                code = vb.main(["see", path, "--task", "ocr", "-q", "自定义问题"])
        self.assertEqual(code, 0)
        self.assertEqual(captured["prompt"], "自定义问题")

    def test_doctor_checks_key(self):
        rt = mock.Mock()
        rt.status.return_value = {"ready": True}
        adapter = mock.Mock()
        adapter.detect.return_value = {
            "base_url": "http://127.0.0.1:19100/v1",
            "catalog_patched": True,
        }
        with mock.patch.object(vb, "ensure_launcher", return_value=Path("launcher")), mock.patch.object(
            vb, "config_home_writable", return_value=True
        ), mock.patch.object(vb, "make_runtime_manager", return_value=rt), mock.patch.object(
            vb, "make_codex_adapter", return_value=adapter
        ), mock.patch.object(vb, "autostart_enabled", return_value=True), mock.patch.object(
            vb, "run_vision_test", return_value={"ok": True}
        ):
            with mock.patch.dict(vb._ENV, {"VISION_API_KEY": "sk-test"}, clear=False):
                self.assertEqual(vb.cmd_doctor(mock.Mock()), 0)
            with mock.patch.dict(vb._ENV, {"VISION_API_KEY": ""}, clear=False):
                self.assertEqual(vb.cmd_doctor(mock.Mock()), 1)


class FakeHeaders:
    def get_content_type(self):
        return "image/png"

    def get(self, key, default=None):
        return default


class FakeResponse:
    headers = FakeHeaders()

    def __init__(self):
        self._done = False

    def read(self, size=-1):
        if self._done:
            return b""
        self._done = True
        return png_bytes()

    def close(self):
        pass


class ImageSourceTests(unittest.TestCase):
    def test_load_url_image_downloads(self):
        with mock.patch.object(vb.urllib.request, "urlopen", return_value=FakeResponse()):
            mime, data = vb.load_url_image("https://example.com/a.png")
        self.assertEqual(mime, "image/png")
        self.assertEqual(data, png_bytes())

    def test_describe_source_url_calls_describe_bytes(self):
        captured = {}

        def fake_describe_bytes(data, mime, prompt, **kwargs):
            captured["data"] = data
            captured["prompt"] = prompt
            return "ok"

        with mock.patch.object(vb, "load_url_image", return_value=("image/png", png_bytes())):
            with mock.patch.object(vb, "describe_bytes", side_effect=fake_describe_bytes):
                text = vb.describe_source("https://example.com/a.png", "描述", api_key="sk-test")
        self.assertEqual(text, "ok")
        self.assertEqual(captured["data"], png_bytes())
        self.assertEqual(captured["prompt"], "描述")

    def test_load_url_image_rejects_non_http_scheme(self):
        with self.assertRaises(ValueError):
            vb.load_url_image("file:///C:/tmp/a.png")

    def test_load_url_image_rejects_oversized_body(self):
        class HugeResponse:
            headers = FakeHeaders()

            def read(self, size=-1):
                return b"x" * (vb.MAX_IMAGE_BYTES + 1)

            def close(self):
                pass

        with mock.patch.object(vb.urllib.request, "urlopen", return_value=HugeResponse()):
            with self.assertRaises(ValueError):
                vb.load_url_image("https://example.com/huge.png")


class LatestImageTests(unittest.TestCase):
    def _session_line(self, image_data: bytes) -> str:
        url = "data:image/png;base64," + base64.b64encode(image_data).decode("ascii")
        return json.dumps(
            {
                "timestamp": "2026-08-04T00:00:00Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "content": [{"type": "input_image", "image_url": url, "detail": "high"}],
                },
            },
            ensure_ascii=False,
        )

    def test_find_latest_pasted_image_parses_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions" / "2026" / "08" / "04"
            session_dir.mkdir(parents=True)
            (session_dir / "rollout.jsonl").write_text(
                self._session_line(png_bytes()) + "\n", encoding="utf-8"
            )
            mime, data = vb.find_latest_pasted_image(session_dir=Path(tmp) / "sessions")
        self.assertEqual(mime, "image/png")
        self.assertEqual(data, png_bytes())

    def test_find_latest_pasted_image_prefers_newest_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions"
            session_dir.mkdir(parents=True)
            older = session_dir / "older.jsonl"
            newer = session_dir / "newer.jsonl"
            older.write_text(self._session_line(png_bytes()) + "\n", encoding="utf-8")
            newer.write_text(self._session_line(b"newest-image") + "\n", encoding="utf-8")
            os.utime(older, (1000000, 1000000))
            os.utime(newer, (2000000, 2000000))
            mime, data = vb.find_latest_pasted_image(session_dir=session_dir)
        self.assertEqual(data, b"newest-image")

    def test_find_latest_skips_corrupt_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions"
            session_dir.mkdir(parents=True)
            (session_dir / "rollout.jsonl").write_text(
                "{not-json}\n" + self._session_line(png_bytes()) + "\n",
                encoding="utf-8",
            )
            mime, data = vb.find_latest_pasted_image(session_dir=session_dir)
        self.assertEqual(data, png_bytes())

    def test_find_latest_skips_non_data_url_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "sessions"
            session_dir.mkdir(parents=True)
            line = json.dumps(
                {
                    "timestamp": "2026-08-04T00:00:00Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "content": [{"type": "input_image", "image_url": "https://example.com/a.png", "detail": "high"}],
                    },
                }
            )
            (session_dir / "rollout.jsonl").write_text(line + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                vb.find_latest_pasted_image(session_dir=session_dir)

    def test_find_latest_pasted_image_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                vb.find_latest_pasted_image(session_dir=Path(tmp) / "sessions")


class ProviderTests(unittest.TestCase):
    def test_builtin_providers_include_zhipu(self):
        providers = vb.all_providers()
        self.assertEqual(providers["zhipu"]["model"], "glm-4v-flash")
        self.assertIn("dashscope", providers)
        self.assertIn("openai", providers)
        self.assertIn("deepseek", providers)
        self.assertEqual(providers["deepseek"]["model"], "deepseek-v4-flash-vision-exp")

    def test_custom_providers_merge(self):
        tmp = Path(__file__).resolve().parent / "tmp-providers.json"
        tmp.write_text(
            json.dumps(
                {
                    "providers": [
                        {
                            "id": "my-provider",
                            "base_url": "https://example.com/v1",
                            "model": "my-vlm",
                            "cost": "free",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        try:
            with mock.patch.object(vb, "CUSTOM_PROVIDERS_FILE", tmp):
                providers = vb.all_providers()
            self.assertEqual(providers["my-provider"]["model"], "my-vlm")
            self.assertEqual(providers["zhipu"]["model"], "glm-4v-flash")
        finally:
            tmp.unlink(missing_ok=True)

    def test_resolve_provider_preset(self):
        base_url, model = vb.resolve_provider("zhipu", None, None)
        self.assertEqual(base_url, "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(model, "glm-4v-flash")

    def test_resolve_provider_unknown_raises(self):
        with self.assertRaises(ValueError):
            vb.resolve_provider("not-a-provider", None, None)

    def test_see_uses_provider_preset(self):
        path = str(Path(__file__).resolve().parent / "sample.png")
        Path(path).write_bytes(png_bytes())
        captured = {}

        def fake_describe_file(image, prompt, **kwargs):
            captured["model"] = kwargs.get("model")
            captured["base_url"] = kwargs.get("base_url")
            return "ok"

        with mock.patch.object(vb, "describe_file", side_effect=fake_describe_file):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                code = vb.main(["see", path, "--provider", "zhipu", "-q", "什么"])
        self.assertEqual(code, 0)
        self.assertEqual(captured["model"], "glm-4v-flash")
        self.assertEqual(captured["base_url"], "https://open.bigmodel.cn/api/paas/v4")

    def test_providers_command_lists_presets(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            code = vb.main(["providers"])
        self.assertEqual(code, 0)
        self.assertIn("zhipu", out.getvalue())
        self.assertIn("dashscope", out.getvalue())


if __name__ == "__main__":
    unittest.main()


class EnhancementTests(unittest.TestCase):
    def test_cache_get_treats_stale_as_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(vb, "_DISK_CACHE_DIR", Path(tmp)), mock.patch.object(vb, "CACHE_TTL_SECONDS", 1):
                path = vb._disk_cache_path("k")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps("old"), encoding="utf-8")
                os.utime(path, (0, 0))
                self.assertIsNone(vb._cache_get("k"))

    def test_proxy_log_rotates_when_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = vb._ProxyLog(Path(tmp) / "proxy.log")
            with mock.patch.object(vb, "LOG_MAX_BYTES", 10):
                logger.write("hello world")
                logger.write("x" * 200)
            self.assertTrue((Path(tmp) / "proxy.log.1").exists())

    def test_should_enable_thinking(self):
        self.assertTrue(vb._should_enable_thinking("glm-4v-flash"))
        self.assertFalse(vb._should_enable_thinking("gpt-5.6-sol"))
        with mock.patch.dict(os.environ, {"VISION_THINKING_MODELS": "5.6"}):
            self.assertTrue(vb._should_enable_thinking("gpt-5.6-sol"))
        with mock.patch.dict(os.environ, {"VISION_FORCE_DISABLE_THINKING": "1"}):
            self.assertFalse(vb._should_enable_thinking("glm-4v-flash"))

    def test_probe_models_returns_nonempty(self):
        models = vb._probe_models()
        self.assertGreaterEqual(len(models), 1)
        self.assertIn("deepseek-v4-flash", models)

    def test_ask_returns_empty_on_eof(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertEqual(vb._ask("? "), "")

    def test_image_url_from_part_accepts_http(self):
        self.assertEqual(vb.image_url_from_part({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}), "https://example.com/a.png")
        self.assertEqual(vb.image_url_from_part({"type": "image_url", "image_url": {"url": data_url()}}), data_url())


class CallOutputNormalizeTests(unittest.TestCase):
    def test_attach_call_id_from_preceding_call(self):
        payload = {
            "input": [
                {"type": "function_call", "id": "fc1", "call_id": "call_1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "id": "fo1", "name": "f", "namespace": "", "output": [{"type": "output_text", "text": "ok"}], "internal_chat_message_metadata_passthrough": {}},
            ]
        }
        changed = vb.normalize_call_outputs(payload)
        self.assertTrue(changed)
        self.assertEqual(payload["input"][1]["call_id"], "call_1")

    def test_drop_dangling_output(self):
        payload = {
            "input": [
                {"type": "function_call_output", "id": "fo1", "name": "f", "namespace": "", "output": [{"type": "output_text", "text": "ok"}]},
            ]
        }
        changed = vb.normalize_call_outputs(payload)
        self.assertTrue(changed)
        self.assertEqual(payload["input"], [])

    def test_output_with_existing_call_id_untouched(self):
        payload = {"input": [{"type": "function_call_output", "id": "fo1", "call_id": "call_1", "output": "ok"}]}
        changed = vb.normalize_call_outputs(payload)
        self.assertFalse(changed)
        self.assertEqual(payload["input"][0]["call_id"], "call_1")

    def test_rewrite_body_removes_dangling(self):
        body = json.dumps(
            {
                "model": "deepseek-v4-flash-vision-exp",
                "input": [{"type": "function_call_output", "id": "fo1", "name": "f", "output": [{"type": "output_text", "text": "ok"}]}],
            }
        ).encode("utf-8")
        with mock.patch.object(vb, "describe_bytes", side_effect=AssertionError("must not convert")):
            new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 0)
        payload = json.loads(new_body.decode("utf-8"))
        self.assertEqual(payload["input"], [])

    def test_output_list_flattened_to_string(self):
        payload = {
            "input": [
                {"type": "function_call_output", "id": "fo1", "call_id": "call_1", "output": [{"type": "output_text", "text": "ok"}, {"type": "output_image", "image_url": ""}]},
            ]
        }
        changed = vb.normalize_call_outputs(payload)
        self.assertTrue(changed)
        self.assertEqual(payload["input"][0]["output"], "ok\n[image]")

    def test_debug_input_summary(self):
        payload = {
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "function_call", "id": "fc", "call_id": "c", "name": "f", "arguments": "{}"},
            ],
            "instructions": "x" * 11,
        }
        s = vb._debug_input_summary(payload)
        self.assertIn("has_user=True", s)
        self.assertIn("last_is_user=False", s)
        self.assertIn("instructions_len=11", s)
