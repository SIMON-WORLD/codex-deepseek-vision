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

    def tearDown(self):
        vb._CACHE.clear()

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

        with mock.patch.object(vb, "find_latest_pasted_image", return_value=("image/png", png_bytes())):
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
