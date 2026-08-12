import base64
import http.client
import io
import json
import tempfile
import threading
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

    def test_function_call_output_image_replaced(self):
        body = json.dumps(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": [{"type": "input_image", "image_url": data_url()}],
                    }
                ]
            }
        ).encode("utf-8")
        with mock.patch.object(vb, "describe_bytes", return_value="一张示意图"):
            new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 1)
        payload = json.loads(new_body.decode("utf-8"))
        out = payload["input"][0]["output"]
        self.assertEqual(out[0]["type"], "input_text")
        self.assertIn("一张示意图", out[0]["text"])
        self.assertNotIn("image_url", json.dumps(payload))

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

    def test_failure_writes_to_log_callback(self):
        body = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url()}}]}
                ]
            }
        ).encode("utf-8")
        lines: list[str] = []
        with mock.patch.object(vb, "describe_bytes", side_effect=RuntimeError("boom")):
            vb.rewrite_body(body, log=lines.append)
        self.assertTrue(any("vision rewrite failed: boom" in line for line in lines))

    def test_oversize_image_replaced_with_marker_without_api_call(self):
        with mock.patch.object(vb, "MAX_IMAGE_BYTES", 16):
            oversized = "data:image/png;base64," + base64.b64encode(b"x" * 32).decode("ascii")
            body = json.dumps(
                {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": oversized}]}]}
            ).encode("utf-8")
            with mock.patch.object(vb, "describe_bytes", side_effect=AssertionError("must not call vision API")):
                new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 1)
        payload = json.loads(new_body.decode("utf-8"))
        part = payload["input"][0]["content"][0]
        self.assertEqual(part["type"], "input_text")
        self.assertIn("image vision conversion failed", part["text"])
        self.assertIn("MB limit", part["text"])

    def test_invalid_image_data_replaced_with_marker_without_api_call(self):
        body = json.dumps(
            {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,!!!!"}]}]}
        ).encode("utf-8")
        with mock.patch.object(vb, "describe_bytes", side_effect=AssertionError("must not call vision API")):
            new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 1)
        payload = json.loads(new_body.decode("utf-8"))
        part = payload["input"][0]["content"][0]
        self.assertIn("image vision conversion failed", part["text"])
        self.assertTrue(any(k in part["text"] for k in ("empty image data", "invalid base64")))

    def test_images_beyond_limit_replaced_with_marker_not_raw(self):
        content = [{"type": "input_image", "image_url": data_url()} for _ in range(5)]
        body = json.dumps({"input": [{"role": "user", "content": content}]}).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(vb, "_DISK_CACHE_DIR", Path(tmp)):
                with mock.patch.object(vb, "call_vision_model_batch", return_value=["图一", "图二", "图三"]):
                    new_body, replaced = vb.rewrite_body(body, max_images=3)
        self.assertEqual(replaced, 3)
        payload = json.loads(new_body.decode("utf-8"))
        parts = payload["input"][0]["content"]
        self.assertEqual(len(parts), 5)
        described = [p for p in parts if p.get("type") == "input_text" and "image described" in p["text"]]
        omitted = [p for p in parts if p.get("type") == "input_text" and "image omitted" in p["text"]]
        self.assertEqual(len(described), 3)
        self.assertEqual(len(omitted), 2)
        self.assertNotIn("image_url", json.dumps(payload))

    def test_five_images_with_default_limit_all_replaced(self):
        content = [{"type": "input_image", "image_url": data_url()} for _ in range(5)]
        body = json.dumps({"input": [{"role": "user", "content": content}]}).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(vb, "_DISK_CACHE_DIR", Path(tmp)):
                with mock.patch.object(
                    vb, "call_vision_model_batch", return_value=["图"] * 5
                ):
                    new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 5)
        self.assertNotIn("image_url", json.dumps(new_body.decode("utf-8")))

    def test_batch_uses_single_call_and_parses_markers(self):
        body = json.dumps(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": data_url()},
                            {"type": "input_image", "image_url": data_url()},
                        ],
                    }
                ]
            }
        ).encode("utf-8")
        calls = {"n": 0}

        def fake_batch(images, prompt, **kwargs):
            calls["n"] += 1
            self.assertEqual(len(images), 2)
            return ["红色", "绿色"]

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(vb, "_DISK_CACHE_DIR", Path(tmp)):
                with mock.patch.object(vb, "call_vision_model_batch", side_effect=fake_batch):
                    new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(replaced, 2)
        payload = json.loads(new_body.decode("utf-8"))
        parts = payload["input"][0]["content"]
        self.assertIn("红色", parts[0]["text"])
        self.assertIn("绿色", parts[1]["text"])
        self.assertNotIn("image_url", json.dumps(payload))

    def test_parse_batch_descriptions(self):
        parsed = vb._parse_batch_descriptions(
            "[IMG1] 第一张\n更多细节[IMG2] 第二张[IMG3] 第三张", 3
        )
        self.assertEqual(parsed, ["第一张\n更多细节", "第二张", "第三张"])
        parsed_missing = vb._parse_batch_descriptions("[IMG1] only", 2)
        self.assertEqual(parsed_missing, ["only", None])

    def test_non_image_content_passes_through(self):
        body = json.dumps(
            {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]}
        ).encode("utf-8")
        new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 0)
        self.assertEqual(new_body, body)

    def test_proxy_log_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = vb._ProxyLog(Path(tmp) / "proxy.log")
            logger.write("hello proxy")
            content = (Path(tmp) / "proxy.log").read_text(encoding="utf-8")
            self.assertIn("hello proxy", content)

    def test_upstream_request_retries_then_succeeds(self):
        calls = {"n": 0}

        class FakeConn:
            def __init__(self, host, port, timeout):
                self.host = host
                self.port = port
                self.timeout = timeout

            def request(self, method, path, body, headers):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise OSError("temporary DNS failure")

            def getresponse(self):
                return "resp"

            def close(self):
                pass

        with mock.patch.object(
            vb.http.client,
            "HTTPSConnection",
            side_effect=lambda *args, **kwargs: FakeConn(*args, **kwargs),
        ):
            conn, response = vb._upstream_request(
                "https://api.deepseek.com", "POST", "/v1/responses", b"{}", {}, retries=3
            )
        self.assertEqual(calls["n"], 3)
        self.assertEqual(response, "resp")
        self.assertEqual(conn.host, "api.deepseek.com")

    def test_upstream_request_raises_after_retries(self):
        class AlwaysFail:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                raise OSError("boom")

            def close(self):
                pass

        with mock.patch.object(
            vb.http.client,
            "HTTPSConnection",
            side_effect=lambda *args, **kwargs: AlwaysFail(),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                vb._upstream_request(
                    "https://api.deepseek.com", "POST", "/v1/responses", b"{}", {}, retries=2
                )
        self.assertIn("upstream unreachable", str(ctx.exception))

    def test_invalid_body_passes_through(self):
        body = b"not json"
        new_body, replaced = vb.rewrite_body(body)
        self.assertEqual(replaced, 0)
        self.assertEqual(new_body, body)


class RealtimeVoiceBlockTests(unittest.TestCase):
    def test_realtime_paths_are_blocked(self):
        self.assertTrue(vb.blocked_realtime_path("/v1/live"))
        self.assertTrue(vb.blocked_realtime_path("/v1/live?model=gpt-live-1"))
        self.assertTrue(vb.blocked_realtime_path("/v1/live/rtc_u0_abc"))
        self.assertTrue(vb.blocked_realtime_path("/v1/realtime"))
        self.assertTrue(vb.blocked_realtime_path("/v1/realtime/calls"))
        self.assertFalse(vb.blocked_realtime_path("/v1/responses"))
        self.assertFalse(vb.blocked_realtime_path("/v1/chat/completions"))
        self.assertFalse(vb.blocked_realtime_path("/"))

    def test_proxy_rejects_live_without_upstream_call(self):
        logger = mock.Mock()
        server = vb.ThreadingHTTPServer(("127.0.0.1", 0), vb.ProxyHandler)
        server.upstream = "https://api.deepseek.com"
        server.max_images = 3
        server.model = "glm-4v-flash"
        server.base_url = "https://open.bigmodel.cn/api/paas/v4"
        server.logger = logger
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                conn.request(
                    "POST",
                    "/v1/live",
                    body=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(response.status, 501)
                self.assertEqual(payload["error"]["code"], "realtime_voice_unsupported")
                self.assertIn("realtime voice", payload["error"]["message"])
            finally:
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
        logger.write.assert_called()
        written = [call.args[0] for call in logger.write.call_args_list]
        self.assertTrue(any("realtime voice request blocked" in line for line in written))


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


class CompatibilityShimTests(unittest.TestCase):
    def test_vision_bridge_reexports_cli(self):
        import vision_bridge
        self.assertIs(vision_bridge.main, vb.main)
        self.assertIs(vision_bridge.describe_bytes, vb.describe_bytes)


if __name__ == "__main__":
    unittest.main()
