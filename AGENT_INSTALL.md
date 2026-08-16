# Agent Installation Guide

Repository: https://github.com/SIMON-WORLD/codex-deepseek-vision

This project is designed so that anyone who can chat with an AI agent can install and configure it. No shell or programming skills are required.

## One-line prompt for the user

Paste this into your agent (Codex, OpenCode, etc.):

```text
Set up agent-vision for me. Read AGENT_INSTALL.md and follow it end to end. Use the free Zhipu provider unless I choose another one.
```

## If GitHub is unreachable

- Install from PyPI instead. Every release is published to PyPI at the same time as the GitHub tag (currently 1.2.1), so GitHub is not required. If official PyPI is slow, use the Tsinghua mirror; note that mirrors can lag official PyPI by a few hours to a few days, so switch back if you get an older version.

  ```bash
  pip install codex-deepseek-vision
  # or
  pip install codex-deepseek-vision -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- A user with Python installed can also paste this into their agent; it does not require GitHub:

  ```text
  Install codex-deepseek-vision with pip (use the Tsinghua mirror if PyPI is slow), then run:
  agent-vision setup --agent codex --provider custom --base-url https://www.rightapi.ai/codex/v1 --model gpt-5.6-sol --api-key <KEY> --yes
  agent-vision autostart --enable
  agent-vision doctor
  Tell me when I need to restart Codex.
  ```

  For the free Zhipu provider, replace the custom provider flags with `--provider free --yes`.
- GitHub is only required when you want code that has not been released to PyPI yet.

## Checklist for the agent

1. Install the package from PyPI (simplest, recommended), or clone the repository and install from source when PyPI is unreachable:

   ```bash
   pip install codex-deepseek-vision
   # or
   pip install codex-deepseek-vision -i https://pypi.tuna.tsinghua.edu.cn/simple
   # or
   git clone https://github.com/SIMON-WORLD/codex-deepseek-vision.git
   cd codex-deepseek-vision
   pip install .
   ```

   Keep all writes inside the project unless a step below explicitly requires a backed-up agent config change.

2. Ask the user which vision provider they want: `zhipu` (free), `dashscope`, `openai`, `gemini`, `groq`, `siliconflow`, `openrouter`, or a custom OpenAI-compatible endpoint. Ask the user to provide the matching API key, or let them paste it into the user config directory's `.env` themselves (`~/.agent-vision/.env`, or `%USERPROFILE%\.agent-vision\.env` on Windows).
3. Run the guided setup:

   ```bash
   agent-vision setup
   ```

   The wizard detects the installed agent, writes `.env` (and `providers.json` for custom providers) into the user config directory, starts the local runtime, and verifies the vision API. For Codex it backs up and only rewrites the active provider's `base_url` (never `wire_api`, model or keys); when Codex loads its model list from a local model catalog (e.g. cc-switch), it also declares image input for the active model so the client accepts pasted images (with a backup). For OpenCode it adds the OpenAI-compatible provider automatically.
4. Verify the pipeline explicitly:

   ```bash
   agent-vision status
   agent-vision see <image-path> -q "What is in this image?"
   agent-vision see https://example.com/image.png --task ocr
   agent-vision see --latest
   ```

   For Codex, ask the user to paste an image as the primary check; for local files use `agent-vision see <path>` as a fallback (the built-in `view_image` tool may be replaced with `[Unsupported Image]` by the current desktop client).
   On Windows, recommend running `agent-vision autostart --enable` once so the proxy starts automatically at login and a watchdog (default 10s) restarts it if 19100 is not listening. Tune with `--watchdog-interval 2-30`, or use `--watchdog-interval 0` for plain start.
   If `agent-vision` is not on PATH, use the stable launcher `%USERPROFILE%\.agent-vision\agent-vision.cmd` (created by setup) or `python -m agent_vision`; do not rely on PATH. Start the proxy with an elevated or normal terminal (`agent-vision start`) so the process is not tied to the Codex session. If the sandbox blocks user config writes, setup writes `agent-vision-finalize.cmd/.ps1` next to the current directory; ask the user to run one of them to finish. After setup, run `agent-vision doctor`; the deployment is complete only when all checks pass.

5. If the user later asks to roll back an auto-patched agent:

   ```bash
   agent-vision rollback codex
   agent-vision rollback opencode
   ```

6. For Claude Code and Cursor, `agent-vision setup --agent claude --dry-run` and `agent-vision setup --agent cursor --dry-run` print the official manual steps. Do not invent config keys for these agents.
7. Never commit or upload `.env`. Keep API keys in the user config directory, not in the repository. Report the chosen provider, the verification result, and any files that were backed up.

## Uninstall / clean removal

Run these from PowerShell in this exact order. `&` and `$env:USERPROFILE` are PowerShell syntax; `%USERPROFILE%` only works in cmd:

```powershell
& "$env:USERPROFILE\.agent-vision\agent-vision.cmd" rollback codex
& "$env:USERPROFILE\.agent-vision\agent-vision.cmd" stop
& "$env:USERPROFILE\.agent-vision\agent-vision.cmd" autostart --disable
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'agent_vision' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
pip uninstall codex-deepseek-vision -y
Remove-Item -LiteralPath "$env:USERPROFILE\.agent-vision" -Recurse -Force
```

Roll back the agent config, stop the proxy, and disable autostart before uninstalling the package. `autostart --disable` only removes the login entry; terminate any running `agent_vision` python processes as well, because a running watchdog restarts the proxy. If the package was already uninstalled, reinstall it first so the CLI/launcher commands work, then run the cleanup above.

## Troubleshooting: image not visible

- If you see `[Unsupported Image]` or `[image vision conversion failed ...]`: vision conversion failed or the client blocked the image; ask the user to re-paste, or run `agent-vision see <path>` instead.
- Failure reasons are logged to `~/.agent-vision/logs/proxy.log`; check this file first when debugging.
- The built-in `view_image` tool result may be replaced with `[Unsupported Image]` by the current desktop client; pasted images are not affected.
- If the user's voice chat fails (`404 /v1/live`, `Voice chat took too long to start`, ...): this is an expected limitation, not a proxy fault. Codex realtime voice uses OpenAI's GPT-Live channel, which DeepSeek does not provide; the proxy intercepts `/v1/live` and returns a clear message. Do not try to "fix" it by changing proxy or network settings; explain that text input should be used instead.
- If Codex cannot chat after a reboot (`stream disconnected`), the local proxy is not running: run `agent-vision start`, or enable `agent-vision autostart --enable` so it starts at login.

## Rules for the agent

- Never modify system proxy, DNS, network adapters, WinHTTP/WinINET proxy, or global HTTP_PROXY/HTTPS_PROXY variables. When the network is unavailable, retry or switch the download source instead of setting a system proxy; otherwise ask the user to fix the network.
- Ask before changing global agent configuration; every auto-patch creates a timestamped backup first.
- Keep API keys inside the local `.env`; never print them.
- If verification fails, stop and explain the error instead of guessing.
