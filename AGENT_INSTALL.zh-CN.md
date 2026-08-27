# Agent 安装指南

仓库地址：https://github.com/SIMON-WORLD/codex-deepseek-vision

这个项目设计给“只要会和 AI Agent 聊天”的用户：不需要会终端，也不需要会编程。

## 用户一句话指令

把下面这句话发给你的 Agent（Codex、OpenCode 等）：

```text
帮我从 https://github.com/SIMON-WORLD/codex-deepseek-vision 安装并配置 agent-vision（包名 codex-deepseek-vision）。请阅读该仓库的 AGENT_INSTALL.zh-CN.md 并从头到尾执行。默认使用智谱免费服务，除非我指定其他服务商。
```

想指定服务商或模型时，把它们和 Key 放进同一句话即可，例如：

```text
帮我从 https://github.com/SIMON-WORLD/codex-deepseek-vision 安装并配置 agent-vision（包名 codex-deepseek-vision），用 RightAPI 的 gpt-5.6-sol 识图，我的 API Key 是 <你的Key>。请阅读该仓库的 AGENT_INSTALL.zh-CN.md 并从头到尾执行。
```

常见自定义服务商参考：

- RightAPI：`base_url=https://www.rightapi.ai/codex/v1`，可用模型如 `gpt-5.6-sol`、`gpt-5.5`、`gpt-5.4-mini`（个别模型名对某些 Key 返回 503，需以 `agent-vision status --test` 实测为准）。
- 其他 OpenAI 兼容接口：按服务商文档提供 `base_url` 与视觉模型名即可。

## 访问不了 GitHub 怎么办

- 直接走 PyPI：每次发版都会同步发布到 PyPI（当前最新 1.2.1），不访问 GitHub 也能安装。国内访问官方 PyPI 慢时，可临时加清华镜像，但镜像同步有时滞后几小时到几天，装到旧版本就等镜像同步或换官方源。

  ```bash
  pip install codex-deepseek-vision
  # 或
  pip install codex-deepseek-vision -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- 已装好 Python 的普通用户也可以让 Agent 直接安装并配置，不需要打开 GitHub。把下面这句话发给 Agent（换成你的服务商和 Key）：

  ```text
  帮我用 pip 安装 codex-deepseek-vision（官方 PyPI 慢就用清华镜像），然后执行：
  agent-vision setup --agent codex --provider custom --base-url https://www.rightapi.ai/codex/v1 --model gpt-5.6-sol --api-key <你的Key> --yes
  agent-vision autostart --enable
  agent-vision doctor
  全部通过后提醒我重启 Codex。
  ```

  默认免费智谱时，把 `--provider custom --base-url ... --model ... --api-key ...` 换成 `--provider free --yes` 即可。
- 只有想用 GitHub 上还没发版的最新代码时，才必须访问 GitHub。

## Agent 执行清单

1. 从 PyPI 安装（最简单，推荐）；如果 PyPI 不可达，再克隆仓库从源码安装：

   ```bash
   pip install codex-deepseek-vision
   # 或
   pip install codex-deepseek-vision -i https://pypi.tuna.tsinghua.edu.cn/simple
   # 或
   git clone https://github.com/SIMON-WORLD/codex-deepseek-vision.git
   cd codex-deepseek-vision
   pip install .
   ```

  除非下面步骤明确要求修改 Agent 配置，否则所有写入都留在项目目录内。

2. 询问用户选择视觉服务商：`zhipu`（免费）、`dashscope`、`openai`、`gemini`、`groq`、`siliconflow`、`openrouter`，或自定义 OpenAI 兼容接口。请用户提供对应的 API Key，或让用户自己粘贴到用户配置目录的 `.env`（`~/.agent-vision/.env`，Windows 为 `%USERPROFILE%\.agent-vision\.env`）。
3. 运行引导式配置：

   ```bash
   agent-vision setup
   ```

   向导会自动检测已安装的 Agent，在用户配置目录写入 `.env`（自定义服务商还会写 `providers.json`），启动本地运行时并验证视觉 API。对 Codex 会先备份，且只改写当前活动 provider 的 `base_url`（绝不改 `wire_api`、模型或 Key）；若 Codex 通过本地模型目录（如 cc-switch）加载模型列表，还会为当前模型声明图片输入，客户端才会允许粘贴图片（同样带备份）。对 OpenCode 会自动添加 OpenAI 兼容 provider。
4. 明确验证链路：

   ```bash
   agent-vision status
   agent-vision see <图片路径> -q "What is in this image?"
   agent-vision see https://example.com/image.png --task ocr
   agent-vision see --latest
   ```

   对 Codex，建议让用户粘贴一张图片验证；需要看本地文件时用 `agent-vision see <图片路径>` 兜底（内置 `view_image` 在当前桌面版可能被客户端替换成 `[Unsupported Image]`）。
   Windows 用户建议执行一次 `agent-vision autostart --enable`，重启电脑后代理会自动启动并由看门狗守护（默认 10 秒检查一次，19100 未监听就自动拉起），避免出现 `stream disconnected`。需要调间隔用 `--watchdog-interval 2-30`，想退回纯启动用 `--watchdog-interval 0`。
   若 `agent-vision` 不在 PATH，使用 `%USERPROFILE%\.agent-vision\agent-vision.cmd`（setup 自动生成）或 `python -m agent_vision`；不要依赖系统 PATH。代理必须用提权或普通终端方式启动（`agent-vision start`），确保进程不挂在 Codex 会话下。若沙箱不允许写用户配置，setup 会在当前目录生成 `agent-vision-finalize.cmd/.ps1`，请让用户跑一次完成配置；完成后运行 `agent-vision doctor`，全绿才算部署完成。

5. 如果用户之后要求回滚被自动修改的 Agent：

   ```bash
   agent-vision rollback codex
   agent-vision rollback opencode
   ```

6. 对 Claude Code 和 Cursor，执行 `agent-vision setup --agent claude --dry-run` 与 `agent-vision setup --agent cursor --dry-run` 会打印官方手动步骤。不要为这两个 Agent 编造配置键。
7. 绝不提交或上传 `.env`。API Key 只放在用户配置目录，不要放进仓库。最后向用户汇报：选择的服务商、验证结果、备份了哪些文件。

## 卸载与彻底清理

在 PowerShell 中按以下顺序执行。必须用 `&` 调用操作符和 `$env:USERPROFILE` 语法；`%USERPROFILE%` 只在 cmd 里有效：

```powershell
& "$env:USERPROFILE\.agent-vision\agent-vision.cmd" rollback codex
& "$env:USERPROFILE\.agent-vision\agent-vision.cmd" stop
& "$env:USERPROFILE\.agent-vision\agent-vision.cmd" autostart --disable
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'agent_vision' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
pip uninstall codex-deepseek-vision -y
Remove-Item -LiteralPath "$env:USERPROFILE\.agent-vision" -Recurse -Force
```

先回滚 Agent 配置、停代理、关自启。`autostart --disable` 只删除登录项，还要结束仍在运行的 `agent_vision` python 进程（看门狗若还在运行会继续把代理拉起），再卸载包，最后删除配置目录。如果已经先卸载了包，先重新安装（`pip install codex-deepseek-vision`）让 CLI/启动器命令可用，再按上面顺序清理。

## DeepSeek 原生识图

DeepSeek 已于 2026-08-21 上线实验性模型 `deepseek-v4-flash-vision-exp`。agent-vision 现已：
- 当请求模型是原生识图模型时，放行原图，让模型直接看图片（默认含 `deepseek-v4-flash-vision-exp`；用 `VISION_PASSTHROUGH_MODELS` 可扩充）。
- 内置 `deepseek` 视觉服务商（`--provider deepseek`），保留 DeepSeek 纯文本主模型、用同一把 Key 识图。
- 设 `VISION_FORCE_CONVERT=1` 可强制把图片转成文字，不走放行。

## 看图失败怎么办

- 若看到 `[Unsupported Image]` 或 `[image vision conversion failed ...]`：视觉转换失败或客户端限制，请让用户重新粘贴图片，或改用 `agent-vision see <图片路径>`。
- 失败原因会写入 `~/.agent-vision/logs/proxy.log`，排查时先看这个文件。
- `view_image` 工具结果在当前桌面版可能被替换为 `[Unsupported Image]`，这是客户端限制；若主模型为纯文本模型，桌面端可能只传文件路径并提示 "image content omitted" 而不是图片字节，此时请用 `agent-vision see <图片路径>` 或 `agent-vision see --latest` 兜底。
- 用户点“语音聊天”报 `404 /v1/live`、`Voice chat took too long to start` 等：这是预期限制，不是代理故障。Codex 实时语音走 OpenAI GPT-Live 通道，DeepSeek 不提供；代理会拦截 `/v1/live` 并返回明确提示。不要尝试修改代理或网络来“修复”，直接说明请用文字输入。
- 重启电脑后 Codex 无法对话（`stream disconnected`）：本地代理未启动，运行 `agent-vision start`，或先执行 `agent-vision autostart --enable` 让登录时自动拉起。

## Agent 守则

- 禁止修改系统代理、DNS、网络适配器、WinHTTP/WinINET 代理或全局 HTTP_PROXY/HTTPS_PROXY 环境变量。网络不通时不要自行设置系统代理；可重试、换官方 PyPI/镜像下载，或提示用户处理网络，然后继续。
- 修改全局 Agent 配置前必须先征得用户同意；每次自动修改前都会先生成带时间戳的备份。
- API Key 只放在本地 `.env`，绝不打印。
- 验证失败就停下来解释原因，不要猜测。
