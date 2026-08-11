# Agent 安装指南

仓库地址：https://github.com/SIMON-WORLD/codex-deepseek-vision

这个项目设计给“只要会和 AI Agent 聊天”的用户：不需要会终端，也不需要会编程。

## 用户一句话指令

把下面这句话发给你的 Agent（Codex、OpenCode 等）：

```text
帮我安装并配置 agent-vision。请阅读 AGENT_INSTALL.zh-CN.md 并从头到尾执行。默认使用智谱免费服务，除非我指定其他服务商。
```

## Agent 执行清单

1. 从 PyPI 安装（最简单）；如果 PyPI 不可达，再克隆仓库从源码安装：

   ```bash
   pip install codex-deepseek-vision
   # 或
   git clone https://github.com/SIMON-WORLD/codex-deepseek-vision.git
   cd codex-deepseek-vision
   pip install .
   ```

   除非下面步骤明确要求修改 Agent 配置，否则所有写入都留在项目目录内。

3. 询问用户选择视觉服务商：`zhipu`（免费）、`dashscope`、`openai`、`gemini`、`groq`、`siliconflow`、`openrouter`，或自定义 OpenAI 兼容接口。请用户提供对应的 API Key，或让用户自己粘贴到用户配置目录的 `.env`（`~/.agent-vision/.env`，Windows 为 `%USERPROFILE%\.agent-vision\.env`）。
4. 运行引导式配置：

   ```bash
   agent-vision setup
   ```

   向导会自动检测已安装的 Agent，在用户配置目录写入 `.env`（自定义服务商还会写 `providers.json`），启动本地运行时并验证视觉 API。对 Codex 会先备份，且只改写当前活动 provider 的 `base_url`（绝不改 `wire_api`、模型或 Key）；若 Codex 通过本地模型目录（如 cc-switch）加载模型列表，还会为当前模型声明图片输入，客户端才会允许粘贴图片（同样带备份）。对 OpenCode 会自动添加 OpenAI 兼容 provider。
5. 明确验证链路：

   ```bash
   agent-vision status
   agent-vision see <图片路径> -q "What is in this image?"
   agent-vision see https://example.com/image.png --task ocr
   agent-vision see --latest
   ```

   对 Codex，建议让用户粘贴一张图片验证；需要看本地文件时用 `agent-vision see <图片路径>` 兜底（内置 `view_image` 在当前桌面版可能被客户端替换成 `[Unsupported Image]`）。
   Windows 用户建议执行一次 `agent-vision autostart --enable`，重启电脑后代理会自动启动并由看门狗守护（默认 10 秒检查一次，19100 未监听就自动拉起），避免出现 `stream disconnected`。需要调间隔用 `--watchdog-interval 2-30`，想退回纯启动用 `--watchdog-interval 0`。
   若 `agent-vision` 不在 PATH，使用 `%USERPROFILE%\.agent-vision\agent-vision.cmd`（setup 自动生成）或 `python -m agent_vision`；不要依赖系统 PATH。代理必须用提权或普通终端方式启动（`agent-vision start`），确保进程不挂在 Codex 会话下。若沙箱不允许写用户配置，setup 会在当前目录生成 `agent-vision-finalize.cmd/.ps1`，请让用户跑一次完成配置；完成后运行 `agent-vision doctor`，全绿才算部署完成。

6. 如果用户之后要求回滚被自动修改的 Agent：

   ```bash
   agent-vision rollback codex
   agent-vision rollback opencode
   ```

7. 对 Claude Code 和 Cursor，执行 `agent-vision setup --agent claude --dry-run` 与 `agent-vision setup --agent cursor --dry-run` 会打印官方手动步骤。不要为这两个 Agent 编造配置键。
8. 绝不提交或上传 `.env`。API Key 只放在用户配置目录，不要放进仓库。最后向用户汇报：选择的服务商、验证结果、备份了哪些文件。

## 看图失败怎么办

- 若看到 `[Unsupported Image]` 或 `[image vision conversion failed ...]`：视觉转换失败或客户端限制，请让用户重新粘贴图片，或改用 `agent-vision see <图片路径>`。
- 失败原因会写入 `~/.agent-vision/logs/proxy.log`，排查时先看这个文件。
- `view_image` 工具结果在当前桌面版可能被替换为 `[Unsupported Image]`，这是客户端限制，粘贴图片不受影响。
- 用户点“语音聊天”报 `404 /v1/live`、`Voice chat took too long to start` 等：这是预期限制，不是代理故障。Codex 实时语音走 OpenAI GPT-Live 通道，DeepSeek 不提供；代理会拦截 `/v1/live` 并返回明确提示。不要尝试修改代理或网络来“修复”，直接说明请用文字输入。
- 重启电脑后 Codex 无法对话（`stream disconnected`）：本地代理未启动，运行 `agent-vision start`，或先执行 `agent-vision autostart --enable` 让登录时自动拉起。

## Agent 守则

- 禁止修改系统代理、DNS、网络适配器、WinHTTP/WinINET 代理或全局 HTTP_PROXY/HTTPS_PROXY 环境变量。网络不通时不要自行设置系统代理；可重试、换官方 PyPI/镜像下载，或提示用户处理网络，然后继续。
- 修改全局 Agent 配置前必须先征得用户同意；每次自动修改前都会先生成带时间戳的备份。
- API Key 只放在本地 `.env`，绝不打印。
- 验证失败就停下来解释原因，不要猜测。
