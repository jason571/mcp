# YuE2 MCP

独立的 Streamable HTTP `/mcp` 服务，使用本地 `m-a-p/YuE2-3B`、`m-a-p/YuE2-Vae`
生成 48 kHz FLAC 歌曲及 MP3 试听文件。单 GPU 串行任务队列，SQLite 持久化任务状态，
支持后台提交、进度与排队位置查询、取消、断线后恢复查询、带随机下载密钥的
音频/歌词/ABC 乐谱文件。

YuE2 模型本身接收曲风和歌词，不负责写歌词。自动写歌词使用**另行配置**的
OpenAI 兼容文本模型；也可以直接传入人工或 Agent 编写的歌词。没有配置文本模型时，
`generate_lyrics` 和省略 `lyrics` 的音乐任务会明确报错。

## 安装

需要 Linux、Python 3.11+、`ffmpeg`（带 `libmp3lame` 编码器）、支持 BF16 的 NVIDIA GPU，以及本地 YuE2 主模型和
**单独的 YuE2-Vae**。官方说明建议 24GB 显存及 24GB 可用内存。

```bash
python3 -m venv .venv
.venv/bin/pip install /home/clhh/.cache/modelscope/models/m-a-p--YuE2-3B/snapshots/master/yue2_infer-0.1.5-py3-none-any.whl
.venv/bin/pip install -e .
cp .env.example .env
# 核对模型、VAE 目录、监听地址、歌词模型接口及令牌
.venv/bin/yue2-mcp
```

运行时只读 `YUE2_MODEL_PATH` 和 `YUE2_VAE_PATH` 指向的本地文件，
`local_files_only=True`；不会在生成期间下载模型。服务默认监听 `127.0.0.1:8767`。
若要让局域网客户端访问，配置 `YUE2_HOST=0.0.0.0` 和客户端可访问的
`YUE2_PUBLIC_URL`。公开网络需要可信网关和身份认证；可设置共享
`YUE2_API_TOKEN` 保护 `/mcp`。

`GET /health` 只有工作循环、推理包、主模型权重与 tokenizer、VAE 权重都存在才返回 200；仍不代表 GPU
推理已通过。模型在首个音乐任务中加载。歌词任务不需要 GPU，但与音乐任务共用
串行队列。任务进度是已进入的真实阶段，无法估计剩余时长。取消在 token、阶段
边界或保存前生效，进行中的 CUDA kernel 不会即时停止。重启后未完成任务标为失败；
已完成任务和文件保留。每个数据目录只能运行一个进程，不要启动多个 Uvicorn worker。

## MCP 工具

| 工具 | 作用 |
| --- | --- |
| `get_music_service_info` | 查询模型、VAE、歌词接口和队列状态 |
| `submit_music_generation` | 提交歌曲并立即返回任务 ID |
| `generate_music` | 等待歌曲完成并发送 MCP 进度通知；默认只把 MP3 作为对话音频展示，FLAC 仍保存在结果中 |
| `submit_lyrics_generation` | 提交歌词任务 |
| `generate_lyrics` | 生成分段歌词并发送进度通知 |
| `get_music_job` | 查询歌曲或歌词任务、真实阶段和结果 |
| `get_music_result` | 获取最终结果；`audio_format=mp3` 默认展示 MP3，显式设为 `flac` 才展示 FLAC |
| `cancel_music_job` | 取消排队任务或请求中断运行任务 |

歌曲参数：`style` 必填；`lyrics` 与 `lyrics_prompt` 二选一。
一次 `generate_music` 只生成一首歌。MP3 和 FLAC 是同一录音的两种编码，不能作为两首候选曲展示；需要两首不同作品时分别提交两次，并使用不同的编曲方向。封面可由另外授权的图片生成能力制作，封面失败不影响歌曲交付。
`cot=full` 规划旋律和和弦，`melody` 只规划旋律，`off` 不规划；
可提供 `abc` 乐谱做再生成或编曲修改，`cot=off` 时不能提供 `abc`。
`seed` 可选，省略时由服务端随机生成并返回。服务返回音频时长、SHA-256、
MP3 试听、FLAC 下载、歌词、ABC 乐谱链接和截断标志。歌词生成使用
`prompt`、`style`、`language`。
`cfg_scale` 可选，范围 0–20；省略时使用官方各 `cot` 模式的默认值。
排队任务返回从 1 开始的 `queue_position`，运行或结束后为 `null`。

```json
{
  "mcpServers": {
    "yue2": {
      "type": "streamableHttp",
      "url": "http://YOUR_SERVER:8767/mcp",
      "headers": {"Authorization": "Bearer YOUR_MCP_TOKEN"},
      "description": "YuE2 音乐生成：自动写歌词、曲风与歌词生成歌曲、ABC 乐谱编辑、排队、进度、取消和 FLAC 下载"
    }
  }
}
```

可直接复制 [examples/mcp-config.json](examples/mcp-config.json) 再填写服务器地址和
`YUE2_API_TOKEN` 对应的令牌。`Authorization` 的值只需要一个 `Bearer ` 前缀，
无需再加引号。不要把实际令牌提交到版本库。

## 配置

参考 [.env.example](.env.example)。`YUE2_LYRICS_BASE_URL` 应指向兼容
`/v1/chat/completions` 的 API 基址，例如 `http://127.0.0.1:8000/v1`；
同时设置 `YUE2_LYRICS_MODEL`，需要鉴权时设置 `YUE2_LYRICS_API_KEY`。
不在日志和任务记录中保存该密钥。歌词和歌曲任务请求会保存在 SQLite，
音频和歌词文件保存在 `YUE2_DATA_DIR/outputs`；按业务需要管理保留期限。
`YUE2_MEMORY_BUDGET_GIB`、`YUE2_QUANTIZATION`（`none`/`fp8`）和
`YUE2_OFFLOAD_AR` 直接传给官方推理管线；默认使用未量化模型。
队列空闲 `YUE2_IDLE_UNLOAD_SECONDS` 秒后卸载模型以释放显存，设为 `0` 可常驻。

## 验证

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check .
curl http://127.0.0.1:8767/health
.venv/bin/python examples/generate.py --style "Mandarin pop, warm female vocal, piano" \
  --lyrics-file lyrics.txt --output artifacts/song.flac
```

CPU 测试使用假后端，验证队列、取消、恢复、MCP 工具、鉴权和文件下载。生产验收应实际提交一首短歌，
确认 `succeeded`、FLAC 可播放、音频长度及歌词内容；健康检查不能替代这一步。

本地模型可通过 `YUE2_MODEL_PATH` 指向现有目录，例如
`/home/flyang/.cache/modelscope/models/m-a-p--YuE2-3B/snapshots/master`。
这台开发机可先复制 [.env.local.example](.env.local.example) 到 `.env`，
其中已填入当前主模型和 VAE 目录；服务器使用 [.env.example](.env.example) 并核对实际路径。
真实音频验证还必须设置本机 `YUE2_VAE_PATH`。没有 VAE 时 CPU 接口测试可运行，
`/health` 会返回 503；这并不表示主模型缺失。运行推理前还需确保 GPU 有足够空闲显存。

生产用户服务模板见 [deploy/yue2-mcp.service.example](deploy/yue2-mcp.service.example)。
部署时按实际目录、模型路径和客户端地址修改模板与 `.env`，并在真实 GPU 验证成功后启用。
安装依赖后，在服务器运行 `bash scripts/install-user-service.sh` 可生成独立令牌、
写入仅用户可读的 `.env` 和 `artifacts/mcp-config.json`，并启用用户级 systemd 服务。
默认只监听服务器本机 `127.0.0.1:8767`；从其他机器访问时可建立 SSH 端口转发。

ExoHarness 的 `yue2-song-production` Skill 默认只生成一首歌；用户明确要两个版本时，
才使用不同编曲生成两首。图片生成能力可用时可在歌曲完成后制作封面，但封面失败不影响歌曲交付。
Skill 源文件位于 ExoHarness 项目的 `examples/web-console/yue2-song-production/SKILL.md`，
需由管理员登记、准备并授权给相应助理。

模型与运行时来源：[YuE2 官方模型说明](https://huggingface.co/m-a-p/YuE2-3B)、
[官方推理代码](https://github.com/multimodal-art-projection/YuE)。
主模型权重遵守其自身的 CC BY-NC 4.0 许可；部署前需确认使用场景符合许可。
