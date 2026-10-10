# Z-Image MCP

使用 **Tongyi-MAI/Z-Image-Turbo** 本地模型生成图片的独立 MCP 服务。
基于官方 MCP Python SDK 和 Hugging Face Diffusers，提供 Streamable HTTP `/mcp`。
无需 ComfyUI，也不调用外部图片生成 API。

## 功能

- 中英文提示词生成 PNG，默认 1024×1024、9 steps、guidance_scale=0。
- 标准 MCP `notifications/progress`，包含任务 ID、真实阶段和采样步进度。
- 同步等待结果或后台提交、查询、取消；任务记录保存到 SQLite。
- 生成结果包含下载 URL、MCP resource link、尺寸、种子、SHA-256、加载和推理耗时。
- `get_image_result(inline=true)` 可返回标准 MCP `ImageContent`，供客户端视觉模型查看。
- 单 GPU 串行队列、参数/像素/队列/磁盘检查、失败恢复、闲置 120 秒卸载模型。
- 只读取本地权重，默认不下载模型；CUDA 推理在工作线程执行，不阻塞 HTTP。

## 安装与运行

需要 Linux、Python 3.11+、NVIDIA CUDA GPU 和完整的 Diffusers 模型目录。
模型目录必须包含 `model_index.json`、`transformer/`、`text_encoder/`、`tokenizer/`、
`scheduler/`、`vae/`。单个 ComfyUI checkpoint 不能作为该目录使用。

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[gpu,dev]'
cp .env.example .env
# 编辑 .env，填写 Z_IMAGE_MODEL_PATH。
bash scripts/start.sh
```

Debian/Ubuntu 缺少 `ensurepip` 时安装相应版本的 `python3-venv`，也可以使用已有
Conda Python 的 `python -m venv .venv` 创建独立环境。
PyTorch 必须使用与显卡、驱动兼容的 CUDA 构建；Blackwell 测试环境使用 `torch==2.13.0`
（CUDA 13.0）。安装器从包索引下载依赖，不会打包或重复下载模型权重。

局域网部署时将 `Z_IMAGE_HOST` 改为服务器 IP 或 `0.0.0.0`，并将
`Z_IMAGE_PUBLIC_URL` 设置为客户端能访问的地址，例如 `http://YOUR_SERVER:8766`。
`public_url` 不包含 `/mcp`、路径或查询参数。

```bash
curl http://127.0.0.1:8766/health
.venv/bin/python examples/generate.py --url http://127.0.0.1:8766/mcp
# 后台提交和轮询：
.venv/bin/python examples/generate.py --detached --output artifacts/background.png
# 连接断开后继续取已有任务，不重复生成：
.venv/bin/python examples/generate.py --job-id JOB_ID --output artifacts/recovered.png
```

示例客户端同时设置 HTTP 读取和 MCP 响应超时为 1800 秒，打印进度、下载 PNG，并校验
图片尺寸与 SHA-256。只调大 MCP 超时，不能覆盖 HTTP 客户端的较短读取超时。

## MCP 客户端配置

在支持 Streamable HTTP 的客户端填写：

```json
{
  "mcpServers": {
    "z-image": {
      "url": "http://192.168.108.50:8766/mcp",
	  "headers": {
			"Authorization": "Bearer YOUR_MCP_TOKEN"
	  },
      "description": "基于 Tongyi-MAI/Z-Image-Turbo 的图片生成服务，支持中英文提示词、自定义图片尺寸、实时生成进度、后台任务查询与取消，以及 PNG 图片下载。"
    }
  }
}
```

导入时保留 `headers` 和 `description` 字段；`description` 用于说明服务用途和支持的能力。
完整模板见 [examples/mcp-config.json](examples/mcp-config.json)，导入前替换地址和令牌。
如设置 `Z_IMAGE_API_TOKEN`，客户端的 Bearer 令牌必须与之相同；未设置时服务不校验令牌。
示例客户端从同名环境变量读取令牌。客户端的具体配置字段可能不同，以其文档为准。

| 工具 | 用途 |
| --- | --- |
| `get_image_service_info` | 查询模型、参数范围、队列与模型加载状态 |
| `generate_image` | 等待完成，持续回报真实进度，返回 PNG 链接和元数据 |
| `submit_image_generation` | 立即返回持久化任务 ID |
| `get_image_job` | 查询任务状态、参数、错误和结果 |
| `get_image_result` | 获取完成结果，`inline=true` 同时附加 MCP 图片内容 |
| `cancel_image_job` | 取消排队或运行任务；已完成任务保持原结果 |

```json
{
  "name": "generate_image",
  "arguments": {
    "prompt": "一只橘猫坐在窗边，窗外是雨后的花园，柔和自然光，真实摄影",
    "width": 1024,
    "height": 1024
  }
}
```

宽、高必须为 16 的倍数，范围 256–2048；默认最多 2,097,152 像素。
横向示例 `1536×864`，竖向示例 `864×1536`。
提示词最多 4000 字符；完整聊天模板超过模型 512 token 限制时会明确报错。
客户端只传 `prompt` 和可选的 `width`、`height`。服务端为每个任务自动生成随机种子，
采样步数固定为官方示例的 9；工具参数中不再暴露 `seed` 和 `steps`。
实际采样参数仍保存在任务记录和图片元数据中，便于排查问题。
Turbo 固定 `guidance_scale=0`，不提供负向提示词和图像编辑参数。

## 配置与运行行为

完整变量见 [.env.example](.env.example)，均以 `Z_IMAGE_` 开头。

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_PATH` | `models/Z-Image-Turbo` | 本地 Diffusers 模型目录 |
| `DATA_DIR` | `var` | SQLite、进程锁和 PNG/JSON 输出 |
| `HOST` / `PORT` | `127.0.0.1` / `8766` | HTTP 监听地址 |
| `PUBLIC_URL` | `http://127.0.0.1:8766` | 返回给客户端的下载地址前缀 |
| `API_TOKEN` | 空 | 可选共享 Bearer 令牌 |
| `DEVICE` / `DTYPE` | `cuda:0` / `bfloat16` | GPU 与权重精度 |
| `CPU_OFFLOAD` | `false` | 显存不足时启用；会增加 RAM 占用和传输耗时 |
| `IDLE_UNLOAD_SECONDS` | `120` | 队列空闲后卸载；0 表示常驻 |
| `MAX_PIXELS` | `2097152` | 单张图片总像素上限 |
| `MAX_PENDING_JOBS` | `16` | 运行和待处理任务总上限 |
| `MIN_FREE_DISK_MB` | `512` | 接收新任务所需的剩余磁盘空间 |

`/health` 表示服务及任务工作循环存活，不等同于 CUDA 推理验证。
`model_files_present` 只检查模型索引是否存在；实际权重在第一次生成时验证和加载。
`model_loaded=false` 在刚启动或空闲卸载后属于正常状态。

GPU 阶段的取消在模型加载完成、采样步结束或保存前等安全边界生效，不能立刻中断
正在执行的 CUDA kernel。直接调用断开后任务继续运行，进度消息包含可恢复查询的
`job_id`；需要停止时显式调用 `cancel_image_job`。

重启时未完成的任务标为 `failed/interrupted`，避免重复生成或错误报告成功。
已经完成的任务和下载链接保留。每个数据目录只能启动一个服务进程；不要配置多个
Uvicorn worker。进度百分比表示已完成的工作阶段，不是预计剩余时间。

图片和任务记录默认持续保留，无自动清理。维护者负责磁盘容量和保留期限；PNG 和
旁边的 JSON 含提示词、种子等生成信息，应和任务数据库一起按需管理。
下载链接内的随机密钥授权读取对应图片，服务默认关闭 HTTP access log，避免记录密钥。

默认只监听本机。无令牌的局域网配置适合可信网络中的单租户使用；服务不提供用户隔离。
如需暴露到互联网，请配置 HTTPS 和网关身份认证/限流；共享 Bearer 令牌不等同于 OAuth。

## systemd 用户服务

```bash
bash scripts/install-service.sh
systemctl --user status z-image-mcp.service
journalctl --user -u z-image-mcp.service -f
systemctl --user restart z-image-mcp.service
systemctl --user stop z-image-mcp.service
```

脚本为当前目录生成 unit，不硬编码用户名、模型路径或服务器地址。
需要退出 SSH 后持续运行时，由管理员启用该用户的 linger。
部署更新前应确认 `/health` 中 `active_jobs=0`；安装脚本不会自动重启已运行的服务。

## 开发与验证

```bash
.venv/bin/pip install -e '.[dev]'
bash scripts/test.sh -q
.venv/bin/ruff check .
```

CPU 测试使用显式注入的假后端，覆盖真实 HTTP MCP 握手、工具发现、进度、结构化结果、
图片内容/下载完整性、后台任务、串行执行、失败恢复、取消、重启、进程锁和鉴权。
运行服务始终使用真实 Diffusers 后端；CPU 测试通过不代表 GPU 已验证。
GPU 验证使用 `examples/generate.py`；生成证据保存在忽略 Git 的 `artifacts/`。
CI 只执行 CPU 测试。

## 开源与模型来源

服务代码使用 [MIT 许可证](LICENSE)。模型权重不包含在本项目中，遵守其自身的
[Apache-2.0 许可证和模型说明](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)。

- [Z-Image 官方项目](https://github.com/Tongyi-MAI/Z-Image)
- [Diffusers Z-Image API](https://huggingface.co/docs/diffusers/api/pipelines/z_image)
- [MCP Python SDK v1 文档](https://py.sdk.modelcontextprotocol.io/v1/)

MCP SDK 固定在仍维护的 v1 系列 `>=1.30,<2`，避免无意升级至具有破坏性 API 变更的 v2。
`.env`、模型、任务、生成图片、安装环境以及具体服务器验证记录均不应提交到公开仓库。
