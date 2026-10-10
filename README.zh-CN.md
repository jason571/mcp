# ExoHarness MCP 服务

[English](README.md) | 简体中文

用于视频、音乐、图片、浏览器操作、教学动画和媒体编排的独立 MCP 服务集合。每个服务通过 MCP 提供工具描述和 JSON Schema，可接入支持相应传输方式的客户端。[ExoHarness](https://github.com/jason571/Exoharness) 负责智能体、授权、执行追踪和界面，本仓库提供可执行能力。

服务独立部署。导入 MCP 配置是在客户端登记连接，不会自动下载模型权重，也不会启动 GPU 推理后端。

## 服务目录

| 组件 | 用途 | 运行依赖 |
| --- | --- | --- |
| [fastvideo-h3-mcp](fastvideo-h3-mcp/) | FastVideo H3 视频与音频生成，支持任务提交、查询、进度和取消 | Python 3.12+、GPU、H3 权重及推理后端 |
| [minimax-h3-mcp](minimax-h3-mcp/) | 基于 ComfyUI 的 MiniMax H3 生成，支持图片输入和持久化任务 | Python 3.12+、ComfyUI 工作流、GPU 和模型文件 |
| [media-workflow-mcp](media-workflow-mcp/) | 持久化多段计划、衔接帧提取、视频拼接及音视频合成 | Python 3.12+、FFmpeg/ffprobe 及上游服务 |
| [lesson-renderer-mcp](lesson-renderer-mcp/README.md) | 教学计划、可局部修复的草稿及带配音的 Manim 动画 | Python 3.12、Manim、FFmpeg、字体、渲染资源及 TTS 音频 |
| [yue2-mcp](yue2-mcp/) | 歌词与演唱歌曲生成、队列、取消及 MP3/FLAC 交付 | Python 3.11+、YuE2 推理包、主模型/VAE 权重、GPU；歌词 LLM 可选 |
| [z-image-mcp](z-image-mcp/) | 本地图片生成和图片文件交付 | Python 3.11+、Z-Image 模型和 GPU 运行环境 |
| [mcp-chrome](mcp-chrome/) | 通过 Chrome 扩展与原生桥接服务操作浏览器 | Node.js、pnpm、Chrome、扩展与 Native Host |
| [weather-forecast-server](weather-forecast-server/) | 通过 wttr.in 查询天气 | Python 和网络访问；上游为 stdio 服务 |
| [skill-workspace-mcp](skill-workspace-mcp/) | 独立任务工作区及隔离的 Python、Shell、TypeScript 执行 | Docker、Broker 配置和任务运行镜像 |

[Tavily](https://docs.tavily.com/documentation/mcp) 为外部托管搜索服务，本仓库只提供占位连接示例，不包含 Tavily 服务端实现。

独立 Workspace MCP 保留给其他客户端和已有集成使用。当前 ExoHarness 已提供原生 Sandbox/Coding 模块，使用这些模块不要求安装该 MCP。

## 安装与运行

```bash
git clone git@github.com:jason571/mcp.git
cd mcp
```

选择服务后，先阅读组件说明并准备推理后端。各服务使用独立虚拟环境；不同组件使用的 MCP SDK 版本不完全一致，不建议共用一个环境安装所有依赖。

以包含 `pyproject.toml` 的 Media Workflow 为例：

```bash
cd media-workflow-mcp
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

按其 [README](media-workflow-mcp/README.md) 设置签名密钥、持久化目录、输入来源白名单、公开文件地址和上游连接，再执行 `.venv/bin/media-workflow-mcp`。其他服务的安装前提和环境变量不同。Python 是这些服务的实现与运行环境，并非仅用于测试。

### Docker 方式

Media Workflow、Lesson Renderer 和独立 Workspace 运行时提供 Dockerfile：

```bash
docker build -t media-workflow:local media-workflow-mcp
docker build -t lesson-renderer:0.1.0-manim0.20.1 lesson-renderer-mcp
docker build -t skill-workspace:local skill-workspace-mcp
```

Media Workflow 镜像包含服务源码。Lesson Renderer 需要只读挂载源码和课件渲染资源，详见[部署说明](lesson-renderer-mcp/README.md)。Workspace 镜像用于执行任务，不是完整的 Broker 服务；其中 `/runner` 由 Broker 提供。部署时需配置持久化卷、凭据和网络策略。GPU 推理后端与模型权重分别安装。

## 接入客户端

1. 启动目标服务，确认监听地址及认证配置。
2. 从[导入示例](examples/README.zh-CN.md)复制对应 JSON，在本地填写实际 URL 和密钥。
3. 在 ExoHarness 技能导入页面登记 MCP，启用后授权给指定助理。
4. 验证 MCP 初始化与 `tools/list`，再测试允许执行的工具。HTTP 健康检查不能代替真实模型推理验收。

```json
{
  "mcpServers": {
    "media-workflow": {
      "type": "streamableHttp",
      "url": "http://127.0.0.1:8791/mcp",
      "headers": {"Authorization": "Bearer YOUR_MCP_TOKEN"}
    }
  }
}
```

`127.0.0.1` 指客户端进程所在主机或网络命名空间。客户端位于容器或另一台主机时，应改为它能访问的地址。生成文件的公开地址单独配置，并保证浏览器与下游服务能访问；下载来源白名单需包含实际输入地址。

天气组件使用 stdio，需要支持 stdio 的客户端或可信传输桥接。仅填写 HTTP URL 不会改变服务端传输方式。

## 运维与安全

- `.env`、Token 文件、私有连接配置、模型权重和生成文件不进入仓库；公开示例只包含占位值。
- 外网服务应配置认证与合适的反向代理。浏览器工具能够操作所连接的登录会话，建议使用独立浏览器配置与最小权限账号。
- 不可信脚本通过隔离运行环境执行。登记 Skill 不代表允许在宿主机直接运行其代码。
- 任务数据库、计划和产物使用持久化卷。任务 ID 不代表成品已完成，应按实时 Schema 查询结果、取消或恢复。
- 仓库包含组件测试。CPU/接口测试与真实 GPU 生成、音视频可播放验收分别进行；权重、实际凭据及生成文件不随代码分发。

## 相关仓库与许可证

- [ExoHarness 主框架](https://github.com/jason571/Exoharness)
- [Skills 集合](https://github.com/jason571/skills)
- [导入配置示例](examples/README.zh-CN.md)
- [第三方来源及许可证](THIRD_PARTY.md)

各组件保留原有许可证与声明，本仓库不使用统一许可证替代上游条款。模型权重、推理包和外部服务各自适用其许可与服务条款。
