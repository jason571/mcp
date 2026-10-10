# ExoHarness MCP Services

English | [简体中文](README.zh-CN.md)

Independent Model Context Protocol (MCP) services for generation, browser interaction and media workflows. Each service exposes its own tool descriptions and JSON Schemas and can be connected to an MCP-compatible client. [ExoHarness](https://github.com/jason571/Exoharness) provides the agent, authorization, tracing and user interface; this repository provides executable capabilities.

Services are deployed separately. Importing an MCP configuration registers a connection; it does not download model weights or start a GPU backend.

## Services

| Component | Purpose | Runtime and dependencies |
| --- | --- | --- |
| [fastvideo-h3-mcp](fastvideo-h3-mcp/) | Queued video/audio generation through a FastVideo H3 backend; submit, query, progress and cancel | Python 3.12+, GPU, H3 weights and configured inference backend |
| [minimax-h3-mcp](minimax-h3-mcp/) | ComfyUI-backed MiniMax H3 generation with image inputs and persistent jobs | Python 3.12+, configured ComfyUI workflow, GPU and model files |
| [media-workflow-mcp](media-workflow-mcp/) | Durable multi-clip plans, continuity frames, video assembly and audio/video composition | Python 3.12+, FFmpeg/ffprobe and configured upstream services |
| [yue2-mcp](yue2-mcp/) | Lyrics and sung music generation, job queue, cancellation and MP3/FLAC results | Python 3.11+, YuE2 inference package, main/VAE weights, GPU; optional lyrics LLM |
| [z-image-mcp](z-image-mcp/) | Local image generation and downloadable image assets | Python 3.11+, Z-Image model files and supported GPU runtime |
| [mcp-chrome](mcp-chrome/) | Browser interaction through a Chrome extension and native bridge | Node.js, pnpm, Chrome and installed extension/native host |
| [weather-forecast-server](weather-forecast-server/) | Weather queries through wttr.in | Python and network access; upstream stdio server |
| [skill-workspace-mcp](skill-workspace-mcp/) | Optional standalone task workspace and isolated Python, Shell and TypeScript execution | Docker, broker configuration and task runtime images |

[Tavily](https://docs.tavily.com/documentation/mcp) is a hosted search service, not an implementation in this repository. A placeholder configuration is included under [examples](examples/README.md).

The workspace MCP is retained for independent clients and existing integrations. Current ExoHarness also provides native Sandbox/Coding tools: installing this MCP is not required to use those modules.

## Quick start

```bash
git clone git@github.com:jason571/mcp.git
cd mcp
```

Choose a component, read its documentation, and prepare its backend before connecting a client. Use a separate virtual environment per service: the packages do not all depend on the same MCP SDK version.

For a Python component with `pyproject.toml`, for example Media Workflow:

```bash
cd media-workflow-mcp
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

Configure the signing key, data directory, allowed input origins, public asset URL and upstream connections as described in its [README](media-workflow-mcp/README.md), then start `.venv/bin/media-workflow-mcp`. Other services have different prerequisites and configuration names. Python is the implementation runtime for these services, not merely a test dependency.

### Docker

Dockerfiles are supplied for Media Workflow and the optional workspace runtime. Build Media Workflow from the repository root:

```bash
docker build -t media-workflow:local media-workflow-mcp
```

Media Workflow includes its source in the image. The optional workspace Dockerfile requires a pre-existing `lesson-renderer:0.1.0-manim0.20.1` base image, which is not distributed in this repository. If that image is already installed, build with `docker build -t skill-workspace:local skill-workspace-mcp`. The workspace image is a **task execution image**, not a complete broker deployment: its `/runner` entrypoint is supplied by the broker. Configure volumes, credentials, network policy and persistent storage for your deployment. GPU inference backends and weights are installed independently.

## Connect to ExoHarness or another MCP client

1. Start the selected service and verify its listening address and authentication.
2. Copy a matching [example configuration](examples/README.md), replacing the URL and token locally.
3. Import that JSON in ExoHarness's capability import page, enable the service and grant the intended assistant access.
4. Verify MCP initialization and `tools/list`, then test an authorized tool. A healthy HTTP process alone does not verify model inference.

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

`127.0.0.1` means the client process's own host/network namespace. In Docker or on another machine, use an address reachable from that client. Configure public artifact URLs separately so browsers and downstream services can retrieve results. Fetch-origin allowlists must cover the actual input origins.

The weather server uses stdio. Use a client supporting stdio or deploy a trusted transport bridge; an HTTP URL example does not convert that server's transport. Consult each client's transport support before importing.

## Operation and security

- Keep `.env`, token files, private connection JSON, model weights and generated assets outside Git. Examples contain placeholders only.
- Protect exposed services with authentication and an appropriate reverse proxy. Browser tools can operate the connected browser session; use a dedicated profile with limited privileges.
- Execute untrusted scripts only through a configured isolated runtime. Registering a Skill is not authorization to execute it on the host.
- Store job databases, plans and artifacts on persistent volumes. A submitted job ID is not a completed deliverable; query the result and handle cancellation/recovery using the service's live schema.
- Separate CPU/interface tests from real GPU generation and playable-output acceptance. Component test directories are included; model weights, service tokens and real generated outputs are not.

## Related projects

- [ExoHarness framework](https://github.com/jason571/Exoharness)
- [Skills collection](https://github.com/jason571/skills)
- [Configuration examples](examples/README.md)
- [Licensing and upstream attribution](THIRD_PARTY.md)

## License

This collection retains component-specific license files and notices. There is no blanket license replacing upstream terms. Model weights, inference packages and third-party services have their own terms; review [THIRD_PARTY.md](THIRD_PARTY.md) and the relevant component before redistribution.
