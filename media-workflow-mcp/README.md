# Media Workflow MCP

## 成品链接有效期

`MEDIA_MCP_ARTIFACT_URL_TTL_SECONDS` 默认 `0`，表示永久签名链接；正整数表示有效秒数。签名绑定文件身份与有效期，永久链接仍需要有效签名。历史限时链接沿用原签名期限，可从已保存工作流重新查询成品以获得当前策略签发的链接，无需重新生成视频。文件删除或签名密钥轮换后，既有链接失效。

A capability-neutral MCP service for media assembly. It does not know about any
video model. It exposes six typed operations:

- `probe_media`: inspect duration, streams, and SHA-256 for an immutable HTTP(S) asset;
- optional `analyze_music_timeline`: disabled by default; when explicitly enabled,
  persist a bounded RMS energy curve as an Artifact without placing the full curve
  in model context or making unsupported semantic claims;
- `validate_and_render_music_video_plan`: validate and persist one complete plan,
  returning only a digest and compact segment references;
- `get_music_video_plan_segment`: retrieve one exact stored prompt immediately
  before generating that segment;
- `extract_last_frame`: publish the last visible frame of a completed clip;
- `assemble_music_video`: normalize ordered silent clips, concatenate them, and
  mux one immutable master audio track into the final MP4.

The service requires `ffmpeg` and `ffprobe`. Remote reads are denied unless the
exact origin appears in `MEDIA_MCP_ALLOWED_FETCH_ORIGINS`; redirects are denied.
Artifacts use expiring HMAC-signed URLs.

## Run

### Harness endpoint binding

Generator orchestration tools advertise their provider-owned selection in MCP
`tools/list` metadata (`connections.py`). A supporting Harness supplies only
authorized endpoints through `x-exoharness-video-generator-url`; the service
keeps its configured `MEDIA_MCP_VIDEO_GENERATOR_URL` fallback when no binding is
available. No model input can set this policy. The generator's input schema,
authentication, duration limits, reference handling and assembly are unchanged.
Probe, frame extraction and assembly-only tools do not request this binding.

### Starting the service

```bash
python -m venv .venv
.venv/bin/pip install -e .
MEDIA_MCP_ARTIFACT_SIGNING_KEY='replace-with-a-random-secret' \
MEDIA_MCP_ALLOWED_FETCH_ORIGINS='http://127.0.0.1:8787,http://127.0.0.1:8765' \
.venv/bin/media-workflow-mcp
```

Configure these values for deployment:

- `MEDIA_MCP_API_TOKEN`: optional bearer token for MCP calls;
- `MEDIA_MCP_PUBLIC_BASE_URL`: browser/FastH3-reachable base URL;
- `MEDIA_MCP_DATA_DIR`: persistent artifacts and validated plans (default `/tmp/media-workflow-mcp`);
- `MEDIA_MCP_ALLOWED_FETCH_ORIGINS`: exact comma-separated input origins;
- `MEDIA_MCP_ARTIFACT_SIGNING_KEY`: required signing key;
- `MEDIA_MCP_OPERATION_TIMEOUT_SECONDS`: ffmpeg/ffprobe and download timeout.
- `MEDIA_MCP_ENABLE_AUDIO_ANALYSIS`: expose the optional RMS analysis tool only
  when set to `true` (default `false`).

The caller must pass complete URLs. Local paths and Base64 media are not accepted.
