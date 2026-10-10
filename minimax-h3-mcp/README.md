# MiniMax H3 MCP

独立的视频生成 MCP：使用 ComfyUI 原生 MiniMax H3 **8 步**图和原生 LoRA，
同一套模型做文生视频和图生视频。对外只运行一个 MCP 服务；MCP 在内部按需
管理本机私有的 ComfyUI Worker 子进程，不依赖另一个 systemd 服务，也不使用
FastVideo 推理引擎。

服务器：`101.34.55.200:6022`（用户 `clhh`），部署目录 `/home/clhh/minimax-h3-mcp`。

## 采样栈

- 底模：`minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`
- SageAttention：`MiniMaxH3MemoryEfficientSageAttentionPatch`
- LoRA：`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`，strength `1.0`
- 采样：`euler` + `beta`，**8 steps**
- 条件节点：无参考图时为 `MiniMaxH3ImageToVideo`；持续参考图时为 `MiniMaxH3ReferenceToVideo`，明确指定的首/尾帧再通过 `MiniMaxH3AddGuide` 加入。

## 路由

| 请求 | 行为 |
|---|---|
| 纯文本 | 去掉 LoadImage，文生视频 |
| 首帧和/或尾帧 | 上传参考图，图生视频 |
| `reference_image_url` / `reference_image_asset_id` | 上传并绑定持续视觉参考，不自动成为首帧 |
| 持续参考 + 明确指定的首/尾帧 | Ref2VA 加帧引导；可用于多段 MV 的段间首尾帧衔接 |
| `model_id=auto/fast/base` | 都走这一套 8 步图 |

例如用户只说“参考我上传的图片生成视频”，调用 `generate_video` 时传
`reference_image_url`，不要传 `first_frame_url`。只有用户明确要求图片作为开场
画面，才传 `first_frame_url`。若下一段需要承接上一段末帧，同时保持原图的人物或
场景参考，可以在同一调用中传原图的 `reference_image_url` 与上一段末帧的
`first_frame_url`。两者作用不同：前者持续参考，后者只锚定第 0 帧。用户提供详细
风格时直接写入 `prompt`；风格缺失时才查询风格技能。MV 仍使用原歌曲作最终音轨，
不会为理解歌曲另做歌词提取。

工具名与旧 FastVideo MCP 相同：`generate_video`、`submit_video_generation`、
`list_video_models` 等。默认端口 **8766**。

### 按 `mv.json` 使用参考图

`mv.json` 是 ComfyUI 的界面工作流参考：其中
`MiniMaxH3ReferenceToVideo.ref_images.ref_image_0` 接收持续参考图。
MCP 提交的是相应 API 图，不直接把界面 JSON 当作执行请求。使用顺序是：

1. 先用已授权的视觉能力观察上传图，提取确实可见的人物、服饰、场景、构图和光线；文件名或 URL 本身不能证明图片内容。再由用户决定哪些元素必须保留。没有视觉能力时先询问图片内容，不要编造。
2. 根据用户描述和已观察的元素生成剧情、分镜；明确的风格直接采用，仅在风格不足以成片时才查询风格技能。没有图时仅根据用户描述创作。歌曲可按歌名理解意境，但不能声称已核实歌词或精确歌词时间轴。
3. 仅说“参考这张图”时，调用时传 `reference_image_url`，不传 `first_frame_url`。明确说“这张图是首帧”才传后者；两种用途可以由用户同时指定。
4. 对多段 MV，每一段都传相同的 `reference_image_url`；第二段起另传上一段实际生成结果的末帧 URL 作为 `first_frame_url`。媒体编排服务持久化原始图、逐段签发可访问 URL，不要求模型每次重读同一张图。

普通参考图的一次调用示例（时长、分辨率以实时 Schema 为准）：

```json
{
  "name": "submit_video_generation",
  "arguments": {
    "prompt": "A continuous story consistent with the observed reference subject and setting; four distinct shots...",
    "reference_image_url": "https://your-asset-origin/v1/attachment-assets/att_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "aspect_ratio": "16:9",
    "resolution": "512p",
    "duration_seconds": 15,
    "audio_enabled": false,
    "model_id": "auto"
  }
}
```

下一段只增加 `first_frame_url`，其值必须来自上一段已完成的视频末帧 Artifact，不能重复使用最初的参考图充当首帧。对完整 MV，优先让 `media-workflow.create_music_video_plan` 接收 `reference_image_attachment_id`，其后续 `render_validated_music_video` 会为每段传持续参考图并处理段间首帧；不要由 Agent 手动逐段重提已提交的任务。

## 生产安装

```bash
cd /home/clhh/minimax-h3-mcp
./scripts/install-user-service.sh
```

MCP Web 进程使用自己的 `.venv`；内部 ComfyUI Worker 使用配置的
`COMFYUI_PYTHON` 与 `COMFYUI_ROOT`。

安装器只在首次部署时生成令牌，保存到 `var/secrets/`（权限 `0600`）。
日常重启：

```bash
cd /home/clhh/minimax-h3-mcp
./restart.sh
```

- `./restart.sh --check`：只检查服务
- `./restart.sh --force`：中断未完成任务并强制重启

MCP：`http://101.34.55.200:8766/mcp`  
健康检查：`http://101.34.55.200:8766/health`（公开，无需 token）。只有模型、
工作流文件存在且 ComfyUI API 可连接时才返回 HTTP 200；依赖未启动时返回
HTTP 503，避免“文件存在但服务实际不可生成”的假健康状态。

`audio_enabled=false` 会从提交给 ComfyUI 的图中移除音频 VAE、音频解码和
`CreateVideo.audio`，用于后续与原始歌曲混音的无声片段。运行中的共享 ComfyUI
若未启用 `COMFYUI_ALLOW_INTERRUPT`，取消请求会等待该图结束后丢弃结果，不会
把仍占用 GPU 的任务错误标记为已经停止。

客户端导入名是 `minimax-h3-mcp`。把 `mcp-import.example.json` 里的
`REPLACE_WITH_H3_MCP_API_TOKEN` 换成服务器 `var/secrets/api-token`。
