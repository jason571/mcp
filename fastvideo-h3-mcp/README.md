# FastVideo H3 MCP

面向 `192.168.108.50` 的视频生成 MCP 服务。FastH3 蒸馏模型和 MiniMax H3 Base FL2VA
均由 MCP 以独立 FastVideo 子进程执行，不依赖 ComfyUI。服务把资源等待、模型加载、采样、
编码和完成状态持续返回给 MCP 客户端。

## 已实现

- `generate_video`：保持调用直到完成，并发送标准 MCP progress notification。
- `submit_video_generation`：后台提交任务，立即返回 `job_id`。
- `get_video_job` / `get_video_result` / `list_video_jobs` / `cancel_video_job`。
- `generate_video` / `submit_video_generation` 可直接从客户端给出的首帧、末帧 HTTP(S) URL
  下载 PNG、JPEG 或 WebP；不需要在 MCP 消息中传 Base64。
- `upload_reference_image`：保留兼容的 Base64 上传方式，返回 MCP 本地 `asset_id`。
- `list_video_models` / `get_video_service_status`。
- SQLite 持久化队列，服务重启后明确标记中断任务；限制 GPU 并发数。
- 单模型 GPU Worker 驻留：同模型连续任务复用已加载实例；切换 FastH3/Base 时先释放旧
  模型；空闲 60 秒、取消、异常或外部 GPU 冲突时释放。
- GPU/主机内存准入：外部 ComfyUI 或其他 CUDA 任务未释放资源时，任务进入
  `waiting_for_resources`，不会同时加载第二套大模型触发系统 OOM。
- MCP HTTP 入口和工具调用均强制 Bearer Token、Host 白名单和带时效签名的 MP4 下载地址。

模型自动路由规则：

| 请求 | 后端 |
|---|---|
| 任何纯文本请求（全部受支持比例和分辨率） | FastH3 快速蒸馏版 |
| 首帧或尾帧参考 | FastVideo MiniMax H3 Base FL2VA |

路由只由生成模态决定，不再由分辨率决定。文生视频即使调用端传入 `model_id=base`，服务也会
归一化为 FastH3；只有实际提供首帧或尾帧资产时才使用 Base。任务记录中的 `model_id` 保存
实际选择结果。

Base 使用 FastVideo Diffusers 格式的 rank-16 AdaLN 检查点，其他组件可只读链接到官方模型：

```text
/mnt/data/models/MiniMax-H3-r16
```

可由 FastVideo 自带转换器从官方 `transformer` 生成该目录的 rank-16 transformer；再把
`text_encoder`、`tokenizer`、`processor`、`vae`、`audio_vae`、两个 scheduler 和
`modular_model_index.json` 只读链接到官方模型。不要直接使用 ComfyUI 的
`int8_convrot`/`nvfp4_awq` 文件，它们不是 FastVideo Diffusers checkpoint 格式。

Base FL2VA 在该模型上加载 LightX2V 的 8-step Turbo LoRA
`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`，使用 8 NFE。ComfyUI
原生键是融合的 `blocks.*.attn.qkv_proj`；Base worker 在交给 FastVideo 之前把它改写成
Diffusers 的 `transformer_blocks.*.attn.to_q/k/v`，并丢掉 FastVideo r16 对不上的
AdaLN / token_refiner / MLP 张量。已经是 Diffusers `to_q` 键的 LoRA 会原样加载。
该 LoRA 针对约 544p 混合宽高比训练，调度器保持视频 shift 12、音频 shift 3，适用于
服务统一的 512p 输出。LoRA 使用 alpha 8、strength 1.0；FastVideo scheduler 接收 9 个
含末端 0 的 sigma 网格点，从而实际执行 8 次 transformer evaluation。
不能只把裸 Base 从 20 步改成 8 步；没有匹配的蒸馏 LoRA 时应判定后端未就绪。
MiniMax H3 pipeline 必须接入 FastVideo 的 `LoRAPipeline`，生产 worker 同时设置
`FASTVIDEO_REQUIRE_COMPLETE_LORA=1`。如果 LoRA tensor 没有全部映射到目标层，任务会直接失败，禁止静默退化为裸 Base 少步推理。
Base worker 不执行 LoRA 热切换，因此关闭 FastVideo 默认的完整 Base CPU 回滚副本，避免在
60 GiB RAM 主机上为适配器切换额外复制数十 GB 权重。同一 Base 配置的普通任务在 worker
内复用已合并 LoRA 的生成器；切换模型时销毁整个进程，不在同一进程内替换适配器。

## 服务器安装

要求 MCP Python 3.12+、FastVideo 自身 venv、`nvidia-smi` 和 `ffmpeg`。服务器当前部署目录为：

```bash
cd /home/clhh/fastvideo-h3-mcp
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
```

编辑 `.env`，重点确认：

```dotenv
FASTVIDEO_MCP_BACKEND=auto
FASTVIDEO_MCP_GPU_IDLE_TIMEOUT_SECONDS=60
FASTVIDEO_MCP_WARM_MAX_DURATION_SECONDS=30
FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS=15
FASTVIDEO_MCP_MAX_DURATION_SECONDS=60
MINIMAX_H3_BASE_BACKEND=fastvideo-native
MINIMAX_H3_PYTHON=/home/clhh/FastVideo/.venv/bin/python
MINIMAX_H3_MODEL_PATH=/mnt/data/models/MiniMax-H3-r16
FASTVIDEO_RESOURCE_ADMISSION_ENABLED=true
```

`fastvideo-native` 在独立 GPU Worker 中加载 Base，支持仅首帧、仅尾帧或首尾帧，并严格使用
MCP 计算出的宽、高、帧数和步数。Base 固定使用 Dense `TORCH_SDPA`，不会继承 FastH3 的
VSA 配置。普通 30 秒及以下任务完成后保留生成器，收到同模型任务时直接复用；空闲 60 秒
或切换模型时退出整个进程。超过 30 秒的任务继续在解码前释放 DiT 并按一次性 worker 退出，
这是 60 秒视频的 OOM 安全边界，不能用驻留策略覆盖。

生成两个互不相同的长随机值，分别设置
`FASTVIDEO_MCP_API_TOKEN` 和 `FASTVIDEO_MCP_ARTIFACT_SIGNING_KEY`。不要提交 `.env`。

旧的 `comfyui-api` 和 `minimax-h3-diffusers` 后端仍作为显式兼容选项保留，但不用于生产。
ComfyUI 兼容说明见 [docs/COMFYUI_WORKFLOW.md](docs/COMFYUI_WORKFLOW.md)。

FastH3 使用现有 ModelScope 快照，不移动模型：

```dotenv
FASTVIDEO_REPO=/home/clhh/FastVideo
FASTVIDEO_PYTHON=/home/clhh/FastVideo/.venv/bin/python
FASTVIDEO_SCRIPT=/home/clhh/FastVideo/examples/inference/basic/basic_minimax_h3_t2v.py
FASTVIDEO_MODEL_PATH=/home/clhh/.cache/modelscope/models/FastVideo--FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree/snapshots/master
FASTVIDEO_MODEL_ID=fasth3-4-step-preview-v1-dense-datafree
FASTVIDEO_NUM_GPUS=1
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
FASTVIDEO_SAGE_ATTENTION=auto
FASTVIDEO_VSA_SPARSITY=0.9
FASTVIDEO_VSA_TILE_SIZE=64
FASTVIDEO_VSA_KERNEL=triton
FASTVIDEO_VSA_CACHE_TILE_BUFFER=false
FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS=32768
FASTVIDEO_ENABLE_FA4=false
FASTVIDEO_DIRECT_DEVICE_LOAD=true
FASTVIDEO_TEXT_ENCODER_FP8=true
FASTVIDEO_MINIMAX_H3_RELEASE_DIT_AFTER_DENOISE=true
```

当前生产配置使用 Dense-DataFree + `TORCH_SDPA`。VSA 代码和配置入口继续保留；切换到
VSA-DataFree 时需要同时把模型路径/ID 改为 VSA 版本，并将后端设为
`VIDEO_SPARSE_ATTN_H3`。VSA 发布合同使用 `0.9` 稀疏率和 `64` tile；服务器的 RTX PRO
6000 Blackwell 可使用 Triton VSA 路径，`sm100a` 仅用于带对应内核扩展的平台。只有选择
VSA 后端时，MCP 才会传递 VSA tile 参数并启用 Triton 稀疏 kernel；Dense 后端不会进入
这些代码分支。`FASTVIDEO_VSA_CACHE_TILE_BUFFER=false` 让 VSA 的 padded QKV tile buffer
只存活于当前 Attention 调用，不再由 metadata builder 持有整个去噪循环；代价是每层会重新
分配 tile buffer。`FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS=32768` 在推理时按 token
分块、原地合并 Attention gated residual，避免在 VSA 全序列输出仍存活时再创建一份完整
gate 乘积。MCP 对 Dense 子进程会强制把两个 VSA 显存参数设为 `0`，所以生产 Dense 路径
保持原有行为。将 cache 参数设为 `true` 可恢复 builder 级复用；将残差分块 token 数设为
`0` 可关闭分块。`FASTVIDEO_ENABLE_FA4=false` 保证 Dense 注意力路径不要求 FA4。
`FASTVIDEO_SAGE_ATTENTION=auto` 只作用于 Dense FastH3：worker venv 里有 SageAttention、
且 MiniMax H3 把 `SAGE_ATTN` / `SAGE_ATTN_THREE` 列为支持后端时才启用；否则保持
`TORCH_SDPA`。它不会改写 `VIDEO_SPARSE_ATTN_H3`，也不会改 Base FL2VA 的 dense SDPA。

服务器只有 60 GiB 系统内存，而 FastH3 transformer 为数百亿参数。
先执行 `scripts/apply-fastvideo-compat.sh /home/clhh/FastVideo`，再启用
`FASTVIDEO_DIRECT_DEVICE_LOAD=true`，可让权重直接进入 96 GiB GPU，避免加载器
保留一份约 66 GB 的 CPU 中间副本。该开关默认关闭，未应用补丁时不会改变上游行为。
同一脚本还提供 `FASTVIDEO_TEXT_ENCODER_FP8=true` 所需的 CLI 兼容补丁，令约
40 GB 的 BF16 文本编码器以 FastVideo `AbsMaxFP8` 路径加载并继续使用 CPU
offload。这样 denoise 时不会让已用完的 GPU 同时常驻文本编码器，系统内存也能
容纳压缩后的权重。此模式用于单卡内存兼容，不应视为 BF16 文本编码器的数值等价证明。
VSA 检查点的压缩门增加了 DiT 常驻显存；单任务 CLI worker 在完成去噪后将已经用完的 DiT
转为 meta 并清空 CUDA cache，再进入视频 VAE 解码，从而不需要把约 70 GiB 权重复制回
只有 60 GiB 的系统内存。此开关只用于当前单卡、单任务、非 FSDP 执行方式。
兼容脚本还为通用 H3 推理脚本补齐 VSA tile 参数，并修复 FSDP CPU offload 替换参数对象后 `AbsMaxFP8` 的
`output_dtype` 元数据丢失问题；缺失时回退到当前输入激活 dtype。

启动：

```bash
set -a
. ./.env
set +a
.venv/bin/fastvideo-h3-mcp
```

MCP 地址为 `http://192.168.108.50:8765/mcp`，健康检查为
`http://192.168.108.50:8765/health`。生产运行可参考
[deploy/fastvideo-h3-mcp.service.example](deploy/fastvideo-h3-mcp.service.example)。

50 服务器的用户级常驻服务配置在 `deploy/production.env`，安装命令为：

```bash
./scripts/install-user-service.sh
```

日常重启在服务器项目根目录执行：

```bash
cd /home/clhh/fastvideo-h3-mcp
./restart.sh
```

脚本使用当前部署的用户级 systemd 服务，重启后自动等待 `/health` 恢复，成功时显示
进程 PID；失败时显示服务状态和最近日志。以 `clhh` 用户执行即可，无需 `sudo`；使用
`sudo ./restart.sh` 时会自动切回项目所属用户。

- `./restart.sh --check`：只检查当前服务与任务状态。
- `./restart.sh --force`：中断未完成任务并强制重启，适合任务卡住时使用。

默认检测到运行中或排队任务时不重启；健康接口不可用时会尝试恢复服务，未完成任务可能
中断。重启使用原生产配置和令牌，不重新安装模型。该脚本在服务器本机运行，不需要外网地址。

安装器只在首次部署时生成令牌，保存到 `var/secrets/` 并设置为 `0600`；不会把
令牌写进代码或 systemd unit。生产环境对 `/mcp` 的初始化、消息和会话请求都在
HTTP 入口校验 `Authorization: Bearer ...`，缺失或错误时直接返回 `401`；`/health`
保持公开。客户端通过 SSH 在服务器读取 `var/secrets/api-token`，不要把令牌提交到
仓库或发到聊天中。

支持导入 JSON 的应用端可直接使用 [mcp-import.example.json](mcp-import.example.json)。
导入前只需把 `REPLACE_WITH_FASTVIDEO_MCP_API_TOKEN` 替换为服务器
`var/secrets/api-token` 的内容。某些客户端把类型名写成 `streamable-http`；如导入器
不识别 `streamableHttp`，只改这一项即可，URL 和请求头保持不变。

## 客户端与实时进度

长调用必须使用 `generate_video` 并提供 progress callback。示例：

```bash
FASTVIDEO_MCP_API_TOKEN='服务端令牌' \
FASTVIDEO_MCP_URL='http://192.168.108.50:8765/mcp' \
python examples/progress_client.py
```

进度总量固定为 100，通知 message 使用 `[stage] 中文说明`，包括：

```text
queued -> routing -> waiting_for_resources/checking_resources
       -> loading_model -> preparing_inputs -> generating -> encoding -> completed
```

FastH3 路径会读取 FastVideo 进程中的真实联合去噪与解码事件：

```text
loading_model -> preparing_inputs -> conditioning -> preparing_latents
              -> joint_av_denoising 1/4 -> 2/4 -> 3/4 -> 4/4
              -> video_decoding -> audio_decoding -> encoding -> muxing -> completed
```

FastH3 的 4 次联合去噪会继续细分到每次前向实际执行的 50 个 Transformer block。长视频
逐 block 更新百分比；短视频每 5 个 block 更新一次以避免通知洪峰。不可再细分的计算期间，
服务继续发送实时递增的已运行秒数心跳，不使用纯计时方式伪造采样完成度。

不适合维持长连接的客户端使用 `submit_video_generation`，随后轮询
`get_video_job`；它返回同样的 `progress`、`stage` 和 `message`，但这不是 MCP 主动推送。

连续提交两个 FastH3 任务并统计排队、单任务和总耗时：

```bash
FASTVIDEO_MCP_API_TOKEN='服务端令牌' \
FASTVIDEO_MCP_URL='http://192.168.108.50:8765/mcp' \
python examples/two_video_benchmark.py
```

客户端直接把可由 MCP 访问的完整 HTTP(S) 地址作为 `first_frame_url`、`last_frame_url`
传入生成工具。MCP 不推导客户端地址，也不拼接附件路径；它在任务入队前下载、限制大小、
校验响应 MIME 与图片文件头，并复制进自己的资产库。首图和尾图必须放在同一条
`generate_video` 消息中；任一 URL 非空即表示参考图生成并自动路由到 Base 后端。没有对应
图片时字段传 `null` 或省略，不要为首图、尾图分别调用两次生成工具：

```json
{
  "name": "generate_video",
  "arguments": {
    "prompt": "Overall visual direction and continuity anchors with a concrete timeline.",
    "negative_prompt": "blur, flicker, watermark, subtitles",
    "soundscape": "A continuous synchronized audio plan following the timeline.",
    "prompt_mode": "simple",
    "aspect_ratio": "16:9",
    "resolution": "512p",
    "duration_seconds": 15,
    "audio_enabled": true,
    "model_id": "auto",
    "first_frame_url": "http://127.0.0.1:8787/v1/attachment-assets/att_<64位十六进制ID>",
    "last_frame_url": "http://127.0.0.1:8787/v1/attachment-assets/att_<64位十六进制ID>"
  }
}
```

当 MCP 与 ExoHarness 不在同一主机时，两项都改用 ExoHarness 配置的 HTTPS 公网域名，路径和
附件 ID 不变。完整随机 ID 是不可猜测的 bearer capability，不要写入公开日志。

只接受 `http://` 和 `https://`，不跟随重定向；默认最大图片大小为 20 MiB。为兼容已经上传到
MCP 资产库的图片，也可以使用 `first_frame_asset_id` / `last_frame_asset_id`，但同一帧的
`*_url` 与 `*_asset_id` 不能同时提供。

## 参数约束

- `duration_seconds`：生成工具 schema 与 `list_video_models` / `get_video_service_status` 都返回数值 `duration_seconds.default`（来自启动配置 `FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS`，默认 15 秒）。单次上限为 `FASTVIDEO_MCP_MAX_DURATION_SECONDS`（默认 60 秒，不能超过 60）。改配置后重启 MCP 生效。按 H3 时序约束对齐到 `17n+5` 帧，响应会返回实际时长。
- 超过上游原训练时长的长视频使用 token 分块执行逐 token 前馈与残差门控，完整注意力不切块；
  60 秒请求下发 1450 帧，响应中的实际时长为约 60.417 秒。
- `audio_enabled`：`true` 保留联合生成音轨；`false` 在结果阶段用 `ffmpeg` 删除音轨。
- `aspect_ratio`：仅支持 `16:9` 横屏或 `9:16` 竖屏；其他比例会在入队前拒绝。
- `resolution`：默认 `512p`；也可显式选择 `640p`、`768p`，均指短边；宽高对齐为 32 的倍数。
  未知值会在入队前自动修正为 `512p`。当 `duration_seconds > 15` 时，误传的 `640p`、`768p`
  也会自动降为 `512p`。该策略同时适用于 `16:9`、`9:16` 和 Fast/Base 路径，避免多片段工作流
  因后续单个片段的分辨率错误而丢弃前面已经完成的结果；任务响应中的 `resolution` 为实际采用值。
- `prompt`：正面提示词；`negative_prompt` 会编入 H3 的 `negative_constraints`，不是 CFG 负向条件。
- `prompt_mode=simple`：服务生成 H3 结构化提示；`prompt_mode=h3`：调用方直接提供 H3 提示。
- `model_id` 保留用于客户端兼容；实际路由由参考帧决定：纯文本为 `fast`，有首帧或尾帧为 `base`。

完整需求和验收口径见 [REQUIREMENTS.md](REQUIREMENTS.md)。

## 本地验证

`mock` 后端不生成真实视频，只用于验证 MCP 协议、任务状态和进度回调：

```bash
FASTVIDEO_MCP_BACKEND=mock .venv/bin/python -m unittest discover -s tests -v
```

真实生成完成的唯一判据是服务返回 `status=completed`，且下载得到非空 MP4；模型文件存在、
工作流入队或进度到 99% 都不算生成成功。

在 GPU 服务器上可先执行只读检查：

```bash
./scripts/verify-server.sh
```

它检查生产 Base 模型组件、FastVideo Python、GPU 探针和 ffmpeg；只有显式选择旧
`comfyui-api` 后端时才检查 ComfyUI 接口。脚本不会下载或修改模型。

服务器真实验证记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。
