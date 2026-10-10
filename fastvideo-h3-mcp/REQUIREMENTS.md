# FastVideo H3 MCP 需求规格

版本：0.1  
目标服务器：`192.168.108.50`  
Base 模型目录：`/mnt/data/models/MiniMax-H3-r16`

## 1. 目标与边界

创建独立 MCP 服务，统一调用：

1. FastVideo FastH3 快速蒸馏版，用于低步数纯文本音视频生成。
2. FastVideo 原生 MiniMax H3 Base，仅用于首尾帧 FL2VA。

MCP 服务负责参数校验、模型路由、任务排队、资源准入、FastVideo 子进程生命周期、实时进度、
取消、结果持久化和安全下载。生产 Base 路径不得依赖 ComfyUI 进程或工作流。

## 2. 必须功能

### FR-01 生成控制

每个生成请求必须支持：

- 正面提示词 `prompt`；
- 负面约束 `negative_prompt`；
- 声音描述 `soundscape`；
- 是否保留音轨 `audio_enabled`；
- 时长 `duration_seconds`，范围 5–60 秒；
- 画幅比例只接受 `16:9` 横屏和 `9:16` 竖屏；
- 默认时长 15 秒，允许 5–60 秒；默认分辨率 `512p`，也支持显式选择 `640p`、`768p`
  （短边定义）；超过 15 秒时只允许 `512p`，`640p` 与
  `768p` 最长 15 秒；
- 随机种子 `seed`；
- 可选首帧和尾帧；
- `model_id=auto|fast|base`。

视频固定 24 fps。请求帧数必须对齐到 H3 支持的 `17n+5`，响应同时返回请求时长、帧数和
实际时长。宽高必须为 32 的倍数。

### FR-02 图生视频与模型路由

FastH3 4-step 蒸馏检查点只作为 T2AV 快速路径，不得接受首帧或尾帧。完整 Base FL2VA
必须支持：仅首帧、仅尾帧、首尾帧同时提供。任何带参考帧的请求必须路由到 Base。

路由只按生成模态决定：

- 没有参考帧的文生视频一律使用 FastH3，支持 `16:9`、`9:16`，分辨率遵守 FR-01 的
  时长—分辨率限制；
- 有首帧或尾帧时一律使用 FastVideo 原生 Base FL2VA；
- 分辨率和画幅不得成为回退 Base 的理由；Fast 不可用时返回明确错误，不回退到20步 Base；
- `model_id` 只作为兼容输入，任务持久化并返回实际解析出的 `fast` 或 `base`。

### FR-03 参考帧资产

生成接口接受客户端提供的完整 HTTP 地址（`http://` 或 `https://`）作为
`first_frame_url` / `last_frame_url`，MCP 不推导客户端地址，也不拼接附件路径。MCP 必须在
入队前下载 PNG、JPEG 或 WebP，默认单文件最大 20 MiB，并校验响应 MIME 与图片文件头；
不得接受其他 URL 协议、服务器文件路径或 HTTP 重定向。下载后复制到 MCP 自己的资产库，
只持久化内部 `asset_id` 和本地路径，不持久化可能包含临时签名的完整 URL。

首图、尾图 URL 必须在同一条生成调用中传递。`upload_reference_image` 和 MCP 本地
`first_frame_asset_id` / `last_frame_asset_id` 仅作为已有客户端的兼容接口保留。

### FR-04 音频

`audio_enabled=true` 保留 H3 联合生成的同步音轨。`audio_enabled=false` 允许底层模型仍完成
联合推理，但交付 MP4 必须不含音轨。当前实现用 `ffmpeg -an` 无损复制视频流。

### FR-05 正负提示词

H3 是联合音视频提示模型。`prompt_mode=simple` 时服务生成
`integrated_multimodal_description` 和 `overall_soundscape`；`prompt_mode=h3` 时直接使用调用方
的 H3 提示。

Base/FastH3 的当前推理不是传统 CFG 负向提示接口，因此 `negative_prompt` 必须明确解释为
写入 `negative_constraints` 的语义约束，不得对外宣称为独立 CFG negative conditioning。

### FR-06 实时进度

`generate_video` 必须在同一 MCP tool call 内发送标准 `notifications/progress`：

- `progress`：0–100，单调不下降；
- `total`：100；
- `message`：包含稳定阶段标识和可读说明；
- 长时间 GPU 阶段至少每 5 秒发送一次心跳；
- 完成、失败、取消都必须以持久化终态结束。

FastVideo Base 后端必须报告资源准入、模型加载、输入准备、生成、编码和释放阶段；
不能只用虚构定时器。FastH3 必须报告每次视频/音频联合去噪前向的开始与完成，并在长前向中
按实际完成的 Transformer block 细分进度，以及报告视频解码、音频解码和封装阶段；不可再细分
的计算期间在当前真实阶段发送实时耗时心跳，不得用纯计时百分比冒充模型采样完成度。

同时提供 detached 模式：提交立即返回 `job_id`，客户端可查询相同的
`status/progress/stage/message` 状态。

### FR-07 队列、取消与恢复

- 默认 GPU 并发为 1，可配置但不得绕过显存容量规划；
- 在启动 FastH3 或 Base 子进程前检测 GPU 已用显存与 Linux `MemAvailable`；外部 ComfyUI
  或其他 CUDA 任务占用超过阈值时进入 `waiting_for_resources`，不得并行加载模型；
- Base 4 步模式必须完整加载并应用对应的 FL2VA 蒸馏 LoRA；如果 pipeline 不支持 LoRA、
  adapter tensor 不成对、存在未映射层或应用层数不完整，必须失败，禁止静默执行裸 Base 4 步；
- 必须连续多次检测为空闲后才启动，以降低资源刚释放时的竞态；检测失败时应阻止启动；
- GPU Worker 同一时刻只允许驻留一个模型；连续同模型任务复用生成器，切换 FastH3/Base
  时先释放旧模型，空闲 60 秒后释放；取消、异常或外部 GPU 冲突立即释放；
- 普通 30 秒及以下任务允许驻留；超过该显存安全阈值的长任务必须保留解码前释放 DiT 的
  保护并在完成后退出，不得为了缓存模型重新引入 60 秒视频解码 OOM；
- 任务状态写入 SQLite；
- 服务重启时，将旧的 queued/running 任务明确标记为 interrupted/failed；
- 取消运行中任务时终止该任务自己的 FastVideo 进程组，不影响外部 ComfyUI；
- 非驻留子进程必须在成功、失败或取消后释放 CUDA 上下文和模型内存。

### FR-08 输出

- 工作流必须保存 MP4；
- 结果必须是非空文件；
- MCP 返回带时效签名的下载地址；
- 下载接口必须验证任务、文件名、输出根目录和签名/令牌；
- 文件缺失时返回明确的 410，而不是虚假完成。

## 3. MCP 工具契约

| 工具 | 行为 |
|---|---|
| `generate_video` | 长调用，标准 MCP 进度推送，完成后返回结果 |
| `submit_video_generation` | 后台提交并立即返回 `job_id` |
| `get_video_job` | 查询单任务及当前进度 |
| `get_video_result` | 完成后刷新下载链接 |
| `cancel_video_job` | 取消排队或执行中的任务 |
| `list_video_jobs` | 查询最近任务 |
| `upload_reference_image` | 兼容旧客户端的 Base64 参考帧上传 |
| `list_video_models` | 返回能力和强制约束 |
| `get_video_service_status` | 队列、Worker 和后端就绪状态 |

## 4. FastVideo Base 集成要求

- Base 通过 FastVideo `VideoGenerator` + `InputConfig(pil_image, last_image)` 执行；
- MCP 必须显式传递宽、高、帧数、24 fps、步数、种子以及首尾参考帧；
- Base 固定使用 Dense `TORCH_SDPA`，不得继承 FastH3 的 VSA 稀疏参数；
- 在 60 GiB 主机内存服务器上启用直接设备加载、文本编码器 FP8、前馈分块，并在去噪后释放
  DiT，再执行 VAE 解码；
- 生产使用 FastVideo Diffusers 格式 rank-16 AdaLN transformer，其他 Base 组件可以只读链接
  到官方模型；不得把 ComfyUI 专用量化 checkpoint 当作 FastVideo Diffusers 权重加载；
- Base FL2VA 必须加载匹配的 8-step 蒸馏 LoRA，并使用 8 NFE、视频 shift 12、
  音频 shift 3、LoRA alpha 8、strength 1.0；FastVideo 的 scheduler 参数应为 9 个含末端
  0 的网格点，从而执行 8 次模型计算；不得把裸 Base 直接降到 8 步；ComfyUI 融合
  `qkv_proj` 适配器必须在 FastVideo 加载前改写成 Diffusers `to_q/k/v`；
- 一次性 Base worker 不保留用于热切换 LoRA 的完整 CPU 权重副本；其他 FastVideo 调用
  仍保持框架默认行为；
- 提示词和路径只通过结构化 JSON 文件传给无 shell 的子进程，用户不得控制 runner 路径；
- `comfyui-api` 仅为显式启用的旧兼容后端，不属于生产依赖。

## 5. 安全与运维

- MCP 工具必须验证 Bearer Token；
- HTTP transport 启用 DNS rebinding protection 和显式 Host 白名单；
- API Token 与下载签名密钥必须独立生成且不进入仓库；
- 对外开放时应由 HTTPS 反向代理终止 TLS；
- 日志不得输出令牌、签名密钥、参考图 Base64 或包含签名参数的参考图 URL；
- 单任务默认超时 3600 秒；提示词、上传大小、列表数量均设上限。

## 6. 验收标准

1. mock 模式下 MCP 客户端收到多条单调进度，最终为 100/completed。
2. FastH3 真实生成一段 `16:9`、`768p`、5 秒、带音频的非空 MP4。
3. Base 使用首帧生成非空 MP4，画面开头与参考帧语义一致。
4. Base 使用首尾帧生成非空 MP4，首尾语义均符合参考图。
5. 生成 `9:16` 和 `16:9` 各一段，输出宽高及方向正确，并验证其他比例在入队前被拒绝。
6. `audio_enabled=false` 的结果经 `ffprobe` 确认无音频流；`true` 确认含音频流。
7. 生成期间客户端收到 FastVideo Base 实际阶段和心跳进度；断开长调用会触发取消请求。
8. 错误令牌不能调用工具或下载文件；过期签名不能下载。
9. 重启服务后历史任务可查询，执行中的旧任务被标记为中断而不是永远 running。
10. FastH3 真实完成一段 60 秒请求的视频，确认下发 1450 帧；客户端收到单调的 block 级
    联合音视频进度，输出经 `ffprobe` 核验约 60.417 秒并包含请求的音频流。
11. 未知 `resolution`，以及 `duration_seconds > 15` 时误传的 `640p|768p`，均在入队前
    自动修正为 `512p`；两个画幅与 Fast/Base 路径执行相同策略。15 秒的 `640p|768p`
    仍按请求值执行，任务响应中的 `resolution` 必须反映实际采用值。
12. 外部 ComfyUI 占用 GPU 时，新任务显示 `waiting_for_resources` 且不启动 FastVideo；
    外部资源释放并连续两次满足阈值后才启动。

模型目录存在、API 返回 `prompt_id`、任务到达 99% 均不是最终验收；必须取得并检查实际 MP4。
