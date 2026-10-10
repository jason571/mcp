# 192.168.108.50 真实验证记录

> 历史记录：本页记录的是切换到 FastVideo 原生 Base 之前的 ComfyUI 验证结果，不能作为
> 当前 `fastvideo-native` 生产后端的验收证据。当前验收必须以新的服务 health、资源准入
> 状态和 FastVideo Base 实际 MP4 为准。

验证日期：2026-09-01  
服务器用户：`clhh`  
ComfyUI：0.32.0，`127.0.0.1:8189`  
GPU：NVIDIA RTX PRO 6000 Blackwell Workstation Edition

## 自动化验证

- 服务器 17 项测试通过；
- MCP SDK 2.1.1 Streamable HTTP 端到端进度回调通过；
- T2VA、仅首帧、仅尾帧、首尾帧工作流均通过服务器 `/object_info` 节点与连接类型验证。

## Base T2VA

- Job：`vid_20260901T043854Z_353dbaf3`
- 请求：5 秒、16:9、512p、带音频、`model_id=base`
- MCP 进度：0 → ComfyUI 节点 → 20/20 采样 → 解码 → 释放显存 → 100
- 文件：`var-smoke/outputs/vid_20260901T043854Z_353dbaf3/result.mp4`
- SHA-256：`b64ba11ebd743ed866b88633547057b02454cfd682e2805edd0a105d55eb0674`
- ffprobe：H.264 896×512、24 fps、5.166667 秒；AAC 5.167 秒；629,216 字节。

## Base 首尾帧 FL2VA

- Job：`vid_20260901T044112Z_fc98c5f5`
- 请求：5 秒、9:16、512p、首帧+尾帧、`audio_enabled=false`
- 两张图片均经 `upload_reference_image` 上传并由 ComfyUI 节点 15/16 执行；
- 文件：`var-smoke/outputs/vid_20260901T044112Z_fc98c5f5/result.mp4`
- SHA-256：`ab2138a442cfb038e643af5f79461fb1ddf12705936fc2e9248155b69a3a94df`
- ffprobe：仅 H.264 视频流，512×896、24 fps、5.166667 秒；324,338 字节；无音频流。
- 抽取首尾帧视觉核对：生成首帧与第一参考图主体/姿态/背景一致；生成尾帧与第二参考图一致。

## Base 19:9 历史验证（当前契约已移除）

这条记录证明 Base 后端曾完成过宽屏输出。产品画幅契约后来更正为仅支持
`16:9` 与 `9:16`，因此当前 MCP 会在入队前拒绝 `19:9`。

- Job：`vid_20260901T053927Z_ae5eb0d8`
- 请求：5 秒、19:9、512p、`model_id=base`、`audio_enabled=false`
- 正式 MCP `192.168.108.50:8765` 完成路由、20/20 采样、显存释放和无损移除音轨。
- 文件：`var/outputs/vid_20260901T053927Z_ae5eb0d8/result.mp4`
- SHA-256：`6ae8578babe3bf8799171a78858f7b361c1597d845a2530d6c905775d2dddb5d`
- ffprobe：仅 H.264 视频流，1088×512、24 fps、5.166667 秒；862,853 字节；无音频流。

## 显存释放

Base 推理前 ComfyUI 已加载模型时约占 41,292 MiB。任务完成、队列为空并调用 `/free` 后，
ComfyUI 进程约占 876 MiB；进程保留，可继续接收后续任务。

## FastH3 4-step T2VA

- Job：`vid_20260901T053118Z_70cb0fc0`
- 请求：5 秒、16:9、768p、带音频、`model_id=fast`
- MCP 进度：排队 → Worker → 模型加载/心跳 → FastH3 denoise → 解码/编码 → 100
- 任务创建至完成约 128 秒；FastVideo 报告生成阶段 66.22 秒、端到端推理 66.93 秒。
- 文件：`var-fast-smoke/outputs/vid_20260901T053118Z_70cb0fc0/result.mp4`
- SHA-256：`28feebc29b3dc4b1ea59e70f2c9bee831485f6b26539acd965aacb42a222c73a`
- ffprobe：H.264 1344×768、24 fps、5.166667 秒；AAC 5.152 秒；3,675,367 字节。
- 音频非静音：164,864 samples，整体 RMS -36.73 dB，峰值 -24.01 dB。
- 抽取首/中/尾帧视觉核对：均为雪地森林中的红狐，主体位置和步态连续变化，无文字或水印。

## FastH3 连续双任务吞吐验证

- 同一时刻提交两个 5 秒、16:9、768p、带音频的 `model_id=fast` 任务；生产配置为单 worker。
- Job 1：`vid_20260901T055055Z_9f13e1c8`，排队 0.008 秒，运行 130.763 秒；
  FastVideo 生成阶段 67.43 秒、端到端推理 68.11 秒。
- Job 2：`vid_20260901T055055Z_5765d99f`，排队 130.740 秒，运行 127.698 秒；
  FastVideo 生成阶段 66.67 秒、端到端推理 67.30 秒。
- 两条总墙钟 259.299 秒。第二条仅比第一条少 3.065 秒，表明当前逐任务启动进程并重新
  加载模型，无法从连续调用获得显著热缓存收益。
- 两份输出均为 H.264 1344×768、24 fps、5.166667 秒，并含 AAC 音轨。
- Job 1 SHA-256：`072f957f7fe270122851a5424b7dbad87f81f2ca57d16f675cdc518d2269d5c3`。
- Job 2 SHA-256：`6faf92b02ad4c325c455bbecfac92edebfba513ba74f1c38b2f09a9166248052`。

## FastH3 视频/音频联合进度验证

- Job：`vid_20260901T083109Z_90653ab3`，使用标准 MCP `generate_video` 长调用及
  `notifications/progress` 回调完成 5 秒、16:9、768p、带音频的真实生成。
- 模型加载后依次返回 43–49 的输入、条件和联合潜变量准备进度。
- 联合生成按真实 Transformer 前向完成事件返回：1/4=58、2/4=66、3/4=74、4/4=82；
  每次前向计算期间继续返回当前步骤及已运行秒数的心跳。
- 后续明确返回视频解码 84→87、音频解码 88→91、整理/编码/封装 92→98、校验 99、完成 100。
- 从接单到完成约 128.034 秒；不再出现整个生成阶段固定停留在 48% 的情况。
- 输出为 H.264 1344×768、24 fps、5.166667 秒，含 5.152 秒 AAC 音轨，文件 7,720,422 字节。
- SHA-256：`8ad7408d93fa41e3dc731d455de3b65e8e5f268f190759095e36275d02955ba9`。

## FastH3 可变分辨率与模态路由验证

服务端故意接收调用方传入的 `model_id=base`，但因两条请求均未提供参考帧，任务记录和
实际后端都归一化为 `model_id=fast`、`fastvideo-cli`，全程返回 4-step 联合进度，没有进入
ComfyUI 20-step Base。

- 横屏 Job：`vid_20260901T090232Z_0d90c05a`，`16:9 + 512p`；输出 H.264 896×512、
  24 fps、5.166667 秒，含 5.152 秒 AAC 音轨。FastVideo 生成阶段 28.50 秒、端到端推理
  28.96 秒，接单至成品约 90.67 秒；SHA-256：
  `c0edceb19a68c394ea2f1ba0982b76e92f5f0f25e91ee56a2973767e5895494d`。
- 竖屏 Job：`vid_20260901T090426Z_ae9a5a48`，`9:16 + 512p`；输出 H.264 512×896、
  24 fps、5.166667 秒，含 5.152 秒 AAC 音轨。FastVideo 生成阶段 27.91 秒、端到端推理
  28.37 秒，接单至成品约 89.57 秒；SHA-256：
  `e9a54cc8a87d334c02a8830b0467626f0748ad702a583e0029034fadee081b68`。

服务器只有 60 GiB 系统内存。上游默认加载曾在约 54.8 GiB RSS 时被内核 OOM killer
终止；仅把 transformer 直接送入 GPU 后，BF16 文本编码器仍令系统内存 OOM。最终验证配置为：

- 复用 ComfyUI Python 3.10 / torch 2.13.0+cu130 的隔离 venv，不修改 ComfyUI 环境；
- Dense 模型使用 `TORCH_SDPA`，不依赖 `fastvideo-kernel`；
- transformer safetensors 直接加载到 GPU；
- 文本编码器使用 FastVideo `AbsMaxFP8` 并保持 CPU offload；
- 修复 FSDP 替换参数对象后 `AbsMaxFP8.output_dtype` 元数据丢失。

以上兼容修改均保存在 `patches/` 并由 `scripts/apply-fastvideo-compat.sh` 幂等应用。
FP8 文本编码器路径完成了真实生成验证，但不代表与 BF16 文本编码器数值等价。

## FastH3 60 秒与细粒度 MCP 进度验证

第一次 60 秒、`16:9 + 512p` 请求正确下发 1450 帧，但未分块的 SwiGLU 前馈在第 1/4 次
联合前向尝试再分配 5.24 GiB 时 CUDA OOM。随后仅对逐 token 的前馈与残差门控启用
32768-token 分块，完整注意力保持不变，并启用 CUDA expandable segments。ComfyUI 与 LiveOS
进程均未停止。

- 最终 Job：`vid_20260901T094723Z_b12bbb3c`；FastH3 文生视频、请求 60 秒、16:9、
  896×512、24 fps、带音频，下发 1450 帧；
- 标准 MCP `generate_video` 长调用收到 282 条通知，0→100 单调不下降；日志中不存在
  `1/20`；
- 联合去噪为真实 4 次 transformer 前向；每次前向的 50 个 Transformer block 均收到
  完成事件，共 200 个 block 完成事件，进度区间 50→82 连续推进；
- 后续收到视频解码 84→87、音频解码 88→91、整理/编码/封装 92→98、校验 99、完成 100；
- FastVideo 生成阶段 928.60 秒；从 MCP 接单到完成约 993.565 秒；
- ffprobe：H.264 896×512、24 fps、60.416667 秒；AAC 60.416 秒；51,402,799 字节；
- SHA-256：`e42b7b0e3d71e2fbcde13496f02d4983a9c7f266a444bf914dfa4429d60a8907`。

60 秒请求按 H3 `17n+5` 时序约束对齐后为 1450 帧，因此交付文件实际时长约 60.417 秒。
客户端验证器会检查进度单调性、真实 Transformer block 事件、视频/音频解码与完成阶段，并
将结果下载 URL 脱敏后写入验证日志。

## 正式服务

## VSA-DataFree 切换与 RTX PRO 6000 实机验证

本节记录 2026-09-01 的 VSA DataFree 实机验证；2026-09-02 生产配置已按下一节切回
Dense DataFree，VSA 实现仍作为可选配置保留。

- 快照：ModelScope `snapshots/master`，不存在 `.incomplete`；共 68 个文件，32 个
  safetensors 文件头可读，索引引用的分片均存在；发布清单中的 17 个文件完成大小与
  SHA-256 校验；
- 执行合同：`VIDEO_SPARSE_ATTN_H3`、稀疏率 `0.9`、tile `64`、Triton VSA；RTX PRO
  6000 Blackwell Workstation Edition 为 SM120，不启用仅面向相应扩展平台的 `sm100a`；
- Triton VSA 内核在该 GPU 上完成独立 smoke test，输出有限值，并与同输入 SDPA 的最大
  绝对误差为 `0.000000`；
- 首次真实生成完成 4 次联合去噪，但因 VSA 压缩门增加 DiT 常驻显存，在视频 VAE 解码时
  出现 CUDA OOM；随后启用单任务 CLI 专用的去噪后 DiT 释放，日志确认解码前 CUDA 已分配
  显存由 77.42 GiB 降至 12.10 GiB；
- 最终 Job：`vid_20260901T134944Z_e9ae2c46`；请求 5 秒、16:9、768p、带音频；
- 任务创建至完成约 118.35 秒；FastVideo 报告生成阶段 55.46 秒、端到端推理 56.12 秒；
- 252 条 MCP 进度事件从 0 到 100 单调不下降，包含 4/4 联合去噪、200 个 Transformer
  block 完成事件、视频解码、音频解码、封装和完成阶段；
- 文件：`var/outputs/vid_20260901T134944Z_e9ae2c46/result.mp4`；
- ffprobe：H.264 1344×768、24 fps、5.166667 秒；AAC 5.152 秒；3,950,493 字节；
- SHA-256：`69dad336ad54ac8a60cdae1f4237c2279cba67ab661cd727f5537f3d64b60c01`；
- 任务结束后没有残留 FastVideo 推理子进程；整机显存为 97,887 MiB，总占用回落到
  1,811 MiB（其余两个既有 Python 进程分别约占 894 MiB 与 876 MiB）。

部署后 `/health` 返回 `status=ok`、`started=true`、`workers=1`、`queue_size=0`、
`active_jobs=0`；Fast 后端检查项（模型、Python、仓库、脚本和 VSA 内核源目录）全部通过。

## Dense-DataFree 生产回切与 5 秒验证

2026-09-02 将生产 FastH3 配置切回
`FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree` + `TORCH_SDPA`。VSA 模型、参数、
Triton 后端和兼容代码均未删除；`FastVideoCliBackend` 仅在
`FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN_H3` 时追加 VSA 参数并启用对应 kernel。

- Job：`vid_20260902T050015Z_70d5acd4`；请求 5 秒、16:9、768p、带音频；
- 任务创建至完成 164.27 秒；FastVideo 生成阶段 75.83 秒、端到端推理 76.49 秒；
- 日志确认 transformer 使用 `TORCH_SDPA`，完成 4/4 次联合去噪和 200 个 Transformer
  block 事件；解码前释放 DiT 时 CUDA allocated 由 61.80 GiB 降至 0.07 GiB；
- 输出：H.264 1344×768、24 fps、5.166667 秒；AAC 5.152 秒；5,893,679 字节；
- SHA-256：`6e1a511340b4370813e996477153fa9138a9d5d4bd02318d1da2862326573a33`；
- 258 条进度通知从 0 到 100 单调不下降，包含联合去噪、视频解码、音频解码和完成阶段；
- 任务结束后 `/health` 为 `started=true`、一个 worker、队列和活动任务均为 0，Fast 后端
  返回 Dense 模型 ID、Dense 快照路径且 ready。

同规格历史 VSA 任务在释放 DiT 前为 77.42 GiB allocated，释放后仍为 12.10 GiB；日志同时
确认 DiT 选择了 `VIDEO_SPARSE_ATTN_H3` 并进入 Triton kernel，并非退回 PyTorch
`masked_fill`。当前 H3 VSA 实现会构造 tile 级完整 score/mask，并缓存 Q/K/V/gate tile
buffer；在单卡 SM120 Triton 路径上，这些开销会抵消稀疏计算的显存收益。官方 VSA 快照的
发布合同仍是 4×B200、SP4/FSDP、`sm100a`，不能将其显存结论直接套用到当前单卡路径。

## VSA 独立显存策略部署（未做生成验证）

2026-09-02 保持生产 `Dense-DataFree + TORCH_SDPA` 不变，为可选 VSA 路径增加两项独立策略：

- `FASTVIDEO_VSA_CACHE_TILE_BUFFER=false`：H3 VSA metadata builder 不再跨层、跨去噪步持有
  padded QKV tile buffer；每次 Attention 使用独立临时 buffer，返回后可由引用计数释放；
- `FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS=32768`：仅在无梯度推理时按 token 分块
  计算 Attention gated residual，并原地写回 hidden states，避免全序列 gate 乘积临时量；
- MCP 仅在 `VIDEO_SPARSE_ATTN_H3` 后端下向子进程传递上述有效值；当前生产
  `TORCH_SDPA` 的实际子进程环境将两项都强制为 `0`，不改变 Dense 执行路径。

本地与服务器项目测试均为 33/33 通过；两份补丁在服务器 FastVideo 源码应用成功，目标文件
通过 `py_compile`。BF16 小张量对照确认分块的“先乘后加”结果与原全序列表达式逐元素一致。
模拟 VSA 子进程环境得到 cache `0`、残差分块 `32768`、Triton `1`；真实 Dense 子进程环境
得到 cache `0`、残差分块 `0`。服务重启后标准 MCP 状态调用和 `/health` 均确认
`started=true`、`workers=1`、队列与活动任务为 0，模型仍为 Dense-DataFree。

以上只证明配置隔离、补丁安装和服务部署正确；尚未切换生产模型，也未提交 VSA GPU 生成，
因此不能据此宣称 30 秒或 60 秒 VSA OOM 已被实机消除。

- MCP：`http://192.168.108.50:8765/mcp`
- systemd 用户服务：`fastvideo-h3-mcp.service`，已启用并运行；`Linger=yes`
- 生产鉴权调用通过；状态为 `started=true`、`workers=1`、Fast/Base 后端均 ready。
- 使用错误 Bearer token 调用 `get_video_service_status` 被拒绝，未返回服务状态。
- API token 与下载签名密钥仅保存在服务器 `var/secrets/`，权限均为 `0600`。

## 长视频显存保护

- `duration_seconds > 15` 时，服务端只接受 `resolution=512p`；`640p`、`768p` 最长 15 秒；
- 限制同时覆盖 `16:9`、`9:16`、Fast 文生视频和 Base 参考帧路径；
- 参数在任务持久化与入队前校验，拒绝请求不会启动模型或占用 GPU；
- MCP 客户端通过 `is_error=true` 收到明确的 512p 限制原因，而不是通用内部错误；
- 自动化测试覆盖 15 秒边界放行，以及 15.01 秒时两个画幅下的 `640p`、`768p` 拒绝。
