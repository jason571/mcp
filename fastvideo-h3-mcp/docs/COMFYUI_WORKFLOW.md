# ComfyUI MiniMax H3 工作流接入

> 兼容后端说明：当前生产配置为 `MINIMAX_H3_BASE_BACKEND=fastvideo-native`，不调用
> ComfyUI。本页仅用于需要显式恢复旧 `comfyui-api` 后端的部署。

## 1. 为什么需要导出工作流

`/mnt/data/ComfyUI-aki-v3/ComfyUI/models` 是模型根目录。项目中的两份 API 工作流已经根据
服务器现有 `video_minimax_h3_r2v.json` 的采样、VAE、音频解码、保存链路，以及官方本地
T2V/I2V 模板构建：

部署文件位置：

```text
/home/clhh/fastvideo-h3-mcp/workflows/video_minimax_h3_t2va_api.json
/home/clhh/fastvideo-h3-mcp/workflows/video_minimax_h3_fl2va_api.json
```

## 2. 替换动态输入

打开导出的 API JSON，只把对应节点 `inputs` 中的动态值替换为下列精确占位符。不要改变
节点 ID、`class_type`、模型文件名或节点连接。

| 占位符 | JSON 类型 | 含义 |
|---|---:|---|
| `{{H3_PROMPT}}` | string | 已组合的 H3 正面、声音和负面约束，推荐使用 |
| `{{POSITIVE_PROMPT}}` | string | 原始正面提示词 |
| `{{NEGATIVE_PROMPT}}` | string | 原始负面约束文本 |
| `{{SOUNDSCAPE}}` | string | 原始声音描述 |
| `{{WIDTH}}` | number | 对齐后的宽度 |
| `{{HEIGHT}}` | number | 对齐后的高度 |
| `{{NUM_FRAMES}}` | number | 对齐后的总帧数 |
| `{{DURATION_SECONDS}}` | number | 请求时长，通常优先使用帧数 |
| `{{SEED}}` | number | 随机种子 |
| `{{STEPS}}` | number | Base 采样步数，服务器当前为 20 |
| `{{AUDIO_ENABLED}}` | boolean | 可供支持开关的节点使用；最终仍由服务校验/移除音轨 |
| `{{FIRST_FRAME}}` | string | ComfyUI input 子目录中的首帧文件名 |
| `{{LAST_FRAME}}` | string | ComfyUI input 子目录中的尾帧文件名 |

纯占位符会保留 JSON 类型。例如：

```json
{
  "42": {
    "inputs": {
      "text": "{{H3_PROMPT}}",
      "width": "{{WIDTH}}",
      "height": "{{HEIGHT}}",
      "length": "{{NUM_FRAMES}}",
      "seed": "{{SEED}}"
    },
    "class_type": "这里保留现有节点类型"
  }
}
```

渲染后 `width/height/length/seed` 会是数字，不是字符串。

FL2VA 的 Load Image 节点示意：

```json
{
  "10": {
    "inputs": {"image": "{{FIRST_FRAME}}"},
    "class_type": "LoadImage"
  },
  "11": {
    "inputs": {"image": "{{LAST_FRAME}}"},
    "class_type": "LoadImage"
  }
}
```

通用 FL2VA 模板包含两个 Load Image 节点。未提供首帧或尾帧时，MCP 在渲染阶段移除对应
Load Image 节点以及 `MiniMaxH3ImageToVideo` 输入连接，因此不会把空字符串提交给 ComfyUI。

## 3. 输出要求

保存节点必须真正写出 `.mp4`，并让文件出现在 ComfyUI history 的 output 信息中。MCP 会：

1. 用 `/upload/image` 上传参考帧；
2. 用 `/prompt` 提交渲染后的 JSON；
3. 从 `/ws` 接收节点和采样进度；
4. 从 `/history/{prompt_id}` 定位 MP4；
5. 用 `/view` 下载为任务目录中的 `result.mp4`。

如果工作流只输出 WebP、GIF、预览图或 WebM，任务会明确失败，不会把其他格式伪装成 MP4。

## 4. 运行前检查

```bash
test -d /mnt/data/ComfyUI-aki-v3/ComfyUI/models
test -f /mnt/data/ComfyUI-aki-v3/ComfyUI/user/default/workflows/video_minimax_h3_t2va_api.json
test -f /mnt/data/ComfyUI-aki-v3/ComfyUI/user/default/workflows/video_minimax_h3_fl2va_api.json
curl -fsS http://127.0.0.1:8189/system_stats
```

工作流模板首次跑通后，检查每个任务目录中的 `workflow.rendered.json`。它用于排查节点映射，
不包含 API Token，但包含提示词，因此任务目录权限不应对无关用户开放。
