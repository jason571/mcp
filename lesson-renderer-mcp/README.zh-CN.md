# Lesson Renderer MCP

[English](README.md) | 简体中文

带认证的 Streamable HTTP MCP 控制器与独立 Manim 渲染 Worker。控制器持久化草稿、计划、已有配音与任务；Worker 使用保存的输入渲染并交付完整视频和音频。配音由外部授权的 TTS 能力提供，服务本身不执行 TTS。

## 依赖与部署结构

- Python 3.12 与 `requirements.lock` 中的依赖。
- Dockerfile 中的系统库、字体、Manim 和 FFmpeg。
- 配套 [elementary-lesson-video Skill](https://github.com/jason571/skills/tree/main/elementary-lesson-video) 的渲染脚本。
- 可写持久化目录，以及控制器与 Worker 共享的任务目录。

在当前组件目录构建：

```bash
docker build -t lesson-renderer:0.1.0-manim0.20.1 .
```

镜像提供运行依赖，源码只读挂载到 `/opt/lesson-renderer/src`；控制器所需的完整 Skill 挂载到 `/opt/lesson-skill`。控制器执行 `python -m lesson_renderer.server`，Worker 执行 `python -m lesson_renderer.worker`。控制器的 `LESSON_DATA_DIR/jobs` 与 Worker 的 `LESSON_JOB_DIR`（默认 `/jobs`）必须指向同一存储，赋予 UID 10001 必要的写权限。输入保存后，Worker 可在禁网环境执行。

## 控制器配置

| 环境变量 | 用途 |
| --- | --- |
| `LESSON_TOKEN_FILE` | 必需的私有 Token 文件，至少 32 个字符；限制权限并只读挂载 |
| `LESSON_ASSET_BASE` | 必需的旧附件输入接口基址 |
| `LESSON_ARTIFACT_BASE` | 签名产物输入基址，默认使用附件基址 |
| `LESSON_PUBLIC_BASE` | 必需的外部可访问成品地址 |
| `LESSON_DATA_DIR` | 控制器持久化目录，默认 `/data` |
| `LESSON_SKILL_DIR` | 渲染资源目录，默认 `/opt/lesson-skill` |
| `LESSON_HOST` / `LESSON_PORT` | 监听地址，默认 `0.0.0.0` / `8793` |
| `LESSON_ALLOWED_HOSTS` | 请求 Host 白名单，按实际代理和域名配置 |
| `LESSON_ARTIFACT_TTL_SECONDS` | 文件链接有效期；`0` 表示无时间过期限制 |

不要把 `artifact-...` 改成旧的 `att_...` ID；使用 Artifact 输入时，提交 TTS 返回的原始音频 URL。草稿、校验与渲染操作按实时工具 Schema 调用。

测试位于 `tests/`。控制器/Schema 测试、实际 Worker 测试与最终视频可播放验收分别完成；HTTP 服务启动成功不代表已生成课件。
