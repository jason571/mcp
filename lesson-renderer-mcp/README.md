# Lesson Renderer MCP

English | [简体中文](README.zh-CN.md)

An authenticated Streamable HTTP MCP controller and a separate Manim worker for narrated teaching animations. The controller persists drafts, plans, supplied speech audio and render jobs; the worker renders saved inputs and publishes the final video and audio. TTS is supplied by an authorized external capability, not performed by this service.

## Dependencies and layout

- Python 3.12 and dependencies in `requirements.lock`.
- System libraries, fonts, Manim and FFmpeg from `Dockerfile`.
- Renderer scripts from the companion [elementary-lesson-video Skill](https://github.com/jason571/skills/tree/main/elementary-lesson-video).
- A writable persistent data directory and a shared job directory between controller and worker.

Build from this directory:

```bash
docker build -t lesson-renderer:0.1.0-manim0.20.1 .
```

The image contains dependencies; source is mounted read-only at `/opt/lesson-renderer/src`. Mount the complete Skill at `/opt/lesson-skill` for the controller. Run the controller with `python -m lesson_renderer.server` and the worker with `python -m lesson_renderer.worker`. The controller's `LESSON_DATA_DIR/jobs` must be the same storage mounted at the worker's `LESSON_JOB_DIR` (default `/jobs`). Give UID 10001 the required write access. The worker can run without network access after inputs have been saved.

## Controller configuration

| Variable | Purpose |
| --- | --- |
| `LESSON_TOKEN_FILE` | Required private token file, at least 32 characters; restrict file permissions and mount read-only |
| `LESSON_ASSET_BASE` | Required base URL for legacy input attachments |
| `LESSON_ARTIFACT_BASE` | Base URL for signed artifact inputs; defaults to the asset base |
| `LESSON_PUBLIC_BASE` | Required externally reachable service URL for published results |
| `LESSON_DATA_DIR` | Persistent controller data, default `/data` |
| `LESSON_SKILL_DIR` | Renderer resource package, default `/opt/lesson-skill` |
| `LESSON_HOST` / `LESSON_PORT` | Listen address, defaults `0.0.0.0` / `8793` |
| `LESSON_ALLOWED_HOSTS` | Allowed request Host values; adjust for the actual proxy/hostname |
| `LESSON_ARTIFACT_TTL_SECONDS` | Artifact URL lifetime; `0` means no time expiry |

Never rename `artifact-...` handles into legacy `att_...` IDs. Submit the exact TTS-returned audio URL when using artifact inputs. Retrieve live tool schemas for draft, validation and render operations.

Tests are under `tests/`. Controller/schema tests, running worker tests and actual playable-video acceptance are separate checks; starting the HTTP server alone does not complete a lesson.
