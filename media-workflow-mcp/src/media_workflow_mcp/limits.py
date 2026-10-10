"""Product duration limits enforced before expensive media generation."""

from __future__ import annotations

import math


MAX_MUSIC_VIDEO_DURATION_SECONDS = 6 * 60
MAX_VIDEO_DURATION_SECONDS = 3 * 60


def reject_overlong_duration(
    duration_seconds: float,
    *,
    maximum_seconds: int,
    deliverable: str,
) -> None:
    """Reject an over-limit deliverable before a plan or GPU job is created."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return
    if duration_seconds <= maximum_seconds:
        return
    minutes = maximum_seconds // 60
    raise ValueError(
        f"{deliverable}最长支持 {minutes} 分钟（{maximum_seconds} 秒）；"
        f"当前时长为 {duration_seconds:.3f} 秒。未创建计划，也未提交视频生成。"
        "请缩短时长后重试。"
    )
