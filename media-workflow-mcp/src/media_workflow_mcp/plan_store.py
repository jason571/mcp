from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


PLAN_DIGEST = re.compile(r"^[0-9a-f]{64}$")
DRAFT_ID = re.compile(r"^mv-draft-[0-9a-f]{24}$")
SEGMENT_ID = re.compile(r"^segment-[0-9]{3,}$")


class MusicVideoPlanStore:
    """Persist validated plans and return one exact segment at a time."""

    def __init__(self, data_dir: Path):
        self.root = data_dir / "plans"
        self.draft_root = data_dir / "plan-drafts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.draft_root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
            self.draft_root.chmod(0o700)
        except OSError:
            pass

    def put_draft(self, plan: dict[str, object]) -> dict[str, str]:
        payload = json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        draft_id = f"mv-draft-{digest[:24]}"
        self._atomic_write(self.draft_root / f"{draft_id}.json", payload)
        return {"draft_id": draft_id, "draft_digest": digest}

    def get_draft(self, draft_id: str, draft_digest: str) -> dict[str, object]:
        if not DRAFT_ID.fullmatch(draft_id):
            raise ValueError("draft_id is invalid")
        digest = self._digest(draft_digest)
        source = self.draft_root / f"{draft_id}.json"
        try:
            payload = source.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError("music-video plan draft was not found") from exc
        except OSError as exc:
            raise ValueError(f"music-video plan draft could not be read: {exc}") from exc
        actual = hashlib.sha256(payload).hexdigest()
        if actual != digest or draft_id != f"mv-draft-{actual[:24]}":
            raise ValueError("music-video plan draft digest does not match")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("stored music-video plan draft is invalid") from exc
        if not isinstance(value, dict):
            raise ValueError("stored music-video plan draft is invalid")
        return value

    def put(self, rendered_plan: dict[str, object]) -> dict[str, object]:
        digest = self._digest(rendered_plan.get("plan_digest"))
        payload = json.dumps(
            rendered_plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        target = self.root / f"{digest}.json"
        self._atomic_write(target, payload)

        segments = rendered_plan.get("segments")
        if not isinstance(segments, list):
            raise ValueError("validated plan has no segment list")
        return {
            "valid": True,
            "plan_id": rendered_plan.get("plan_id"),
            "plan_digest": digest,
            "master_duration_ms": rendered_plan.get("master_duration_ms"),
            "segment_count": len(segments),
            "segments": [self._segment_reference(item) for item in segments],
            "segment_tool": "get_music_video_plan_segment",
        }

    @staticmethod
    def _atomic_write(target: Path, payload: bytes) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent
        )
        try:
            os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, target)
        except BaseException:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def get_segment(self, plan_digest: str, segment_id: str) -> dict[str, object]:
        digest = self._digest(plan_digest)
        if not SEGMENT_ID.fullmatch(segment_id):
            raise ValueError("segment_id must match segment-NNN")
        source = self.root / f"{digest}.json"
        try:
            plan = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError("validated music-video plan was not found") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"validated music-video plan could not be read: {exc}") from exc

        segments = plan.get("segments")
        if not isinstance(segments, list):
            raise ValueError("stored music-video plan is invalid")
        segment = next(
            (
                item
                for item in segments
                if isinstance(item, dict) and item.get("segment_id") == segment_id
            ),
            None,
        )
        if segment is None:
            raise ValueError(f"segment {segment_id} was not found in the validated plan")
        return {
            "valid": True,
            "plan_id": plan.get("plan_id"),
            "plan_digest": digest,
            "segment_count": len(segments),
            "segment": segment,
        }

    def get(self, plan_digest: str) -> dict[str, object]:
        """Return the complete immutable validated plan to trusted service code."""
        digest = self._digest(plan_digest)
        source = self.root / f"{digest}.json"
        try:
            plan = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError("validated music-video plan was not found") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"validated music-video plan could not be read: {exc}"
            ) from exc
        if plan.get("plan_digest") != digest or not isinstance(plan.get("segments"), list):
            raise ValueError("stored music-video plan is invalid")
        return plan

    @staticmethod
    def _digest(value: object) -> str:
        if not isinstance(value, str) or not PLAN_DIGEST.fullmatch(value):
            raise ValueError("plan_digest must be a 64-character lowercase SHA-256")
        return value

    @staticmethod
    def _segment_reference(segment: object) -> dict[str, object]:
        if not isinstance(segment, dict):
            raise ValueError("validated plan contains an invalid segment")
        return {
            key: segment.get(key)
            for key in (
                "segment_id",
                "master_start_ms",
                "master_end_ms",
                "generation_duration_seconds",
                "storyboard_shot_count",
            )
        }
