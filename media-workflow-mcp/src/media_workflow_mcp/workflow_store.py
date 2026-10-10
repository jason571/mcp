from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


WORKFLOW_ID = re.compile(r"^mv_[0-9a-f]{64}$")


class MusicVideoWorkflowStore:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "workflows"
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def load(self, workflow_id: str) -> dict[str, Any] | None:
        workflow_id = self.validate_id(workflow_id)
        source = self.root / f"{workflow_id}.json"
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"music-video workflow state could not be read: {exc}") from exc
        if not isinstance(value, dict) or value.get("workflow_id") != workflow_id:
            raise ValueError("stored music-video workflow state is invalid")
        return value

    def save(self, state: dict[str, Any]) -> None:
        workflow_id = self.validate_id(state.get("workflow_id"))
        payload = json.dumps(
            state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{workflow_id}.", suffix=".tmp", dir=self.root
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.root / f"{workflow_id}.json")
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            Path(temporary).unlink(missing_ok=True)
            raise

    @staticmethod
    def validate_id(value: object) -> str:
        if not isinstance(value, str) or not WORKFLOW_ID.fullmatch(value):
            raise ValueError("workflow_id must be mv_ followed by a lowercase SHA-256")
        return value
