import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_INFERENCE_STEPS = 9
TERMINAL = {"succeeded", "failed", "canceled"}


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int = Field(default_factory=lambda: secrets.randbelow(2**63), ge=0, lt=2**63)
    steps: int = Field(default=DEFAULT_INFERENCE_STEPS, ge=2, le=20)

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value.strip()

    @field_validator("width", "height")
    @classmethod
    def aligned_dimension(cls, value: int) -> int:
        if value % 16:
            raise ValueError("width and height must be multiples of 16")
        return value


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "cancel_requested", "succeeded", "failed", "canceled"]
    request: GenerateRequest
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: float = 0
    stage: str = "queued"
    message: str = "Waiting for the GPU worker"
    download_token: str
    result: dict | None = None
    error: str | None = None
