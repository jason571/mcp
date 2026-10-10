from typing import Literal

from pydantic import BaseModel, Field, model_validator

TERMINAL = {"succeeded", "failed", "canceled"}


class MusicRequest(BaseModel):
    style: str = Field(min_length=1, max_length=2000)
    lyrics: str | None = Field(default=None, max_length=16000)
    lyrics_prompt: str | None = Field(default=None, max_length=4000)
    language: str = Field(default="Chinese", min_length=2, max_length=40)
    cot: Literal["full", "melody", "off"] = "full"
    abc: str | None = Field(default=None, max_length=20000)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    cfg_scale: float | None = Field(default=None, ge=0, le=20)

    @model_validator(mode="after")
    def validate_lyrics(self):
        if not self.style.strip():
            raise ValueError("style cannot be blank")
        if self.lyrics is not None and not self.lyrics.strip():
            raise ValueError("lyrics cannot be blank")
        if self.lyrics_prompt is not None and not self.lyrics_prompt.strip():
            raise ValueError("lyrics_prompt cannot be blank")
        if not self.lyrics and not self.lyrics_prompt:
            raise ValueError("Provide lyrics or lyrics_prompt for automatic lyrics")
        if self.abc is not None and not self.abc.strip():
            raise ValueError("abc cannot be blank")
        if self.abc and self.cot == "off":
            raise ValueError("abc needs cot=full or melody")
        return self


class LyricsRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    style: str = Field(default="", max_length=2000)
    language: str = Field(default="Chinese", min_length=2, max_length=40)

    @model_validator(mode="after")
    def validate_prompt(self):
        if not self.prompt.strip():
            raise ValueError("prompt cannot be blank")
        return self


class Job(BaseModel):
    id: str
    kind: Literal["music", "lyrics"]
    status: Literal["queued", "running", "cancel_requested", "succeeded", "failed", "canceled"]
    progress: int = 0
    stage: str = "queued"
    message: str = "Waiting in queue"
    request: dict
    result: dict | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    download_token: str
