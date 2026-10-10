from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="Z_IMAGE_", env_file=".env", extra="ignore")

    model_path: Path = Path("models/Z-Image-Turbo")
    data_dir: Path = Path("var")
    host: str = "127.0.0.1"
    port: int = Field(default=8766, ge=1, le=65535)
    public_url: str = "http://127.0.0.1:8766"
    api_token: SecretStr = SecretStr("")
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    cpu_offload: bool = False
    idle_unload_seconds: int = Field(default=120, ge=0)
    max_pixels: int = Field(default=2097152, ge=65536, le=4194304)
    max_pending_jobs: int = Field(default=16, ge=1, le=256)
    min_free_disk_mb: int = Field(default=512, ge=0)

    @model_validator(mode="after")
    def validate_settings(self):
        url = urlsplit(self.public_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("public_url must be an http(s) origin without credentials or a path")
        self.public_url = self.public_url.rstrip("/")
        if not self.device.startswith("cuda"):
            raise ValueError("This service requires a CUDA device, for example cuda:0")
        return self
