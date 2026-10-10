from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YUE2_", env_file=".env", extra="ignore")

    model_path: Path = Path("/home/clhh/.cache/modelscope/models/m-a-p--YuE2-3B/snapshots/master")
    vae_path: Path = Path("/home/clhh/.cache/modelscope/models/m-a-p--YuE2-Vae/snapshots/master")
    data_dir: Path = Path("var")
    host: str = "127.0.0.1"
    port: int = Field(default=8767, ge=1, le=65535)
    public_url: str = "http://127.0.0.1:8767"
    api_token: SecretStr = SecretStr("")
    device: str = "cuda:0"
    memory_budget_gib: float = Field(default=24, gt=2)
    quantization: Literal["none", "fp8"] = "none"
    offload_ar: bool = False
    idle_unload_seconds: int = Field(default=120, ge=0)
    max_pending_jobs: int = Field(default=16, ge=1, le=128)
    min_free_disk_mb: int = Field(default=1024, ge=0)
    lyrics_base_url: str = ""
    lyrics_model: str = ""
    lyrics_api_key: SecretStr = SecretStr("")

    @model_validator(mode="after")
    def validate_settings(self):
        url = urlsplit(self.public_url)
        if (url.scheme not in {"http", "https"} or not url.netloc or url.username
                or url.password or url.query or url.fragment or url.path not in {"", "/"}):
            raise ValueError("public_url must be an http(s) origin")
        self.public_url = self.public_url.rstrip("/")
        if not self.device.startswith("cuda"):
            raise ValueError("YuE2 needs a CUDA device")
        if bool(self.lyrics_base_url) != bool(self.lyrics_model):
            raise ValueError("lyrics_base_url and lyrics_model must be configured together")
        if self.lyrics_base_url:
            parsed = urlsplit(self.lyrics_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("lyrics_base_url must be http(s)")
        return self
