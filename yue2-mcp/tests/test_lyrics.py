from threading import Event

import httpx
import pytest

from yue2_mcp.backend import GenerationCanceled, YuEBackend
from yue2_mcp.config import Settings
from yue2_mcp.models import LyricsRequest


def test_lyrics_provider_request_and_cancel(monkeypatch, tmp_path):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "[Verse 1]\n雨"}}]})

    transport = httpx.MockTransport(handle)
    real_client = httpx.Client
    monkeypatch.setattr(
        "yue2_mcp.backend.httpx.Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )
    backend = YuEBackend(Settings(_env_file=None, data_dir=tmp_path,
                                  lyrics_base_url="http://llm.test/v1",
                                  lyrics_model="local-text", lyrics_api_key="secret"))
    request = LyricsRequest(prompt="rain at night", style="soft jazz", language="Chinese")
    assert backend.lyrics(request, Event()) == "[Verse 1]\n雨"
    assert seen[0].url == "http://llm.test/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert "rain at night" in seen[0].content.decode()
    canceled = Event()
    canceled.set()
    with pytest.raises(GenerationCanceled):
        backend.lyrics(request, canceled)
    assert len(seen) == 1
