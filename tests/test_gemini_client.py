import pytest

from core import gemini_client


class DummyClient:
    pass


def test_get_gemini_llm_uses_env(monkeypatch):
    """get_gemini_llm builds client with API key from env."""

    captured = {}

    def fake_build(api_key=None, **kwargs):
        captured["api_key"] = api_key
        return DummyClient()

    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setattr(gemini_client.genaix, "build_generative_service", fake_build)

    client = gemini_client.get_gemini_llm()

    assert isinstance(client, DummyClient)
    assert captured["api_key"] == "key"


def test_get_gemini_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        gemini_client.get_gemini_llm()

