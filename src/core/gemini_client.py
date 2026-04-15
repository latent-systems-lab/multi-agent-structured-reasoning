"""Gemini client utilities."""

from __future__ import annotations

import os
from typing import Any

from google.ai.generativelanguage_v1beta import GenerativeServiceClient


class _GeminiServiceFactory:
    @staticmethod
    def build_generative_service(api_key: str, **kwargs: Any) -> GenerativeServiceClient:
        client_options = dict(kwargs.pop("client_options", {}) or {})
        client_options.setdefault("api_key", api_key)
        return GenerativeServiceClient(client_options=client_options, **kwargs)


genaix = _GeminiServiceFactory()


def get_gemini_llm() -> Any:
    """Return a configured Gemini service client.

    Returns
    -------
    google.ai.generativelanguage_v1beta.GenerativeServiceClient
        Configured Gemini client ready for text generation.

    Raises
    ------
    RuntimeError
        If the ``GEMINI_API_KEY`` environment variable is not set.
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    return genaix.build_generative_service(api_key=api_key)

