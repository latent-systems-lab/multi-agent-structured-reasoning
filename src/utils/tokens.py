"""Token and latency accounting helpers."""

from __future__ import annotations

from typing import Dict, Mapping, Any


def account_tokens(
    latency_ms: int,
    tokens_in: int,
    tokens_out: int,
    usage: Mapping[str, Any] | None = None,
    prompt: str | None = None,
    completion: str | None = None,
) -> Dict[str, int]:
    """Return accounting record for token usage and latency.

    Parameters
    ----------
    latency_ms:
        Observed end-to-end latency for the call.
    tokens_in / tokens_out:
        Existing token counts on the decision object.
    usage:
        Optional usage metadata from Gemini responses.  Known keys such as
        ``prompt_token_count`` and ``candidates_token_count`` are aggregated
        into the totals.
    prompt / completion:
        Optional input and output strings used for token counting when
        ``usage`` metadata is unavailable.
    """

    if usage:
        tokens_in += int(
            usage.get("prompt_token_count")
            or usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        tokens_out += int(
            usage.get("candidates_token_count")
            or usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
    else:
        if prompt or completion:
            try:
                import tiktoken

                encoding = tiktoken.get_encoding("cl100k_base")
                if prompt:
                    tokens_in += len(encoding.encode(prompt))
                if completion:
                    tokens_out += len(encoding.encode(completion))
            except Exception:
                # If the tokenizer is unavailable, fall back to the provided counts
                pass

    return {
        "latency_ms": latency_ms,
        "token_in": tokens_in,
        "token_out": tokens_out,
    }
