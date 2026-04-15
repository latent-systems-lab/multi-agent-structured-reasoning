from utils.tokens import account_tokens
import tiktoken
import pytest


def test_account_tokens_with_usage():
    usage = {"prompt_token_count": 2, "candidates_token_count": 3}
    acc = account_tokens(100, 1, 1, usage=usage)
    assert acc["token_in"] == 3
    assert acc["token_out"] == 4


def test_account_tokens_without_usage():
    prompt = "hello"
    completion = "world"
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - network issues
        pytest.skip("tiktoken encoding unavailable")
    acc = account_tokens(100, 0, 0, prompt=prompt, completion=completion)
    assert acc["token_in"] == len(enc.encode(prompt))
    assert acc["token_out"] == len(enc.encode(completion))
