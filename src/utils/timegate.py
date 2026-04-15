"""Ensure data is not from the future relative to ``as_of``."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")


def timegate(as_of_field: str = "as_of") -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator asserting returned data is not newer than ``as_of``."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            as_of_val: Any = kwargs.get(as_of_field)
            if as_of_val is None and args:
                as_of_val = args[-1]

            def _check(obj: Any, as_of: str) -> None:
                if isinstance(obj, dict):
                    for v in obj.values():
                        _check(v, as_of)
                elif isinstance(obj, list):
                    for item in obj:
                        _check(item, as_of)
                elif isinstance(obj, tuple) and obj and isinstance(obj[0], str):
                    if obj[0] > as_of:
                        raise ValueError(f"timegate violation: {obj[0]} > {as_of}")

            if isinstance(as_of_val, str):
                _check(result, as_of_val)
            return result

        return cast(Callable[P, R], wrapper)

    return decorator
