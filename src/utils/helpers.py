import json
import re
from typing import Any, Callable, Sequence, Type, List
from pydantic import BaseModel

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)

def _try_parse_json_str(s: str) -> Any:
    """Parse raw or fenced JSON string into Python, or return None if not JSON."""
    if not isinstance(s, str):
        return None
    m = _CODE_FENCE_RE.match(s)
    if m:
        s = m.group(1)
    s = s.strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _coerce_to_schema(
    result: Any, schema: Type[BaseModel], logger: Any | None = None
) -> BaseModel:
    """Unwrap LangGraph outputs and coerce strings / fenced JSON into the Pydantic schema."""
    if isinstance(result, BaseModel):
        return result

    if isinstance(result, dict):
        try:
            return schema.model_validate(result)
        except Exception:
            pass

        if "structured_response" in result and result["structured_response"] is not None:
            sr = result["structured_response"]
            if isinstance(sr, dict) and "parsed" in sr and sr["parsed"] is not None:
                return _coerce_to_schema(sr["parsed"], schema, logger)
            if isinstance(sr, str):
                parsed = _try_parse_json_str(sr)
                if parsed is not None:
                    return _coerce_to_schema(parsed, schema, logger)
                try:
                    return schema.model_validate({
                        "weights": sr,
                        "claim": "coerced from string",
                        "rationale": "parsed from string payload in structured_response",
                        "confidence": 0.3,
                    })
                except Exception:
                    pass
            try:
                return _coerce_to_schema(sr, schema, logger)
            except Exception:
                pass

        for key in ("return_values", "structured_output", "output", "parsed"):
            if key in result and result[key] is not None:
                val = result[key]
                if isinstance(val, dict) and "parsed" in val and val["parsed"] is not None:
                    return _coerce_to_schema(val["parsed"], schema, logger)
                if isinstance(val, str):
                    parsed = _try_parse_json_str(val)
                    if parsed is not None:
                        return _coerce_to_schema(parsed, schema, logger)
                try:
                    return _coerce_to_schema(val, schema, logger)
                except Exception:
                    pass

    if isinstance(result, str):
        parsed = _try_parse_json_str(result)
        if parsed is not None:
            return _coerce_to_schema(parsed, schema, logger)

        try:
            model_fields = set(schema.model_fields.keys())
            if "weights" in model_fields and ":" in result and "," in result:
                items = [p.strip() for p in result.split(",") if p.strip()]
                weights: dict[str, float] = {}
                for it in items:
                    if ":" in it:
                        k, v = it.split(":", 1)
                        try:
                            weights[k.strip()] = float(v.strip())
                        except Exception:
                            pass
                if weights:
                    return schema.model_validate({
                        "weights": weights,
                        "claim": "coerced from weights string",
                        "rationale": "parsed 'SYM: value' pairs from string payload",
                        "confidence": 0.3,
                    })
        except Exception:
            pass

    # Could not parse; log helpful keys.
    if logger is not None:
        keys = list(result.keys()) if isinstance(result, dict) else []
        logger.error(
            "structured_output_extraction_failed",
            available_keys=keys,
            result_type=type(result).__name__,
        )

    raise TypeError("Unexpected agent return type; cannot coerce to schema")
