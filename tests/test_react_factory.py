import sys
import types
import pytest
from pydantic import BaseModel


def _stub_modules():
    sys.modules.setdefault("langgraph", types.ModuleType("langgraph"))
    prebuilt = types.ModuleType("prebuilt")
    prebuilt.create_react_agent = lambda *args, **kwargs: None
    sys.modules["langgraph.prebuilt"] = prebuilt
    sys.modules["langgraph"].prebuilt = prebuilt

    lg_google = types.ModuleType("langchain_google_genai")
    class _DummyLLM:
        pass
    lg_google.ChatGoogleGenerativeAI = _DummyLLM
    sys.modules["langchain_google_genai"] = lg_google

    msgs = types.ModuleType("messages")
    msgs.SystemMessage = object
    msgs.HumanMessage = object
    msgs.BaseMessage = object
    sys.modules["langchain_core.messages"] = msgs

    tools = types.ModuleType("tools")
    tools.BaseTool = object
    sys.modules["langchain_core.tools"] = tools

    callbacks = types.ModuleType("callbacks")
    class _DummyCB:
        pass
    callbacks.BaseCallbackHandler = _DummyCB
    sys.modules["langchain_core.callbacks"] = callbacks


def test_create_react_agent_rejects_dict_schema():
    _stub_modules()
    from core.react_factory import create_react_agent
    with pytest.raises(TypeError):
        create_react_agent("prompt", [], {"type": "object"})


def test_structured_output_extraction_variants():
    _stub_modules()
    from utils.logging import get_logger
    from core.react_factory import _coerce_to_schema

    class Demo(BaseModel):
        a: int

    logger = get_logger("test")

    inst = Demo(a=1)
    assert _coerce_to_schema(inst, Demo, logger) == inst

    envelope = {"structured_output": {"a": 2}}
    assert _coerce_to_schema(envelope, Demo, logger) == Demo(a=2)

    plain = {"a": 3}
    assert _coerce_to_schema(plain, Demo, logger) == Demo(a=3)
