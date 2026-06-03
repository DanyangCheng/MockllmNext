import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..models import FunctionCall, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class ActiveToolCall:
    """Transient state stored between a tool-call response and its result."""

    arguments: Dict[str, Any]
    matched_pattern: Optional[Dict[str, Any]] = None


class MockArgumentBuilder:
    """Builds mock arguments for a single function schema."""

    _DEFAULTS: Dict[str, Any] = {
        "string": lambda name: f"mock_{name}_value",
        "number": lambda _: 42,
        "integer": lambda _: 42,
        "boolean": lambda _: True,
        "array": lambda _: ["mock_item"],
    }

    @classmethod
    def build(
        cls,
        properties: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a dict of mock values for every property in *properties*.

        Values in *overrides* take priority; missing ones are synthesized
        from the property type.
        """
        result = dict(overrides or {})
        for prop_name, prop_schema in properties.items():
            if prop_name in result:
                continue
            prop_type = prop_schema.get("type", "string")
            factory = cls._DEFAULTS.get(prop_type, lambda _: {})
            result[prop_name] = factory(prop_name)
        return result


class ToolCallRegistry:
    """Async-safe registry for tracking active tool calls."""

    def __init__(self) -> None:
        self._store: Dict[str, ActiveToolCall] = {}
        self._lock = asyncio.Lock()

    async def register(self, tool_id: str, call: ActiveToolCall) -> None:
        async with self._lock:
            self._store[tool_id] = call

    async def pop(self, tool_id: str) -> Optional[ActiveToolCall]:
        async with self._lock:
            entry = self._store.pop(tool_id, None)
            if entry is not None:
                logger.info("Tool call %s consumed from registry.", tool_id)
            return entry

    @staticmethod
    def new_id() -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    async def _build_tool_call(
        self,
        func_name: str,
        properties: Dict[str, Any],
        overrides: Optional[Dict[str, Any]],
        matched_pattern: Optional[Dict[str, Any]],
    ) -> ToolCall:
        """Create one ToolCall and register it for later result lookup."""
        arguments = MockArgumentBuilder.build(properties, overrides)
        tool_id = self.new_id()
        call = ActiveToolCall(arguments, matched_pattern)
        await self.register(tool_id, call)
        return ToolCall(
            id=tool_id,
            type="function",
            function=FunctionCall(
                name=func_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )


def resolve_final_text(
    cached_call: Optional[ActiveToolCall],
    matched_pattern: Optional[Dict[str, Any]],
    tool_result_content: str,
) -> Optional[str]:
    """Determine the scripted reply text that follows a tool-result message.

    Looks up ``final_text`` from either the cached ActiveToolCall's
    matched_pattern (preferred) or the current matched_pattern, and
    substitutes ``{{result}}`` with the actual tool output.
    """
    active_pattern = (
        cached_call.matched_pattern
        if (cached_call and cached_call.matched_pattern)
        else (matched_pattern if matched_pattern and matched_pattern.get("type") == "tool_call" else None)
    )
    if active_pattern is None:
        return None

    final_text: Optional[str] = active_pattern.get("final_text")
    if final_text and tool_result_content:
        final_text = final_text.replace("{{result}}", tool_result_content)
    return final_text
