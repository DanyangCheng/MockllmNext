import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ..config import ResponseConfig
from ..models import (
    AnthropicChatRequest,
    AnthropicChatResponse,
)
from ..utils import count_tokens
from ._shared import ActiveToolCall, ToolCallRegistry, resolve_final_text
from .base import LLMProvider

logger = logging.getLogger(__name__)

_ANTHROPIC_TOOL_ID_PREFIX = "toolu_"


class AnthropicStreamChunkFactory:
    """Produces SSE-formatted chunks for the Anthropic streaming protocol."""

    @staticmethod
    def message_start(model: str) -> str:
        payload = {
            "type": "message_start",
            "message": {
                "id": f"msg_{id(model)}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def content_block_start(index: int, content_block: Dict[str, Any]) -> str:
        payload = {"type": "content_block_start", "index": index, "content_block": content_block}
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def content_block_delta_text(index: int, text: str) -> str:
        payload = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def content_block_delta_json(index: int, partial_json: str) -> str:
        payload = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        }
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def content_block_stop(index: int) -> str:
        payload = {"type": "content_block_stop", "index": index}
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def message_delta(stop_reason: str, usage: Dict[str, int]) -> str:
        payload = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage,
        }
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def message_stop() -> str:
        return 'data: {"type": "message_stop"}\n\n'

    DONE = "data: [DONE]\n\n"


def _extract_user_content(messages: list) -> Optional[str]:
    """Extract the text content from the last user message."""
    last_user = next(
        (m for m in reversed(messages) if m.role == "user"), None
    )
    if last_user is None:
        return None
    content = last_user.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return ""


def _extract_tool_result(messages: list) -> Optional[tuple]:
    """Extract tool_use_id and content from the last user message's tool_result block."""
    last_user = next(
        (m for m in reversed(messages) if m.role == "user"), None
    )
    if last_user is None:
        return None
    content = last_user.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return (block.get("tool_use_id"), block.get("content", ""))
    return None


class AnthropicProvider(LLMProvider):
    def __init__(self, response_config: ResponseConfig) -> None:
        self.response_config = response_config
        self._registry = ToolCallRegistry()
        self._tool_id_counter = 0

    def _next_tool_id(self) -> str:
        self._tool_id_counter += 1
        return f"{_ANTHROPIC_TOOL_ID_PREFIX}{self._tool_id_counter:04d}"

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def generate_stream_response(
        self, content: str, model: str
    ) -> AsyncGenerator[str, None]:
        """Generate Anthropic-style streaming response in SSE format."""
        yield AnthropicStreamChunkFactory.message_start(model)

        async for chunk in self.response_config.get_streaming_response_with_lag(
            content
        ):
            yield AnthropicStreamChunkFactory.content_block_delta_text(0, chunk)

        yield AnthropicStreamChunkFactory.message_delta(
            "end_turn",
            {"input_tokens": 0, "output_tokens": count_tokens(content, model)},
        )
        yield AnthropicStreamChunkFactory.message_stop()
        yield AnthropicStreamChunkFactory.DONE

    async def _stream_text(self, text: str, model: str) -> AsyncGenerator[str, None]:
        """Stream a plain text response."""
        yield AnthropicStreamChunkFactory.message_start(model)
        async for chunk in self.response_config.stream_raw_text_with_lag(text):
            yield AnthropicStreamChunkFactory.content_block_delta_text(0, chunk)
        yield AnthropicStreamChunkFactory.message_delta(
            "end_turn",
            {"input_tokens": 0, "output_tokens": count_tokens(text, model)},
        )
        yield AnthropicStreamChunkFactory.message_stop()
        yield AnthropicStreamChunkFactory.DONE

    async def _stream_tool_calls(
        self, tool_calls: List[Dict[str, Any]], model: str
    ) -> AsyncGenerator[str, None]:
        """Stream tool use response blocks."""
        yield AnthropicStreamChunkFactory.message_start(model)

        for i, tc in enumerate(tool_calls):
            content_block = {
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": {},
            }
            yield AnthropicStreamChunkFactory.content_block_start(i, content_block)
            async for fragment in self.response_config.stream_raw_text_with_lag(
                tc["arguments"]
            ):
                yield AnthropicStreamChunkFactory.content_block_delta_json(i, fragment)
            yield AnthropicStreamChunkFactory.content_block_stop(i)

        yield AnthropicStreamChunkFactory.message_delta(
            "tool_use",
            {"input_tokens": 0, "output_tokens": 0},
        )
        yield AnthropicStreamChunkFactory.message_stop()
        yield AnthropicStreamChunkFactory.DONE

    # ------------------------------------------------------------------
    # Tool-call generation
    # ------------------------------------------------------------------

    async def _build_tool_use_blocks(
        self,
        tools: List[Dict[str, Any]],
        matched_pattern: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build Anthropic tool_use content blocks.

        If a matched_pattern with type="tool_call" is provided, use its
        function definition. Otherwise generate mock calls for all tools.
        """
        if matched_pattern and matched_pattern.get("type") == "tool_call":
            func = matched_pattern.get("function", {})
            func_name = func.get("name", "mock_function")
            raw_args = func.get("arguments", "{}")
            overrides: Dict[str, Any] = (
                json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            )
            if not isinstance(overrides, dict):
                overrides = {}

            matched_tool = next(
                (
                    t for t in tools
                    if t.get("name") == func_name
                ),
                None,
            )
            properties = (
                matched_tool.get("input_schema", {}).get("properties", {})
                if matched_tool
                else {}
            )

            tc = await self._registry._build_tool_call(
                func_name, properties, overrides, matched_pattern
            )
            return [{
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }]

        # Generate for all tools
        blocks = []
        for tool in tools:
            name = tool.get("name", "mock_function")
            properties = tool.get("input_schema", {}).get("properties", {})
            tc = await self._registry._build_tool_call(
                name, properties, overrides=None, matched_pattern=matched_pattern,
            )
            blocks.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })
        return blocks

    # ------------------------------------------------------------------
    # Request routing
    # ------------------------------------------------------------------

    async def handle_chat_completion(
        self, request: AnthropicChatRequest
    ) -> Union[Dict[str, Any], StreamingResponse]:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages found in request")

        user_content = _extract_user_content(request.messages)
        if user_content is None:
            raise HTTPException(status_code=400, detail="No user message found in request")

        matched_pattern = self.response_config.match_pattern(user_content)

        # Check for tool_result in the last user message
        tool_result = _extract_tool_result(request.messages)
        is_tool_response = tool_result is not None
        tool_use_id, tool_result_content = tool_result if tool_result else (None, "")

        # ---- branch: emit tool calls ----
        trigger_tool = (
            not is_tool_response
            and (
                (matched_pattern and matched_pattern.get("type") == "tool_call")
                or bool(request.tools)
            )
        )

        if trigger_tool:
            tool_blocks = await self._build_tool_use_blocks(
                request.tools or [], matched_pattern
            )
            if request.stream:
                return StreamingResponse(
                    self._stream_tool_calls(tool_blocks, request.model),
                    media_type="text/event-stream",
                )

            content = [
                {
                    "type": "tool_use",
                    "id": tb["id"],
                    "name": tb["name"],
                    "input": json.loads(tb["arguments"]),
                }
                for tb in tool_blocks
            ]
            prompt_tokens = count_tokens(str(request.messages), request.model)
            return AnthropicChatResponse(
                model=request.model,
                content=content,
                stop_reason="tool_use",
                usage={
                    "input_tokens": prompt_tokens,
                    "output_tokens": 0,
                    "total_tokens": prompt_tokens,
                },
            ).model_dump(exclude_none=True)

        # ---- branch: handle tool result ----
        cached_call: Optional[ActiveToolCall] = None
        if is_tool_response and tool_use_id:
            cached_call = await self._registry.pop(tool_use_id)

        final_text = resolve_final_text(cached_call, matched_pattern, tool_result_content)

        # ---- stream or return plain text ----
        if request.stream:
            if final_text is not None:
                return StreamingResponse(
                    self._stream_text(final_text, request.model),
                    media_type="text/event-stream",
                )
            return StreamingResponse(
                self.generate_stream_response(user_content, request.model),
                media_type="text/event-stream",
            )

        response_content = (
            final_text
            if final_text is not None
            else await self.response_config.get_response_with_lag(user_content)
        )

        prompt_tokens = count_tokens(str(request.messages), request.model)
        completion_tokens = count_tokens(response_content, request.model)
        total_tokens = prompt_tokens + completion_tokens

        return AnthropicChatResponse(
            model=request.model,
            content=[{"type": "text", "text": response_content}],
            usage={
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        ).model_dump()
