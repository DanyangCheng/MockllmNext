import json
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ..config import ResponseConfig
from ..models import (
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    FunctionCall,
    OpenAIChatChoice,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIDeltaMessage,
    OpenAIMessage,
    OpenAIStreamChoice,
    OpenAIStreamResponse,
    ToolCall,
)
from ..utils import count_tokens
from .base import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActiveToolCall:
    """Transient state stored between a tool-call response and its result."""
    arguments: Dict[str, Any]
    matched_pattern: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helper: build mock tool-call arguments
# ---------------------------------------------------------------------------

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
        """
        Return a dict of mock values for every property in *properties*.
        Values present in *overrides* take priority; missing ones are
        synthesised from the property type.
        """
        result = dict(overrides or {})
        for prop_name, prop_schema in properties.items():
            if prop_name in result:
                continue
            prop_type = prop_schema.get("type", "string")
            factory = cls._DEFAULTS.get(prop_type, lambda _: {})
            result[prop_name] = factory(prop_name)
        return result


# ---------------------------------------------------------------------------
# Helper: tool-call registry (replaces the raw dict on the provider)
# ---------------------------------------------------------------------------

class ToolCallRegistry:
    """Thread-unsafe but sufficient for single-process mock servers."""

    def __init__(self) -> None:
        self._store: Dict[str, ActiveToolCall] = {}

    def register(self, tool_id: str, call: ActiveToolCall) -> None:
        self._store[tool_id] = call

    def pop(self, tool_id: str) -> Optional[ActiveToolCall]:
        entry = self._store.pop(tool_id, None)
        if entry is not None:
            logger.info("Tool call %s consumed from registry.", tool_id)
        return entry

    @staticmethod
    def new_id() -> str:
        return f"call_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Helper: streaming chunks
# ---------------------------------------------------------------------------

class StreamChunkFactory:
    """Produces SSE-formatted chunks for the OpenAI streaming protocol."""

    @staticmethod
    def _sse(payload: OpenAIStreamResponse) -> str:
        return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"

    @classmethod
    def role_header(cls, model: str) -> str:
        return cls._sse(
            OpenAIStreamResponse(
                model=model,
                choices=[OpenAIStreamChoice(delta=OpenAIDeltaMessage(role="assistant"))],
            )
        )

    @classmethod
    def content_chunk(cls, model: str, text: str) -> str:
        return cls._sse(
            OpenAIStreamResponse(
                model=model,
                choices=[OpenAIStreamChoice(delta=OpenAIDeltaMessage(content=text))],
            )
        )

    @classmethod
    def stop_chunk(cls, model: str) -> str:
        return cls._sse(
            OpenAIStreamResponse(
                model=model,
                choices=[OpenAIStreamChoice(delta=OpenAIDeltaMessage(), finish_reason="stop")],
            )
        )

    @classmethod
    def tool_header(cls, model: str, tool: ToolCall) -> str:
        return cls._sse(
            OpenAIStreamResponse(
                model=model,
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDeltaMessage(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id=tool.id,
                                    type="function",
                                    function=ChoiceDeltaFunctionCall(
                                        name=tool.function.name,
                                        arguments="",
                                    ),
                                )
                            ],
                        ),
                    )
                ],
            )
        )

    @classmethod
    def tool_args_chunk(cls, model: str, args_fragment: str) -> str:
        return cls._sse(
            OpenAIStreamResponse(
                model=model,
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDeltaMessage(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    function=ChoiceDeltaFunctionCall(arguments=args_fragment),
                                )
                            ]
                        ),
                    )
                ],
            )
        )

    @classmethod
    def tool_stop_chunk(cls, model: str) -> str:
        return cls._sse(
            OpenAIStreamResponse(
                model=model,
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDeltaMessage(),
                        finish_reason="tool_calls",
                    )
                ],
            )
        )

    DONE = "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Main provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    def __init__(self, response_config: ResponseConfig) -> None:
        self.response_config = response_config
        self._registry = ToolCallRegistry()

    # ------------------------------------------------------------------
    # Tool-call generation
    # ------------------------------------------------------------------

    def _build_tool_call(
        self,
        func_name: str,
        properties: Dict[str, Any],
        overrides: Optional[Dict[str, Any]],
        matched_pattern: Optional[Dict[str, Any]],
    ) -> ToolCall:
        """Create one ToolCall and register it for later result lookup."""
        arguments = MockArgumentBuilder.build(properties, overrides)
        tool_id = ToolCallRegistry.new_id()
        self._registry.register(tool_id, ActiveToolCall(arguments, matched_pattern))
        return ToolCall(
            id=tool_id,
            type="function",
            function=FunctionCall(
                name=func_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _tool_call_from_regex(
        self,
        tools: List[Dict[str, Any]],
        regex_tool_info: Dict[str, Any],
        matched_pattern: Optional[Dict[str, Any]],
    ) -> ToolCall:
        """Build a ToolCall when a regex pattern specified the function name/args."""
        func_name = regex_tool_info.get("name", "mock_function")
        raw_args = regex_tool_info.get("arguments", "{}")

        overrides: Dict[str, Any] = (
            json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        )
        if not isinstance(overrides, dict):
            overrides = {}

        # Attempt to find the matching schema so we can fill missing properties.
        matched_tool = next(
            (
                t for t in tools
                if t.get("type") == "function"
                and t.get("function", {}).get("name") == func_name
            ),
            None,
        )
        properties = (
            matched_tool.get("function", {}).get("parameters", {}).get("properties", {})
            if matched_tool
            else {}
        )
        return self._build_tool_call(func_name, properties, overrides, matched_pattern)

    def _tool_call_from_schema(
        self,
        tools: List[Dict[str, Any]],
        matched_pattern: Optional[Dict[str, Any]],
    ) -> Optional[ToolCall]:
        """Build a ToolCall from the first tool in the schema list."""
        if not tools:
            return None
        tool = tools[0]
        if tool.get("type") != "function":
            return None
        func_info = tool.get("function", {})
        properties = func_info.get("parameters", {}).get("properties", {})
        return self._build_tool_call(
            func_info.get("name", "mock_function"),
            properties,
            overrides=None,
            matched_pattern=matched_pattern,
        )

    def _generate_mock_tool_calls(
        self,
        tools: List[Dict[str, Any]],
        regex_tool_info: Optional[Dict[str, Any]] = None,
        matched_pattern: Optional[Dict[str, Any]] = None,
    ) -> List[ToolCall]:
        if regex_tool_info:
            return [self._tool_call_from_regex(tools, regex_tool_info, matched_pattern)]
        tool = self._tool_call_from_schema(tools, matched_pattern)
        return [tool] if tool else []

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def generate_stream_response(
        self,
        content: Optional[str],
        model: str,
        tool_calls: Optional[List[ToolCall]] = None,
        raw_text: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if tool_calls:
            yield StreamChunkFactory.tool_header(model, tool_calls[0])
            async for fragment in self.response_config.stream_raw_text_with_lag(
                tool_calls[0].function.arguments
            ):
                yield StreamChunkFactory.tool_args_chunk(model, fragment)
            yield StreamChunkFactory.tool_stop_chunk(model)
            yield StreamChunkFactory.DONE
            return

        # Plain-text streaming (raw_text bypasses the response_config templating)
        text_to_stream = raw_text if raw_text is not None else content or ""
        stream_fn = (
            self.response_config.stream_raw_text_with_lag
            if raw_text is not None
            else self.response_config.get_streaming_response_with_lag
        )

        yield StreamChunkFactory.role_header(model)
        async for chunk in stream_fn(text_to_stream):
            yield StreamChunkFactory.content_chunk(model, chunk)
        yield StreamChunkFactory.stop_chunk(model)
        yield StreamChunkFactory.DONE

    # ------------------------------------------------------------------
    # Request routing
    # ------------------------------------------------------------------

    def _resolve_final_text(
        self,
        cached_call: Optional[ActiveToolCall],
        matched_pattern: Optional[Dict[str, Any]],
        tool_result_content: str,
    ) -> Optional[str]:
        """
        Determine the scripted reply text (if any) that should follow a
        tool-result message, substituting {{result}} with the actual content.
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

    async def handle_chat_completion(
        self, request: OpenAIChatRequest
    ) -> Union[Dict[str, Any], StreamingResponse]:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages found in request")

        last_user_msg = next(
            (m for m in reversed(request.messages) if m.role == "user"), None
        )
        if last_user_msg is None:
            raise HTTPException(status_code=400, detail="No user message found in request")

        matched_pattern = (
            self.response_config.match_pattern(last_user_msg.content)
            if last_user_msg.content
            else None
        )
        last_msg = request.messages[-1]

        # ---- branch: emit a tool call ----
        is_tool_response = last_msg.role == "tool"
        trigger_tool = (
            not is_tool_response
            and (
                (matched_pattern and matched_pattern.get("type") == "tool_call")
                or bool(request.tools and request.tool_choice != "none")
            )
        )

        if trigger_tool:
            regex_tool_info = (
                matched_pattern.get("function", {})
                if matched_pattern and matched_pattern.get("type") == "tool_call"
                else None
            )
            tool_calls = self._generate_mock_tool_calls(
                request.tools or [], regex_tool_info, matched_pattern
            )
            if request.stream:
                return StreamingResponse(
                    self.generate_stream_response(None, request.model, tool_calls=tool_calls),
                    media_type="text/event-stream",
                )
            prompt_tokens = count_tokens(str(request.messages), request.model)
            return OpenAIChatResponse(
                model=request.model,
                choices=[
                    OpenAIChatChoice(
                        index=0,
                        message=OpenAIMessage(
                            role="assistant", content=None, tool_calls=tool_calls
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 45,
                    "total_tokens": prompt_tokens + 45,
                },
            ).model_dump(exclude_none=True)

        # ---- branch: handle tool result ----
        tool_result_content = ""
        cached_call: Optional[ActiveToolCall] = None

        if is_tool_response:
            tool_call_id = last_msg.tool_call_id
            tool_result_content = last_msg.content or ""
            if tool_call_id:
                cached_call = self._registry.pop(tool_call_id)

        final_text = self._resolve_final_text(cached_call, matched_pattern, tool_result_content)

        # ---- stream or return plain text ----
        if request.stream:
            return StreamingResponse(
                self.generate_stream_response(
                    content=None if final_text is not None else last_user_msg.content,
                    model=request.model,
                    raw_text=final_text,
                ),
                media_type="text/event-stream",
            )

        response_content = (
            final_text
            if final_text is not None
            else await self.response_config.get_response_with_lag(last_user_msg.content or "")
        )
        prompt_tokens = count_tokens(str(request.messages), request.model)
        completion_tokens = count_tokens(response_content, request.model)
        return OpenAIChatResponse(
            model=request.model,
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=response_content),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        ).model_dump(exclude_none=True)