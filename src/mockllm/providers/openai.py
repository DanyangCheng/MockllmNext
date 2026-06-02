from typing import Any, AsyncGenerator, Dict, Union, List, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import uuid
import json
from ..config import ResponseConfig
from ..models import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChatChoice,
    OpenAIMessage,
    OpenAIDeltaMessage,
    OpenAIStreamChoice,
    OpenAIStreamResponse,
    ToolCall,
    FunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaFunctionCall,
)
from ..utils import count_tokens
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, response_config: ResponseConfig):
        self.response_config = response_config
    def _generate_mock_tool_calls(self, tools: List[Dict[str, Any]]) -> List[ToolCall]:
        tool_calls = []
        if tools:
            tool = tools[0]
            if tool.get("type") == "function":
                func_info = tool.get("function", {})
                func_name = func_info.get("name", "mock_function")
                
                parameters = func_info.get("parameters", {})
                properties = parameters.get("properties", {})
                
                mock_arguments = {}
                for prop_name, prop_schema in properties.items():
                    prop_type = prop_schema.get("type", "string")
                    if prop_type == "string":
                        mock_arguments[prop_name] = f"mock_{prop_name}_value"
                    elif prop_type in ["number", "integer"]:
                        mock_arguments[prop_name] = 42
                    elif prop_type == "boolean":
                        mock_arguments[prop_name] = True
                    elif prop_type == "array":
                        mock_arguments[prop_name] = ["mock_item"]
                    else:
                        mock_arguments[prop_name] = {}

                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",  
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=json.dumps(mock_arguments, ensure_ascii=False)
                        )
                    )
                )
        return tool_calls
    
    async def generate_stream_response(
        self, content: Optional[str], model: str, tool_calls: Optional[List[ToolCall]] = None
    ) -> AsyncGenerator[str, None]:
        
        if tool_calls:
            first_tool = tool_calls[0]
            first_chunk = OpenAIStreamResponse(
                model=model,
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDeltaMessage(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id=first_tool.id,
                                    type="function",
                                    function=ChoiceDeltaFunctionCall(
                                        name=first_tool.function.name,
                                        arguments=""
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
            yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

            args_str = first_tool.function.arguments
            async for arg_chunk in self.response_config.get_streaming_response_with_lag(args_str):
                chunk_response = OpenAIStreamResponse(
                    model=model,
                    choices=[
                        OpenAIStreamChoice(
                            index=0,
                            delta=OpenAIDeltaMessage(
                                tool_calls=[
                                    ChoiceDeltaToolCall(
                                        index=0,
                                        function=ChoiceDeltaFunctionCall(arguments=arg_chunk)
                                    )
                                ]
                            )
                        )
                    ]
                )
                yield f"data: {chunk_response.model_dump_json(exclude_none=True)}\n\n"

            final_chunk = OpenAIStreamResponse(
                model=model,
                choices=[
                    OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIDeltaMessage(),
                        finish_reason="tool_calls"
                    )
                ]
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

        else:
            first_chunk = OpenAIStreamResponse(
                model=model,
                choices=[OpenAIStreamChoice(delta=OpenAIDeltaMessage(role="assistant"))],
            )
            yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

            async for chunk in self.response_config.get_streaming_response_with_lag(content or ""):
                chunk_response = OpenAIStreamResponse(
                    model=model,
                    choices=[OpenAIStreamChoice(delta=OpenAIDeltaMessage(content=chunk))],
                )
                yield f"data: {chunk_response.model_dump_json(exclude_none=True)}\n\n"

            final_chunk = OpenAIStreamResponse(
                model=model,
                choices=[OpenAIStreamChoice(delta=OpenAIDeltaMessage(), finish_reason="stop")],
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

    async def handle_chat_completion(
        self, request: OpenAIChatRequest
    ) -> Union[Dict[str, Any], StreamingResponse]:
        if not request.messages:
            raise HTTPException(
                status_code=400, detail="No messages found in request"
            )
        last_message = next(
            (msg for msg in reversed(request.messages) if msg.role == "user"), request.messages[-1]
        )


        is_tool_triggered = bool(
            request.tools 
            and request.tool_choice != "none" 
            and request.messages[-1].role != "tool"
        )

        if is_tool_triggered and request.tools:
            tool_calls = self._generate_mock_tool_calls(request.tools)
            
            if request.stream:
                return StreamingResponse(
                    self.generate_stream_response(None, request.model, tool_calls=tool_calls),
                    media_type="text/event-stream",
                )
            
            prompt_tokens = count_tokens(str(request.messages), request.model)
            completion_tokens = 45
            
            return OpenAIChatResponse(
                model=request.model,
                choices=[
                    OpenAIChatChoice(
                        index=0,
                        message=OpenAIMessage(role="assistant", content=None, tool_calls=tool_calls),
                        finish_reason="tool_calls",
                    )
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            ).model_dump(exclude_none=True)

        if request.stream:
            return StreamingResponse(
                self.generate_stream_response(last_message.content or "", request.model),
                media_type="text/event-stream",
            )

        response_content = await self.response_config.get_response_with_lag(
            last_message.content or ""
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