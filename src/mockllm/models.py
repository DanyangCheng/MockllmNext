import time
import uuid
from typing import Dict, List, Literal, Optional, Any, Union
from pydantic import BaseModel, Field

class FunctionCall(BaseModel):
    name: str
    arguments: str
    
    
class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall
    
# OpenAI Models
class OpenAIMessage(BaseModel):
    """OpenAI chat message model."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class OpenAIChatRequest(BaseModel):
    """OpenAI chat completion request model."""

    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=150)
    stream: Optional[bool] = Field(default=False)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class ChoiceDeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None

class ChoiceDeltaToolCall(BaseModel):
    index: int
    id: Optional[str] = None
    type: Optional[Literal["function"]] = "function"
    function: Optional[ChoiceDeltaFunctionCall] = None

class OpenAIDeltaMessage(BaseModel):
    """OpenAI streaming delta message model."""

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None


class OpenAIStreamChoice(BaseModel):
    """OpenAI streaming choice model."""

    delta: OpenAIDeltaMessage
    index: int = 0
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None


class OpenAIChatChoice(BaseModel):
    """OpenAI regular chat choice model."""

    message: OpenAIMessage
    index: int = 0
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"


class OpenAIChatResponse(BaseModel):
    """OpenAI chat completion response model."""

    id: str = Field(default_factory=lambda: f"mock-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[OpenAIChatChoice]
    usage: Dict[str, int]


class OpenAIStreamResponse(BaseModel):
    """OpenAI streaming response model."""

    id: str = Field(default_factory=lambda: f"mock-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[OpenAIStreamChoice]


# Anthropic Models
class AnthropicMessage(BaseModel):
    """Anthropic message model."""

    role: Literal["user", "assistant"]
    content: str


class AnthropicChatRequest(BaseModel):
    """Anthropic chat completion request model."""

    model: str
    max_tokens: Optional[int] = Field(default=1024)
    messages: List[AnthropicMessage]
    stream: Optional[bool] = Field(default=False)
    temperature: Optional[float] = Field(default=1.0)


class AnthropicChatResponse(BaseModel):
    """Anthropic chat completion response model."""

    id: str = Field(default_factory=lambda: f"mock-{uuid.uuid4()}")
    type: str = "message"
    role: str = "assistant"
    model: str
    content: List[Dict[str, str]]
    stop_reason: Optional[str] = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Dict[str, int]


class AnthropicStreamDelta(BaseModel):
    """Anthropic streaming delta model."""

    type: str = "content_block_delta"
    index: int = 0
    delta: Dict[str, str]


class AnthropicStreamResponse(BaseModel):
    """Anthropic streaming response model."""

    type: str = "message_delta"
    id: str = Field(default_factory=lambda: f"mock-{uuid.uuid4()}")
    delta: AnthropicStreamDelta
    usage: Optional[Dict[str, int]] = None


# For backward compatibility
Message = OpenAIMessage
ChatRequest = OpenAIChatRequest
DeltaMessage = OpenAIDeltaMessage
StreamChoice = OpenAIStreamChoice
ChatChoice = OpenAIChatChoice
ChatResponse = OpenAIChatResponse
StreamResponse = OpenAIStreamResponse
