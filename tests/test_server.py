import json
from unittest.mock import mock_open, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

MOCK_YAML_CONTENT = """
defaults:
  unknown_response: "I don't know the answer to that."

patterns:
  - regex: "weather and time"
    type: "tool_call"

  - regex: "search and fetch"
    type: "tool_call"

  - regex: "weather|tool call"
    type: "tool_call"
    function:
      name: "get_weather_info"
      arguments: '{"location": "test", "days": 42}'
"""


def _mock_stat():
    st = MagicMock()
    st.st_mtime = 9999999999
    return st


with patch("builtins.open", mock_open(read_data=MOCK_YAML_CONTENT)), patch(
    "io.open", mock_open(read_data=MOCK_YAML_CONTENT)
), patch("os.path.exists", return_value=True), patch(
    "pathlib.Path.stat", return_value=_mock_stat()
):
    from mockllm.server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_responses_file():
    with patch("builtins.open", mock_open(read_data=MOCK_YAML_CONTENT)), patch(
        "io.open", mock_open(read_data=MOCK_YAML_CONTENT)
    ), patch("os.path.exists", return_value=True), patch(
        "pathlib.Path.stat", return_value=_mock_stat()
    ):
        yield



def test_openai_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "test message"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert "usage" in data


def test_anthropic_chat_completion():
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "test message"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert len(data["content"]) > 0
    assert "usage" in data


def test_openai_streaming():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "test message"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_anthropic_streaming():
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "test message"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


def test_invalid_request():
    response = client.post(
        "/v1/chat/completions", json={"model": "mock-llm", "messages": []}
    )
    assert response.status_code in [400, 500]


@pytest.fixture
def sample_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather_info",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "days": {"type": "integer"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]


def test_openai_tool_calling_standard(sample_tools):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "what's the weather like in beijing today?"}],
            "tools": sample_tools,
            "tool_choice": "auto"
        },
    )
    assert response.status_code == 200
    data = response.json()
    

    assert data["choices"][0]["finish_reason"] == "tool_calls"
    message = data["choices"][0]["message"]
    assert "tool_calls" in message
    

    tool_call = message["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather_info"
    

    args = json.loads(tool_call["function"]["arguments"])
    assert "location" in args
    assert "days" in args
    assert args["days"] == 42


def test_openai_tool_calling_streaming(sample_tools):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "Trigger streaming tool call"}],
            "tools": sample_tools,
            "stream": True
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    lines = [line for line in response.iter_lines()]
    data_chunks = [line for line in lines if line.startswith("data: ") and "[DONE]" not in line]
    
    assert len(data_chunks) >= 2 

    first_chunk = json.loads(data_chunks[0].replace("data: ", ""))
    first_delta = first_chunk["choices"][0]["delta"]
    assert first_delta["role"] == "assistant"
    assert "tool_calls" in first_delta
    assert first_delta["tool_calls"][0]["id"].startswith("call_")
    assert first_delta["tool_calls"][0]["function"]["name"] == "get_weather_info"

    last_chunk = json.loads(data_chunks[-1].replace("data: ", ""))
    assert last_chunk["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_agent_loop_break(sample_tools):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [
                {"role": "user", "content": "what's the weather like in beijing today?"},
                {
                    "role": "assistant", 
                    "content": None, 
                    "tool_calls": [{
                        "id": "call_123", 
                        "type": "function", 
                        "function": {"name": "get_weather_info", "arguments": "{}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_123", "name": "get_weather_info", "content": '{"weather": "sunny"}'}
            ],
            "tools": sample_tools
        },
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "content" in data["choices"][0]["message"]
    assert data["choices"][0]["message"]["content"] == "I don't know the answer to that."


def test_openai_multi_tool_schema():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                },
            },
        },
    ]
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "weather and time please"}],
            "tools": tools,
            "tool_choice": "auto",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    tool_calls = data["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert tool_calls[1]["function"]["name"] == "get_time"


def test_openai_multi_tool_streaming():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_page",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                },
            },
        },
    ]
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "search and fetch"}],
            "tools": tools,
            "stream": True,
        },
    )
    assert response.status_code == 200
    lines = [line for line in response.iter_lines()]
    data_lines = [l for l in lines if l.startswith("data: ") and "[DONE]" not in l]
    chunks = [json.loads(l.replace("data: ", "")) for l in data_lines]

    tool_names = set()
    for c in chunks:
        for choice in c.get("choices", []):
            delta = choice.get("delta", {})
            for tc in delta.get("tool_calls", []):
                name = tc.get("function", {}).get("name")
                if name:
                    tool_names.add(name)

    assert tool_names == {"search_web", "fetch_page"}


def test_openai_tool_choice_none(sample_tools):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": sample_tools,
            "tool_choice": "none",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in data["choices"][0]["message"]


def test_openai_streaming_text_content():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "test message"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    lines = [line for line in response.iter_lines()]
    data_lines = [l for l in lines if l.startswith("data: ") and "[DONE]" not in l]
    assert len(data_lines) >= 3  # role_header + content chunk(s) + stop

    chunks = [json.loads(l.replace("data: ", "")) for l in data_lines]
    roles = []
    contents = []
    for c in chunks:
        delta = c["choices"][0]["delta"]
        if delta.get("role"):
            roles.append(delta["role"])
        if delta.get("content"):
            contents.append(delta["content"])
    assert "assistant" in roles
    assert len(contents) > 0

    last_chunk = chunks[-1]
    assert last_chunk["choices"][0]["finish_reason"] == "stop"