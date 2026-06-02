import json
from unittest.mock import mock_open, patch

import pytest
from fastapi.testclient import TestClient

MOCK_YAML_CONTENT = """
responses:
  default: "I don't know the answer to that."
"""

with patch("builtins.open", mock_open(read_data=MOCK_YAML_CONTENT)), patch(
    "os.path.exists", return_value=True
), patch("mockllm.config.ResponseConfig.load_responses"):
    from mockllm.server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_responses_file():
    with patch("builtins.open", mock_open(read_data=MOCK_YAML_CONTENT)), patch(
        "os.path.exists", return_value=True
    ), patch("mockllm.config.ResponseConfig.load_responses"):
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