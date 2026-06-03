import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

YAML_CONTENT = """
responses:
  "exact match prompt": "exact match response"
  "stream this please": "streaming content for testing"

defaults:
  unknown_response: "default unknown response"

patterns:
  - regex: "sensitive|secret|password"
    type: "text"
    text: "Blocked for security"

  - regex: "get weather in (.*)"
    type: "tool_call"
    function:
      name: "get_weather"
      arguments: '{"location": "$1"}'
    final_text: "Weather in $1: {{result}}"

  - regex: "find (.*) in (.*)"
    type: "tool_call"
    function:
      name: "search_files"
      arguments: '{"pattern": "$1", "directory": "$2"}'
    final_text: "Found $1 in $2: {{result}}"

settings:
  lag_enabled: false
  lag_factor: 10
"""

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "directory": {"type": "string"},
                },
            },
        },
    },
]


@pytest.fixture
def yaml_file(tmp_path):
    p = tmp_path / "test_responses.yml"
    p.write_text(YAML_CONTENT)
    return str(p)


@pytest.fixture
def configured_client(yaml_file):
    from mockllm.config import ResponseConfig
    from mockllm.providers.anthropic import AnthropicProvider
    from mockllm.providers.openai import OpenAIProvider
    from mockllm.server import app

    config = ResponseConfig(yaml_file)
    oai = OpenAIProvider(config)
    ant = AnthropicProvider(config)

    with patch("mockllm.server.response_config", config), patch(
        "mockllm.server.openai_provider", oai
    ), patch("mockllm.server.anthropic_provider", ant):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Exact match / default responses
# ---------------------------------------------------------------------------


def test_exact_match_response(configured_client):
    resp = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "exact match prompt"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "exact match response"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_default_response_openai(configured_client):
    resp = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "some random query"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "default unknown response"


def test_default_response_anthropic(configured_client):
    resp = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "some random query"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"][0]["text"] == "default unknown response"


# ---------------------------------------------------------------------------
# Pattern matching — text type
# ---------------------------------------------------------------------------


def test_pattern_text_blocking_openai(configured_client):
    resp = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "my password is hunter2"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Blocked for security"


def test_pattern_text_blocking_anthropic(configured_client):
    resp = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "reveal the secret code"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"][0]["text"] == "Blocked for security"


# ---------------------------------------------------------------------------
# Pattern matching — tool_call with capture groups (OpenAI)
# ---------------------------------------------------------------------------


def test_openai_pattern_tool_call_single_capture(configured_client):
    resp = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "get weather in Beijing"}],
            "tools": SAMPLE_TOOLS,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    tc = data["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"

    args = json.loads(tc["function"]["arguments"])
    assert args["location"] == "Beijing"


def test_openai_pattern_tool_call_multi_capture(configured_client):
    resp = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "find hello.py in /home/user"}],
            "tools": SAMPLE_TOOLS,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    tc = data["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "search_files"

    args = json.loads(tc["function"]["arguments"])
    assert args["pattern"] == "hello.py"
    assert args["directory"] == "/home/user"


def test_openai_pattern_tool_call_streaming(configured_client):
    resp = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "get weather in Tokyo"}],
            "tools": SAMPLE_TOOLS,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    lines = [l for l in resp.iter_lines()]
    data_lines = [l for l in lines if l.startswith("data: ") and "[DONE]" not in l]

    first_chunk = json.loads(data_lines[0].replace("data: ", ""))
    name = first_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
    assert name == "get_weather"

    last_chunk = json.loads(data_lines[-1].replace("data: ", ""))
    assert last_chunk["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# Agent loop — final_text with {{result}} substitution (OpenAI)
# ---------------------------------------------------------------------------


def test_openai_agent_loop_with_final_text(configured_client):
    resp1 = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "get weather in Shanghai"}],
            "tools": SAMPLE_TOOLS,
        },
    )
    assert resp1.status_code == 200
    tc = resp1.json()["choices"][0]["message"]["tool_calls"][0]

    resp2 = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [
                {"role": "user", "content": "get weather in Shanghai"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": tc["function"],
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": "get_weather",
                    "content": '{"temp": 25, "condition": "sunny"}',
                },
            ],
            "tools": SAMPLE_TOOLS,
        },
    )
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "Weather in Shanghai:" in data["choices"][0]["message"]["content"]
    assert "sunny" in data["choices"][0]["message"]["content"]


def test_openai_agent_loop_with_final_text_streaming(configured_client):
    resp1 = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "get weather in London"}],
            "tools": SAMPLE_TOOLS,
        },
    )
    tc = resp1.json()["choices"][0]["message"]["tool_calls"][0]

    resp2 = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [
                {"role": "user", "content": "get weather in London"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": tc["function"],
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": "get_weather",
                    "content": '{"temp": 18}',
                },
            ],
            "tools": SAMPLE_TOOLS,
            "stream": True,
        },
    )
    assert resp2.status_code == 200
    lines = [l for l in resp2.iter_lines()]
    data_lines = [l for l in lines if l.startswith("data: ") and "[DONE]" not in l]

    chunks = [json.loads(l.replace("data: ", "")) for l in data_lines]
    all_content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks
    )
    assert "Weather in London:" in all_content


# ---------------------------------------------------------------------------
# Multi-tool call with regex pattern (OpenAI)
# ---------------------------------------------------------------------------


def test_openai_multi_tool_pattern_single(configured_client):
    """Regex tool_call always produces a single tool call for the matched function."""
    resp = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "get weather in Paris"}],
            "tools": SAMPLE_TOOLS,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    tool_calls = data["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Anthropic tool calling via schema
# ---------------------------------------------------------------------------

ANTHROPIC_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        },
    }
]


def test_anthropic_tool_calling_schema(configured_client):
    resp = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "what is the weather"}],
            "tools": ANTHROPIC_TOOLS,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["stop_reason"] == "tool_use"
    assert len(data["content"]) == 1
    block = data["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "get_weather"
    assert "location" in block["input"]
    assert "unit" in block["input"]


def test_anthropic_tool_calling_streaming(configured_client):
    resp = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "weather please"}],
            "tools": ANTHROPIC_TOOLS,
            "stream": True,
        },
    )
    assert resp.status_code == 200
    lines = [l for l in resp.iter_lines() if l]
    data_lines = [l for l in lines if l.startswith("data: ") and "[DONE]" not in l]

    types_found = set()
    for line in data_lines:
        chunk = json.loads(line.replace("data: ", ""))
        t = chunk.get("type", "")
        types_found.add(t)
        # Check tool_use name in content_block_start
        if t == "content_block_start":
            cb = chunk.get("content_block", {})
            if cb.get("type") == "tool_use":
                assert cb["name"] == "get_weather"

    assert "message_start" in types_found
    assert "content_block_start" in types_found
    assert "content_block_delta" in types_found
    assert "message_stop" in types_found


# ---------------------------------------------------------------------------
# Anthropic regex pattern tool call
# ---------------------------------------------------------------------------


def test_anthropic_pattern_tool_call(configured_client):
    resp = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "get weather in Tokyo"}],
            "tools": ANTHROPIC_TOOLS,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["stop_reason"] == "tool_use"
    block = data["content"][0]
    assert block["name"] == "get_weather"
    assert block["input"]["location"] == "Tokyo"


def test_anthropic_agent_loop_final_text(configured_client):
    resp1 = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "get weather in Kyoto"}],
            "tools": ANTHROPIC_TOOLS,
        },
    )
    assert resp1.status_code == 200
    block1 = resp1.json()["content"][0]
    tool_use_id = block1["id"]

    resp2 = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [
                {"role": "user", "content": "get weather in Kyoto"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": "get_weather",
                            "input": block1["input"],
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": '{"temp": 30, "condition": "clear"}',
                        }
                    ],
                },
            ],
            "tools": ANTHROPIC_TOOLS,
        },
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["stop_reason"] == "end_turn"
    text = data2["content"][0]["text"]
    assert "Weather in Kyoto:" in text
    assert "clear" in text


# ---------------------------------------------------------------------------
# Anthropic multi-tool schema
# ---------------------------------------------------------------------------

ANTHROPIC_MULTI_TOOLS = [
    {
        "name": "get_weather",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    },
    {
        "name": "get_time",
        "input_schema": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
        },
    },
]


def test_anthropic_multi_tool_schema(configured_client):
    resp = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "give me weather and time"}],
            "tools": ANTHROPIC_MULTI_TOOLS,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["stop_reason"] == "tool_use"
    assert len(data["content"]) == 2
    names = {b["name"] for b in data["content"]}
    assert names == {"get_weather", "get_time"}


# ---------------------------------------------------------------------------
# Hot-reload — config updates when YAML file changes
# ---------------------------------------------------------------------------


def test_hot_reload(configured_client, yaml_file):
    resp1 = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "exact match prompt"}],
        },
    )
    assert resp1.json()["choices"][0]["message"]["content"] == "exact match response"

    new_yaml = YAML_CONTENT.replace(
        "exact match response", "updated exact match response"
    )
    Path(yaml_file).write_text(new_yaml)

    resp2 = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "user", "content": "exact match prompt"}],
        },
    )
    assert resp2.json()["choices"][0]["message"]["content"] == "updated exact match response"


# ---------------------------------------------------------------------------
# Streaming text response — Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_streaming_text(configured_client):
    resp = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "stream this please"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    lines = [l for l in resp.iter_lines() if l]
    data_lines = [l for l in lines if l.startswith("data: ") and "[DONE]" not in l]

    chunks = [json.loads(l.replace("data: ", "")) for l in data_lines]
    types = {c.get("type") for c in chunks}
    assert "message_start" in types
    assert "content_block_delta" in types
    assert "message_delta" in types
    assert "message_stop" in types


# ---------------------------------------------------------------------------
# Anthropic plain-string content backward compatibility
# ---------------------------------------------------------------------------


def test_anthropic_plain_string_content(configured_client):
    resp = configured_client.post(
        "/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "exact match prompt"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"][0]["text"] == "exact match response"


# ---------------------------------------------------------------------------
# Invalid requests
# ---------------------------------------------------------------------------


def test_openai_no_user_message(configured_client):
    resp = configured_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-llm",
            "messages": [{"role": "system", "content": "be helpful"}],
        },
    )
    assert resp.status_code in [400, 500]


def test_openai_empty_messages(configured_client):
    resp = configured_client.post(
        "/v1/chat/completions",
        json={"model": "mock-llm", "messages": []},
    )
    assert resp.status_code in [400, 500]
