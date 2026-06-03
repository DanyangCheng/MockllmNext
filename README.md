# Mock LLM Next

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![mockllm-logo](assets/logo.png)

**Mock LLM Next** is an enhanced LLM simulator fork of the original [stacklok/mockllm](https://github.com/stacklok/mockllm) project. It perfectly mimics OpenAI and Anthropic API formats using predefined responses and regular expression routing from a YAML configuration file instead of calling an actual upstream large language model.

This next-generation fork is specifically tailored for deterministic testing, interactive demos, and local development/debugging of complex LLM Agents and Multi-Agent workflows (e.g., LangGraph, LangChain) by providing advanced tool-calling simulation and stateful session tracking.

## Features

- **OpenAI and Anthropic compatible API endpoints**
- **Streaming support** (character-by-character response streaming)
- **OpenAI Tool Calling (Function Calling)** support for both standard and text/event-stream chunks.
- **Advanced Regular Expression Patterns Engine** for dynamic tool/text interception and user intent capturing.
- **Stateful Multi-Turn Agent Loop Tracking** to store, simulate, and gracefully terminate local tool executions without endless loops.
- **Configurable responses via YAML file** with hot-reloading support.
- **Mock token counting**.

## Configuration

### Response Configuration

Responses are configured in `responses.yml`. The file has four main sections:

1. `responses`: Maps precise input prompts to predefined responses.
2. `patterns`: Advanced regular expression matching rules for routing to text guardrails or complex tool calling interactions.
3. `defaults`: Contains default configurations like the unknown response message.
4. `settings`: Contains server behavior settings like network lag simulation.

Example `responses.yml`:

```yaml

responses:
  "write a python function to calculate factorial": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"
  "what colour is the sky?": "The sky is purple except on Tuesday when it is  hue green."

# Advanced Pattern Matching & Tool Calling Engine
patterns:
  # Route 1: Force a Tool Call & capture dynamic arguments
  - regex: "^List the files in (.*)$"
    type: "tool_call"
    function:
      name: "list_dir"
      arguments: '{"path": "$1", "recursive": false}'
    final_text: "Successfully invoked list_dir on directory $1, execution result: {{result}}"

  # Route 2: Short-circuit user request with a static text guardrail
  - regex: "password|key|token"
    type: "text"
    text: "Alert: Sensitive content detected. Mockllm declined to respond due to privacy concerns."

defaults:
  unknown_response: "I don't know the answer to that. This is a mock response."

settings:
  lag_enabled: true
  lag_factor: 10  # Higher values = faster responses (10 = fast, 1 = slow)
```

Tool Calling & Regex Patterns Deep Dive

1. Client Requirement (OpenAI API Compliance)
    - To enjoy automatic schema matching and dynamic parameter backfilling, the LLM client (e.g., LangChain or LangGraph) MUST register the tool descriptions inside the tools field of the incoming request payload. Mockllm will automatically search the tools array by the function name, inspect the required properties' data types, and patch any missing parameters.

2. Multi-Turn Agent Loop Lifecycle (final_text)
    - When type is set to "tool_call", Mockllm coordinates with your local Agent workflow in a sandbox environment:  

    - Turn 1 (Invocation): The user prompt matches a regex pattern. Mockllm replies to the Agent framework with a structured tool call payload and finish_reason="tool_calls". Behind the scenes, the context is safely cached using a unique tool call ID.  

    - Local Execution: Your Agent catches the intent and executes the corresponding codebase function locally.

    - Turn 2 (Completion & Destruct): The Agent appends a message with role="tool" containing the outcome and submits it back to Mockllm. Mockllm captures this, pops/destroys the cached session to avoid memory leaks, breaks out of the execution loop, and renders the final_text.

3. Dynamic Placeholders
    - $1, $2, ...: Replaced dynamically by regex capture groups parsed from the original user query.

    - {{result}}: Replaced dynamically by the real output payload submitted back from your Agent's local tool node.

## Hot Reloading

The server automatically detects changes to responses.yml and reloads the configuration without restarting the server.

## Installation

From Source
Clone the repository:

```Bash
git clone [https://github.com/DanyangCheng/mockllm.git](https://github.com/DanyangCheng/mockllm.git)
cd mockllm
pip install -e .
```

## Usage

CLI Commands
MockLLM provides a command-line interface for managing the server and validating configurations:

```Bash
# Show available commands and options
mockllm --help

# Show version
mockllm --version

# Start the server with default settings
mockllm start

# Start with custom responses file
mockllm start --responses custom_responses.yml

# Start with custom host and port
mockllm start --host localhost --port 3000

# Validate a responses file
mockllm validate responses.yml
```

## Quick Start

Set up the responses.yml:

```yaml

Validate your responses file (optional):

```Bash
mockllm validate custom_responses.yml
```

Start the server:

```Bash
mockllm start --responses responses.yml
The server will start on http://localhost:8000 by default.
```

## API Endpoints

### OpenAI Format

Regular Request:

```Bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mock-llm",
    "messages": [
      {"role": "user", "content": "what colour is the sky?"}
    ]
  }'
```

Streaming Request:

```Bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mock-llm",
    "messages": [
      {"role": "user", "content": "what colour is the sky?"}
    ],
    "stream": true
  }'
```

OpenAI Tool Calling Interception Request:

```Bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mock-llm",
    "messages": [
      {"role": "user", "content": "List the files in the current directory."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "list_dir",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string"},
              "recursive": {"type": "boolean"}
            }
          }
        }
      }
    ]
  }'
```

### Anthropic Format

Regular Request:

```Bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "messages": [
      {"role": "user", "content": "what colour is the sky?"}
    ]
  }'
```

Streaming Request:

```Bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "messages": [
      {"role": "user", "content": "what colour is the sky?"}
    ],
    "stream": true
  }'
```

## Testing

To run the tests:

```Bash
poetry run pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a PR.

Check out the CodeGate project when you're done here!

## License

This project is licensed under the Apache 2.0 License.
