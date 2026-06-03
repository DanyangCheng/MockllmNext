import asyncio
import logging
import os
import random
import re
import copy
import json
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator, Optional, cast, Any

import yaml
from pythonjsonlogger.json import JsonFormatter


class ConfigError(Exception):
    """Raised when response configuration cannot be loaded."""


log_handler = logging.StreamHandler()
log_handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[log_handler])
logger = logging.getLogger(__name__)


class ResponseConfig:
    """Handles loading and managing response configurations from YAML."""

    def __init__(self, yaml_path: Optional[str] = None):
        self.yaml_path = cast(
            str, yaml_path or os.getenv("MOCKLLM_RESPONSES_FILE", "responses.yml")
        )
        self.last_modified = 0
        self.responses: Dict[str, str] = {}
        self.patterns: list = []
        self.default_response = "I don't know the answer to that."
        self.lag_enabled = False
        self.lag_factor = 10
        self.load_responses()

    def load_responses(self) -> None:
        """Load or reload responses from YAML file if modified."""
        try:
            path = Path(self.yaml_path)
            current_mtime = path.stat().st_mtime
            if current_mtime > self.last_modified:
                with path.open("r") as f:
                    data = yaml.safe_load(f)
                    self.responses = data.get("responses", {})
                    self.patterns = data.get("patterns", [])
                    self.default_response = data.get("defaults", {}).get(
                        "unknown_response", self.default_response
                    )
                    settings = data.get("settings", {})
                    self.lag_enabled = settings.get("lag_enabled", False)
                    self.lag_factor = settings.get("lag_factor", 10)
                self.last_modified = int(current_mtime)
                logger.info(
                    f"Loaded {len(self.responses)} responses, {len(self.patterns)} regex patterns from {self.yaml_path}"
                )
        except Exception as e:
            logger.error(f"Error loading responses: {str(e)}")
            raise ConfigError(
                "Failed to load response configuration"
            ) from e

    def match_pattern(self, prompt: str) -> Optional[Dict[str, Any]]:
        self.load_responses()
        for pattern in self.patterns:
            regex = pattern.get("regex")
            if not regex:
                continue
            try:
                match = re.search(regex, prompt, re.IGNORECASE)
                if match:
                    matched_config = copy.deepcopy(pattern)
                    
                    if matched_config.get("type") == "text" and "text" in matched_config:
                        text_val = matched_config["text"]
                        for idx, group_val in enumerate(match.groups(), start=1):
                            text_val = text_val.replace(f"${idx}", group_val)
                        matched_config["text"] = text_val

                    elif matched_config.get("type") == "tool_call" and "function" in matched_config:
                        func_node = matched_config["function"]
                        args = func_node.get("arguments", "{}")
                        
                        if isinstance(args, str):
                            for idx, group_val in enumerate(match.groups(), start=1):
                                args = args.replace(f"${idx}", group_val)
                            func_node["arguments"] = args
                            
                        elif isinstance(args, dict):
                            args_str = json.dumps(args, ensure_ascii=False)
                            for idx, group_val in enumerate(match.groups(), start=1):
                                args_str = args_str.replace(f"${idx}", group_val)
                            func_node["arguments"] = args_str
                            
                        if "final_text" in matched_config:
                            final_text_val = matched_config["final_text"]
                            for idx, group_val in enumerate(match.groups(), start=1):
                                final_text_val = final_text_val.replace(f"${idx}", group_val)
                            matched_config["final_text"] = final_text_val
                            
                    return matched_config
            except Exception as e:
                logger.error(f"Regex error for pattern '{regex}': {str(e)}")
        return None

    def get_response(self, prompt: str) -> str:
        """Get response for a given prompt."""
        self.load_responses()  # Check for updates
        if prompt in self.responses:
            return self.responses[prompt]
        
        matched = self.match_pattern(prompt)
        if matched and matched.get("type") == "text":
            return matched.get("text", self.default_response)
            
        return self.default_response

    def get_streaming_response(
        self, prompt: str, chunk_size: Optional[int] = None
    ) -> Generator[str, None, None]:
        """Generator that yields response content
        character by character or in chunks."""
        response = self.get_response(prompt)
        if chunk_size:
            # Yield response in chunks
            for i in range(0, len(response), chunk_size):
                yield response[i : i + chunk_size]
        else:
            for char in response:
                yield char

    async def get_response_with_lag(self, prompt: str) -> str:
        """Get response with artificial lag for non-streaming responses."""
        response = self.get_response(prompt)
        if self.lag_enabled:
            delay = len(response) / (self.lag_factor * 10)
            await asyncio.sleep(delay)
        return response

    async def get_streaming_response_with_lag(
        self, prompt: str, chunk_size: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Generator that yields response content with artificial lag."""
        response = self.get_response(prompt)

        if chunk_size:
            for i in range(0, len(response), chunk_size):
                chunk = response[i : i + chunk_size]
                if self.lag_enabled:
                    delay = len(chunk) / (self.lag_factor * 10)
                    await asyncio.sleep(delay)
                yield chunk
        else:
            for char in response:
                if self.lag_enabled:
                    # Add random variation to character delay
                    base_delay = 1 / (self.lag_factor * 10)
                    variation = random.uniform(-0.5, 0.5) * base_delay
                    delay = max(0, base_delay + variation)
                    await asyncio.sleep(delay)
                yield char
                
    async def stream_raw_text_with_lag(
        self, text: str, chunk_size: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Generator that yields text content with artificial lag."""
        if chunk_size:
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                if self.lag_enabled:
                    delay = len(chunk) / (self.lag_factor * 10)
                    await asyncio.sleep(delay)
                yield chunk
        else:
            for char in text:
                if self.lag_enabled:
                    base_delay = 1 / (self.lag_factor * 10)
                    variation = random.uniform(-0.5, 0.5) * base_delay
                    delay = max(0, base_delay + variation)
                    await asyncio.sleep(delay)
                yield char