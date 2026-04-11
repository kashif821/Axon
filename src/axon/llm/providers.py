from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

import litellm
from litellm import AuthenticationError, RateLimitError, BadRequestError

litellm.suppress_debug_info = True
litellm.set_verbose = False

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

from axon.config.settings import settings
from axon.llm.base import (
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceDelta,
    FunctionCall,
    LLMProvider,
    Message,
    MessageRole,
    StreamResponse,
    ToolCall,
    Usage,
)


class LiteLLMProvider(LLMProvider):
    def __init__(self, default_model: str = "gpt-4o-mini"):
        self._default_model = default_model
        self._fallback_models = [
            {"model": "groq/llama-3.1-8b-instant"},
            {"model": "openai/gpt-4o-mini"},
            {"model": "anthropic/claude-3-haiku-20240307"},
        ]

    @property
    def provider_name(self) -> str:
        return "litellm"

    @property
    def default_model(self) -> str:
        return self._default_model

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
        tools: list[dict] | None = None,
    ) -> ChatCompletionResponse:
        model = model or self._default_model

        litellm_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                litellm_messages.append(msg)
            else:
                msg_dict = {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": msg.content,
                }
                if getattr(msg, "tool_calls", None):
                    msg_dict["tool_calls"] = msg.tool_calls
                if getattr(msg, "tool_call_id", None):
                    msg_dict["tool_call_id"] = msg.tool_call_id
                if getattr(msg, "name", None):
                    msg_dict["name"] = msg.name
                litellm_messages.append(msg_dict)

        try:
            response = await litellm.acompletion(
                model=model,
                messages=litellm_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                fallbacks=self._fallback_models,
                api_base="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY"),
            )

            return self._parse_response(response)

        except AuthenticationError as e:
            raise LLMConfigurationError(
                f"Authentication failed. Please check your API key for {model.split('/')[0]}. "
                f"Update your .env file with the correct API key."
            ) from e
        except RateLimitError as e:
            raise LLMError(f"Rate limit exceeded. Please try again later.") from e
        except BadRequestError as e:
            raise LLMError(f"Bad request: {e}") from e
        except Exception as e:
            raise LLMError(f"Unexpected error during LLM call: {e}") from e

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
    ) -> AsyncIterator[StreamResponse]:
        model = model or self._default_model

        litellm_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                litellm_messages.append(msg)
            else:
                msg_dict = {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": msg.content,
                }
                if getattr(msg, "tool_calls", None):
                    msg_dict["tool_calls"] = msg.tool_calls
                if getattr(msg, "tool_call_id", None):
                    msg_dict["tool_call_id"] = msg.tool_call_id
                if getattr(msg, "name", None):
                    msg_dict["name"] = msg.name
                litellm_messages.append(msg_dict)

        try:
            response = await litellm.acompletion(
                model=model,
                messages=litellm_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                fallbacks=self._fallback_models,
                api_base="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY"),
            )

            async for chunk in response:
                yield self._parse_stream_chunk(chunk)

        except AuthenticationError as e:
            raise LLMConfigurationError(
                f"Authentication failed. Please check your API key for {model.split('/')[0]}. "
                f"Update your .env file with the correct API key."
            ) from e
        except RateLimitError as e:
            raise LLMError(f"Rate limit exceeded. Please try again later.") from e
        except BadRequestError as e:
            raise LLMError(f"Bad request: {e}") from e
        except Exception as e:
            raise LLMError(f"Unexpected error during streaming LLM call: {e}") from e

    def _parse_response(self, response: Any) -> ChatCompletionResponse:
        choice = response["choices"][0]
        message = choice["message"]

        litellm_tool_calls = message.get("tool_calls")
        parsed_tool_calls = None
        if litellm_tool_calls:
            parsed_tool_calls = []
            for tc in litellm_tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                tc_type = (
                    tc.get("type", "function") if isinstance(tc, dict) else tc.type
                )
                func = tc.get("function", {}) if isinstance(tc, dict) else tc.function
                func_name = func.get("name") if isinstance(func, dict) else func.name
                func_args = (
                    func.get("arguments") if isinstance(func, dict) else func.arguments
                )
                parsed_tool_calls.append(
                    {
                        "id": tc_id,
                        "type": tc_type,
                        "function": {"name": func_name, "arguments": func_args},
                    }
                )
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in parsed_tool_calls
            ]
        else:
            tool_calls = None

        usage = None
        if "usage" in response and response["usage"]:
            usage = Usage(
                prompt_tokens=response["usage"].get("prompt_tokens"),
                completion_tokens=response["usage"].get("completion_tokens"),
                total_tokens=response["usage"].get("total_tokens"),
            )

        return ChatCompletionResponse(
            model=response.get("model", self._default_model),
            choices=[
                Choice(
                    index=choice.get("index", 0),
                    message=Message(
                        role=MessageRole(message.get("role", "assistant")),
                        content=message.get("content"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            ],
            usage=usage,
        )

    def _parse_stream_chunk(self, chunk: Any) -> StreamResponse:
        delta = chunk["choices"][0].get("delta", {})
        finish_reason = chunk["choices"][0].get("finish_reason")

        raw_role = delta.get("role")
        safe_role = raw_role if raw_role is not None else "assistant"

        raw_content = delta.get("content")
        safe_content = raw_content if raw_content is not None else ""

        reasoning_content = delta.get("reasoning_content")

        return StreamResponse(
            model=chunk.get("model", self._default_model),
            choices=[
                ChoiceDelta(
                    index=chunk["choices"][0].get("index", 0),
                    delta=Message(
                        role=MessageRole(safe_role),
                        content=safe_content,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            reasoning_content=reasoning_content,
        )


class LLMError(Exception):
    pass


class LLMConfigurationError(LLMError):
    pass


_llm_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        default_model = _get_default_model()
        _llm_provider = LiteLLMProvider(default_model=default_model)
    return _llm_provider


def _get_default_model() -> str:
    if settings.nvidia_api_key and settings.nvidia_api_key != "your_nvidia_key_here":
        return "openai/moonshotai/kimi-k2-thinking"
    elif settings.gemini_api_key and settings.gemini_api_key != "your_gemini_key_here":
        return "gemini/gemini-2.5-flash"
    elif settings.openai_api_key and settings.openai_api_key != "your_openai_key_here":
        return "gpt-4o-mini"
    elif (
        settings.anthropic_api_key
        and settings.anthropic_api_key != "your_anthropic_key_here"
    ):
        return "claude-3-haiku-20240307"
    elif settings.groq_api_key and settings.groq_api_key != "your_groq_key_here":
        return "groq/llama-3.1-8b-instant"
    return "openai/moonshotai/kimi-k2-thinking"
