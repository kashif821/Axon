from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Optional

import litellm
from litellm import AuthenticationError, RateLimitError, BadRequestError

litellm.suppress_debug_info = True
litellm.set_verbose = False

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

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
    def __init__(self, default_model: str):
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        return "litellm"

    @property
    def default_model(self) -> str:
        return self._default_model

    def _get_litellm_kwargs(self, model: str) -> dict:
        from axon.config.loader import get_api_base, get_env_key
        import os

        kwargs = {}
        api_base = get_api_base(model)
        if api_base:
            kwargs["api_base"] = api_base
        env_key = get_env_key(model)
        if env_key:
            api_key = os.getenv(env_key)
            if api_key:
                kwargs["api_key"] = api_key
        return kwargs

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
        tools: list[dict] | None = None,
    ) -> ChatCompletionResponse:
        model = model or self._default_model

        litellm_messages = self._prepare_messages(messages)
        litellm_kwargs = self._get_litellm_kwargs(model)

        try:
            response = await litellm.acompletion(
                model=model,
                messages=litellm_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **litellm_kwargs,
            )

            return self._parse_response(response)

        except AuthenticationError as e:
            raise LLMConfigurationError(
                f"Model {model} failed: missing or invalid API key. "
                f"Set the appropriate API key in your .env file."
            ) from e
        except RateLimitError as e:
            raise LLMError(
                f"Model {model} failed: rate limit exceeded. Please try again later."
            ) from e
        except BadRequestError as e:
            raise LLMError(f"Model {model} failed: bad request - {e}") from e
        except Exception as e:
            raise LLMError(f"Model {model} failed: {e}") from e

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
    ) -> AsyncIterator[StreamResponse]:
        model = model or self._default_model

        litellm_messages = self._prepare_messages(messages)
        litellm_kwargs = self._get_litellm_kwargs(model)

        try:
            response = await litellm.acompletion(
                model=model,
                messages=litellm_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **litellm_kwargs,
            )

            async for chunk in response:
                yield self._parse_stream_chunk(chunk)

        except AuthenticationError as e:
            raise LLMConfigurationError(
                f"Model {model} failed: missing or invalid API key. "
                f"Set the appropriate API key in your .env file."
            ) from e
        except RateLimitError as e:
            raise LLMError(
                f"Model {model} failed: rate limit exceeded. Please try again later."
            ) from e
        except BadRequestError as e:
            raise LLMError(f"Model {model} failed: bad request - {e}") from e
        except Exception as e:
            raise LLMError(f"Model {model} failed: {e}") from e

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[dict]:
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
        return litellm_messages

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


_llm_provider: Optional[LiteLLMProvider] = None


def get_llm_provider(model: Optional[str] = None) -> LiteLLMProvider:
    global _llm_provider

    if not model:
        from axon.config.loader import get_config

        config = get_config()
        model = config.default_model

    if _llm_provider is None or _llm_provider.default_model != model:
        _llm_provider = LiteLLMProvider(default_model=model)

    return _llm_provider
