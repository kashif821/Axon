from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: "FunctionCall"


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Message(BaseModel):
    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatMessage(BaseModel):
    role: MessageRole
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=4096, ge=1)
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    model: str
    choices: list[Choice]
    usage: Usage | None = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class ChoiceDelta(BaseModel):
    index: int
    delta: Message
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class StreamResponse(BaseModel):
    model: str
    choices: list[ChoiceDelta]


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
        tools: list[dict] | None = None,
    ) -> ChatCompletionResponse:
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
    ) -> AsyncIterator[StreamResponse]:
        pass
