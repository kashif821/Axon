from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProviderSpec:
    name: str
    env_keys: list[str]
    api_base: str | None = None
    extra_headers: dict = field(default_factory=dict)
    model_prefix: str = ""
    notes: str = ""


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "google": ProviderSpec(
        name="Google Gemini",
        env_keys=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        model_prefix="gemini/",
        notes="Free tier available. Rate limits per minute.",
    ),
    "groq": ProviderSpec(
        name="Groq",
        env_keys=["GROQ_API_KEY"],
        model_prefix="groq/",
        notes="Fast inference. Free tier available.",
    ),
    "nvidia_nim": ProviderSpec(
        name="NVIDIA NIM",
        env_keys=["NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"],
        api_base="https://integrate.api.nvidia.com/v1",
        model_prefix="nvidia_nim/",
        notes="Requires api_base. Free tier available.",
    ),
    "openai": ProviderSpec(
        name="OpenAI",
        env_keys=["OPENAI_API_KEY"],
        model_prefix="openai/",
        notes="Paid. GPT-4o and variants.",
    ),
    "anthropic": ProviderSpec(
        name="Anthropic",
        env_keys=["ANTHROPIC_API_KEY"],
        model_prefix="anthropic/",
        notes="Paid. Claude models.",
    ),
    "cerebras": ProviderSpec(
        name="Cerebras",
        env_keys=["CEREBRAS_API_KEY"],
        model_prefix="cerebras/",
        notes="Fast inference. Free tier.",
    ),
    "together_ai": ProviderSpec(
        name="Together AI",
        env_keys=["TOGETHER_API_KEY"],
        model_prefix="together_ai/",
        notes="Open source models. Pay per token.",
    ),
    "openrouter": ProviderSpec(
        name="OpenRouter",
        env_keys=["OPENROUTER_API_KEY"],
        api_base="https://openrouter.ai/api/v1",
        model_prefix="openrouter/",
        notes="100+ models. Unified API.",
    ),
    "mistral": ProviderSpec(
        name="Mistral",
        env_keys=["MISTRAL_API_KEY"],
        model_prefix="mistral/",
        notes="Mistral models. European provider.",
    ),
    "deepseek": ProviderSpec(
        name="DeepSeek",
        env_keys=["DEEPSEEK_API_KEY"],
        model_prefix="deepseek/",
        notes="Coding focused. Cheap.",
    ),
    "cohere": ProviderSpec(
        name="Cohere",
        env_keys=["COHERE_API_KEY"],
        model_prefix="cohere/",
        notes="RAG focused models.",
    ),
    "perplexity": ProviderSpec(
        name="Perplexity",
        env_keys=["PERPLEXITY_API_KEY"],
        model_prefix="perplexity/",
        notes="Online models with web search.",
    ),
    "xai": ProviderSpec(
        name="xAI Grok",
        env_keys=["XAI_API_KEY"],
        model_prefix="xai/",
        notes="Grok models by xAI.",
    ),
    "ollama": ProviderSpec(
        name="Ollama (Local)",
        env_keys=[],
        api_base="http://localhost:11434",
        model_prefix="ollama/",
        notes="Free. Runs locally. No internet needed.",
    ),
    "lmstudio": ProviderSpec(
        name="LM Studio (Local)",
        env_keys=[],
        api_base="http://127.0.0.1:1234/v1",
        model_prefix="openai/",
        notes="Free. Local LM Studio server.",
    ),
}


def get_provider_for_model(model: str) -> str | None:
    for provider_id, spec in PROVIDER_REGISTRY.items():
        if spec.model_prefix and model.startswith(spec.model_prefix):
            return provider_id
    return None


def get_spec_for_model(model: str) -> ProviderSpec | None:
    provider_id = get_provider_for_model(model)
    if provider_id:
        return PROVIDER_REGISTRY[provider_id]
    return None


def is_valid_key(value: str | None) -> bool:
    if not value or len(value) < 8:
        return False
    bad = ["your_", "placeholder", "_here", "insert", "example", "sk-xxx"]
    return not any(b in value.lower() for b in bad)


def has_valid_key(spec: ProviderSpec) -> bool:
    if not spec.env_keys:
        return True
    return any(is_valid_key(os.environ.get(k)) for k in spec.env_keys)


def get_api_key(spec: ProviderSpec) -> str | None:
    for key_name in spec.env_keys:
        val = os.environ.get(key_name)
        if is_valid_key(val):
            return val
    return None


def get_litellm_kwargs(model: str) -> dict:
    spec = get_spec_for_model(model)
    kwargs = {}
    if spec:
        if spec.api_base:
            kwargs["api_base"] = spec.api_base
        api_key = get_api_key(spec)
        if api_key:
            kwargs["api_key"] = api_key
        if spec.extra_headers:
            kwargs["extra_headers"] = spec.extra_headers
    return kwargs


def get_available_providers() -> list[tuple[str, ProviderSpec]]:
    available = []
    for provider_id, spec in PROVIDER_REGISTRY.items():
        if has_valid_key(spec):
            available.append((provider_id, spec))
    return available


def build_fallback_chain(primary_model: str, yaml_providers: dict) -> list[str]:
    chain = [primary_model]
    candidates = []

    for provider_name, models in yaml_providers.items():
        if not isinstance(models, dict):
            continue

        spec = None
        for pid, s in PROVIDER_REGISTRY.items():
            if (
                s.name.lower() == provider_name.lower()
                or pid == provider_name.lower().replace(" ", "_")
            ):
                spec = s
                break

        if spec and not has_valid_key(spec):
            continue

        for model_id, info in models.items():
            if not isinstance(info, dict):
                continue
            if model_id == primary_model:
                continue
            cost = info.get("cost_in_1m", 0)
            candidates.append((model_id, cost))

    candidates.sort(key=lambda x: x[1])
    chain += [m[0] for m in candidates]
    return chain
