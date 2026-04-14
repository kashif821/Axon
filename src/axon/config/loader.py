from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    api_base: str | None = None
    env_key: str | None = None


class ProviderConfigs(BaseModel):
    nvidia_nim: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            api_base="https://integrate.api.nvidia.com/v1",
            env_key="NVIDIA_API_KEY",
        )
    )
    openai: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(api_base=None, env_key="OPENAI_API_KEY")
    )
    ollama: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            api_base="http://localhost:11434/v1", env_key=None
        )
    )


class ModesConfig(BaseModel):
    planner: str = "gemini/gemini-2.5-flash"
    builder: str = "groq/llama-3.1-70b-versatile"
    brain: str = "gemini/gemini-2.5-flash"
    chat: str = "gemini/gemini-2.5-flash"


class AxonConfig(BaseModel):
    default_model: str = "gemini/gemini-2.5-flash"
    modes: ModesConfig = Field(default_factory=ModesConfig)
    provider_configs: ProviderConfigs = Field(default_factory=ProviderConfigs)
    brain: dict[str, Any] = Field(
        default_factory=lambda: {"idle_seconds": 180, "max_files": 10}
    )
    limits: dict[str, int] = Field(
        default_factory=lambda: {"max_tokens": 8000, "max_iterations": 15}
    )

    def get_mode_model(self, mode: str) -> str:
        return getattr(self.modes, mode, self.default_model)

    def merge_cli(self, model: Optional[str] = None) -> str:
        return model or self.default_model


API_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "ollama": "OLLAMA_API_BASE",
    "azure": "AZURE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
}


_loaded_config: Optional[AxonConfig] = None


def load_config(search_path: str | None = None) -> AxonConfig:
    global _loaded_config

    if _loaded_config is not None:
        return _loaded_config

    config_data = {
        "default_model": "gemini/gemini-2.5-flash",
        "modes": {
            "planner": "gemini/gemini-2.5-flash",
            "builder": "groq/llama-3.1-70b-versatile",
            "brain": "gemini/gemini-2.5-flash",
            "chat": "gemini/gemini-2.5-flash",
        },
    }

    search_dirs = []
    if search_path:
        search_dirs.append(Path(search_path))
    search_dirs.append(Path.cwd())
    search_dirs.append(Path.home() / ".axon")

    for config_dir in search_dirs:
        yaml_path = config_dir / "axon.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path) as f:
                    user_config = yaml.safe_load(f) or {}
                if "modes" in user_config and isinstance(user_config["modes"], dict):
                    user_modes = user_config.pop("modes")
                    user_config["modes"] = ModesConfig(**user_modes)
                if "provider_configs" in user_config and isinstance(
                    user_config["provider_configs"], dict
                ):
                    user_providers = user_config.pop("provider_configs")
                    user_config["provider_configs"] = ProviderConfigs(
                        **{k: ProviderConfig(**v) for k, v in user_providers.items()}
                    )
                config_data = _deep_merge(config_data, user_config)
                break
            except Exception:
                pass

    _loaded_config = AxonConfig(**config_data)
    return _loaded_config


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
            and not isinstance(result[key], BaseModel)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def reload_config() -> None:
    global _loaded_config
    _loaded_config = None


def get_config() -> AxonConfig:
    return load_config()


def get_environment_keys() -> dict[str, bool]:
    found = {}
    for provider, var_names in API_KEY_VARS.items():
        if isinstance(var_names, list):
            found[provider] = any(os.getenv(v) for v in var_names)
        else:
            found[provider] = bool(os.getenv(var_names))
    return found


def get_model_api_key(model: str) -> str | None:
    model_lower = model.lower()
    for provider, var_names in API_KEY_VARS.items():
        if provider in model_lower:
            if isinstance(var_names, list):
                for v in var_names:
                    key = os.getenv(v)
                    if key:
                        return v
            else:
                key = os.getenv(var_names)
                if key:
                    return var_names
    return None


def get_provider_for_model(model: str) -> str:
    if "/" in model:
        return model.split("/")[0]
    return model


def get_provider_config(model: str) -> ProviderConfig | None:
    config = get_config()
    provider = get_provider_for_model(model)
    return getattr(config.provider_configs, provider, None)


def get_api_base(model: str) -> str | None:
    pc = get_provider_config(model)
    if pc:
        return pc.api_base
    return None


def get_env_key(model: str) -> str | None:
    pc = get_provider_config(model)
    if pc:
        return pc.env_key
    return None
