from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    axon_mode: str = "standard"
    axon_db_path: str = ".axon/memory.sqlite"
    
    # API Keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
