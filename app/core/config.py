from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_backend_key: str = "dev-key"
    gemini_api_key: str | None = None

settings = Settings()
