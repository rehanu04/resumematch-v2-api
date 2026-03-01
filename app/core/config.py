from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_backend_key: str
    gemini_api_key: str | None = None

    jd_proxy_url: str | None = None
    jd_proxy_key: str | None = None

settings = Settings()
