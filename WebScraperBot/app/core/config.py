from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    grok_api_key: str = Field(..., alias="GROK_API_KEY")
    default_scrape_url: str = Field(..., alias="DEFAULT_SCRAPE_URL")
    grok_model: str = Field(default="llama-3.1-8b-instant", alias="GROK_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
