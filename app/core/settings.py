from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Use Docker-exposed GraphDB port on localhost for now
    graphdb_base_url: str = "http://localhost:7200"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()