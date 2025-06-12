from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    env: str = "development"
    db_user: str = "root"
    db_password: str = "password"
    db_host: str = "localhost"
    db_name: str = "ai_db"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings() 