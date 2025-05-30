import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    env: str
    db_user: str
    db_password: str
    db_host: str
    db_name: str

    @property
    def database_url(self):
        return f"mysql+aiomysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}"

    class Config:
        env_file = f".env.{os.getenv('ENV', 'local')}"