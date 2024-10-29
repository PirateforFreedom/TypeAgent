from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="typeagent_")
    typeagent_dir: Optional[Path] = Field(Path.home() / ".typeagent", env="TYPEAGENT_DIR")
    debug: Optional[bool] = False
    server_pass: Optional[str] = None
    pg_db: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    pg_host: Optional[str] = None
    pg_port: Optional[int] = None
    pg_uri: Optional[str] = None  # option to specifiy full uri
    cors_origins: Optional[list] = ["http://typeagent.localhost", "http://localhost:8283", "http://localhost:8083"]

    @property
    # def pg_uri(self) -> str:
    #     return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
    def memgpt_pg_uri_no_default(self) -> str:
        if self.pg_uri:
            return self.pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return None
    @property
    def memgpt_pg_uri(self) -> str:
        if self.pg_uri:
            return self.pg_uri
        elif self.pg_db and self.pg_user and self.pg_password and self.pg_host and self.pg_port:
            return f"postgresql+pg8000://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        else:
            return f"postgresql+pg8000://typeagent:typeagent@localhost:5432/typeagent"

# singleton
settings = Settings()
