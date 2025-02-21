from pydantic_settings import BaseSettings

class Setting(BaseSettings):
    DATABASE_URL: str
    DB_ECHO: bool
    LLM_MODEL: str
    EMBEDDING_DIMENSIONS: int
    EMBEDDING_MODEL: str
    MAX_TOKENS: int
    TOKEN_EMBEDDING_MODEL: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Setting()