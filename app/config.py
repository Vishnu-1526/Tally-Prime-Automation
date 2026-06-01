from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and/or a local .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/tally_prime",
        description="SQLAlchemy Database connection string (asyncpg driver is required)."
    )
    OPENAI_API_KEY: str = Field(
        default="",
        description="API Key for OpenAI services."
    )
    GOOGLE_API_KEY: str = Field(
        default="",
        description="API Key for Google services (Gemini)."
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for the locally running Ollama service."
    )
    TALLY_HOST: str = Field(
        default="localhost",
        description="Hostname/IP of the Tally Prime server."
    )
    TALLY_PORT: int = Field(
        default=9000,
        description="XML port configured in Tally Prime."
    )
    COMPANY_GSTIN: str = Field(
        default="",
        description="GSTIN of the company executing voucher postings."
    )
    COMPANY_STATE_CODE: str = Field(
        default="",
        description="Two-digit state code of the company."
    )
    AUTO_POST_THRESHOLD: float = Field(
        default=0.95,
        description="Confidence threshold above which vouchers are automatically posted to Tally."
    )


settings = Settings()
