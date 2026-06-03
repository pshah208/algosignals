"""Application configuration loaded from environment / .env file."""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration object for AlgoSignals."""

    # Flask
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///algosignals.db")

    # LLM / GitHub Models
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "") or os.getenv("GITHUB_MODELS_TOKEN", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://models.github.ai/inference")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    @property
    def llm_token(self) -> str:
        """Return whichever LLM token is available."""
        return self.GITHUB_TOKEN or self.OPENAI_API_KEY

    @property
    def llm_enabled(self) -> bool:
        """True when at least one LLM auth token is configured."""
        return bool(self.llm_token)

    # News
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    # Scheduler — IST = UTC + 5:30
    SCHEDULE_HOUR_IST: int = int(os.getenv("SCHEDULE_HOUR_IST", "9"))
    SCHEDULE_MINUTE_IST: int = int(os.getenv("SCHEDULE_MINUTE_IST", "0"))

    # API key for read-only JSON endpoint
    APP_API_KEY: str = os.getenv("APP_API_KEY", "")


settings = Config()
