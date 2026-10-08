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
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "models").strip().lower()
    COPILOT_MODEL: str = os.getenv("COPILOT_MODEL", "gpt-4.1")
    GITHUB_CLIENT_ID: str = os.getenv("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET", "")
    GITHUB_REDIRECT_URI: str = os.getenv("GITHUB_REDIRECT_URI", "")
    # Optional trusted, administrator-configured TradingView MCP server.
    TRADINGVIEW_MCP_URL: str = os.getenv("TRADINGVIEW_MCP_URL", "")
    # Comma-separated list of selectable model identifiers shown in the UI.
    # LLM_MODEL is always included (prepended) if not already present.
    LLM_AVAILABLE_MODELS: str = os.getenv(
        "LLM_AVAILABLE_MODELS",
        "openai/gpt-4.1-mini,openai/gpt-4.1,openai/gpt-4o,openai/gpt-5,openai/gpt-5-mini,openai/o4-mini,"
        "anthropic/claude-sonnet-4.6,anthropic/claude-opus-4.6,microsoft/mai-code-1-flash,microsoft/phi-4",
    )

    @property
    def llm_available_models(self) -> list[str]:
        """Return the list of selectable model identifiers.

        ``LLM_MODEL`` is always the first entry and always present.
        """
        raw = [m.strip() for m in self.LLM_AVAILABLE_MODELS.split(",") if m.strip()]
        if self.LLM_MODEL not in raw:
            raw.insert(0, self.LLM_MODEL)
        elif raw[0] != self.LLM_MODEL:
            raw.remove(self.LLM_MODEL)
            raw.insert(0, self.LLM_MODEL)
        return raw

    @property
    def llm_token(self) -> str:
        """Return whichever LLM token is available."""
        return self.GITHUB_TOKEN or self.OPENAI_API_KEY

    @property
    def llm_enabled(self) -> bool:
        """True when at least one LLM auth token is configured."""
        return self.LLM_PROVIDER == "copilot" or bool(self.llm_token)

    # News
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    # Scheduler — IST = UTC + 5:30
    SCHEDULE_HOUR_IST: int = int(os.getenv("SCHEDULE_HOUR_IST", "9"))
    SCHEDULE_MINUTE_IST: int = int(os.getenv("SCHEDULE_MINUTE_IST", "0"))

    # API key for read-only JSON endpoint
    APP_API_KEY: str = os.getenv("APP_API_KEY", "")


settings = Config()
