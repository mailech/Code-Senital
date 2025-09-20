from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


class Settings(BaseModel):
    app_name: str = Field(default="Self-Healing Codebase Sentinel")
    environment: str = Field(default=os.getenv("ENVIRONMENT", "development"))

    # API keys and tokens
    github_token: str | None = Field(default=os.getenv("GITHUB_TOKEN"))
    slack_token: str | None = Field(default=os.getenv("SLACK_BOT_TOKEN"))
    openai_api_key: str | None = Field(default=os.getenv("OPENAI_API_KEY"))
    hf_api_token: str | None = Field(default=os.getenv("HF_API_TOKEN"))

    # Slack config
    slack_channel: str = Field(default=os.getenv("SLACK_CHANNEL", "#general"))

    # GitHub repo config
    github_owner: str | None = Field(default=os.getenv("GITHUB_OWNER"))
    github_repo: str | None = Field(default=os.getenv("GITHUB_REPO"))
    default_branch: str = Field(default=os.getenv("DEFAULT_BRANCH", "main"))

    # Security & multi-repo
    webhook_secret: str | None = Field(default=os.getenv("SENTINEL_WEBHOOK_SECRET"))
    allowed_repos: str = Field(default=os.getenv("ALLOWED_REPOS", ""))  # comma-separated list

    # Confidence slider
    confidence_threshold: float = Field(default=float(os.getenv("CONFIDENCE_THRESHOLD", "0.8")))

    # Server
    host: str = Field(default=os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default=int(os.getenv("PORT", "8000")))


settings = Settings()
