"""All configuration and every decision threshold, loaded from the environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    llm_provider: Literal["anthropic", "openai", "heuristic"] = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    anthropic_server_fallback: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    llm_max_retries: int = 4
    llm_timeout_seconds: float = 120.0
    extraction_validation_retries: int = 2
    policy_llm_review: bool = True

    # --- Embeddings / KB ---
    embedding_provider: Literal["local", "openai", "none"] = "local"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_path: str = "storage/chroma"
    seed_dir: str = "data/seed"
    rag_enabled: bool = True

    # --- Decision thresholds ---
    critical_field_confidence_min: float = 0.80
    price_tolerance_pct: float = 10.0
    auto_approval_limit: float = 10_000.0
    single_po_hard_limit: float = 50_000.0
    vendor_match_high: float = 90.0
    vendor_match_medium: float = 75.0
    vendor_ambiguity_margin: float = 5.0
    catalog_match_min: float = 80.0
    arithmetic_abs_tolerance: float = 0.05
    arithmetic_rel_tolerance: float = 0.001
    allowed_payment_terms: str = "Net 30,Net 45,Net 60"
    base_currency: str = "USD"

    # --- Storage ---
    database_url: str = "sqlite:///storage/po_agent.db"
    storage_dir: str = "storage"
    max_attachment_mb: float = 20.0

    # --- Intake ---
    webhook_provider: Literal["mailgun", "sendgrid", "generic"] = "mailgun"
    mailgun_signing_key: str = ""
    webhook_shared_secret: str = ""
    webhook_max_age_seconds: int = 900
    allowed_sender_domains: str = ""
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    imap_user: str = ""
    imap_password: str = ""
    imap_folder: str = "INBOX"
    imap_poll_seconds: int = 30

    # --- Notifications ---
    slack_webhook_url: str = ""
    review_ui_url: str = "http://localhost:8501"

    # --- Google Sheets mirror (optional) ---
    google_sheets_enabled: bool = False
    google_sheets_spreadsheet_id: str = ""
    google_service_account_file: str = ""

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True

    supported_extensions: tuple[str, ...] = Field(
        default=(".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".csv")
    )

    # ---- derived helpers ----
    @property
    def allowed_terms_list(self) -> list[str]:
        return [t.strip() for t in self.allowed_payment_terms.split(",") if t.strip()]

    @property
    def allowed_sender_domain_list(self) -> list[str]:
        return [d.strip().lower() for d in self.allowed_sender_domains.split(",") if d.strip()]

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def storage_path(self) -> Path:
        return self.path(self.storage_dir)

    @property
    def chroma_dir(self) -> Path:
        return self.path(self.chroma_path)

    @property
    def seed_path(self) -> Path:
        return self.path(self.seed_dir)

    @property
    def resolved_database_url(self) -> str:
        """Make relative sqlite paths relative to the project root, not the cwd."""
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            raw = self.database_url[len(prefix):]
            if raw and raw != ":memory:" and not Path(raw).is_absolute():
                return prefix + (PROJECT_ROOT / raw).as_posix()
        return self.database_url

    @property
    def llm_available(self) -> bool:
        if self.llm_provider == "anthropic":
            return True  # SDK also resolves credentials from `ant auth` profiles
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return False


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
