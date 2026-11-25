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
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    anthropic_server_fallback: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    llm_max_retries: int = 4
    llm_timeout_seconds: float = 120.0
    extraction_validation_retries: int = 2
    policy_max_tokens: int = 16_000
    policy_llm_review: bool = True

    # --- Embeddings / KB ---
    embedding_provider: Literal["local", "openai", "none"] = "local"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_path: str = "storage/chroma"
    seed_dir: str = "data/seed"
    rag_enabled: bool = True

    # --- Decision thresholds ---
    # The limits written in policy.md are enforced; the values below are *caps* - whichever is stricter
    # wins, so neither an edited policy nor an edited .env can silently loosen the other.
    critical_field_confidence_min: float = 0.80
    price_tolerance_pct: float = 10.0
    auto_approval_limit: float = 10_000.0
    single_po_hard_limit: float = 50_000.0
    vendor_match_high: float = 90.0
    vendor_match_medium: float = 75.0
    vendor_ambiguity_margin: float = 5.0
    vendor_semantic_min: float = 0.86       # calibrated on the dev split of data/eval/retrieval_cases.csv
    catalog_match_min: float = 80.0
    catalog_ambiguity_margin: float = 5.0
    catalog_semantic_min: float = 0.81      # calibrated on the dev split
    catalog_semantic_margin: float = 0.08
    po_split_window_days: int = 7
    arithmetic_abs_tolerance: float = 0.05
    arithmetic_rel_tolerance: float = 0.001
    allowed_payment_terms: str = "Net 30,Net 45,Net 60"
    base_currency: str = "USD"

    # --- Storage ---
    database_url: str = "sqlite:///storage/po_agent.db"
    storage_dir: str = "storage"
    max_attachment_mb: float = 20.0

    # --- Jobs ---
    job_max_attempts: int = 3
    job_queued_grace_seconds: int = 60       # queued longer than this = its background task was lost
    job_stale_seconds: int = 900             # processing longer than this = its worker died
    job_sweep_seconds: int = 30
    intake_stale_seconds: int = 120          # an email claim older than this without jobs is taken over

    # --- Intake ---
    webhook_provider: Literal["mailgun", "sendgrid", "generic"] = "mailgun"
    webhook_extra_providers: str = ""        # e.g. "generic" to also accept the n8n workflow
    mailgun_signing_key: str = ""
    webhook_shared_secret: str = ""
    webhook_max_age_seconds: int = 900
    allowed_sender_domains: str = ""
    api_key: str = ""                        # X-API-Key for every non-webhook endpoint; empty = those are disabled
    review_ui_password: str = ""
    api_base_url: str = "http://localhost:8000"   # where the review UI finds the API
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

    @property
    def enabled_webhook_providers(self) -> set[str]:
        extra = {p.strip().lower() for p in self.webhook_extra_providers.split(",") if p.strip()}
        return {self.webhook_provider} | extra

    @staticmethod
    def real_secret(value: str) -> str:
        """Placeholders copied from .env.example are public knowledge - treat them as not configured."""
        v = (value or "").strip()
        return "" if v.lower().startswith("change-me") else v

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
