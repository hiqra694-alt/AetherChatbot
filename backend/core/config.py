import os
from functools import lru_cache
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration management for the application.
    Loads secrets from .env file and validates them on startup.
    """
    gemini_api_key: str = Field("", validation_alias=AliasChoices("gemini_api_key", "GEMINI_API_KEY"))
    openai_api_key: str = Field("", validation_alias=AliasChoices("openai_api_key", "OPENAI_API_KEY"))
    anthropic_api_key: str = Field("", validation_alias=AliasChoices("anthropic_api_key", "ANTHROPIC_API_KEY"))
    groq_api_key: str = Field("", validation_alias=AliasChoices("groq_api_key", "GROQ_API_KEY"))
    voyage_api_key: str = Field("", validation_alias=AliasChoices("voyage_api_key", "VOYAGE_API_KEY"))

    supabase_url: str = Field("", validation_alias=AliasChoices("supabase_url", "SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"))
    supabase_anon_key: str = Field("", validation_alias=AliasChoices("supabase_anon_key", "SUPABASE_ANON_KEY", "NEXT_PUBLIC_SUPABASE_ANON_KEY"))

    # Used only by the background task scheduler (core/scheduler.py), which
    # runs with no per-request user JWT and so has nothing to authenticate an
    # anon-key client with -- it needs RLS bypassed to poll due tasks across
    # every user. Never used to build a per-request client; those always stay
    # scoped to the caller's own JWT via supabase_anon_key. Empty by default,
    # in which case the scheduler logs a warning and polls are a no-op.
    supabase_service_role_key: str = Field(
        "", validation_alias=AliasChoices("supabase_service_role_key", "SUPABASE_SERVICE_ROLE_KEY")
    )

    # Comma-separated list of external MCP server URLs (HTTP/SSE transport),
    # e.g. a Brave Search MCP endpoint. Empty by default -- MCPClientManager
    # simply connects to nothing until this is populated.
    mcp_server_urls: str = Field("", validation_alias=AliasChoices("mcp_server_urls", "MCP_SERVER_URLS"))

    # GitHub MCP connector: both must be set for MCPClientManager to dial it
    # (see mcp_integration.mcp_manager._build_default_server_specs). The
    # token is sent as a Bearer header on the SSE connection, never logged.
    github_personal_access_token: str = Field(
        "", validation_alias=AliasChoices("github_personal_access_token", "GITHUB_PERSONAL_ACCESS_TOKEN")
    )
    github_mcp_server_url: str = Field(
        "", validation_alias=AliasChoices("github_mcp_server_url", "GITHUB_MCP_SERVER_URL")
    )

    # Google Workspace MCP connector (Phase 4): unlike GitHub, this has no
    # single shared token -- MCPClientManager.create_workspace_session opens
    # a fresh, per-request session against this URL using that request's own
    # google_access_token, so only the URL (never a token) is static config.
    google_workspace_mcp_url: str = Field(
        "", validation_alias=AliasChoices("google_workspace_mcp_url", "GOOGLE_WORKSPACE_MCP_URL")
    )

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def mcp_server_url_list(self) -> list[str]:
        """mcp_server_urls split on commas, trimmed, empty entries dropped."""
        return [u.strip() for u in self.mcp_server_urls.split(",") if u.strip()]

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the settings.
    Ensures configuration is only read and validated once at application startup.
    """
    return Settings()
