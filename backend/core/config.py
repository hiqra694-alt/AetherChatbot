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

    # Server-side refresh-token fallback (mcp_integration.mcp_manager
    # .create_workspace_session / _refresh_google_access_token): used to
    # exchange a stored user_oauth_tokens.refresh_token for a fresh Google
    # access token via Google's own token endpoint when a request doesn't
    # carry a live one. Must match the OAuth client configured in Supabase's
    # Google provider settings, since the refresh_token was minted for that
    # client. Both empty by default -- the fallback is simply skipped (same
    # as no token at all) until both are set.
    google_client_id: str = Field(
        "", validation_alias=AliasChoices("google_client_id", "GOOGLE_CLIENT_ID")
    )
    google_client_secret: str = Field(
        "", validation_alias=AliasChoices("google_client_secret", "GOOGLE_CLIENT_SECRET")
    )

    # Gmail MCP connector (Phase 1, mcp_integration/gmail_mcp.py +
    # gmail_mcp_router.py): a fully separate OAuth client from
    # google_client_id/google_client_secret above -- this one talks
    # directly to Google's OAuth endpoints (never through Supabase's own
    # provider-linking flow) and its refresh token is stored under the
    # distinct 'google_gmail_mcp' user_oauth_tokens row, never the plain
    # 'google' row. All three must be set for /api/connectors/gmail-mcp/
    # authorize to work; empty by default -- that route 503s until
    # configured.
    gmail_mcp_client_id: str = Field(
        "", validation_alias=AliasChoices("gmail_mcp_client_id", "GMAIL_MCP_CLIENT_ID")
    )
    gmail_mcp_client_secret: str = Field(
        "", validation_alias=AliasChoices("gmail_mcp_client_secret", "GMAIL_MCP_CLIENT_SECRET")
    )
    gmail_mcp_redirect_uri: str = Field(
        "", validation_alias=AliasChoices("gmail_mcp_redirect_uri", "GMAIL_MCP_REDIRECT_URI")
    )

    # Local-development-only escape hatch (mcp_integration/gmail_mcp_router.py):
    # when true, GET /api/connectors/gmail-mcp/authorize additionally accepts
    # a `dev_user_id` query param as an UNVERIFIED substitute for a real
    # Supabase `access_token`, so the Google consent/redirect/callback flow
    # can be exercised by pasting a URL straight into a browser instead of
    # extracting a live JWT first. False by default -- must be explicitly
    # set to true, and must never be set in any deployed environment, since
    # it lets the caller claim to be any user_id with zero verification.
    gmail_mcp_dev_mode: bool = Field(
        False, validation_alias=AliasChoices("gmail_mcp_dev_mode", "GMAIL_MCP_DEV_MODE")
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
