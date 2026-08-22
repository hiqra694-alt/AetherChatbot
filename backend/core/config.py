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

    # Voice agent (Phase 2, voice/agent.py): STT/TTS credentials for the
    # Deepgram and Cartesia LiveKit plugins. Passed explicitly into
    # deepgram.STT(api_key=...)/cartesia.TTS(api_key=...) rather than left
    # for those plugins to fall back to reading os.environ directly --
    # nothing in this app ever populates os.environ from .env (pydantic-settings
    # parses .env into this model only), so an implicit os.environ fallback
    # would silently see nothing no matter what's set in .env.
    deepgram_api_key: str = Field("", validation_alias=AliasChoices("deepgram_api_key", "DEEPGRAM_API_KEY"))
    cartesia_api_key: str = Field("", validation_alias=AliasChoices("cartesia_api_key", "CARTESIA_API_KEY"))

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

    # Native Google Workspace connector (Phase 2, api/connectors/google_oauth.py
    # + connector_integrations/google_tools.py): one shared OAuth client
    # backing the native Gmail/Calendar/Drive tools, requesting all four
    # scopes (gmail.modify, calendar.events, drive.readonly, drive.file) up
    # front so the frontend's independent google_gmail/google_calendar/
    # google_drive toggles never need to re-prompt for consent. Its refresh
    # token is stored under the distinct 'google_workspace' user_oauth_tokens
    # row -- never the plain 'google' row, nor 'google_drive' (Canvas's own,
    # narrower-scoped client above). All three must be set for
    # /api/connectors/google/authorize to work; empty by default -- that
    # route 503s until configured.
    google_workspace_client_id: str = Field(
        "", validation_alias=AliasChoices("google_workspace_client_id", "GOOGLE_WORKSPACE_CLIENT_ID")
    )
    google_workspace_client_secret: str = Field(
        "", validation_alias=AliasChoices("google_workspace_client_secret", "GOOGLE_WORKSPACE_CLIENT_SECRET")
    )
    google_workspace_redirect_uri: str = Field(
        "", validation_alias=AliasChoices("google_workspace_redirect_uri", "GOOGLE_WORKSPACE_REDIRECT_URI")
    )

    # Local-development-only escape hatch (api/connectors/google_oauth.py):
    # when true, GET /api/connectors/google/authorize additionally accepts a
    # `dev_user_id` query param as an UNVERIFIED substitute for a real
    # Supabase `access_token`. False by default -- must never be set in any
    # deployed environment.
    google_workspace_dev_mode: bool = Field(
        False, validation_alias=AliasChoices("google_workspace_dev_mode", "GOOGLE_WORKSPACE_DEV_MODE")
    )

    # Base URL of the Next.js frontend -- used by api/connectors/google_oauth.py's
    # /callback to send the browser back to the app once the connection is
    # saved (it lands there as a plain top-level redirect, not through the
    # Next.js middleware's backend rewrite, so it needs the frontend's own
    # origin, not this API's). Defaults to the local Next dev server; must be
    # set to the deployed frontend origin (e.g. the Vercel domain) in
    # production.
    frontend_url: str = Field(
        "http://localhost:3000", validation_alias=AliasChoices("frontend_url", "FRONTEND_URL")
    )

    # Canvas Drive connector (Phase 1, canvas/drive_oauth.py + canvas/router.py):
    # its own dedicated OAuth client, talking directly to Google's OAuth
    # endpoints requesting the restrictive drive.file scope, with its
    # refresh token stored under the distinct 'google_drive' user_oauth_tokens
    # row. All three must be set for /api/canvas/drive/authorize to work;
    # empty by default -- that route 503s until configured.
    canvas_drive_client_id: str = Field(
        "", validation_alias=AliasChoices("canvas_drive_client_id", "CANVAS_DRIVE_CLIENT_ID")
    )
    canvas_drive_client_secret: str = Field(
        "", validation_alias=AliasChoices("canvas_drive_client_secret", "CANVAS_DRIVE_CLIENT_SECRET")
    )
    canvas_drive_redirect_uri: str = Field(
        "", validation_alias=AliasChoices("canvas_drive_redirect_uri", "CANVAS_DRIVE_REDIRECT_URI")
    )

    # Local-development-only escape hatch (canvas/router.py): when true, GET
    # /api/canvas/drive/authorize additionally accepts a `dev_user_id` query
    # param as an UNVERIFIED substitute for a real Supabase `access_token`.
    # False by default -- must never be set in any deployed environment.
    canvas_drive_dev_mode: bool = Field(
        False, validation_alias=AliasChoices("canvas_drive_dev_mode", "CANVAS_DRIVE_DEV_MODE")
    )

    # Voice agent (Phase 1, voice/router.py): LiveKit room/token issuance for
    # the realtime voice pipeline. All three must be set for
    # POST /api/voice/token to work; missing config is caught inside the
    # endpoint and surfaced as a 500 rather than failing app startup.
    livekit_url: str = Field("", validation_alias=AliasChoices("livekit_url", "LIVEKIT_URL"))
    livekit_api_key: str = Field("", validation_alias=AliasChoices("livekit_api_key", "LIVEKIT_API_KEY"))
    livekit_api_secret: str = Field("", validation_alias=AliasChoices("livekit_api_secret", "LIVEKIT_API_SECRET"))

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
