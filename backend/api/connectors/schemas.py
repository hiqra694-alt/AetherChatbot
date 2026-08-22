from enum import Enum
from pydantic import BaseModel, Field


class OAuthProvider(str, Enum):
    GOOGLE = "google"
    GITHUB = "github"
    GOOGLE_WORKSPACE = "google_workspace"


class StoreTokenRequest(BaseModel):
    """
    Payload for POST /api/connectors/store-token. Sent once, right after a
    successful linkIdentity redirect (see src/app/page.tsx's mount effect),
    with whatever provider_refresh_token Supabase attached to that session --
    only present when the linkIdentity call requested offline access
    (queryParams: access_type=offline, prompt=consent).
    """
    provider: OAuthProvider = Field(..., description="Which OAuth provider this refresh token belongs to")
    refresh_token: str = Field(..., min_length=1, description="The long-lived OAuth refresh token to store")


class StoreTokenResponse(BaseModel):
    """Payload for POST /api/connectors/store-token."""
    stored: bool = Field(..., description="Whether the refresh token was saved")


class DisconnectResponse(BaseModel):
    """Payload for DELETE /api/connectors/disconnect."""
    status: str = Field("success", description="Outcome status")
    message: str = Field(..., description="Human-readable outcome message")


class ConnectorStatusResponse(BaseModel):
    """
    Payload for GET /api/connectors/status. Reports, per provider, whether
    this user has a *stored* refresh_token the backend can use to mint fresh
    access tokens on its own (mcp_manager._refresh_google_access_token) --
    not merely whether the provider identity is linked (see
    src/app/page.tsx's `linked` state, sourced from user.identities).

    The two can diverge: a provider identity linked before this refresh
    fallback existed (or via a plain sign-in through that provider rather
    than this app's own connector-linking flow) is "linked" but has no
    stored refresh_token, since Supabase's linkIdentity() only ever
    requests/returns one when explicitly asked for offline access, and
    linkIdentity() itself refuses to re-run for an identity already linked
    to the current user -- so the frontend can't just retry linking to fix
    this. It needs this endpoint to tell it a disconnect+reconnect is
    actually required, rather than silently toggling a connector on that
    can never open a Workspace session.
    """
    google: bool = Field(..., description="Whether a usable Google refresh_token is stored server-side")
    github: bool = Field(..., description="Whether a usable GitHub refresh_token is stored server-side")
    google_workspace: bool = Field(
        ...,
        description=(
            "Whether a usable native Google Workspace refresh_token (Gmail/Calendar/Drive, see "
            "api/connectors/google_oauth.py) is stored server-side -- distinct from `google` above, "
            "which reflects Supabase's own 'google' sign-in identity, not this connector."
        ),
    )
