from pydantic import BaseModel, ConfigDict, Field


class CanvasExportRequest(BaseModel):
    """Payload for POST /api/canvas/export."""
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, description="Raw Markdown content to export as a Google Doc")
    title: str = Field("Untitled Canvas Document", min_length=1, description="Title for the created Google Doc")


class CanvasExportResponse(BaseModel):
    """Payload for POST /api/canvas/export."""
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., description="The Google Drive file id of the newly created Google Doc")
    web_view_link: str = Field(..., description="A browser-openable URL to the newly created Google Doc")
