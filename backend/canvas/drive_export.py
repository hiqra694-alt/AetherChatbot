"""
Google Drive multipart upload for the Canvas export endpoint
(POST /api/canvas/export -- see canvas/router.py).
"""

import json
import logging
import uuid
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"

# The exact trigger that forces Drive to natively convert the raw Markdown
# upload into an editable Google Doc, rather than storing it as an opaque
# .md file -- see the metadata part built in _build_multipart_body.
GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"

# The source format of the uploaded bytes (the media part). Drive uses this,
# together with the metadata part's target mimeType above, to decide how to
# convert the content on import.
MARKDOWN_MIME_TYPE = "text/markdown"

_HTTP_TIMEOUT_SECONDS = 30.0


class DriveExportError(Exception):
    """Raised when the Drive API rejects or fails an export upload."""


def _build_multipart_body(*, title: str, content: str, boundary: str) -> bytes:
    """
    Hand-builds a `multipart/related` body per Drive's multipart upload spec
    (https://developers.google.com/drive/api/guides/manage-uploads#multipart) --
    httpx's own `files=` helper builds `multipart/form-data` with per-part
    Content-Disposition headers Drive doesn't expect, so this constructs the
    two required parts (a JSON metadata part, then the raw media part)
    directly instead.
    """
    metadata = {"name": title, "mimeType": GOOGLE_DOC_MIME_TYPE}
    parts = [
        f"--{boundary}\r\n"
        f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{json.dumps(metadata)}\r\n",
        f"--{boundary}\r\n"
        f"Content-Type: {MARKDOWN_MIME_TYPE}; charset=UTF-8\r\n\r\n"
        f"{content}\r\n",
        f"--{boundary}--",
    ]
    return "".join(parts).encode("utf-8")


async def export_markdown_as_google_doc(access_token: str, content: str, title: str) -> dict:
    """
    Uploads `content` (raw Markdown) to Google Drive as a new file whose
    metadata part declares mimeType=application/vnd.google-apps.document --
    the signal that makes Drive natively convert the Markdown into an
    editable Google Doc on import, rather than storing it as a plain file.

    Returns the parsed Drive API response (contains at least `id`, and
    `webViewLink` since both are explicitly requested via `fields`). Raises
    DriveExportError on any non-2xx response or network failure -- the
    caller (canvas/router.py) is responsible for turning that into a
    user-facing error.
    """
    boundary = uuid.uuid4().hex
    body = _build_multipart_body(title=title, content=content, boundary=boundary)

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                DRIVE_UPLOAD_URL,
                params={"uploadType": "multipart", "fields": "id,webViewLink"},
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": f"multipart/related; boundary={boundary}",
                },
                content=body,
                timeout=_HTTP_TIMEOUT_SECONDS,
            )
    except httpx.HTTPError as exc:
        raise DriveExportError("Network error while uploading to Google Drive.") from exc

    if response.status_code >= 400:
        logger.error("Canvas Drive export failed (%s): %s", response.status_code, response.text)
        raise DriveExportError(f"Google Drive rejected the upload (HTTP {response.status_code}).")

    payload = response.json()
    if not payload.get("id"):
        raise DriveExportError("Google Drive's response did not include a document id.")

    return payload
