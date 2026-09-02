"""
Pydantic schemas for Document request validation and response serialisation.

DocumentCreate — validates incoming POST /documents requests.
DocumentResponse — serialises database rows back to API consumers.
"""

from datetime import datetime
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator


class DocumentCreate(BaseModel):
    """
    Schema for creating a new document via POST /documents.

    The client provides url, title, text, and optionally canonical_url.
    All other fields (id, domain, word_count, content_hash, timestamps,
    status) are derived by the backend.
    """

    url: str
    title: str
    text: str
    canonical_url: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Reject empty or obviously malformed URLs."""
        v = v.strip()
        if not v:
            raise ValueError("URL must not be empty.")
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Unsupported URL scheme '{parsed.scheme}'. "
                "Only http and https are accepted."
            )
        if not parsed.netloc:
            raise ValueError("URL is missing a hostname.")
        return v

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Reject empty titles."""
        v = v.strip()
        if not v:
            raise ValueError("Title must not be empty.")
        return v

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Reject empty body text."""
        v = v.strip()
        if not v:
            raise ValueError("Text must not be empty.")
        return v


class DocumentResponse(BaseModel):
    """
    Full document representation returned by the API.

    Configured with ``from_attributes=True`` so Pydantic can serialise
    SQLAlchemy model instances directly.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    canonical_url: str | None
    domain: str
    title: str
    text: str
    word_count: int
    content_hash: str
    created_at: datetime
    updated_at: datetime
    status: str
