"""
Pydantic schemas for search API responses.
"""

from pydantic import BaseModel


class SearchResultItem(BaseModel):
    """A single search result in the API response."""

    document_id: int
    title: str
    url: str
    score: float


class SearchResponse(BaseModel):
    """Full search API response with pagination."""

    query: str
    total: int
    page: int
    limit: int
    results: list[SearchResultItem]
    took_ms: float
