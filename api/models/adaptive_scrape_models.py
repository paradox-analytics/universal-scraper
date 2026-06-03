from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl

class AdaptiveScrapeRequest(BaseModel):
    """Request model for adaptive scraping"""
    url: HttpUrl = Field(..., description="Target URL to scrape")
    fields: Optional[List[str]] = Field(None, description="List of fields to extract")
    max_attempts: int = Field(5, description="Maximum number of adaptive attempts")
    initial_preset: str = Field("balanced", description="Initial configuration preset (stealth, balanced, aggressive)")
    use_cache: bool = Field(True, description="Whether to use cached results")

    # Optional overrides
    proxy_config: Optional[Dict[str, Any]] = Field(None, description="Custom proxy configuration")
    webhook_url: Optional[HttpUrl] = Field(None, description="Webhook URL for async results")

class AdaptiveScrapeResponse(BaseModel):
    """Response model for adaptive scraping"""
    success: bool = Field(..., description="Whether the scrape was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Extracted data")
    html: Optional[str] = Field(None, description="Raw HTML content (if requested)")

    # Metadata
    status_code: int = Field(..., description="HTTP status code")
    strategy_used: str = Field(..., description="Strategy that succeeded (learned, stealth, etc.)")
    attempts_made: int = Field(..., description="Number of attempts made")
    time_taken: float = Field(..., description="Time taken in seconds")

    # Insights
    blocking_analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis of blocking encountered")
    recommendations: Optional[List[str]] = Field(None, description="Recommendations for future scrapes")
