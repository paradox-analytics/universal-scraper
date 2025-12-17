"""
Content Type Detector
Detects whether a URL points to HTML, PDF, DOCX, or other content types
"""

import logging
import requests
from typing import Tuple, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ContentDetector:
    """
    Detect content type from URL or response headers
    """
    
    SUPPORTED_TYPES = {
        'text/html': 'html',
        'application/xhtml+xml': 'html',
        'application/pdf': 'pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
        'application/json': 'json',
        'text/plain': 'text',
    }
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def detect(self, url: str, headers: Optional[dict] = None) -> Tuple[str, Optional[str]]:
        """
        Detect content type from URL
        
        Args:
            url: Target URL
            headers: Optional request headers
            
        Returns:
            Tuple of (content_type, content_subtype)
            Examples: ('html', None), ('pdf', None), ('html', 'application/json')
        """
        # First, check file extension
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        if path.endswith('.pdf'):
            logger.info(f" Detected PDF from URL extension: {url}")
            return ('pdf', None)
        elif path.endswith('.docx'):
            logger.info(f" Detected DOCX from URL extension: {url}")
            return ('docx', None)
        elif path.endswith('.xlsx'):
            logger.info(f" Detected XLSX from URL extension: {url}")
            return ('xlsx', None)
        
        # If no extension match, make HEAD request to check Content-Type header
        try:
            head_headers = headers or {}
            head_headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            response = requests.head(
                url, 
                headers=head_headers,
                timeout=self.timeout,
                allow_redirects=True
            )
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Parse content type (remove charset, etc.)
            if ';' in content_type:
                content_type = content_type.split(';')[0].strip()
            
            # Map to our supported types
            for mime_type, type_name in self.SUPPORTED_TYPES.items():
                if mime_type in content_type:
                    logger.info(f" Detected {type_name.upper()} from Content-Type header: {url}")
                    return (type_name, None)
            
            # Default to HTML if text/html-like or unknown
            if 'html' in content_type or 'text' in content_type:
                return ('html', None)
            
            logger.warning(f"  Unknown content type: {content_type}, defaulting to HTML")
            return ('html', None)
            
        except Exception as e:
            logger.warning(f"  Failed to detect content type via HEAD request: {e}")
            logger.info("   Defaulting to HTML")
            return ('html', None)
    
    def is_pdf(self, url: str) -> bool:
        """Quick check if URL is PDF"""
        content_type, _ = self.detect(url)
        return content_type == 'pdf'
    
    def is_document(self, url: str) -> bool:
        """Check if URL is a document (PDF, DOCX, XLSX)"""
        content_type, _ = self.detect(url)
        return content_type in ['pdf', 'docx', 'xlsx']


