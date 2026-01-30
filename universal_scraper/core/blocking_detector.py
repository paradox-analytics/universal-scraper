"""
Blocking Detector

Detects and classifies different types of anti-bot protection and blocking.
Supports Cloudflare, Kasada, DataDome, CAPTCHAs, and generic rate limiting.
"""

import re
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class BlockingType(Enum):
    """Types of blocking detected"""
    NONE = "none"
    CLOUDFLARE = "cloudflare"
    KASADA = "kasada"
    DATADOME = "datadome"
    PERIMETER_X = "perimeterx"
    RECAPTCHA = "recaptcha"
    HCAPTCHA = "hcaptcha"
    RATE_LIMIT = "rate_limit"
    IP_BAN = "ip_ban"
    AKAMAI = "akamai"
    GENERIC_BLOCK = "generic_block"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"


class BlockingDetector:
    """
    Detects various types of anti-bot protection and blocking mechanisms.
    
    Analyzes HTML content, status codes, headers, and error messages to
    identify the specific blocking technology in use.
    """
    
    # Detection patterns for different blocking types
    PATTERNS = {
        BlockingType.CLOUDFLARE: [
            r'cloudflare',
            r'cf-ray',
            r'__cf_bm',
            r'cf_clearance',
            r'checking your browser',
            r'ddos protection by cloudflare',
            r'attention required.*cloudflare',
        ],
        BlockingType.KASADA: [
            r'kasada',
            r'x-kpsdk',
            r'/_kpsdk/',
            r'kpsdk-ct',
            r'window\.KPSDK',
        ],
        BlockingType.DATADOME: [
            r'datadome',
            r'dd-request-id',
            r'datadome\.co',
            r'window\.DD',
        ],
        BlockingType.AKAMAI: [
            r'akamai',
            r'edgesuite\.net',
            r'ak-grs',
            r'reference #\d+\.\w+',
        ],
        BlockingType.PERIMETER_X: [
            r'perimeterx',
            r'_px\d',
            r'pxhd',
            r'px-captcha',
        ],
        BlockingType.RECAPTCHA: [
            r'recaptcha',
            r'google\.com/recaptcha',
            r'g-recaptcha',
        ],
        BlockingType.HCAPTCHA: [
            r'hcaptcha',
            r'h-captcha',
        ],
        BlockingType.RATE_LIMIT: [
            r'rate limit',
            r'too many requests',
            r'slow down',
            r'429',
        ],
        BlockingType.IP_BAN: [
            r'ip.*banned',
            r'ip.*blocked',
            r'access denied',
            r'forbidden',
        ],
    }
    
    def __init__(self):
        """Initialize blocking detector"""
        logger.info("🔍 Blocking Detector initialized")
    
    def detect(
        self,
        html: str = "",
        status_code: int = 0,
        headers: Optional[Dict[str, str]] = None,
        error_message: str = ""
    ) -> Dict[str, Any]:
        """
        Detect blocking type from response data.
        
        Args:
            html: HTML content
            status_code: HTTP status code
            headers: Response headers
            error_message: Error message if any
            
        Returns:
            Detection result with type, confidence, and details
        """
        headers = headers or {}
        
        # Check for timeout/connection errors first
        if error_message:
            if 'timeout' in error_message.lower():
                return self._create_result(BlockingType.TIMEOUT, 1.0, "Request timed out")
            if 'connection' in error_message.lower() or 'refused' in error_message.lower():
                return self._create_result(BlockingType.CONNECTION_ERROR, 1.0, "Connection error")
        
        # Check status codes
        if status_code == 403:
            return self._detect_403_blocking(html, headers)
        elif status_code == 429:
            return self._create_result(BlockingType.RATE_LIMIT, 1.0, "HTTP 429 Too Many Requests")
        elif status_code in [503, 504]:
            return self._detect_503_blocking(html, headers)
        elif status_code == 0:
            return self._create_result(BlockingType.CONNECTION_ERROR, 0.8, "No response received")
        
        # Check HTML content for blocking patterns
        if html:
            blocking_type = self._detect_from_html(html, headers)
            if blocking_type != BlockingType.NONE:
                return blocking_type
        
        # Check headers
        blocking_type = self._detect_from_headers(headers)
        if blocking_type != BlockingType.NONE:
            return blocking_type
        
        # No blocking detected
        return self._create_result(BlockingType.NONE, 1.0, "No blocking detected")
    
    def _detect_403_blocking(self, html: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Detect specific blocking type for 403 responses"""
        # Check for Cloudflare
        if any(key.lower().startswith('cf-') for key in headers.keys()):
            return self._create_result(BlockingType.CLOUDFLARE, 0.9, "Cloudflare 403 block")
        
        # Check HTML patterns
        html_lower = html.lower()
        
        for blocking_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, html_lower, re.IGNORECASE):
                    return self._create_result(blocking_type, 0.8, f"403 with {blocking_type.value} detected")
        
        # Generic 403
        return self._create_result(BlockingType.GENERIC_BLOCK, 0.7, "Generic 403 Forbidden")
    
    def _detect_503_blocking(self, html: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Detect specific blocking type for 503 responses"""
        # Cloudflare often returns 503
        if any(key.lower().startswith('cf-') for key in headers.keys()):
            return self._create_result(BlockingType.CLOUDFLARE, 0.9, "Cloudflare 503 challenge")
        
        return self._create_result(BlockingType.GENERIC_BLOCK, 0.6, "503 Service Unavailable")
    
    def _detect_from_html(self, html: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Detect blocking from HTML content"""
        html_lower = html.lower()
        
        # Check each blocking type
        for blocking_type, patterns in self.PATTERNS.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, html_lower, re.IGNORECASE):
                    matches += 1
            
            if matches > 0:
                confidence = min(0.6 + (matches * 0.1), 1.0)
                return self._create_result(
                    blocking_type,
                    confidence,
                    f"Detected {blocking_type.value} ({matches} pattern matches)"
                )
        
        return self._create_result(BlockingType.NONE, 1.0, "No blocking patterns in HTML")
    
    def _detect_from_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Detect blocking from response headers"""
        headers_lower = {k.lower(): v.lower() for k, v in headers.items()}
        
        # Cloudflare headers
        if any(h.startswith('cf-') for h in headers_lower.keys()):
            return self._create_result(BlockingType.CLOUDFLARE, 0.8, "Cloudflare headers detected")
        
        # DataDome headers
        if 'x-datadome' in headers_lower or 'dd-request-id' in headers_lower:
            return self._create_result(BlockingType.DATADOME, 0.9, "DataDome headers detected")
        
        # PerimeterX headers
        if any('px' in h for h in headers_lower.keys()):
            return self._create_result(BlockingType.PERIMETER_X, 0.7, "PerimeterX headers detected")
        
        return self._create_result(BlockingType.NONE, 1.0, "No blocking headers")
    
    def _create_result(
        self,
        blocking_type: BlockingType,
        confidence: float,
        details: str
    ) -> Dict[str, Any]:
        """Create a detection result"""
        return {
            'type': blocking_type,
            'type_name': blocking_type.value,
            'confidence': confidence,
            'details': details,
            'is_blocked': blocking_type != BlockingType.NONE
        }
    
    def get_bypass_recommendations(self, blocking_type: BlockingType) -> List[str]:
        """
        Get recommendations for bypassing a specific blocking type.
        
        Args:
            blocking_type: Type of blocking detected
            
        Returns:
            List of recommended actions
        """
        recommendations = {
            BlockingType.CLOUDFLARE: [
                "Use residential proxies",
                "Enable full fingerprinting randomization",
                "Rotate browser profiles",
                "Add random delays between requests",
                "Consider using Web Unblocker"
            ],
            BlockingType.KASADA: [
                "Use Web Unblocker (most effective)",
                "Enable advanced fingerprinting",
                "Use residential proxies",
                "Rotate proxies per request",
                "Enable JavaScript execution"
            ],
            BlockingType.DATADOME: [
                "Use residential proxies",
                "Enable canvas and audio fingerprinting",
                "Rotate browser profiles",
                "Add realistic mouse movements"
            ],
            BlockingType.RATE_LIMIT: [
                "Increase delays between requests",
                "Rotate proxies more frequently",
                "Use distributed scraping",
                "Reduce concurrency"
            ],
            BlockingType.TIMEOUT: [
                "Increase timeout duration",
                "Check proxy health",
                "Try different proxy zone",
                "Use faster proxy type (datacenter)"
            ],
            BlockingType.AKAMAI: [
                "Use Web Unblocker (highly recommended)",
                "Use high-quality residential proxies",
                "Disable stealth_mode (can trigger Akamai)",
                "Enable humanize features",
                "Rotate IP addresses frequently"
            ],
            BlockingType.CONNECTION_ERROR: [
                "Check proxy configuration",
                "Verify proxy credentials",
                "Try different proxy",
                "Check network connectivity"
            ]
        }
        
        return recommendations.get(blocking_type, [
            "Try different browser configuration",
            "Use residential proxies",
            "Enable anti-detection features"
        ])
