"""
JSON Detector - Priority JSON Extraction
Detects and extracts JSON data before resorting to HTML parsing
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class JSONDetector:
    """Detects and extracts JSON data from various sources"""
    
    def __init__(self):
        self.json_sources = []
    
    def detect_and_extract(self, html: str, url: str) -> Dict[str, Any]:
        """
        Detect and extract JSON from multiple sources
        
        Args:
            html: Raw HTML content
            url: Source URL for context
            
        Returns:
            Dict with 'json_found', 'sources', 'data' keys
        """
        results = {
            'json_found': False,
            'sources': [],
            'data': []
        }
        
        # Priority 1: JSON-LD structured data
        json_ld_data = self._extract_json_ld(html)
        if json_ld_data:
            results['json_found'] = True
            results['sources'].append('json-ld')
            results['data'].extend(json_ld_data)
            logger.info(f"✅ Found {len(json_ld_data)} JSON-LD objects")
        
        # Priority 2: Embedded JSON in script tags
        embedded_json = self._extract_embedded_json(html)
        if embedded_json:
            results['json_found'] = True
            results['sources'].append('embedded-json')
            results['data'].extend(embedded_json)
            logger.info(f"✅ Found {len(embedded_json)} embedded JSON objects")
        
        # Priority 3: Next.js/React props
        nextjs_data = self._extract_nextjs_data(html)
        if nextjs_data:
            results['json_found'] = True
            results['sources'].append('nextjs')
            results['data'].append(nextjs_data)
            logger.info("✅ Found Next.js data")
        
        # Priority 4: GraphQL detection (endpoints only, not data)
        graphql_endpoints = self._detect_graphql(html)
        if graphql_endpoints:
            results['sources'].append('graphql')
            results['data'].append({'graphql_endpoints': graphql_endpoints})
            logger.info(f"✅ Found {len(graphql_endpoints)} GraphQL endpoint(s)")
        
        # Priority 5: API endpoints in HTML
        api_endpoints = self._detect_api_endpoints(html, url)
        if api_endpoints:
            results['sources'].append('api-endpoints')
            results['data'].append({'api_endpoints': api_endpoints})
            logger.info(f"✅ Found {len(api_endpoints)} API endpoint(s)")
        
        if not results['json_found'] and not results['sources']:
            logger.info("❌ No JSON sources detected, will use HTML parsing")
        
        return results
    
    def _extract_json_ld(self, html: str) -> List[Dict]:
        """Extract JSON-LD structured data"""
        soup = BeautifulSoup(html, 'html.parser')
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        
        json_data = []
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                json_data.append(data)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON-LD: {str(e)[:50]}")
                continue
        
        return json_data
    
    def _extract_embedded_json(self, html: str) -> List[Dict]:
        """Extract JSON embedded in JavaScript variables"""
        json_data = []
        
        # Common patterns for embedded JSON
        patterns = [
            r'var\s+\w+\s*=\s*(\{.+?\});',
            r'const\s+\w+\s*=\s*(\{.+?\});',
            r'let\s+\w+\s*=\s*(\{.+?\});',
            r'window\.\w+\s*=\s*(\{.+?\});',
            r'data\s*:\s*(\{.+?\}),',
            r'props\s*:\s*(\{.+?\}),',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, html, re.DOTALL)
            for match in matches:
                try:
                    json_str = match.group(1)
                    # Try to parse as JSON
                    data = json.loads(json_str)
                    if isinstance(data, dict) and len(data) > 0:
                        json_data.append(data)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        return json_data
    
    def _extract_nextjs_data(self, html: str) -> Optional[Dict]:
        """Extract Next.js __NEXT_DATA__ object"""
        # Look for __NEXT_DATA__ script
        pattern = r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>'
        match = re.search(pattern, html, re.DOTALL)
        
        if match:
            try:
                data = json.loads(match.group(1))
                return data
            except json.JSONDecodeError:
                logger.debug("Failed to parse __NEXT_DATA__")
        
        return None
    
    def _detect_graphql(self, html: str) -> List[str]:
        """Detect GraphQL endpoints"""
        endpoints = []
        
        # Common GraphQL endpoint patterns
        patterns = [
            r'["\']([^"\']*?graphql[^"\']*?)["\']',
            r'["\']([^"\']*?/gql[^"\']*?)["\']',
            r'endpoint\s*:\s*["\']([^"\']+)["\']',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, html, re.IGNORECASE)
            for match in matches:
                endpoint = match.group(1)
                if endpoint and endpoint not in endpoints:
                    # Filter out false positives
                    if 'graphql' in endpoint.lower() or '/gql' in endpoint.lower():
                        endpoints.append(endpoint)
        
        return endpoints
    
    def _detect_api_endpoints(self, html: str, base_url: str) -> List[str]:
        """Detect REST API endpoints"""
        endpoints = []
        
        # Common API patterns
        patterns = [
            r'["\']([^"\']*?/api/[^"\']+)["\']',
            r'["\']([^"\']*?/v\d+/[^"\']+)["\']',
            r'fetch\(["\']([^"\']+)["\']',
            r'axios\.get\(["\']([^"\']+)["\']',
            r'\.get\(["\']([^"\']+)["\']',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, html)
            for match in matches:
                endpoint = match.group(1)
                if endpoint and endpoint not in endpoints:
                    # Convert to absolute URL if relative
                    if not endpoint.startswith('http'):
                        endpoint = urljoin(base_url, endpoint)
                    
                    # Filter out non-API URLs
                    parsed = urlparse(endpoint)
                    if any(api_indicator in parsed.path.lower() for api_indicator in ['/api/', '/v1/', '/v2/', '/v3/']):
                        endpoints.append(endpoint)
        
        return endpoints
    
    def extract_from_json(self, json_data: List[Dict], fields: List[str]) -> List[Dict]:
        """
        Extract specified fields from JSON data
        
        Args:
            json_data: List of JSON objects
            fields: Fields to extract
            
        Returns:
            List of dicts with extracted data
        """
        extracted = []
        
        for data in json_data:
            item = {}
            for field in fields:
                value = self._find_field_in_json(data, field)
                if value is not None:
                    item[field] = value
            
            if item:  # Only add if we found at least one field
                extracted.append(item)
        
        return extracted
    
    def _find_field_in_json(self, data: Any, field: str) -> Any:
        """Recursively find field in JSON structure"""
        if isinstance(data, dict):
            # Direct match
            if field in data:
                return data[field]
            
            # Case-insensitive match
            for key in data.keys():
                if key.lower() == field.lower():
                    return data[key]
            
            # Partial match (e.g., 'name' matches 'product_name')
            for key in data.keys():
                if field.lower() in key.lower() or key.lower() in field.lower():
                    return data[key]
            
            # Recurse into nested dicts
            for value in data.values():
                result = self._find_field_in_json(value, field)
                if result is not None:
                    return result
        
        elif isinstance(data, list):
            # Check first item in list
            if data and len(data) > 0:
                return self._find_field_in_json(data[0], field)
        
        return None
    
    def is_json_sufficient(self, json_results: Dict, fields: List[str]) -> bool:
        """
        Check if JSON data contains sufficient information
        
        Args:
            json_results: Results from detect_and_extract
            fields: Required fields
            
        Returns:
            True if JSON contains most of the required fields
        """
        if not json_results['json_found']:
            return False
        
        # Extract data from JSON
        extracted = self.extract_from_json(json_results['data'], fields)
        
        if not extracted:
            return False
        
        # Check if we have at least 50% of requested fields
        sample_item = extracted[0]
        fields_found = len(sample_item)
        fields_required = len(fields)
        
        coverage = fields_found / fields_required if fields_required > 0 else 0
        
        logger.info(f"📊 JSON field coverage: {coverage:.1%} ({fields_found}/{fields_required})")
        
        return coverage >= 0.5  # At least 50% coverage

