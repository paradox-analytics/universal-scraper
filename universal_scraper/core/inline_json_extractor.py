"""
Inline JSON Extractor - Extract JSON embedded directly in HTML body
Targeting Next.js 13+ RSC payloads, streaming data, and other inline patterns
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class InlineJSONExtractor:
    """
    Extracts JSON objects and arrays embedded directly in HTML text.
    Essential for modern frameworks (Next.js 13+, Nuxt 3) that embed data 
    in the HTML body rather than in script tags.
    """
    
    def __init__(self):
        # Patterns that suggest the start of interesting JSON data
        self.candidate_patterns = [
            # Next.js 13+ / GraphQL patterns
            r'"items":\s*\[\s*\{',
            r'"products":\s*\[\s*\{',
            r'"posts":\s*\[\s*\{',
            r'"results":\s*\[\s*\{',
            r'\{"__typename":\s*"[A-Z]\w+"',
            
            # Common data patterns
            r'"props":\s*\{',
            r'"pageProps":\s*\{',
            r'"state":\s*\{',
            
            # Streaming patterns
            r'self\.__next_f\.push\('
        ]
        
        # Patterns to ignore (analytics, tracking, etc.)
        self.ignore_patterns = [
            r'google_tag_manager',
            r'gtag\(',
            r'fbq\(',
            r'analytics',
            r'sentry',
            r'datadog',
            r'intercom',
            r'segment',
            r'hotjar'
        ]

    def extract(self, html: str, url: str = "") -> List[Dict[str, Any]]:
        """
        Main entry point: Extract all valid, high-quality JSON objects from HTML
        """
        results = []
        
        # 1. Extract from Next.js RSC streaming format
        rsc_data = self._extract_rsc_payload(html)
        if rsc_data:
            results.extend(rsc_data)
            logger.info(f" Extracted {len(rsc_data)} items from Next.js RSC payload")
            
        # 2. Scan for inline JSON objects/arrays
        inline_data = self._scan_for_inline_json(html)
        if inline_data:
            results.extend(inline_data)
            logger.info(f" Extracted {len(inline_data)} items from inline JSON scan")
            
        return results

    def _extract_rsc_payload(self, html: str) -> List[Dict[str, Any]]:
        """
        Extract data from Next.js 13+ self.__next_f.push() calls
        """
        results = []
        pattern = r'self\.__next_f\.push\((.*?)\)\s*(?:</script>|$)'
        matches = re.findall(pattern, html, re.DOTALL)
        
        for match in matches:
            try:
                # The match is usually a JSON array like: [1,"some data"]
                # or [1, "1:[\"$\",\"div\",null,{\"children\":...}]"]
                parsed = json.loads(match)
                
                if isinstance(parsed, list) and len(parsed) >= 2:
                    chunk_data = parsed[1]
                    
                    if isinstance(chunk_data, str):
                        # RSC data is often a string containing JSON-like structures
                        # We need to find the actual data objects within this string
                        extracted_objects = self._extract_objects_from_string(chunk_data)
                        results.extend(extracted_objects)
                        
            except json.JSONDecodeError:
                continue
                
        return results

    def _scan_for_inline_json(self, html: str) -> List[Dict[str, Any]]:
        """
        Scan HTML for JSON-like patterns and extract balanced blocks
        """
        results = []
        seen_hashes = set()
        
        # Limit scan to reasonable size to avoid performance issues
        # Most critical data is usually in the first 2MB
        scan_limit = 2 * 1024 * 1024
        content_to_scan = html[:scan_limit]
        
        for pattern in self.candidate_patterns:
            for match in re.finditer(pattern, content_to_scan):
                start_pos = match.start()
                
                # If we matched a key like "items":, we want to start extracting
                # from the value (the [ or {)
                value_start = content_to_scan.find(':', start_pos) + 1
                while value_start < len(content_to_scan) and content_to_scan[value_start].isspace():
                    value_start += 1
                
                if value_start >= len(content_to_scan):
                    continue
                    
                # Extract the full JSON object/array
                json_str = self._extract_balanced_block(content_to_scan, value_start)
                
                if json_str:
                    try:
                        data = json.loads(json_str)
                        
                        # Validate quality
                        if self._validate_content_quality(data):
                            # Deduplicate
                            data_hash = hash(json.dumps(data, sort_keys=True))
                            if data_hash not in seen_hashes:
                                results.append({
                                    '_source': 'inline_json',
                                    '_pattern': pattern,
                                    'data': data
                                })
                                seen_hashes.add(data_hash)
                                
                    except json.JSONDecodeError:
                        continue
                        
        return results

    def _extract_objects_from_string(self, text: str) -> List[Dict[str, Any]]:
        """
        Find JSON objects embedded within a string (common in RSC payloads)
        """
        results = []
        
        # Look for GraphQL-like objects with __typename
        # This is a strong signal for data
        pattern = r'\{[^{}]*"__typename"\s*:\s*"[^"]+"'
        
        for match in re.finditer(pattern, text):
            start_pos = match.start()
            json_str = self._extract_balanced_block(text, start_pos)
            
            if json_str:
                try:
                    data = json.loads(json_str)
                    if self._validate_content_quality(data):
                        results.append({
                            '_source': 'rsc_payload',
                            'data': data
                        })
                except json.JSONDecodeError:
                    continue
                    
        return results

    def _extract_balanced_block(self, text: str, start_index: int) -> Optional[str]:
        """
        Extract a balanced JSON block ({...} or [...]) starting at start_index
        """
        if start_index >= len(text):
            return None
            
        start_char = text[start_index]
        if start_char == '{':
            end_char = '}'
        elif start_char == '[':
            end_char = ']'
        else:
            return None
            
        stack = 1
        in_string = False
        escape = False
        
        for i in range(start_index + 1, len(text)):
            char = text[i]
            
            if escape:
                escape = False
                continue
                
            if char == '\\':
                escape = True
                continue
                
            if char == '"':
                in_string = not in_string
                continue
                
            if not in_string:
                if char == start_char:
                    stack += 1
                elif char == end_char:
                    stack -= 1
                    if stack == 0:
                        return text[start_index:i+1]
                        
        return None

    def _validate_content_quality(self, data: Any) -> bool:
        """
        Check if extracted data looks like useful content (not analytics/junk)
        """
        if isinstance(data, list):
            # Arrays should have multiple items
            if len(data) < 2:
                return False
            # Check first item
            return self._validate_content_quality(data[0])
            
        if isinstance(data, dict):
            # Check for analytics keywords
            json_str = json.dumps(data).lower()
            for pattern in self.ignore_patterns:
                if re.search(pattern, json_str):
                    return False
            
            # Must have some fields
            if len(data.keys()) < 2:
                return False
                
            # Bonus: Check for common content fields
            common_fields = ['id', 'name', 'title', 'description', 'slug', 'price', 'image']
            has_common_field = any(f in data for f in common_fields)
            
            # Bonus: Check for GraphQL typename
            has_typename = '__typename' in data
            
            return has_common_field or has_typename
            
        return False
