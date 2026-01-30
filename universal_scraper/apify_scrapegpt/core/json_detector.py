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
from universal_scraper.core.inline_json_extractor import InlineJSONExtractor

logger = logging.getLogger(__name__)


class JSONDetector:
    """
    Universal JSON Detector - Pattern-based JSON extraction
    Works for ANY website with embedded JSON (Next.js, Nuxt, React, Vue, Angular, etc.)
    """
    
    # Universal patterns for embedded JSON frameworks
    JSON_PATTERNS = {
        'nextjs': {
            'pattern': r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            'paths': ['props.pageProps.menuData', 'props.pageProps.data', 'props.pageProps', 'props.initialProps', 'props'],
            'priority': 1
        },
        'nuxtjs': {
            'pattern': r'window\.__NUXT__\s*=\s*(.*?);',
            'paths': ['data', 'state', 'fetch'],
            'priority': 2
        },
        'react_apollo': {
            'pattern': r'window\.__APOLLO_STATE__\s*=\s*(.*?);',
            'paths': ['ROOT_QUERY', 'data'],
            'priority': 3
        },
        'react_redux': {
            'pattern': r'window\.__INITIAL_STATE__\s*=\s*(.*?);',
            'paths': ['data', 'entities', 'state'],
            'priority': 4
        },
        'vue_initial': {
            'pattern': r'window\.__INITIAL_DATA__\s*=\s*(.*?);',
            'paths': ['data', 'state'],
            'priority': 5
        },
        'generic_window': {
            'pattern': r'window\.(?:initialData|pageData|appData|serverData)\s*=\s*(\{.*?\});',
            'paths': ['data', 'items', 'products', 'results'],
            'priority': 6
        }
    }
    
    # Common item array field names (universal patterns)
    ITEM_ARRAY_FIELDS = [
        'items', 'products', 'results', 'data', 'list', 'entries',
        'menuItems', 'menuitems', 'menu_items', 'listings', 'posts', 'articles', 'records',
        'content', 'children', 'nodes', 'edges'
    ]
    
    def __init__(self):
        self.json_sources = []
        self.discovered_patterns = {}  # Cache successful patterns
        self.inline_extractor = InlineJSONExtractor()
    
    def detect_and_extract(self, html: str, url: str, captured_json: Optional[List] = None) -> Dict[str, Any]:
        """
        Detect and extract JSON from multiple sources
        
        Universal approach: Treat all JSON the same, whether from HTML or API responses
        
        Args:
            html: Raw HTML content
            url: Source URL for context
            captured_json: Optional list of JSON blobs (from API responses, etc.)
            
        Returns:
            Dict with 'json_found', 'sources', 'data' keys
        """
        results = {
            'json_found': False,
            'sources': [],
            'data': []
        }
        
        # Priority 0: Captured JSON blobs (from API responses during pagination)
        if captured_json:
            for idx, json_blob in enumerate(captured_json):
                try:
                    # If it's a string, try to parse it
                    if isinstance(json_blob, str):
                        try:
                            json_blob = json.loads(json_blob)
                        except json.JSONDecodeError:
                            logger.debug(f"Skipping non-JSON blob {idx}")
                            continue
                    
                    # Only process dict or list
                    if not isinstance(json_blob, (dict, list)):
                        continue
                    
                    # Wrap in framework metadata (treat as generic captured JSON)
                    results['json_found'] = True
                    results['sources'].append(f"captured_json_{idx}")
                    results['data'].append({
                        '_framework': 'captured',
                        '_data': json_blob,
                        '_paths': []
                    })
                    logger.info(f" Found captured JSON blob #{idx + 1}")
                except (TypeError, AttributeError) as e:
                    logger.debug(f"Skipping invalid JSON blob {idx}: {e}")
                    continue
        
        # Priority 1: JSON-LD structured data
        json_ld_data = self._extract_json_ld(html)
        if json_ld_data:
            results['json_found'] = True
            results['sources'].append('json-ld')
            results['data'].extend(json_ld_data)
            logger.info(f" Found {len(json_ld_data)} JSON-LD objects")
        
        # Priority 2: Embedded JSON in script tags
        embedded_json = self._extract_embedded_json(html)
        if embedded_json:
            results['json_found'] = True
            results['sources'].append('embedded-json')
            results['data'].extend(embedded_json)
            logger.info(f" Found {len(embedded_json)} embedded JSON objects")
        
        # Priority 3: Next.js/React props
        nextjs_data = self._extract_nextjs_data(html)
        if nextjs_data:
            results['json_found'] = True
            results['sources'].append('nextjs')
            results['data'].append(nextjs_data)
            logger.info(" Found Next.js data")
        
        # Priority 4: Inline JSON (Next.js 13+ RSC, etc.)
        inline_data = self._extract_inline_json(html)
        if inline_data:
            results['json_found'] = True
            results['sources'].append('inline-json')
            results['data'].extend(inline_data)
            logger.info(f" Found {len(inline_data)} inline JSON objects (Next.js 13+ / RSC)")
        
        # Priority 5: GraphQL detection (endpoints only, not data)
        graphql_endpoints = self._detect_graphql(html)
        if graphql_endpoints:
            results['sources'].append('graphql')
            results['data'].append({'graphql_endpoints': graphql_endpoints})
            logger.info(f" Found {len(graphql_endpoints)} GraphQL endpoint(s)")
        
        # Priority 6: API endpoints in HTML
        api_endpoints = self._detect_api_endpoints(html, url)
        if api_endpoints:
            results['sources'].append('api-endpoints')
            results['data'].append({'api_endpoints': api_endpoints})
            logger.info(f" Found {len(api_endpoints)} API endpoint(s)")
        
        if not results['json_found'] and not results['sources']:
            logger.info(" No JSON sources detected, will use HTML parsing")
        
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
        """
        Universal framework data extraction
        Extracts from Next.js, Nuxt, React, Vue, Angular patterns
        """
        # Try all patterns in priority order
        for pattern_name, config in sorted(
            self.JSON_PATTERNS.items(),
            key=lambda x: x[1]['priority']
        ):
            pattern = config['pattern']
            match = re.search(pattern, html, re.DOTALL)
            
            if match:
                try:
                    json_str = match.group(1)
                    data = json.loads(json_str)
                    
                    # Cache successful pattern for this session
                    self.discovered_patterns[pattern_name] = config
                    
                    logger.info(f" Extracted {pattern_name} data")
                    return {
                        '_framework': pattern_name,
                        '_data': data,
                        '_paths': config['paths']
                    }
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse {pattern_name} data")
                    continue
        
        return None
    
    def _extract_inline_json(self, html: str) -> List[Dict]:
        """
        Extract inline JSON using the dedicated InlineJSONExtractor
        Handles Next.js 13+ RSC payloads, streaming data, and other embedded patterns
        """
        try:
            return self.inline_extractor.extract(html)
        except Exception as e:
            logger.error(f"Error extracting inline JSON: {e}")
            return []
    
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
    
    def extract_from_json(self, json_data: List[Dict], fields: List[str], llm_field_mappings: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        UNIVERSAL JSON extraction - uses semantic field mapping like HTML extraction
        Works for ANY JSON structure (Next.js, Nuxt, React, etc.)
        
        Args:
            json_data: JSON data to extract from (can be list of items or nested structure)
            fields: Fields to extract
            llm_field_mappings: Optional LLM-discovered field mappings from JSONStructureAnalyzer
                              Format: {"field": {"json_key": "actual_key", "path": "nested.path", "confidence": 0.0-1.0}}
                              When provided, used as fallback when hardcoded synonyms fail (more universal)
        
        This now follows the same methodology as HTML extraction:
        1. Find item arrays (like finding containers in HTML)
        2. Use semantic field mapping (like semantic strategies in HTML)
        3. Apply field validation and normalization
        
        Args:
            json_data: List of JSON objects
            fields: Fields to extract
            
        Returns:
            List of dicts with extracted data
        """
        extracted = []
        
        # OPTIMIZATION: Score all sources and extract only from the BEST one
        # This prevents mixing products with filters/navigation/metadata
        source_candidates = []  # List of (score, items, source_name) tuples
        
        for idx, data in enumerate(json_data):
            source_items = []
            source_name = f"source_{idx}"
            
            # Check if this is framework data with metadata
            if isinstance(data, dict) and '_framework' in data:
                framework = data['_framework']
                actual_data = data['_data']
                suggested_paths = data['_paths']
                source_name = framework
                
                logger.info(f" Analyzing {framework} data structure")
                
                # Try suggested paths first
                for path in suggested_paths:
                    items_array = self._get_nested_value(actual_data, path)
                    if items_array:
                        # Check if this path leads to an items array
                        found_items = self._find_item_arrays(items_array)
                        if found_items:
                            logger.info(f" Found items at path: {path}")
                            source_items = found_items
                            break
                
                # If no items found in suggested paths, search entire structure
                if not source_items:
                    logger.info(" Searching entire JSON structure for items...")
                    all_items = self._find_item_arrays(actual_data)
                    if all_items:
                        source_items = all_items
            else:
                # Regular JSON extraction - also use semantic approach
                if not fields:
                    # Auto-extraction: return all data
                    source_items = [data] if data else []
                else:
                    # UNIVERSAL: Use semantic extraction for single item
                    if isinstance(data, dict):
                        items = self._find_item_arrays(data)
                        if items:
                            source_items = items
                        else:
                            # Single item, not array
                            # But first check if it's newsletter/subscription/metadata content
                            data_str = str(data).lower()
                            is_newsletter = any(kw in data_str for kw in ['newsletter', 'subscribe', 'email_signup', 'signup_section'])
                            is_metadata = '@context' in data or '@type' in data or 'schema.org' in data_str
                            
                            if is_newsletter:
                                logger.debug(f"   ⏭  Skipping single item (newsletter/subscription content)")
                            elif is_metadata:
                                logger.debug(f"   ⏭  Skipping single item (schema.org metadata)")
                            else:
                                item = self._extract_single_item_semantically(data, fields)
                                if item:
                                    source_items = [item]
            
            # Score this source's items
            if source_items:
                # Score based on array size, field richness, and content quality
                score = 0.0
                
                if source_items and isinstance(source_items[0], dict):
                    # Sample first 10 items to analyze structure
                    sample_keys = set()
                    sample_item = source_items[0]
                    
                    for item in source_items[:10]:
                        if isinstance(item, dict):
                            sample_keys.update(item.keys())
                    
                    # SIZE: Larger arrays score higher, but cap the bonus to prevent metadata arrays from dominating
                    # Use logarithmic scaling: 10 items = 10, 100 items = 20, 1000 items = 30
                    size_bonus = min(len(source_items), 100) + (len(source_items) > 100 and 10) + (len(source_items) > 1000 and 10)
                    score += size_bonus
                    
                    # FIELD RICHNESS: More unique fields = richer data
                    score += len(sample_keys) * 2
                    
                    # CONTENT QUALITY BONUS: Arrays with product-like fields score much higher
                    product_indicators = ['title', 'name', 'price', 'product', 'slug', 'url', 'link', 'image', 'images', 
                                         'color', 'variant', 'description', 'cents', 'productType', 'parentProduct']
                    has_product_fields = any(key.lower() in product_indicators for key in sample_keys)
                    if has_product_fields:
                        score += 500  # Strong bonus for product-like arrays
                    
                    # METADATA PENALTY: Arrays that look like image metadata, tracking, or analytics
                    metadata_indicators = ['src', 'alt', 'approved', 'decorative', 'missingalt', 'tracking', 'analytics',
                                          'session', 'token', 'cookie', 'correlation', 'guid', 'config']
                    has_metadata_fields = any(key.lower() in metadata_indicators for key in sample_keys)
                    if has_metadata_fields and not has_product_fields:
                        score -= 1000  # Strong penalty for metadata arrays without product fields
                    
                    # FIELD NAME BONUS: Known product field names get bonus
                    if source_name.lower() in ['nextjs', 'gridproducts', 'products', 'items', 'data']:
                        score += 200
                    
                    # FIELD NAME PENALTY: Known metadata field names get penalty
                    if source_name.lower() in ['missingalts', 'tracking', 'analytics', 'metadata']:
                        score -= 500
                else:
                    # Non-dict items: minimal score
                    score = len(source_items) * 0.1
                
                source_candidates.append((score, source_items, source_name))
                logger.info(f"    Source '{source_name}': {len(source_items)} items, score={score:.1f}")
        
        # Select the BEST source (highest score)
        if source_candidates:
            source_candidates.sort(reverse=True, key=lambda x: x[0])
            best_score, best_items, best_name = source_candidates[0]
            
            logger.info(f" Selected BEST source: '{best_name}' (score={best_score:.1f}, {len(best_items)} items)")
            
            # Extract from best source only
            if best_items:
                if fields:
                    # Use semantic extraction
                    extracted = self._extract_fields_semantically(best_items, fields, llm_field_mappings=llm_field_mappings)
                else:
                    # Auto-extraction: return items as-is
                    extracted = best_items
        else:
            logger.warning("  No valid items found in any JSON source")
        
        return extracted
    
    def _find_item_arrays(self, data: Any, max_depth: int = 10, current_depth: int = 0) -> List[Dict]:
        """
        UNIVERSAL item array finder - finds the BEST array of items
        
        Prioritizes arrays by:
        1. Size (larger is better - more products/items)
        2. Field richness (more unique fields = more data)
        3. Known field names (products, items, results, etc.)
        
        Works with ANY JSON structure
        
        Returns:
            List of items (the best/most relevant array found)
        """
        if current_depth > max_depth:
            return []
        
        candidates = []  # List of (score, array, name) tuples
        
        # If this is already an array of dicts, score it
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Check if items have multiple fields (not just IDs)
            if len(data[0].keys()) >= 2:
                score = self._score_array(data, 'root')
                candidates.append((score, data, 'root'))
        
        # If this is a dict, search for item arrays
        if isinstance(data, dict):
            # First, check common item array field names
            for field_name in self.ITEM_ARRAY_FIELDS:
                if field_name in data:
                    value = data[field_name]
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict) and len(value[0].keys()) >= 2:
                            score = self._score_array(value, field_name)
                            candidates.append((score, value, field_name))
            
            # Search recursively through all values
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    found = self._find_item_arrays(value, max_depth, current_depth + 1)
                    if found:
                        score = self._score_array(found, key)
                        candidates.append((score, found, key))
        
        # Return the BEST array (highest score)
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            best_score, best_array, best_name = candidates[0]
            
            # CRITICAL: Reject arrays with negative scores (breadcrumbs, navigation, etc.)
            if best_score < 0:
                logger.info(f"  Found {len(candidates)} arrays, but BEST has negative score: '{best_name}' (score={best_score:.1f})")
                logger.info(f"    Rejecting all arrays - likely breadcrumbs/navigation")
                return []
            
            # GRAPHQL FIX: Unwrap edges[].node pattern
            # GraphQL responses use { edges: [{ node: { ...actualData } }] }
            # We need to extract the actual data from inside the 'node' objects
            unwrapped_array = self._unwrap_graphql_nodes(best_array, best_name)
            
            logger.info(f" Found {len(candidates)} arrays, selected BEST: '{best_name}' (score={best_score:.1f}, {len(unwrapped_array)} items)")
            return unwrapped_array
        
        return []
    
    def _unwrap_graphql_nodes(self, array: List[Dict], field_name: str) -> List[Dict]:
        """
        Unwrap GraphQL edge/node pattern to get actual item data.
        
        GraphQL typically returns:
        {
            edges: [
                { node: { name: "...", votesCount: 123 }, cursor: "..." },
                { node: { name: "...", votesCount: 456 }, cursor: "..." }
            ]
        }
        
        We need to extract the 'node' objects as the actual items.
        Also handles 'nodes' pattern (some GraphQL APIs use this directly).
        """
        if not array or not isinstance(array[0], dict):
            return array
        
        first_item = array[0]
        
        # Check if this is edges pattern (items have 'node' key with dict value)
        if 'node' in first_item and isinstance(first_item.get('node'), dict):
            unwrapped = []
            for edge in array:
                node = edge.get('node')
                if isinstance(node, dict) and len(node.keys()) >= 2:
                    unwrapped.append(node)
            
            if unwrapped:
                logger.info(f" GraphQL: Unwrapped {len(unwrapped)} items from 'edges[].node' pattern")
                return unwrapped
        
        # Check if field_name is 'edges' and items might be wrappers
        if field_name.lower() == 'edges':
            # Even if 'node' key doesn't exist, check for nested data patterns
            # Some APIs use { id: ..., data: {...} } or { cursor: ..., item: {...} }
            for wrapper_key in ['node', 'item', 'data', 'content']:
                if wrapper_key in first_item and isinstance(first_item.get(wrapper_key), dict):
                    unwrapped = []
                    for edge in array:
                        inner = edge.get(wrapper_key)
                        if isinstance(inner, dict) and len(inner.keys()) >= 2:
                            unwrapped.append(inner)
                    
                    if unwrapped:
                        logger.info(f" GraphQL: Unwrapped {len(unwrapped)} items from 'edges[].{wrapper_key}' pattern")
                        return unwrapped
        
        # No unwrapping needed
        return array
    
    def _is_navigation_data(self, sample_item: Dict, field_name: str) -> bool:
        """
        Detect if an array contains navigation/filter data instead of actual content
        
        Examples of navigation data:
        - {"job title": "Belgium (English)", "job url": "https://www.monster.be/en/"}
        - {"title": "Filter by Location", "url": "/jobs?location=..."}
        - Country/language selectors, filter options, etc.
        
        Returns:
            True if this looks like navigation/filter data
        """
        if not sample_item or not isinstance(sample_item, dict):
            return False
        
        # Check for navigation patterns
        item_str = str(sample_item).lower()
        item_values = [str(v).lower() for v in sample_item.values() if v]
        
        # Pattern 1: Country/language selectors (e.g., "Belgium (English)")
        navigation_patterns = [
            r'\(english\)', r'\(french\)', r'\(spanish\)', r'\(german\)',
            r'belgium', r'france', r'spain', r'germany', r'italy', r'netherlands',
            r'filter by', r'select', r'choose', r'options', r'categories'
        ]
        if any(re.search(pattern, val) for val in item_values for pattern in navigation_patterns):
            return True
        
        # Pattern 2: URLs that are clearly navigation (country sites, filter pages)
        url_fields = ['url', 'link', 'href', 'job url', 'product url']
        for url_field in url_fields:
            if url_field in sample_item:
                url_val = str(sample_item[url_field]).lower()
                # Navigation URLs: country sites, filter pages, category pages
                if any(pattern in url_val for pattern in ['/en/', '/fr/', '/de/', '/jobs?', '/filter', '/category', '/search?']):
                    # But only if it's NOT a product/job detail page
                    if not any(pattern in url_val for pattern in ['/product/', '/job/', '/detail/', '/item/']):
                        return True
        
        # Pattern 3: Field names that suggest navigation
        nav_field_names = ['filter', 'option', 'category', 'location', 'country', 'language', 'region']
        if any(nav_field in field_name.lower() for nav_field in nav_field_names):
            # Check if it's actually content (has title/name + other fields) or just navigation
            has_content_fields = any(field in sample_item for field in ['title', 'name', 'product', 'job'])
            if not has_content_fields:
                return True
        
        # Pattern 4: Very few fields (navigation usually has 1-2 fields: title + url)
        if len(sample_item) <= 2 and 'url' in str(sample_item).lower():
            # Check if it looks like a content item (has description, price, etc.)
            has_content_indicators = any(field in sample_item for field in ['description', 'price', 'company', 'location', 'salary'])
            if not has_content_indicators:
                return True
        
        return False
    
    def _score_array(self, array: List[Dict], field_name: str) -> float:
        """
        Score an array to determine if it's likely the main content array
        
        Higher scores indicate better/more relevant arrays:
        - Larger arrays are usually main content
        - Arrays with richer data (more unique fields) score higher
        - Arrays from known field names (products, items) score higher
        - Arrays with breadcrumb/navigation data score LOWER
        
        NEW: Detects navigation/filter data and heavily penalizes it
        """
        if not array or not isinstance(array[0], dict):
            return 0.0
        
        score = 0.0
        
        # CRITICAL: Check if this is navigation/filter data (like Monster.com's "Belgium (English)")
        sample = array[0] if array else {}
        if self._is_navigation_data(sample, field_name):
            logger.warning(f"    Navigation/filter data detected in '{field_name}' - heavily penalizing")
            return -1000  # Massive penalty - navigation data should NEVER be selected
        
        # SIZE: Larger arrays are usually main content
        # Use logarithmic scaling to prevent huge arrays from dominating
        import math
        score += math.log(len(array) + 1) * 10
        
        # FIELD RICHNESS: More unique fields = richer data
        # Count unique keys across all items
        all_keys = set()
        for item in array[:10]:  # Sample first 10 items
            all_keys.update(item.keys())
        
        # 10 unique fields = 20 points, 20 fields = 40 points
        score += len(all_keys) * 2
        
        # FIELD NAME BONUS: Known content field names get bonus points (UNIVERSAL)
        # These work across all website types: e-commerce, airlines, news, social media, etc.
        universal_content_fields = ['products', 'items', 'results', 'listings', 'entries', 'data', 'menuitems', 
                                     'flights', 'trips', 'routes', 'articles', 'posts', 'comments', 'reviews',
                                     'events', 'venues', 'shows', 'movies', 'hotels', 'restaurants']
        if field_name.lower() in universal_content_fields:
            score += 50
        elif field_name.lower() in ['menu', 'catalog', 'inventory']:
            score += 30
        
        # FIELD NAME PENALTY: Navigation/meta data gets penalized (STRONGLY)
        if field_name.lower() in ['breadcrumb', 'navigation', 'nav', 'menu', 'footer', 'header', 'itemlistelement']:
            score -= 200  # Increased to ensure breadcrumbs lose
        
        # CONTENT PENALTY: Penalize arrays that look like navigation/breadcrumbs
        sample = array[0] if array else {}
        
        # UNIVERSAL CONTENT BONUS: Arrays with rich, structured data score higher
        # This works for ANY website type (e-commerce, airlines, news, etc.)
        # Detects arrays that contain actual content items, not just reference data
        content_indicators = [
            # Universal indicators (work across all site types)
            'title', 'name', 'description', 'url', 'link', 'id', 'slug',
            # E-commerce specific (but won't hurt other sites)
            'price', 'product', 'image',
            # Travel/airline specific
            'flight', 'departure', 'arrival', 'airline', 'route',
            # News/social specific
            'article', 'post', 'author', 'date', 'published',
            # Generic structured data
            'data', 'attributes', 'metadata'
        ]
        # Count how many content indicators are present (more = richer data)
        content_field_count = sum(1 for key in sample.keys() if any(indicator in key.lower() for indicator in content_indicators))
        if content_field_count >= 3:  # At least 3 content fields = likely main content array
            score += 100  # Strong bonus for rich content arrays
            logger.debug(f"    Rich content array detected in '{field_name}' ({content_field_count} content fields)")
        
        # REFERENCE DATA PENALTY: Arrays that are just reference/lookup data
        # (like colorOrder, printOrder, airportCodes) should not be selected over actual content arrays
        reference_data_fields = ['colororder', 'color_order', 'printorder', 'print_order', 'variants', 'colors',
                                  'airportcodes', 'airport_codes', 'airlines', 'airline_list', 'currencies', 'currency_list']
        if field_name.lower() in reference_data_fields:
            # Only penalize if it doesn't have rich content fields
            if content_field_count < 3:
                score -= 150  # Strong penalty for reference arrays without rich content
                logger.debug(f"     Reference data array detected in '{field_name}' (penalized)")
        
        # Breadcrumb indicators (VERY strong penalty)
        if 'position' in sample and 'item' in sample and '@type' in sample:
            score -= 300  # Massively increased - breadcrumbs should NEVER win
            logger.debug(f"     BREADCRUMB pattern detected in '{field_name}'")
        
        # ListItem with URL items = breadcrumb
        if sample.get('@type') == 'ListItem' or (isinstance(sample.get('item'), str) and sample.get('item', '').startswith('http')):
            score -= 300
            logger.debug(f"     ListItem breadcrumb detected in '{field_name}'")
        
        # Newsletter/subscription indicators  
        if any(key in str(sample).lower() for key in ['newsletter', 'subscribe', 'email', 'signup']):
            if len(array) < 5:  # Small arrays with newsletter data
                score -= 100
        
        # Schema.org structured data (often not main content, but not as bad as breadcrumbs)
        if '@context' in sample or ('@type' in sample and sample.get('@type') not in ['Product', 'Offer']):
            score -= 80
        
        logger.debug(f"   Array '{field_name}': {len(array)} items, {len(all_keys)} fields → score={score:.1f}")
        
        return score
    
    def _extract_fields_from_items(self, items: List[Dict], fields: List[str]) -> List[Dict]:
        """
        Extract specified fields from array of items
        Uses fuzzy matching for field names
        
        If fields is empty (auto-extraction mode), extracts ALL fields
        """
        extracted = []
        
        # Auto-extraction mode: extract ALL fields
        if not fields:
            return items  # Return items as-is with all their fields
        
        # Manual mode: extract only specified fields
        for item in items:
            extracted_item = {}
            for field in fields:
                value = self._find_field_in_json(item, field)
                if value is not None:
                    extracted_item[field] = value
            
            if extracted_item:  # Only add if we found at least one field
                extracted.append(extracted_item)
        
        return extracted
    
    def _get_nested_value(self, data: Any, path: str) -> Any:
        """
        Get value from nested dict using dot notation
        Example: 'props.pageProps.menuData' -> data['props']['pageProps']['menuData']
        """
        if not path or not isinstance(data, dict):
            return data
        
        parts = path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
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
    
    def _requires_authentication(self, json_results: Dict, html: str) -> bool:
        """
        Detect if JSON/page requires authentication FOR PRIMARY CONTENT
        
        Only returns True if the MAIN content is genuinely blocked, not just:
        - Login links in navigation
        - Purchase buttons requiring auth
        - Partial features requiring auth
        
        Common auth indicators:
        - JSON with error messages about auth
        - HTML with NO visible data + prominent auth prompts
        - Empty/minimal data with auth keywords
        
        Returns:
            True if authentication is required for main content
        """
        # Convert JSON results to string for searching
        json_str = json.dumps(json_results.get('data', [])).lower()
        
        # Strong signals: Auth errors in JSON (genuine blocks)
        strong_auth_phrases = ['unauthorized', '401', '403', 'session expired', 'access denied']
        for phrase in strong_auth_phrases:
            if phrase in json_str:
                logger.warning(f" Auth required detected in JSON: '{phrase}'")
                return True
        
        # Check HTML for GENUINE content blocks (not Amazon-style "sign in to buy")
        html_lower = html.lower()
        
        # These phrases indicate the VIEWING of content is blocked (not just purchasing)
        genuine_blocks = [
            'login to view',
            'sign in to view',
            'authenticate to view',
            'sign in to access this',
            'you must sign in to view',
            'authentication required to view',
            'this content requires authentication'
        ]
        
        for block_phrase in genuine_blocks:
            if block_phrase in html_lower:
                logger.warning(f" Auth wall detected: '{block_phrase}'")
                return True
        
        # If we see "sign in to shop" (Amazon-style), check if there's actual visible data
        # This is a purchase auth, not a viewing auth
        if 'sign in to shop' in html_lower or 'sign-in to shop' in html_lower:
            # Check if there are product indicators (prices, "Add to cart", product names)
            has_products = any([
                re.search(r'\$\d+', html),  # Dollar prices
                'add to cart' in html_lower,
                'buy now' in html_lower,
                re.search(r'<h\d[^>]*>[^<]{5,100}</h\d>', html),  # Product titles
            ])
            
            if has_products:
                logger.info(" Auth is for purchasing only - products are visible")
                return False  # Don't block - data is visible
            else:
                logger.warning(" Auth required: No visible product data")
                return True
        
        return False
    
    def is_json_sufficient(self, json_results: Dict, fields: List[str]) -> bool:
        """
        Universal sufficiency check - works for ANY JSON structure
        
        Args:
            json_results: Results from detect_and_extract
            fields: Required fields
            
        Returns:
            True if JSON contains sufficient items and fields
        """
        if not json_results['json_found']:
            return False
        
        # Extract data from JSON
        extracted = self.extract_from_json(json_results['data'], fields)
        
        if not extracted:
            logger.info(" No items extracted from JSON")
            return False
        
        # Check item count and field coverage
        item_count = len(extracted)
        sample_item = extracted[0]
        fields_found = len(sample_item)
        fields_required = len(fields)
        
        # Auto-extraction mode (no fields specified)
        if fields_required == 0:
            logger.info(f" JSON extraction (auto-mode): {item_count} items with {fields_found} fields each")
            if item_count >= 1:
                logger.info(f" JSON sufficient: Found {item_count} items (auto-extraction)")
                return True
            return False
        
        # Manual extraction mode (specific fields requested)
        coverage = fields_found / fields_required if fields_required > 0 else 0
        logger.info(f" JSON extraction: {item_count} items, {coverage:.1%} field coverage ({fields_found}/{fields_required})")
        
        # CRITICAL: Validate content quality (not just quantity!)
        # Check if extracted data is meaningful or just analytics/tracking junk
        if item_count >= 2:
            quality_score = self._validate_content_quality(extracted, fields)
            logger.info(f" Content quality score: {quality_score:.1%}")
            
            if quality_score >= 0.6:
                logger.info(f" JSON sufficient: {item_count} items with good content quality")
                return True
            else:
                logger.warning(f" JSON rejected: Content appears to be analytics/tracking data (quality={quality_score:.1%})")
                logger.warning(f"   Sample values: {self._get_sample_values(extracted)}")
                return False
        
        if coverage >= 0.5:
            logger.info(f" JSON sufficient: {coverage:.1%} field coverage")
            return True
        
        logger.info(f" JSON insufficient: {item_count} items, {coverage:.1%} coverage")
        return False
    
    def _extract_fields_semantically(self, items: List[Dict], fields: List[str], llm_field_mappings: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        UNIVERSAL SEMANTIC EXTRACTION for JSON
        
        Uses the same methodology as HTML semantic extraction:
        - Intelligent field name matching (fuzzy, synonym-aware)
        - Multi-strategy fallback (primary + fallbacks)
        - Type-aware extraction (currency, date, number patterns)
        - Validation and normalization
        
        Args:
            items: Array of JSON objects (like HTML containers)
            fields: Requested fields to extract
            
        Returns:
            List of extracted items with only the requested fields
        """
        if not fields or not isinstance(fields, list) or len(fields) == 0:
            # Auto-extraction mode: return items as-is
            logger.warning(f"  Semantic extraction in auto-mode (fields={fields}). Returning all fields.")
            logger.warning(f"   Sample item keys: {list(items[0].keys())[:10] if items else 'No items'}")
            return items
        
        logger.info(f" Semantic extraction starting: {len(items)} items, {len(fields)} fields requested")
        logger.info(f"   Fields: {fields}")
        if items:
            logger.info(f"   Sample item keys: {list(items[0].keys())[:15]}")
        
        # CRITICAL: Detect minified JSON and infer key mappings ONCE
        key_mapping = None
        if items and len(items) > 0:
            single_char_keys = [k for k in items[0].keys() if len(k) == 1 and k.isalpha()]
            if len(single_char_keys) >= 5:
                logger.info(f"    Minified JSON detected ({len(single_char_keys)} single-char keys)!")
                
                # Filter out breadcrumb/navigation items before analysis
                product_items = []
                for item in items:
                    # Skip breadcrumbs (ListItem type, mostly URLs)
                    if isinstance(item.get('p'), str) and item.get('p') == 'ListItem':
                        continue
                    if isinstance(item.get('E'), str) and item.get('E') == 'ListItem':
                        continue
                    product_items.append(item)
                
                logger.info(f"    Filtered to {len(product_items)} product items (removed {len(items) - len(product_items)} breadcrumbs)")
                
                if product_items:
                    key_mapping = self._infer_key_mapping(product_items, fields)
                    if key_mapping:
                        logger.info(f"    Inferred mappings: {key_mapping}")
                else:
                    logger.warning(f"     All items filtered out! Using original items for inference.")
                    key_mapping = self._infer_key_mapping(items, fields)
                    if key_mapping:
                        logger.info(f"    Inferred mappings: {key_mapping}")
        
        extracted = []
        
        for idx, item in enumerate(items):
            # PRE-PASS: Check for data-name in any dict field BEFORE extraction
            # This ensures we can use it for title improvement
            data_name_candidate = None
            for key, value in item.items():
                if isinstance(value, dict):
                    # Check for data-name in any dict field
                    if 'data-name' in value and isinstance(value['data-name'], str):
                        candidate = value['data-name']
                        # Prefer longer, descriptive names (at least 30 chars)
                        if len(candidate) > 30:
                            if data_name_candidate is None or len(candidate) > len(data_name_candidate):
                                data_name_candidate = candidate
            
            extracted_item = {}
            
            for field in fields:
                # STRATEGY 1: Use inferred key mapping if available (FASTEST)
                if key_mapping and field in key_mapping:
                    minified_key = key_mapping[field]
                    if minified_key in item:
                        value = self._normalize_value(item[minified_key])
                        if value is not None:
                            extracted_item[field] = value
                            if idx == 0:
                                logger.info(f"    '{field}' → key['{minified_key}'] = '{value}'")
                            continue
                
                # STRATEGY 2: Fallback to semantic search (includes LLM mappings if available)
                value = self._extract_field_semantically(item, field, llm_field_mappings=llm_field_mappings)
                
                if value is not None:
                    extracted_item[field] = value
                    if idx == 0:
                        logger.debug(f"    Matched '{field}' → '{value}'")
            
            # POST-PASS: Improve title extraction using pre-found data-name
            # If title is short and we found a data-name candidate, use it
            if 'title' in extracted_item and len(str(extracted_item['title'])) < 30:
                if data_name_candidate and len(data_name_candidate) > len(str(extracted_item['title'])) + 20:
                    extracted_item['title'] = data_name_candidate
                    if idx == 0:
                        logger.info(f"    Improved title from {len(str(extracted_item.get('title', '')))} to {len(data_name_candidate)} chars using data-name")
            
            # Only include items that have at least one requested field
            if extracted_item:
                extracted.append(extracted_item)
            elif idx == 0:
                logger.warning(f"    First item had ZERO field matches!")
        
        logger.info(f" Semantic JSON extraction: {len(extracted)}/{len(items)} items extracted")
        if extracted:
            logger.info(f"   Sample extracted item: {list(extracted[0].keys())}")
        else:
            logger.warning(f"    NO items extracted! All fields failed to match.")
        
        return extracted
    
    def _extract_single_item_semantically(self, item: Dict, fields: List[str]) -> Dict:
        """Extract fields from a single JSON object using semantic strategies"""
        extracted_item = {}
        
        # First pass: extract all fields
        for field in fields:
            value = self._extract_field_semantically(item, field)
            if value is not None:
                extracted_item[field] = value
        
        # Second pass: improve title extraction by checking other fields
        # If title is short, look for longer product names in any dict field (data-name, etc.)
        if 'title' in extracted_item and len(str(extracted_item['title'])) < 30:
            # Check ALL fields in the original item for data-name or longer name fields
            best_title = extracted_item['title']
            best_title_len = len(str(best_title))
            
            for key, value in item.items():
                if isinstance(value, dict):
                    # Check for data-name in any dict field (not just url/product fields)
                    for name_key in ['data-name', 'name', 'title', 'productName', 'product_name', 'fullName', 'full_name']:
                        if name_key in value and isinstance(value[name_key], str):
                            candidate_title = value[name_key]
                            # Prefer significantly longer names (at least 20 chars longer)
                            if len(candidate_title) > best_title_len + 20 and len(candidate_title) > 30:
                                best_title = candidate_title
                                best_title_len = len(candidate_title)
                                break
                elif isinstance(value, str) and len(value) > best_title_len + 20:
                    # Also check if any string field is significantly longer and looks like a product name
                    # (contains product-related keywords and is longer)
                    if len(value) > 30 and any(kw in value.lower() for kw in ['product', 'variety', 'pack', 'food', 'cat', 'dog']):
                        # Check if key suggests it's a name field
                        if any(kw in key.lower() for kw in ['name', 'title', 'product']):
                            best_title = value
                            best_title_len = len(value)
            
            # Update title if we found a better one
            if best_title != extracted_item['title']:
                extracted_item['title'] = best_title
        
        return extracted_item if extracted_item else None
    
    def _extract_field_semantically(self, data: Dict, field: str, llm_field_mappings: Optional[Dict[str, Any]] = None) -> Any:
        """
        UNIVERSAL SEMANTIC FIELD EXTRACTION
        
        Uses intelligent strategies to find the field, similar to HTML semantic extraction:
        
        Strategies (in order):
        0. MINIFIED KEY DETECTION (Next.js, React minification)
        1. Exact match (case-insensitive)
        2. Synonym matching (price → cost, name → title) - hardcoded
        3. Partial matching (product_name → name)
        4. Nested search (look in child objects)
        5. Pattern matching (for special types like currency, dates)
        6. Context-aware search (look for most relevant field)
        7. LLM-discovered mappings (dynamic, universal) - NEW: Fallback for edge cases
        
        Args:
            data: JSON object to search
            field: Field name to find (e.g., "product_name", "price")
            llm_field_mappings: Optional LLM-discovered field mappings from JSONStructureAnalyzer
                              Format: {"field": {"json_key": "actual_key", "path": "nested.path", "confidence": 0.0-1.0}}
            
        Returns:
            Field value or None if not found
        """
        if not isinstance(data, dict):
            return None
        
        field_lower = field.lower()
        
        # NOTE: Minified key detection is now handled via batch _infer_key_mapping()
        # in _extract_fields_semantically() for better accuracy
        
        # STRATEGY 1: Exact match (case-insensitive)
        for key in data.keys():
            if key.lower() == field_lower:
                return self._normalize_value(data[key])
        
        # STRATEGY 2: Synonym matching (universal field mappings)
        synonyms = self._get_field_synonyms(field_lower)
        for synonym in synonyms:
            # Exact match on synonym
            for key in data.keys():
                if key.lower() == synonym.lower():
                    return self._normalize_value(data[key])
            
            # Partial match on synonym (e.g. 'latestScore' matches 'score')
            # Only do this if we haven't found an exact match
            for key in data.keys():
                if synonym.lower() in key.lower():
                    # Avoid boolean flags for numeric synonyms
                    if isinstance(data[key], bool) and synonym in ['score', 'count', 'price', 'votes']:
                        continue
                    return self._normalize_value(data[key])
        
        # STRATEGY 3: Partial matching (product_name matches 'name')
        # Extract the core field name (e.g., "product_name" → "name")
        field_parts = field_lower.replace('_', ' ').replace('-', ' ').split()
        core_field = field_parts[-1] if field_parts else field_lower
        
        # Try core field match
        for key in data.keys():
            key_lower = key.lower()
            if core_field in key_lower or key_lower in core_field:
                value = data[key]
                
                # AVOID BOOLEAN FLAGS: e.g. 'hideVotesCount' when looking for 'votesCount'
                if isinstance(value, bool) and ('hide' in key_lower or 'show' in key_lower or 'is' in key_lower):
                    continue
                    
                # Prefer string values over objects for basic fields
                if not isinstance(value, (dict, list)):
                    return self._normalize_value(value)
        
        # STRATEGY 4: Nested search (look in common nested locations)
        # Many JSON structures nest data in 'product', 'item', 'strain', etc.
        nested_keys = ['product', 'item', 'strain', 'listing', 'data', 'attributes', 'fields']
        for nested_key in nested_keys:
            if nested_key in data and isinstance(data[nested_key], dict):
                # Recursively search in nested object
                nested_value = self._extract_field_semantically(data[nested_key], field)
                if nested_value is not None:
                    return nested_value
        
        # STRATEGY 5: Pattern matching for special types
        # UNIVERSAL NESTED OBJECT EXTRACTION: When a field value is an object, extract nested string values
        # This handles: color={colorName: "Navy"}, variant={name: "Large"}, price={amount: 52}, etc.
        # Works for ANY field type, not just color/variant
        for key, value in data.items():
            if isinstance(value, dict):
                # Check if this key matches the requested field (exact or semantic match)
                key_lower = key.lower()
                field_matches = (
                    key_lower == field_lower or
                    field_lower in key_lower or
                    key_lower in field_lower or
                    any(synonym in key_lower for synonym in self._get_field_synonyms(field_lower))
                )
                
                if field_matches:
                    # UNIVERSAL: Extract nested string value from object
                    # Priority order: field-specific keys, then common nested keys
                    nested_keys_to_try = []
                    
                    # Field-specific nested keys (e.g., colorName for color, variantName for variant)
                    if 'color' in field_lower or 'colour' in field_lower:
                        nested_keys_to_try = ['colorName', 'name', 'value', 'title', 'label', 'displayName', 'color']
                    elif 'variant' in field_lower:
                        nested_keys_to_try = ['variantName', 'name', 'value', 'title', 'label', 'displayName', 'variant']
                    elif 'price' in field_lower or 'cost' in field_lower:
                        nested_keys_to_try = ['amount', 'value', 'price', 'cost', 'total']
                    else:
                        # Generic nested keys for any field
                        nested_keys_to_try = ['name', 'value', 'title', 'label', 'displayName', 'text', 'content']
                    
                    # Try nested keys in priority order
                    for nested_key in nested_keys_to_try:
                        if nested_key in value:
                            nested_value = value[nested_key]
                            # Prefer string values, but also accept numbers for price fields
                            if isinstance(nested_value, str) and len(nested_value) > 0:
                                return self._normalize_value(nested_value)
                            elif isinstance(nested_value, (int, float)) and ('price' in field_lower or 'cost' in field_lower):
                                return self._normalize_value(nested_value)
                    
                    # Fallback: Get first string value from nested object
                    for nested_value in value.values():
                        if isinstance(nested_value, str) and len(nested_value) > 0:
                            return self._normalize_value(nested_value)
        
        if any(keyword in field_lower for keyword in ['price', 'cost', 'amount']):
            # Look for currency patterns
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    return self._normalize_value(value)
                if isinstance(value, str) and ('$' in value or '€' in value or '£' in value):
                    return self._normalize_value(value)
        
        if any(keyword in field_lower for keyword in ['name', 'title', 'heading']):
            # Look for the most prominent string field
            # Prefer longer, descriptive strings
            candidates = []
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 3 and len(value) < 500:
                    # Score based on key relevance and value length
                    score = 0
                    if any(kw in key.lower() for kw in ['name', 'title', 'label']):
                        score += 10
                    # Prefer longer strings (full product names over short labels)
                    score += min(len(value) / 10, 10)  # Longer is better (up to a point)
                    # Penalize very short strings (likely labels, not full names)
                    if len(value) < 15:
                        score -= 5
                    candidates.append((score, value))
                elif isinstance(value, dict):
                    # Check for data-name or similar in nested dicts (e.g., product url dict)
                    # This handles cases where full product name is in data attributes
                    for nested_key in ['data-name', 'name', 'title', 'productName', 'product_name']:
                        if nested_key in value and isinstance(value[nested_key], str):
                            nested_value = value[nested_key]
                            if len(nested_value) > 10:  # Prefer longer names
                                # Higher score for data-name (often contains full product names)
                                score = 20 if nested_key == 'data-name' else 15
                                score += min(len(nested_value) / 10, 10)
                                candidates.append((score, nested_value))
                                break
            
            if candidates:
                candidates.sort(reverse=True, key=lambda x: x[0])
                best_title = candidates[0][1]
                # If best title is very short (< 20 chars) and we have a longer candidate, prefer the longer one
                if len(best_title) < 20 and len(candidates) > 1:
                    for score, title in candidates[1:]:
                        if len(title) > len(best_title) + 10:  # Significantly longer
                            best_title = title
                            break
                return self._normalize_value(best_title)
        
        if any(keyword in field_lower for keyword in ['description', 'desc', 'summary', 'details']):
            # Look for longer text fields
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 20:
                    if any(kw in key.lower() for kw in ['description', 'desc', 'summary', 'detail', 'about', 'info']):
                        return self._normalize_value(value)
        
        # STRATEGY 6: URL extraction (special handling for product url, url, link fields)
        if any(keyword in field_lower for keyword in ['url', 'link', 'href', 'uri']):
            # Look for URL strings first
            url_candidates = []
            for key, value in data.items():
                if isinstance(value, str):
                    # Check if it's a URL
                    if value.startswith(('http://', 'https://', '/')) or 'chewy.com' in value.lower() or 'amazon.com' in value.lower():
                        # Score based on key relevance
                        score = 10 if any(kw in key.lower() for kw in ['url', 'link', 'href', 'uri']) else 5
                        url_candidates.append((score, value))
                elif isinstance(value, dict):
                    # Extract URL from dictionary (handles data-* attributes)
                    extracted_url = self._extract_url_from_dict(value)
                    if extracted_url:
                        url_candidates.append((15, extracted_url))  # Higher score for extracted URLs
            
            if url_candidates:
                url_candidates.sort(reverse=True, key=lambda x: x[0])
                return self._normalize_value(url_candidates[0][1])
        
        # STRATEGY 7: LLM-discovered field mappings (dynamic, universal fallback)
        # Use LLM-discovered mappings when hardcoded synonyms fail
        # This handles domain-specific, non-English, or obscure field names
        if llm_field_mappings and field in llm_field_mappings:
            mapping = llm_field_mappings[field]
            json_key = mapping.get('json_key')
            path = mapping.get('path', json_key)
            confidence = mapping.get('confidence', 0.0)
            
            # Only use if confidence is reasonable (≥0.5) and we haven't found a match yet
            if confidence >= 0.5:
                # Try direct key match first
                if json_key and json_key in data:
                    value = data[json_key]
                    if value is not None:
                        logger.debug(f"    LLM mapping: '{field}' → '{json_key}' (confidence: {confidence:.2f})")
                        return self._normalize_value(value)
                
                # Try nested path if provided
                if path and path != json_key:
                    path_parts = path.split('.')
                    current = data
                    try:
                        for part in path_parts:
                            if isinstance(current, dict) and part in current:
                                current = current[part]
                            else:
                                current = None
                                break
                        if current is not None:
                            logger.debug(f"    LLM mapping (nested): '{field}' → '{path}' (confidence: {confidence:.2f})")
                            return self._normalize_value(current)
                    except (TypeError, AttributeError):
                        pass
        
        # STRATEGY 8: Last resort - fuzzy match on any field
        for key, value in data.items():
            # If field name contains any word from the requested field
            for word in field_parts:
                if len(word) > 2 and word in key.lower():
                    # Prefer non-object values
                    if not isinstance(value, (dict, list)):
                        return self._normalize_value(value)
        
        return None
    
    def _extract_url_from_dict(self, data: Dict) -> Optional[str]:
        """
        Extract URL from a dictionary (handles data-* attributes, nested structures)
        
        Universal solution for extracting URLs from:
        - data-href attributes
        - href keys
        - data-id (construct URL)
        - nested url/link/href fields
        
        Args:
            data: Dictionary that may contain URL information
            
        Returns:
            URL string or None
        """
        if not isinstance(data, dict):
            return None
        
        # Priority 1: Direct href/url/link fields
        for key in ['href', 'url', 'link', 'uri', 'permalink', 'productUrl', 'product_url']:
            if key in data:
                value = data[key]
                if isinstance(value, str) and (value.startswith(('http://', 'https://', '/')) or len(value) > 5):
                    return value
        
        # Priority 2: data-href or data-url attributes
        for key in data.keys():
            if key.lower() in ['data-href', 'data-url', 'data-link', 'data-uri']:
                value = data[key]
                if isinstance(value, str) and (value.startswith(('http://', 'https://', '/')) or len(value) > 5):
                    return value
        
        # Priority 3: Construct URL from data-id (universal pattern)
        # Many e-commerce sites use data-id to construct product URLs
        data_id = None
        base_url = None
        
        # Look for data-id
        for key in ['data-id', 'id', 'productId', 'product_id', 'itemId', 'item_id']:
            if key in data:
                data_id = str(data[key])
                break
        
        # Try to find base URL from data-impression-tracker or other URL fields
        for key, value in data.items():
            if isinstance(value, str) and ('http' in value or 'chewy.com' in value.lower() or 'amazon.com' in value.lower()):
                # Extract domain from URL
                if 'chewy.com' in value.lower():
                    base_url = 'https://www.chewy.com'
                elif 'amazon.com' in value.lower():
                    base_url = 'https://www.amazon.com'
                else:
                    # Try to extract base URL
                    import re
                    match = re.search(r'(https?://[^/]+)', value)
                    if match:
                        base_url = match.group(1)
                break
        
        # Construct URL if we have both id and base_url
        if data_id and base_url:
            # Common e-commerce URL patterns
            if 'chewy.com' in base_url:
                return f"{base_url}/dp/{data_id}"  # Amazon-style
            else:
                return f"{base_url}/product/{data_id}"  # Generic pattern
        
        # Priority 4: Look for URL in nested structures
        for key, value in data.items():
            if isinstance(value, dict):
                nested_url = self._extract_url_from_dict(value)
                if nested_url:
                    return nested_url
        
        # Priority 5: Check data-name for URL-like strings (sometimes URLs are stored there)
        if 'data-name' in data and isinstance(data['data-name'], str):
            value = data['data-name']
            if value.startswith(('http://', 'https://')):
                return value
        
        return None
    
    def _get_field_synonyms(self, field: str) -> List[str]:
        """
        Get semantic synonyms for field names
        
        Examples:
        - price → cost, pricing, amount, value
        - name → title, label, heading
        - description → desc, summary, details, about
        """
        synonyms_map = {
            'price': ['cost', 'pricing', 'amount', 'value', 'rate', 'fee'],
            'name': ['title', 'label', 'heading', 'caption', 'displayname', 'display_name'],
            'title': ['name', 'label', 'heading', 'caption', 'displayname', 'display_name', 'productName', 'product_name'],  # NEW: Reverse mapping for title → name
            'product': ['item', 'listing', 'offer', 'sku', 'strain'],
            'description': ['desc', 'summary', 'details', 'about', 'info', 'text', 'content'],
            'image': ['img', 'picture', 'photo', 'thumbnail', 'icon', 'avatar'],
            'url': ['link', 'href', 'uri', 'path', 'permalink', 'productUrl', 'product_url'],
            'product url': ['link', 'href', 'uri', 'path', 'permalink', 'productUrl', 'product_url', 'url'],
            'date': ['time', 'timestamp', 'created', 'updated', 'published'],
            'category': ['type', 'kind', 'class', 'tag', 'genre'],
            'brand': ['manufacturer', 'maker', 'vendor', 'company'],
            'rating': ['score', 'stars', 'review', 'reviews'],
            'votes': ['score', 'points', 'likes', 'upvotes', 'count', 'rating'],
            'comments': ['replies', 'discussion', 'feedback', 'posts'],
            'quantity': ['amount', 'count', 'number', 'stock'],
            'id': ['identifier', 'code', 'key', 'uuid', 'slug'],
        }
        
        # Check if field or any part of it has synonyms
        for key, synonyms in synonyms_map.items():
            if key in field.lower() or field.lower() in key:
                return synonyms
        
        return []
    
    def _normalize_value(self, value: Any) -> Any:
        """
        Normalize extracted value (like HTML semantic extractor does)
        
        - Clean whitespace
        - Convert numbers appropriately
        - Handle None/empty values
        - Extract text from nested structures if needed
        """
        # Handle None
        if value is None:
            return None
        
        # Handle strings - clean whitespace
        if isinstance(value, str):
            cleaned = ' '.join(value.split())  # Normalize whitespace
            return cleaned if cleaned else None
        
        # Handle numbers - return as-is
        if isinstance(value, (int, float, bool)):
            return value
        
        # Handle nested dict - try to extract a meaningful string
        if isinstance(value, dict):
            # For URL fields, try to extract URL from dict
            # (This handles the case where product url is a dict with data-* attributes)
            if any(kw in str(value).lower() for kw in ['url', 'href', 'link', 'http']):
                extracted_url = self._extract_url_from_dict(value)
                if extracted_url:
                    return extracted_url
            
            # Look for common text fields in the nested object
            for key in ['name', 'title', 'text', 'value', 'label', 'data-name']:
                if key in value and isinstance(value[key], str):
                    return self._normalize_value(value[key])
            # If no text field, return the dict (user can handle it)
            return value
        
        # Handle lists - join if strings, otherwise return first item
        if isinstance(value, list):
            if not value:
                return None
            if all(isinstance(item, str) for item in value):
                return ', '.join(value)
            # Return first non-None item
            return value[0] if value else None
        
        return value
    
    def _infer_key_mapping(self, items: List[Dict], fields: List[str]) -> Dict[str, str]:
        """
        UNIVERSAL KEY MAPPING INFERENCE
        
        Analyzes ALL items to infer which minified keys correspond to requested fields.
        This is done ONCE per page for maximum efficiency.
        
        Strategy: Statistical analysis of values across all items:
        - If key "t" always contains 5-50 char strings → likely product name
        - If key "n" always contains short brand names → likely brand
        - If key contains nested objects → check nested fields
        
        Args:
            items: List of JSON objects to analyze
            fields: Requested fields to map
        
        Returns:
            Dictionary mapping field names to minified keys (e.g., {"product name": "t"})
        """
        if not items or len(items) == 0:
            return {}
        
        logger.info(f"    Analyzing {len(items)} items to infer key mappings...")
        
        # Collect statistics for each key across all items
        key_stats = {}
        sample_size = min(20, len(items))  # Analyze first 20 items
        
        for item in items[:sample_size]:
            for key, value in item.items():
                if key not in key_stats:
                    key_stats[key] = {
                        'types': {},
                        'lengths': [],
                        'samples': [],
                        'has_url': 0,
                        'has_currency': 0,
                        'is_numeric': 0
                    }
                
                # Track value type
                vtype = type(value).__name__
                key_stats[key]['types'][vtype] = key_stats[key]['types'].get(vtype, 0) + 1
                
                # Track string characteristics
                if isinstance(value, str):
                    key_stats[key]['lengths'].append(len(value))
                    if len(key_stats[key]['samples']) < 3:
                        key_stats[key]['samples'].append(value)
                    if value.startswith(('http', '/')):
                        key_stats[key]['has_url'] += 1
                    if any(c in value for c in ['$', '€', '£']):
                        key_stats[key]['has_currency'] += 1
                
                # Track numeric characteristics
                if isinstance(value, (int, float)):
                    key_stats[key]['is_numeric'] += 1
        
        # Now map fields to keys based on statistics
        mapping = {}
        
        for field in fields:
            field_lower = field.lower()
            best_key = None
            best_score = 0
            
            # Analyze each key to see if it matches the field
            for key, stats in key_stats.items():
                score = 0
                
                # PRODUCT NAME / TITLE
                if any(kw in field_lower for kw in ['name', 'product', 'title', 'strain']):
                    if 'str' in stats['types'] and stats['lengths']:
                        avg_len = sum(stats['lengths']) / len(stats['lengths'])
                        # Medium-length strings (5-100 chars), not URLs
                        if 5 <= avg_len <= 100 and stats['has_url'] < sample_size * 0.3:
                            score += 50
                            # Boost common name keys
                            if key in ['t', 'title', 'name', 'n']:
                                score += 20
                            # Check sample values
                            for sample in stats['samples']:
                                if sample and not sample.lower() in ['listitem', 'home']:
                                    score += 5
                
                # PRICE
                elif any(kw in field_lower for kw in ['price', 'cost', 'amount']):
                    if stats['has_currency'] > 0:
                        score += 100  # Strong signal
                    elif stats['is_numeric'] > sample_size * 0.5:
                        # Check if numeric values are in reasonable price range
                        score += 30
                
                # DESCRIPTION
                elif any(kw in field_lower for kw in ['description', 'desc', 'summary', 'details']):
                    if 'str' in stats['types'] and stats['lengths']:
                        avg_len = sum(stats['lengths']) / len(stats['lengths'])
                        # Long text (>50 chars), not URLs
                        if avg_len > 50 and stats['has_url'] < sample_size * 0.3:
                            score += 50
                
                # BRAND
                elif any(kw in field_lower for kw in ['brand', 'manufacturer', 'vendor']):
                    if 'str' in stats['types'] and stats['lengths']:
                        avg_len = sum(stats['lengths']) / len(stats['lengths'])
                        # Short strings (2-30 chars), not URLs
                        if 2 <= avg_len <= 30 and stats['has_url'] == 0:
                            score += 50
                            if key in ['n', 'brand', 'vendor', 'm', 'b']:
                                score += 20
                
                # IMAGE
                elif any(kw in field_lower for kw in ['image', 'img', 'photo', 'picture']):
                    if stats['has_url'] > sample_size * 0.5:
                        # Check if URLs contain image indicators
                        for sample in stats['samples']:
                            if 'jpg' in sample or 'png' in sample or 'image' in sample or 'imgix' in sample:
                                score += 100
                                break
                
                # ID / SKU
                elif any(kw in field_lower for kw in ['id', 'sku', 'code']):
                    if stats['is_numeric'] > sample_size * 0.7:
                        score += 50
                        if key in ['i', 'id', 'x', 'sku']:
                            score += 20
                
                # Track best match
                if score > best_score:
                    best_score = score
                    best_key = key
            
            # Only add mapping if we have reasonable confidence (score > 20)
            if best_key and best_score > 20:
                mapping[field] = best_key
                logger.info(f"      '{field}' → '{best_key}' (confidence={best_score})")
            else:
                logger.warning(f"      '{field}' → NO MATCH (best_key={best_key}, best_score={best_score})")
        
        if not mapping:
            logger.warning(f"    NO key mappings inferred! Field matching failed.")
            # Debug: show key statistics
            logger.warning(f"    Available keys and their characteristics:")
            for key, stats in list(key_stats.items())[:10]:
                if stats['lengths']:
                    avg_len = sum(stats['lengths']) / len(stats['lengths'])
                    logger.warning(f"      '{key}': type={list(stats['types'].keys())}, avg_len={avg_len:.1f}, samples={stats['samples'][:2]}")
        
        return mapping
    
    def _validate_content_quality(self, extracted: List[Dict], fields: List[str]) -> float:
        """
        UNIVERSAL CONTENT QUALITY VALIDATOR
        
        Detects if extracted JSON is meaningful product/content data
        vs analytics/tracking/configuration junk.
        
        Returns score 0.0-1.0:
        - 1.0 = High quality product/content data
        - 0.0 = Analytics/tracking garbage
        
        Patterns that indicate LOW quality:
        - UUIDs and long hashes
        - Internal metric names (desktop_optimistic, correlation_id)
        - Tracking IDs (si=, c=, operationId=)
        - Configuration keys
        - Very short or very long gibberish text
        
        Patterns that indicate HIGH quality:
        - Human-readable product names
        - Proper prices with currency
        - Descriptive text
        - Brand names
        """
        if not extracted or len(extracted) == 0:
            return 0.0
        
        quality_signals = []
        
        # Sample first few items
        sample_size = min(5, len(extracted))
        for item in extracted[:sample_size]:
            item_score = 0.0
            value_count = 0
            
            for field in fields:
                if field in item and field != '_metadata':
                    value = item[field]
                    value_str = str(value).lower()
                    value_count += 1
                    
                    # NEGATIVE SIGNALS (analytics/tracking patterns)
                    
                    # Pattern 1: UUIDs (32-36 char hex strings with dashes)
                    if len(value_str) >= 32 and '-' in value_str and value_str.replace('-', '').replace(' ', '').isalnum():
                        if value_str.count('-') >= 4:  # UUID pattern
                            item_score -= 1.0
                            continue
                    
                    # Pattern 2: Long hashes/tracking IDs (>40 chars of alphanumeric)
                    if len(value_str) > 40 and value_str.replace('=', '').replace(',', '').isalnum():
                        item_score -= 1.0
                        continue
                    
                    # Pattern 3: Internal metric names
                    analytics_keywords = ['_optimistic_', '_correlation_', 'operationid', 'serviceid', 
                                        'traceid', 'sessionid', '_internal_', 'byg_desktop', 'metric_']
                    if any(kw in value_str for kw in analytics_keywords):
                        item_score -= 0.8
                        continue
                    
                    # Pattern 4: Key-value pair strings (si=xxx,c=xxx)
                    if value_str.count('=') >= 2 and value_str.count(',') >= 1:
                        item_score -= 0.8
                        continue
                    
                    # Pattern 5: Very short gibberish (1-3 chars, not numbers)
                    if len(value_str) <= 3 and not value_str.isdigit() and value_str not in ['yes', 'no', 'new', 'old']:
                        item_score -= 0.5
                        continue
                    
                    # POSITIVE SIGNALS (real product/content data)
                    
                    # Pattern 1: Proper product names (5-100 chars, readable)
                    if 5 <= len(value_str) <= 100 and ' ' in value_str:
                        item_score += 1.0
                    
                    # Pattern 2: Currency/prices
                    if isinstance(value, (int, float)) and 0.1 <= value <= 100000:
                        item_score += 1.0
                    elif '$' in value_str or '€' in value_str or '£' in value_str:
                        item_score += 1.0
                    
                    # Pattern 3: Descriptive text (>20 chars with multiple words)
                    if len(value_str) > 20 and value_str.count(' ') >= 3:
                        item_score += 0.8
                    
                    # Pattern 4: Brand/proper names (capitalized words)
                    words = value_str.split()
                    if len(words) >= 2 and any(w[0].isupper() for w in words if len(w) > 0):
                        item_score += 0.5
            
            # Normalize by number of fields checked
            if value_count > 0:
                normalized_score = item_score / value_count
                # Clamp to 0-1 range
                normalized_score = max(0.0, min(1.0, (normalized_score + 1.0) / 2.0))
                quality_signals.append(normalized_score)
        
        # Average quality across all sampled items
        if quality_signals:
            return sum(quality_signals) / len(quality_signals)
        return 0.0
    
    def _get_sample_values(self, extracted: List[Dict]) -> str:
        """Get sample values for debugging content quality issues"""
        if not extracted:
            return "No items"
        
        sample = extracted[0]
        values = []
        for key, value in sample.items():
            if key != '_metadata':
                value_str = str(value)[:50]
                values.append(f"{key}={value_str}")
        
        return ", ".join(values[:3])
    
    def _detect_minified_field(self, data: Dict, field: str) -> Any:
        """
        DEPRECATED: Use _infer_key_mapping instead for better accuracy.
        
        Single-item minified field detection (less reliable than batch analysis).
        """
        # Detect minified structure
        single_char_keys = [k for k in data.keys() if len(k) == 1 and k.isalpha()]
        if len(single_char_keys) < 3:
            return None
        
        # 1. NAME / TITLE (5-100 char strings, not URLs)
        if any(kw in field for kw in ['name', 'product', 'title', 'strain']):
            candidates = []
            for key, value in data.items():
                if isinstance(value, str) and 5 <= len(value) <= 100:
                    if not value.startswith(('http', '/')):
                        if value.lower() not in ['listitem', 'home', 'dispensaries']:
                            score = len(value) / 50
                            if key in ['t', 'title', 'n', 'name']:
                                score += 10
                            candidates.append((score, value))
            if candidates:
                candidates.sort(reverse=True)
                return self._normalize_value(candidates[0][1])
        
        # 2. PRICE (currency symbols or numbers)
        if any(kw in field for kw in ['price', 'cost', 'amount']):
            for key, value in data.items():
                if isinstance(value, str) and any(c in value for c in ['$', '€', '£']):
                    return self._normalize_value(value)
                if isinstance(value, (int, float)) and 0.1 <= value <= 10000:
                    return self._normalize_value(value)
        
        # 3. DESCRIPTION (long text > 50 chars)
        if any(kw in field for kw in ['description', 'desc', 'summary', 'details']):
            candidates = []
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 50 and not value.startswith('http'):
                    candidates.append((len(value), value))
            if candidates:
                candidates.sort(reverse=True)
                return self._normalize_value(candidates[0][1])
        
        # 4. BRAND (2-30 char strings, capitalized)
        if any(kw in field for kw in ['brand', 'manufacturer', 'vendor']):
            for key, value in data.items():
                if isinstance(value, str) and 2 <= len(value) <= 30:
                    if not value.startswith('http') and '/' not in value:
                        if value[0].isupper() or ' ' not in value:
                            return self._normalize_value(value)
        
        return None

