"""
Field Discovery - Suggest fields to extract from a webpage
Uses lightweight analysis without full scraping
"""
import logging
import json
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class FieldDiscovery:
    """
    Discovers available fields on a webpage using lightweight analysis.
    
    Strategy (fastest to slowest):
    1. JSON source analysis (instant if JSON found)
    2. HTML structure analysis (fast, no LLM)
    3. LLM-based discovery (slower but more accurate)
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name
    
    async def discover_fields(
        self,
        html: str,
        url: str,
        use_llm: bool = False,
        target: Optional[str] = None  # NEW: Target hint (e.g., 'products')
    ) -> Dict[str, Any]:
        """
        Discover available fields from HTML.
        
        Universal approach - works for any website type:
        - E-commerce (products, prices, sellers)
        - Travel (flights, hotels, destinations)
        - News (articles, authors, dates)
        - Social media (posts, users, engagement)
        - Search results (titles, snippets, URLs)
        - And any other type of content
        
        Args:
            html: HTML content
            url: Source URL
            use_llm: Whether to use LLM for discovery (slower but more accurate)
            
        Returns:
            Dict with 'fields' (list of field names), 'confidence' (0-1), 'source' (json/html/llm)
        """
        json_fields = None
        html_fields = None
        
        # Step 1: Try JSON sources (only if they contain actual content, not config)
        json_fields = self._discover_from_json(html, url)
        if json_fields and len(json_fields.get('fields', [])) >= 3:
            logger.info(f"✅ Field discovery: Found {len(json_fields['fields'])} fields from JSON")
            # Only return JSON fields if they look like actual content fields
            content_indicators = ['title', 'name', 'price', 'description', 'image', 'date', 'author', 'rating', 'location']
            has_content_fields = any(ind in ' '.join(json_fields['fields']).lower() for ind in content_indicators)
            if has_content_fields and not use_llm:
                return json_fields
            # Otherwise, JSON found config data - continue to other methods
            if not has_content_fields:
                json_fields = None
        
        # Step 2: HTML structure analysis (fast, universal)
        html_fields = self._discover_from_html_structure(html, url)
        if html_fields and len(html_fields.get('fields', [])) >= 3:
            logger.info(f"✅ Field discovery: Found {len(html_fields['fields'])} fields from HTML structure")
            if not use_llm:
                return html_fields
        
        # Step 3: LLM-based discovery (most accurate for any website type)
        # Use LLM if:
        # - Explicitly requested
        # - Previous methods found too few fields
        # - We have an API key
        should_use_llm = use_llm or (
            self.api_key and 
            (not json_fields or len(json_fields.get('fields', [])) < 3) and
            (not html_fields or len(html_fields.get('fields', [])) < 3)
        )
        
        if should_use_llm and self.api_key:
            llm_fields = await self._discover_with_llm(html, url, target=target)
            if llm_fields and len(llm_fields.get('fields', [])) > 0:
                logger.info(f"✅ Field discovery: Found {len(llm_fields['fields'])} fields using LLM")
                return {
                    'fields': llm_fields.get('fields', []),
                    'confidence': llm_fields.get('confidence', 0.85),
                    'source': 'llm',
                    'reasoning': llm_fields.get('reasoning', 'LLM analyzed page content')
                }
        
        # Return best available result
        if html_fields and len(html_fields.get('fields', [])) > 0:
            return html_fields
        if json_fields and len(json_fields.get('fields', [])) > 0:
            return json_fields
        
        # Fallback: Return generic universal fields
        fallback_fields = ['title', 'description', 'url', 'image', 'date']
        
        url_preview = url.lower()[:50] if url else 'unknown'
        return {
            'fields': fallback_fields,
            'confidence': 0.3,
            'source': 'fallback',
            'reasoning': f'Using generic field suggestions for {url_preview}...'
        }
    
    def _discover_from_json(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Discover fields from JSON sources (fastest method)"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # 1. Check JSON-LD (Primary source for Schema.org)
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                # NEW: Deep Schema.org analysis
                schema_fields = self._discover_from_schema_org(data)
                if schema_fields:
                    return {
                        'fields': schema_fields,
                        'confidence': 0.95,
                        'source': 'json_ld_schema',
                        'reasoning': f"Found rich Schema.org data ({data.get('@type', 'unknown')})"
                    }
                
                # Fallback to generic JSON extraction
                fields = self._extract_fields_from_json(data)
                if fields:
                    return {
                        'fields': fields,
                        'confidence': 0.85,
                        'source': 'json_ld',
                        'reasoning': 'Found fields in JSON-LD'
                    }
            except Exception:
                continue

        # 2. Check __NEXT_DATA__
        next_data = soup.find('script', id='__NEXT_DATA__')
        if next_data:
            try:
                data = json.loads(next_data.string)
                fields = self._extract_fields_from_json(data)
                if fields:
                    return {
                        'fields': fields,
                        'confidence': 0.9,
                        'source': 'json_next_data',
                        'reasoning': 'Found fields in __NEXT_DATA__'
                    }
            except Exception as e:
                logger.debug(f"Failed to parse __NEXT_DATA__: {e}")
        
        # 3. Check inline JSON in scripts
        for script in soup.find_all('script'):
            if script.string and len(script.string) > 100:
                # Look for JSON object assignments
                json_matches = re.findall(r'(?:window\.\w+|var \w+)\s*=\s*(\{[\s\S]*?\});', script.string)
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        fields = self._extract_fields_from_json(data)
                        if fields:
                            return {
                                'fields': fields,
                                'confidence': 0.8,
                                'source': 'json_script',
                                'reasoning': 'Found fields in inline JSON'
                            }
                    except Exception:
                        continue
        
        return None

    def _discover_from_schema_org(self, data: Any) -> Optional[List[str]]:
        """Deeply analyze Schema.org data to discover high-value fields"""
        if not isinstance(data, dict):
            return None
            
        # Handle @graph or list of objects
        if '@graph' in data and isinstance(data['@graph'], list):
            for item in data['@graph']:
                fields = self._discover_from_schema_org(item)
                if fields: return fields
            return None
            
        schema_type = data.get('@type')
        if not schema_type:
            return None
            
        # Standard Schema.org types and their high-value fields
        # This allows us to "know" what to look for even if the site uses weird keys
        schema_blueprints = {
            'Product': ['name', 'description', 'brand', 'sku', 'mpn', 'gtin', 'image', 'offers', 'aggregateRating', 'review'],
            'Recipe': ['name', 'description', 'author', 'prepTime', 'cookTime', 'recipeYield', 'recipeIngredient', 'recipeInstructions', 'aggregateRating'],
            'Article': ['headline', 'description', 'author', 'datePublished', 'dateModified', 'image', 'publisher', 'articleBody'],
            'NewsArticle': ['headline', 'description', 'author', 'datePublished', 'image', 'publisher'],
            'Event': ['name', 'description', 'startDate', 'endDate', 'location', 'image', 'offers', 'performer'],
            'LocalBusiness': ['name', 'address', 'telephone', 'openingHours', 'priceRange', 'image', 'aggregateRating'],
            'Review': ['itemReviewed', 'reviewRating', 'author', 'reviewBody', 'datePublished']
        }
        
        # If it's a known type, extract fields intelligently
        if schema_type in schema_blueprints:
            fields = set()
            blueprint = schema_blueprints[schema_type]
            
            for field_name in blueprint:
                # 1. Check if field exists directly
                if field_name in data:
                    fields.add(self._normalize_field_name(field_name) or field_name)
                    
                    # 2. Handle nested objects (offers, brand, etc.)
                    nested_val = data[field_name]
                    if isinstance(nested_val, dict):
                        if field_name == 'offers':
                            # Extract price, availability, etc. from offers
                            for offer_key in ['price', 'priceCurrency', 'availability', 'itemCondition']:
                                if offer_key in nested_val:
                                    fields.add(self._normalize_field_name(offer_key) or offer_key)
                        elif field_name == 'brand':
                            if 'name' in nested_val: fields.add('brand')
                        elif field_name == 'aggregateRating':
                            for rating_key in ['ratingValue', 'reviewCount', 'bestRating']:
                                if rating_key in nested_val:
                                    fields.add(self._normalize_field_name(rating_key) or rating_key)
                    elif isinstance(nested_val, list) and len(nested_val) > 0:
                        # Handle list of offers or reviews
                        first_item = nested_val[0]
                        if isinstance(first_item, dict):
                            if field_name == 'offers':
                                for offer_key in ['price', 'priceCurrency', 'availability']:
                                    if offer_key in first_item:
                                        fields.add(self._normalize_field_name(offer_key) or offer_key)
            
            if fields:
                return sorted(list(fields))
                
        return None
    
    def _extract_fields_from_json(self, data: Any, max_depth: int = 5) -> List[str]:
        """Extract field names from JSON structure - only from actual content arrays"""
        if max_depth <= 0:
            return []
        
        fields = set()
        
        if isinstance(data, dict):
            # Check for Schema.org @type first
            if '@type' in data:
                schema_fields = self._discover_from_schema_org(data)
                if schema_fields:
                    return schema_fields

            # ONLY look for actual content arrays (not config objects)
            # These keys typically contain actual page content items
            content_array_keys = ['itemListElement', 'items', 'products', 'posts', 'articles', 
                                  'results', 'listings', 'offers', 'variants', 'entries',
                                  'flights', 'hotels', 'restaurants', 'reviews', 'comments',
                                  'searchResults', 'pageProps']
            
            for key in content_array_keys:
                if key in data:
                    value = data[key]
                    # Handle pageProps specially - look inside it for content arrays
                    if key == 'pageProps' and isinstance(value, dict):
                        nested_fields = self._extract_fields_from_json(value, max_depth - 1)
                        if nested_fields:
                            fields.update(nested_fields)
                        continue
                    
                    if isinstance(value, list) and len(value) > 0:
                        # Found a content array - extract fields from items
                        for item in value[:3]:
                            if isinstance(item, dict):
                                # Only extract if item looks like content (has title/name/price/etc)
                                content_indicators = ['name', 'title', 'price', 'description', 'image', 
                                                     'url', 'date', 'author', 'rating', 'location', '@type']
                                item_keys_lower = [k.lower() for k in item.keys()]
                                has_content = any(ind in ' '.join(item_keys_lower) for ind in content_indicators)
                                
                                if has_content:
                                    # If item has @type, try schema discovery on it
                                    if '@type' in item:
                                        item_schema = self._discover_from_schema_org(item)
                                        if item_schema:
                                            fields.update(item_schema)
                                            continue

                                    for field_key in item.keys():
                                        normalized = self._normalize_field_name(field_key)
                                        if normalized:
                                            fields.add(normalized)
                        if fields:
                            return sorted(list(fields))  # Found content, return early
            
            # Check for single-item content structures
            content_object_keys = ['product', 'listing', 'flight', 'hotel', 'article', 'post']
            for key in content_object_keys:
                if key in data and isinstance(data[key], dict):
                    content_data = data[key]
                    # Try schema discovery on the content object
                    if '@type' in content_data:
                        item_schema = self._discover_from_schema_org(content_data)
                        if item_schema:
                            return item_schema

                    for field_key in content_data.keys():
                        normalized = self._normalize_field_name(field_key)
                        if normalized:
                            fields.add(normalized)
                    if fields:
                        return sorted(list(fields))
        
        elif isinstance(data, list) and len(data) > 0:
            # Only extract if items look like content
            for item in data[:3]:
                if isinstance(item, dict):
                    content_indicators = ['name', 'title', 'price', 'description', 'image', 
                                         'url', 'date', 'author', 'rating', 'location', '@type']
                    item_keys_lower = [k.lower() for k in item.keys()]
                    has_content = any(ind in ' '.join(item_keys_lower) for ind in content_indicators)
                    
                    if has_content:
                        if '@type' in item:
                            item_schema = self._discover_from_schema_org(item)
                            if item_schema:
                                fields.update(item_schema)
                                continue

                        for field_key in item.keys():
                            normalized = self._normalize_field_name(field_key)
                            if normalized:
                                fields.add(normalized)
        
        return sorted(list(fields))
    
    def _normalize_field_name(self, name: str) -> Optional[str]:
        """Normalize field names to common formats"""
        if not name or len(name) < 2:
            return None
        
        # Remove common prefixes (product, item, etc.)
        name = re.sub(r'^(product|item|entry|post|article)\s*', '', name, flags=re.I)
        
        # Convert camelCase/snake_case to lowercase with spaces
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        name = name.replace('_', ' ').replace('-', ' ')
        name = name.lower().strip()
        
        # Skip ONLY truly internal/structural fields (minimal list to stay universal)
        skip_fields = {
            # Structural/internal only
            'id', 'type', 'class', 'style', 'props', 'children', 'key', 'ref',
            '@type', '@context', '@id', 'sameas', 'mainentity',
            'itemlistelement', 'listitem', 'schema', 'metadata', 'context'
        }
        if name in skip_fields or name.startswith('_') or name.startswith('$') or name.startswith('@'):
            return None
        
        # Map common variations to standard names
        field_mapping = {
            'name': 'title',
            'link': 'url',
            'image': 'image',
            'img': 'image',
            'rating': 'rating',
            'score': 'review score',
            'launch date': 'date',
            'published date': 'date',
            'created date': 'date',
            'category': 'category',
            'tags': 'tags',
            'description': 'description',
            'desc': 'description',
            'summary': 'description',
            'author': 'author',
            'creator': 'author',
            'user': 'author',
            'price': 'price',
            'cost': 'price',
        }
        
        # Check if name matches any mapped field
        for key, mapped_value in field_mapping.items():
            if key in name or name in key:
                return mapped_value
        
        return name
    
    def _extract_fields_from_tables(self, soup: BeautifulSoup) -> List[str]:
        """Extract field names from HTML table headers (th elements)"""
        fields = set()
        
        # Find all tables
        tables = soup.find_all('table')
        
        for table in tables:
            # Look for header row (thead > tr > th or tbody > tr:first-child > th)
            header_row = None
            
            # Check thead first
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
            
            # If no thead, check first row of tbody
            if not header_row:
                tbody = table.find('tbody')
                if tbody:
                    header_row = tbody.find('tr')
                else:
                    # No tbody, check first tr in table
                    header_row = table.find('tr')
            
            if header_row:
                # Extract th or td elements from header row
                headers = header_row.find_all(['th', 'td'])
                for header in headers:
                    text = header.get_text(strip=True)
                    if text and len(text) > 0:
                        # Normalize the field name
                        normalized = self._normalize_field_name(text)
                        if normalized:
                            # Filter out navigation/header elements
                            nav_keywords = ['menu', 'home', 'about', 'contact', 'login', 'sign up', 
                                          'search', 'filter', 'sort', 'view', 'download', 'share',
                                          'navigation', 'nav', 'header', 'footer', 'sidebar']
                            if not any(keyword in normalized.lower() for keyword in nav_keywords):
                                fields.add(normalized)
        
        return list(fields)
    
    def _extract_fields_from_dl(self, soup: BeautifulSoup) -> List[str]:
        """Extract field names from definition lists (dt elements)"""
        fields = set()
        
        # Find all definition lists
        dl_elements = soup.find_all('dl')
        
        for dl in dl_elements:
            # Extract dt (definition term) elements as field names
            dt_elements = dl.find_all('dt')
            for dt in dt_elements:
                text = dt.get_text(strip=True)
                if text and len(text) > 0:
                    # Normalize the field name
                    normalized = self._normalize_field_name(text)
                    if normalized:
                        # Filter out navigation/header elements
                        nav_keywords = ['menu', 'home', 'about', 'contact', 'login', 'sign up',
                                      'search', 'filter', 'sort', 'view', 'download', 'share',
                                      'navigation', 'nav', 'header', 'footer', 'sidebar']
                        if not any(keyword in normalized.lower() for keyword in nav_keywords):
                            fields.add(normalized)
        
        return list(fields)
    
    def _discover_from_microdata(self, soup: BeautifulSoup) -> Optional[List[str]]:
        """Extract field names from HTML Microdata (itemprop attributes)"""
        fields = set()
        
        # Find elements with itemprop
        props = soup.find_all(attrs={"itemprop": True})
        for prop in props:
            name = prop.get("itemprop")
            if name:
                normalized = self._normalize_field_name(name)
                if normalized:
                    fields.add(normalized)
        
        return sorted(list(fields)) if fields else None

    def _discover_from_rdfa(self, soup: BeautifulSoup) -> Optional[List[str]]:
        """Extract field names from HTML RDFa (property attributes)"""
        fields = set()
        
        # Find elements with property
        props = soup.find_all(attrs={"property": True})
        for prop in props:
            name = prop.get("property")
            if name:
                # RDFa properties often have prefixes like 'og:', 'schema:', etc.
                if ':' in name:
                    name = name.split(':')[-1]
                
                normalized = self._normalize_field_name(name)
                if normalized:
                    fields.add(normalized)
        
        return sorted(list(fields)) if fields else None

    def _discover_from_html_structure(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Discover fields from HTML structure (fast, no LLM)"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # PRIORITY 0: Check for Microdata/RDFa (Schema-First)
        microdata_fields = self._discover_from_microdata(soup)
        if microdata_fields and len(microdata_fields) >= 3:
            logger.info(f"✅ Found {len(microdata_fields)} fields from Microdata")
            return {
                'fields': microdata_fields,
                'confidence': 0.98,
                'source': 'html_microdata',
                'reasoning': 'Extracted fields from HTML Microdata (itemprop)'
            }
            
        rdfa_fields = self._discover_from_rdfa(soup)
        if rdfa_fields and len(rdfa_fields) >= 3:
            logger.info(f"✅ Found {len(rdfa_fields)} fields from RDFa")
            return {
                'fields': rdfa_fields,
                'confidence': 0.98,
                'source': 'html_rdfa',
                'reasoning': 'Extracted fields from HTML RDFa (property)'
            }
        
        # PRIORITY 1: Check for data tables (most reliable for structured data)
        table_fields = self._extract_fields_from_tables(soup)
        if table_fields and len(table_fields) >= 3:
            logger.info(f"✅ Found {len(table_fields)} fields from data tables")
            return {
                'fields': table_fields,
                'confidence': 0.95,
                'source': 'html_table',
                'reasoning': 'Extracted fields from data table headers'
            }
        
        # PRIORITY 2: Check for definition lists (common in data pages)
        dl_fields = self._extract_fields_from_dl(soup)
        if dl_fields and len(dl_fields) >= 3:
            logger.info(f"✅ Found {len(dl_fields)} fields from definition lists")
            return {
                'fields': dl_fields,
                'confidence': 0.9,
                'source': 'html_dl',
                'reasoning': 'Extracted fields from definition list structure'
            }
        
        # Enhanced field patterns with more specific selectors
        field_patterns = {
            'title': [
                'h1', 'h2', 'h3', '[class*="title"]', '[class*="name"]', '[id*="title"]', '[id*="name"]',
                '[data-title]', '[data-name]', 'title', 'meta[property="og:title"]',
                '[class*="product-title"]', '[class*="item-title"]', '[class*="listing-title"]'
            ],
            'description': [
                'p', '[class*="description"]', '[class*="desc"]', '[class*="summary"]', '[class*="details"]',
                '[data-description]', 'meta[property="og:description"]', 'meta[name="description"]',
                '[class*="product-description"]', '[class*="item-description"]', '[id*="description"]'
            ],
            'price': [
                '[class*="price"]', '[class*="cost"]', '[class*="amount"]', '[class*="value"]',
                '[data-price]', '[data-cost]', '[id*="price"]', '[id*="cost"]',
                '[class*="product-price"]', '[class*="item-price"]', '[class*="sale-price"]',
                'span[class*="currency"]', '[class*="dollar"]', '[class*="usd"]'
            ],
            'condition': [
                '[class*="condition"]', '[class*="state"]', '[data-condition]', '[id*="condition"]',
                '[class*="item-condition"]', '[class*="product-condition"]'
            ],
            'seller': [
                '[class*="seller"]', '[class*="vendor"]', '[class*="merchant"]', '[class*="store"]',
                '[data-seller]', '[data-vendor]', '[id*="seller"]', '[class*="seller-name"]',
                '[class*="seller-info"]', '[class*="username"]', '[class*="user-name"]'
            ],
            'shipping': [
                '[class*="shipping"]', '[class*="delivery"]', '[class*="ship"]', '[data-shipping]',
                '[id*="shipping"]', '[class*="shipping-cost"]', '[class*="delivery-cost"]',
                '[class*="shipping-info"]', '[class*="free-shipping"]'
            ],
            'image': [
                'img[src]', '[data-image]', '[data-img]', 'meta[property="og:image"]',
                '[class*="image"]', '[class*="photo"]', '[class*="picture"]', '[id*="image"]',
                '[class*="product-image"]', '[class*="item-image"]', '[class*="main-image"]'
            ],
            'url': [
                'a[href]', '[data-url]', '[data-link]', '[data-href]', 'link[rel="canonical"]'
            ],
            'author': [
                '[class*="author"]', '[class*="user"]', '[class*="creator"]', '[class*="posted-by"]',
                '[data-author]', '[data-user]', '[data-creator]', '[id*="author"]'
            ],
            'date': [
                'time[datetime]', '[class*="date"]', '[class*="time"]', '[class*="posted"]',
                '[data-date]', '[data-time]', '[data-published]', '[id*="date"]'
            ],
            'rating': [
                '[class*="rating"]', '[class*="score"]', '[class*="stars"]', '[class*="review-rating"]',
                '[data-rating]', '[data-score]', '[id*="rating"]', '[aria-label*="rating"]'
            ],
            'review score': [
                '[class*="review"]', '[class*="rating"]', '[data-review]', '[data-rating]',
                '[class*="review-count"]', '[class*="reviews"]', '[id*="review"]'
            ],
            'category': [
                '[class*="category"]', '[class*="tag"]', '[class*="breadcrumb"]', '[data-category]', '[data-tag]',
                '[id*="category"]', '[class*="product-category"]', '[class*="item-category"]'
            ],
            'quantity': [
                '[class*="quantity"]', '[class*="stock"]', '[class*="available"]', '[data-quantity]',
                '[id*="quantity"]', '[class*="in-stock"]', '[class*="availability"]', '[class*="qty"]',
                '[class*="available-quantity"]', '[class*="stock-quantity"]'
            ],
            'location': [
                '[class*="location"]', '[class*="address"]', '[class*="city"]', '[data-location]',
                '[id*="location"]', '[class*="seller-location"]', '[class*="item-location"]',
                '[class*="shipping-location"]', '[class*="from"]'
            ],
            'seller name': [
                '[class*="seller"]', '[class*="vendor"]', '[class*="merchant"]', '[class*="store"]',
                '[data-seller]', '[data-vendor]', '[id*="seller"]', '[class*="seller-name"]',
                '[class*="seller-info"]', '[class*="username"]', '[class*="user-name"]',
                '[class*="seller-username"]', 'a[href*="seller"]', 'a[href*="user"]'
            ],
            'seller rating': [
                '[class*="seller-rating"]', '[class*="seller-score"]', '[class*="feedback"]',
                '[data-seller-rating]', '[class*="seller-feedback"]', '[class*="seller-reputation"]'
            ],
            'shipping cost': [
                '[class*="shipping"]', '[class*="delivery"]', '[class*="ship"]', '[data-shipping]',
                '[id*="shipping"]', '[class*="shipping-cost"]', '[class*="delivery-cost"]',
                '[class*="shipping-info"]', '[class*="free-shipping"]', '[class*="shipping-price"]',
                '[class*="ship-cost"]', 'span[class*="shipping"]'
            ],
            'brand': [
                '[class*="brand"]', '[data-brand]', '[id*="brand"]', '[class*="manufacturer"]',
                '[class*="product-brand"]', '[class*="item-brand"]', 'meta[property*="brand"]'
            ],
            'model': [
                '[class*="model"]', '[data-model]', '[id*="model"]', '[class*="product-model"]',
                '[class*="item-model"]', '[class*="model-number"]'
            ],
            'sku': [
                '[class*="sku"]', '[data-sku]', '[id*="sku"]', '[class*="product-sku"]',
                '[class*="item-sku"]', '[class*="product-code"]'
            ],
            'specifications': [
                '[class*="spec"]', '[class*="specification"]', '[class*="details"]', '[class*="features"]',
                '[data-spec]', '[id*="spec"]', '[class*="product-spec"]', '[class*="item-spec"]',
                'table[class*="spec"]', 'dl[class*="spec"]'
            ]
        }
        
        found_fields = []
        confidence_scores = {}
        
        # Check for repeating item containers (list pages)
        item_selectors = [
            'article', '[role="article"]', '[class*="item"]', '[class*="card"]',
            '[class*="product"]', '[class*="post"]', 'li[class*="item"]'
        ]
        
        items_found = 0
        items = []
        for selector in item_selectors:
            items = soup.select(selector)
            if len(items) >= 3:  # At least 3 items suggests a list
                items_found = len(items)
                break
        
        # If no repeating items found, check if it's a single-item page (product page)
        is_single_item_page = False
        if items_found == 0:
            # Check for common single-item page indicators
            single_item_indicators = [
                soup.find('main'),
                soup.find('article'),
                soup.find('[class*="product"]'),
                soup.find('[class*="item-detail"]'),
                soup.find('[id*="product"]'),
                soup.find('[id*="item"]')
            ]
            if any(indicator for indicator in single_item_indicators if indicator):
                is_single_item_page = True
                # Use the entire page or main content area
                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                if main_content:
                    items = [main_content]
                else:
                    items = [soup]
        
        if not items:
            return None
        
        # Check which fields are present
        search_area = items[0] if is_single_item_page else items[:3]  # Single item or first 3 items
        
        for field_name, patterns in field_patterns.items():
            found_count = 0
            search_items = [search_area] if is_single_item_page else search_area
            
            for item in search_items:
                for pattern in patterns:
                    try:
                        matches = item.select(pattern)
                        if matches:
                            # Verify it's not empty
                            for match in matches:
                                text = match.get_text(strip=True) if hasattr(match, 'get_text') else str(match)
                                if text and len(text) > 0:
                                    found_count += 1
                                    break
                            if found_count > 0:
                                break
                    except Exception:
                        continue
                if found_count > 0:
                    break
            
            # For single-item pages, field is present if found at least once
            # For list pages, field is present if found in at least 2/3 items
            threshold = 1 if is_single_item_page else 2
            if found_count >= threshold:
                found_fields.append(field_name)
                confidence_scores[field_name] = min(1.0, found_count / max(1, len(search_items)))
        
        if found_fields:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.7
            page_type = "single-item page" if is_single_item_page else f"{items_found} items"
            return {
                'fields': found_fields,
                'confidence': avg_confidence,
                'source': 'html_structure',
                'reasoning': f'Found {len(found_fields)} fields in HTML structure from {page_type}'
            }
        
        return None
    
    async def _discover_with_llm(
        self, 
        html: str, 
        url: str,
        target: Optional[str] = None  # NEW: Target hint
    ) -> Optional[Dict[str, Any]]:
        """Discover fields using LLM (slower but more accurate)"""
        if not self.api_key:
            return None
        
        try:
            import litellm
            
            # Use a small sample of HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Get main content area (universal - no site-specific logic)
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                html_sample = str(main_content)[:10000]  # Get enough content for LLM analysis
            else:
                html_sample = html[:10000]
            
            # Universal prompt - let LLM analyze content to determine page type and fields
            # No hardcoded site detection - LLM analyzes HTML content intelligently
            target_instruction = f"The user is specifically looking for: {target.upper()}" if target else "Suggest ALL relevant fields that can be extracted."
            
            prompt = f"""Analyze this HTML content and intelligently suggest fields for extraction.
{target_instruction}

URL: {url}

HTML sample:
{html_sample}

TASK:
1. PRIORITY: Look for DATA TABLES first (with <table>, <thead>, <th> elements)
   - Extract field names from table headers (<th> elements)
   - These are the PRIMARY fields to extract
   - Ignore navigation tables, menu tables, footer tables

2. SECONDARY: Look for definition lists (<dl>, <dt> elements)
   - Extract field names from definition terms (<dt> elements)
   - These often contain structured data fields

3. TERTIARY: Analyze the HTML content to determine what TYPE of page this is:
   - Is it a data/table page (government data, financial data, datasets)?
   - Is it an e-commerce product listing page (with prices, sellers, shipping)?
   - Is it a product discovery/showcase platform (showing products/tools without prices)?
   - Is it a movie/TV show listing page?
   - Is it a news/article listing page?
   - Is it a job listing page?
   - Is it a social media feed?
   - Is it something else?

4. Based on the page type you identified (and the target hint if provided), suggest appropriate fields:
   - If a TARGET was provided ({target if target else 'none'}), ONLY suggest fields relevant to that target.
   - For DATA/TABLE pages: Extract ALL column headers from tables as fields
   - For E-COMMERCE pages: price, condition, seller name, shipping cost, brand, etc.
   - For PRODUCT DISCOVERY platforms: maker/creator, upvotes, tagline, launch date, etc. (NOT price/seller)
   - For MOVIE/TV pages: metascore, director, cast, release date, poster image, etc.
   - For NEWS/ARTICLE pages: author, publish date, category, excerpt, etc.
   - For SOCIAL MEDIA: username, post content, likes, shares, timestamp, etc.

5. Use SPECIFIC, DESCRIPTIVE field names:
   - Instead of generic "image", use "poster image", "cover image", "thumbnail", "product image", etc.
   - Instead of generic "title", use "movie title", "product title", "article title", "show title", etc.
   - Instead of generic "description", use "synopsis", "product description", "summary", "tagline", etc.
   - Instead of generic "rating", use "metascore", "critic score", "user rating", "upvotes", etc.

6. CRITICAL FILTERING: EXCLUDE these navigation/header elements:
   - Menu, Home, About, Contact, Login, Sign Up, Search, Filter, Sort, View, Download, Share
   - Navigation, Nav, Header, Footer, Sidebar
   - Any field that appears in page navigation or site-wide headers/footers

7. Extract 8-20 fields depending on page complexity
8. Focus on visible, extractable DATA fields in the HTML
9. Prioritize fields that appear in DATA TABLES or DATA STRUCTURES over generic page elements
10. Be comprehensive but accurate - only suggest fields that contain actual data, not UI elements

Return a JSON object with:
{{
  "page_type": "your analysis of what type of page this is",
  "fields": ["specific field 1", "specific field 2", "specific field 3", ...],
  "reasoning": "Brief explanation of page type and why these fields are relevant"
}}

Return ONLY the JSON, no other text."""

            response = await litellm.acompletion(
                model=self.model_name,
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing web page structure and suggesting relevant fields to extract."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            fields = result.get('fields', [])
            if fields:
                # Normalize field names to user-friendly format and deduplicate
                normalized_fields = []
                seen = set()
                for f in fields:
                    normalized = self._normalize_field_name(f)
                    if normalized and normalized not in seen:
                        normalized_fields.append(normalized)
                        seen.add(normalized)
                
                if normalized_fields:
                    return {
                        'fields': normalized_fields,
                        'confidence': 0.85,
                        'source': 'llm',
                        'reasoning': result.get('reasoning', 'LLM-suggested fields')
                    }
        
        except Exception as e:
            logger.warning(f"LLM field discovery failed: {e}")
        
        return None

