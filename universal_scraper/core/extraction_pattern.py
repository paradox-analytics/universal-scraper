"""
Extraction Pattern - Stores deterministic extraction strategies

This is the core of the caching system. Instead of caching LLM results,
we cache the STRATEGY/PATTERN that can be reused for future scrapes.

Pattern Types:
1. JSON_API: Direct JSON endpoint (fastest, most reliable)
2. JSON_EMBEDDED: JSON-LD or embedded JSON in HTML
3. CSS_SELECTORS: BeautifulSoup CSS selectors (generated code)
4. XPATH: XPath expressions (for complex structures)
5. ATTRIBUTE_MAP: Data attributes (data-*, aria-*, etc.)

The system learns from Direct LLM extraction and generates reusable patterns.
"""

import json
import logging
import hashlib
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Types of extraction patterns"""
    JSON_API = "json_api"           # Direct JSON endpoint
    JSON_EMBEDDED = "json_embedded"  # JSON-LD, __NEXT_DATA__, etc.
    CSS_SELECTORS = "css_selectors"  # BeautifulSoup selectors
    XPATH = "xpath"                  # XPath expressions
    ATTRIBUTE_MAP = "attribute_map"  # Data attributes extraction
    HYBRID = "hybrid"                # Combination of methods


@dataclass
class FieldExtractor:
    """
    Defines how to extract a single field.
    This is the atomic unit of extraction.
    """
    field_name: str

    # Primary extraction method
    selector: Optional[str] = None       # CSS selector
    xpath: Optional[str] = None          # XPath expression
    json_path: Optional[str] = None      # JSON path (e.g., "data.items[*].title")
    attribute: Optional[str] = None      # HTML attribute (e.g., "data-author")
    regex: Optional[str] = None          # Regex pattern for text extraction

    # Post-processing
    extract_text: bool = True            # Extract .text or use attribute
    strip: bool = True                   # Strip whitespace
    default_value: Optional[str] = None  # Default if not found

    # Type conversion
    data_type: str = "string"            # string, int, float, bool, date, url

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldExtractor":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExtractionPattern:
    """
    A complete extraction pattern for a domain/structure.

    This is what gets cached and reused. It contains:
    1. How to find the container (list of items)
    2. How to extract each field from an item
    3. Metadata for validation and versioning
    """
    # Identity
    pattern_id: str                      # Unique ID (hash of domain + structure)
    domain: str                          # Domain this pattern applies to
    structure_hash: str                  # Hash of the page structure

    # Pattern type
    pattern_type: PatternType

    # Container selection (how to find the list of items)
    container_selector: Optional[str] = None   # CSS selector for container
    container_xpath: Optional[str] = None      # XPath for container
    item_selector: Optional[str] = None        # CSS selector for each item
    item_xpath: Optional[str] = None           # XPath for each item
    json_items_path: Optional[str] = None      # JSON path to items array

    # Field extractors
    field_extractors: List[FieldExtractor] = field(default_factory=list)

    # Generated code (for CSS_SELECTORS type)
    extraction_code: Optional[str] = None      # Python code to execute

    # JSON endpoint (for JSON_API type)
    json_endpoint: Optional[str] = None        # Direct JSON URL
    json_method: str = "GET"                   # HTTP method
    json_headers: Optional[Dict[str, str]] = None

    # Metadata
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Validation
    expected_fields: List[str] = field(default_factory=list)
    min_items: int = 1                         # Minimum expected items
    sample_output: Optional[List[Dict]] = None # Sample of extracted data

    # Source tracking
    learned_from_url: Optional[str] = None     # URL where pattern was learned
    learned_via: str = "direct_llm"            # How pattern was created

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'pattern_id': self.pattern_id,
            'domain': self.domain,
            'structure_hash': self.structure_hash,
            'pattern_type': self.pattern_type.value,
            'container_selector': self.container_selector,
            'container_xpath': self.container_xpath,
            'item_selector': self.item_selector,
            'item_xpath': self.item_xpath,
            'json_items_path': self.json_items_path,
            'field_extractors': [fe.to_dict() for fe in self.field_extractors],
            'extraction_code': self.extraction_code,
            'json_endpoint': self.json_endpoint,
            'json_method': self.json_method,
            'json_headers': self.json_headers,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'use_count': self.use_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'expected_fields': self.expected_fields,
            'min_items': self.min_items,
            'sample_output': self.sample_output,
            'learned_from_url': self.learned_from_url,
            'learned_via': self.learned_via,
        }
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionPattern":
        # Convert field extractors
        field_extractors = [
            FieldExtractor.from_dict(fe)
            for fe in data.get('field_extractors', [])
        ]

        return cls(
            pattern_id=data['pattern_id'],
            domain=data['domain'],
            structure_hash=data['structure_hash'],
            pattern_type=PatternType(data['pattern_type']),
            container_selector=data.get('container_selector'),
            container_xpath=data.get('container_xpath'),
            item_selector=data.get('item_selector'),
            item_xpath=data.get('item_xpath'),
            json_items_path=data.get('json_items_path'),
            field_extractors=field_extractors,
            extraction_code=data.get('extraction_code'),
            json_endpoint=data.get('json_endpoint'),
            json_method=data.get('json_method', 'GET'),
            json_headers=data.get('json_headers'),
            created_at=data.get('created_at', time.time()),
            last_used=data.get('last_used', time.time()),
            use_count=data.get('use_count', 0),
            success_count=data.get('success_count', 0),
            failure_count=data.get('failure_count', 0),
            expected_fields=data.get('expected_fields', []),
            min_items=data.get('min_items', 1),
            sample_output=data.get('sample_output'),
            learned_from_url=data.get('learned_from_url'),
            learned_via=data.get('learned_via', 'direct_llm'),
        )


class PatternGenerator:
    """
    Generates extraction patterns from successful LLM extractions.

    This is the "learning" component - it observes what the LLM extracted
    and reverse-engineers a deterministic pattern that can be reused.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name

    async def generate_pattern_from_extraction(
        self,
        html: str,
        extracted_items: List[Dict[str, Any]],
        fields: List[str],
        url: str,
        structure_hash: str
    ) -> Optional[ExtractionPattern]:
        """
        Generate a reusable pattern from a successful LLM extraction.

        This is the key method - it takes what the LLM found and creates
        a deterministic pattern that can be reused without LLM.

        Args:
            html: The HTML that was scraped
            extracted_items: Items extracted by Direct LLM
            fields: Fields that were requested
            url: Source URL
            structure_hash: Hash of the page structure

        Returns:
            ExtractionPattern that can be cached and reused
        """
        if not extracted_items:
            return None

        domain = urlparse(url).netloc
        pattern_id = self._generate_pattern_id(domain, structure_hash, fields)

        logger.info(f"🔬 Generating extraction pattern for {domain}...")
        logger.info(f"   Analyzing {len(extracted_items)} extracted items with {len(fields)} fields")

        # Step 1: Try to detect JSON source first (fastest path)
        json_pattern = await self._detect_json_pattern(html, extracted_items, fields, url)
        if json_pattern:
            json_pattern.pattern_id = pattern_id
            json_pattern.domain = domain
            json_pattern.structure_hash = structure_hash
            json_pattern.learned_from_url = url
            json_pattern.expected_fields = fields
            json_pattern.sample_output = extracted_items[:3]
            logger.info(f"✅ Generated JSON pattern: {json_pattern.pattern_type.value}")
            return json_pattern

        # Step 2: Generate CSS selector pattern (most common path)
        css_pattern = await self._generate_css_pattern(html, extracted_items, fields, url, structure_hash)
        if css_pattern:
            css_pattern.pattern_id = pattern_id
            css_pattern.domain = domain
            css_pattern.structure_hash = structure_hash
            css_pattern.learned_from_url = url
            css_pattern.expected_fields = fields
            css_pattern.sample_output = extracted_items[:3]
            logger.info("✅ Generated CSS selector pattern")
            return css_pattern

        # Step 3: Generate attribute-based pattern (for custom elements)
        attr_pattern = await self._generate_attribute_pattern(html, extracted_items, fields, url, structure_hash)
        if attr_pattern:
            attr_pattern.pattern_id = pattern_id
            attr_pattern.domain = domain
            attr_pattern.structure_hash = structure_hash
            attr_pattern.learned_from_url = url
            attr_pattern.expected_fields = fields
            attr_pattern.sample_output = extracted_items[:3]
            logger.info("✅ Generated attribute pattern")
            return attr_pattern

        logger.warning(f"⚠️ Could not generate deterministic pattern for {domain}")
        return None

    def _generate_pattern_id(self, domain: str, structure_hash: str, fields: List[str]) -> str:
        """Generate unique pattern ID"""
        fields_str = ",".join(sorted(fields))
        content = f"{domain}:{structure_hash}:{fields_str}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _detect_json_pattern(
        self,
        html: str,
        extracted_items: List[Dict],
        fields: List[str],
        url: str
    ) -> Optional[ExtractionPattern]:
        """
        Detect if data came from embedded JSON (fastest extraction method).

        Checks for:
        - JSON-LD (<script type="application/ld+json">)
        - Next.js data (__NEXT_DATA__)
        - Nuxt.js data (__NUXT__)
        - Generic data scripts
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Check JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if self._json_contains_items(data, extracted_items, fields):
                    logger.info("   Found data in JSON-LD")
                    return ExtractionPattern(
                        pattern_id="",
                        domain="",
                        structure_hash="",
                        pattern_type=PatternType.JSON_EMBEDDED,
                        json_items_path="@graph" if "@graph" in data else ".",
                        learned_via="json_ld_detection"
                    )
            except json.JSONDecodeError:
                continue

        # Check __NEXT_DATA__
        next_data = soup.find('script', id='__NEXT_DATA__')
        if next_data:
            try:
                data = json.loads(next_data.string)
                items_path = self._find_items_in_json(data, extracted_items, fields)
                if items_path:
                    logger.info(f"   Found data in __NEXT_DATA__ at path: {items_path}")
                    return ExtractionPattern(
                        pattern_id="",
                        domain="",
                        structure_hash="",
                        pattern_type=PatternType.JSON_EMBEDDED,
                        json_items_path=f"__NEXT_DATA__.{items_path}",
                        learned_via="next_data_detection"
                    )
            except json.JSONDecodeError:
                pass

        # Check generic data scripts
        for script in soup.find_all('script'):
            if script.string and len(script.string) > 100:
                # Look for JSON object assignments
                json_matches = re.findall(r'(?:window\.\w+|var \w+)\s*=\s*(\{[\s\S]*?\});', script.string)
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        items_path = self._find_items_in_json(data, extracted_items, fields)
                        if items_path:
                            logger.info(f"   Found data in script at path: {items_path}")
                            return ExtractionPattern(
                                pattern_id="",
                                domain="",
                                structure_hash="",
                                pattern_type=PatternType.JSON_EMBEDDED,
                                json_items_path=items_path,
                                learned_via="script_json_detection"
                            )
                    except json.JSONDecodeError:
                        continue

        return None

    def _json_contains_items(self, data: Any, extracted_items: List[Dict], fields: List[str]) -> bool:
        """Check if JSON data contains the extracted items"""
        if not extracted_items:
            return False

        first_item = extracted_items[0]

        # Check if any field value from extracted items exists in JSON
        def search_json(obj, depth=0):
            if depth > 10:
                return False

            if isinstance(obj, dict):
                # Check if this dict matches our extracted item
                matches = 0
                for field in fields[:3]:  # Check first 3 fields
                    if field in first_item and first_item[field]:
                        value = str(first_item[field])[:50]
                        if any(value in str(v) for v in obj.values()):
                            matches += 1
                if matches >= 2:
                    return True

                # Recurse into values
                for v in obj.values():
                    if search_json(v, depth + 1):
                        return True

            elif isinstance(obj, list):
                for item in obj[:10]:  # Check first 10 items
                    if search_json(item, depth + 1):
                        return True

            return False

        return search_json(data)

    def _find_items_in_json(self, data: Any, extracted_items: List[Dict], fields: List[str], path: str = "") -> Optional[str]:
        """Find the JSON path where extracted items are located"""
        if not extracted_items:
            return None

        first_item = extracted_items[0]

        def search(obj, current_path, depth=0):
            if depth > 10:
                return None

            if isinstance(obj, list) and len(obj) > 0:
                # Check if this list contains our items
                matches = 0
                for list_item in obj[:5]:
                    if isinstance(list_item, dict):
                        for field in fields[:3]:
                            if field in first_item and first_item[field]:
                                value = str(first_item[field])[:30]
                                if any(value.lower() in str(v).lower() for v in list_item.values()):
                                    matches += 1
                                    break

                if matches >= min(2, len(obj)):
                    return current_path

            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    result = search(value, new_path, depth + 1)
                    if result:
                        return result

            elif isinstance(obj, list):
                for i, item in enumerate(obj[:5]):
                    result = search(item, f"{current_path}[{i}]", depth + 1)
                    if result:
                        # Return the array path, not the specific index
                        return current_path

            return None

        return search(data, path)

    async def _generate_css_pattern(
        self,
        html: str,
        extracted_items: List[Dict],
        fields: List[str],
        url: str,
        structure_hash: str
    ) -> Optional[ExtractionPattern]:
        """
        Generate CSS selector pattern using LLM to analyze HTML structure.

        This asks the LLM to generate BeautifulSoup code that can extract
        the same data deterministically.
        """
        import litellm

        if not self.api_key:
            return None

        # Prepare a sample of the HTML (truncate for LLM context)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Find elements that might contain our data
        sample_html = self._get_relevant_html_sample(soup, extracted_items, fields)

        if not sample_html or len(sample_html) < 100:
            return None

        # Ask LLM to generate extraction code
        prompt = f"""Analyze this HTML and generate BeautifulSoup Python code to extract data.

The LLM previously extracted these items:
{json.dumps(extracted_items[:3], indent=2)}

Fields to extract: {fields}

HTML sample:
{sample_html[:8000]}

Generate a Python function called `extract_data(soup)` that:
1. Finds the container element holding all items (MAIN CONTENT AREA ONLY)
2. Iterates through each item
3. Extracts each field using CSS selectors or attributes
4. Returns a list of dictionaries

CRITICAL - EXCLUDE NAVIGATION/FOOTER:
- DO NOT extract items from <nav>, <footer>, <header>, or sidebar elements
- DO NOT extract navigation links (paths like "/categories", "/products/.../reviews", etc.)
- DO NOT extract footer links (links with ?ref=footer, /legal, /about, etc.)
- ONLY extract items from the MAIN CONTENT AREA (usually <main>, <article>, or main container)
- Skip items where title/url is just a path (e.g., "/categories/vibe-coding")
- Skip items where description is None/null and title is just a URL or path

IMPORTANT:
- Use .select() for CSS selectors, not .find_all() with complex logic
- Check for data-* attributes (e.g., elem.get('data-author'))
- Handle None values gracefully - fields are OPTIONAL, extract what exists
- Capture ALL items from main content, even if some don't have all fields
- Return empty list if no items found
- If a field doesn't exist for an item, set it to None (don't skip the item)

Return ONLY the Python code, no explanations."""

        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )

            code = response.choices[0].message.content

            # Clean up code
            code = self._clean_code(code)

            # Validate the code works
            if self._validate_code(code, html, extracted_items, fields):
                # Extract selectors from the code for the pattern
                item_selector = self._extract_item_selector(code)
                field_extractors = self._extract_field_extractors(code, fields)

                return ExtractionPattern(
                    pattern_id="",
                    domain="",
                    structure_hash=structure_hash,
                    pattern_type=PatternType.CSS_SELECTORS,
                    item_selector=item_selector,
                    field_extractors=field_extractors,
                    extraction_code=code,
                    learned_via="llm_code_generation"
                )

        except Exception as e:
            logger.error(f"Failed to generate CSS pattern: {e}")

        return None

    def _get_relevant_html_sample(self, soup, extracted_items: List[Dict], fields: List[str]) -> str:
        """Get a relevant sample of HTML containing the extracted data"""
        if not extracted_items:
            return str(soup)[:10000]

        first_item = extracted_items[0]

        # Find elements containing our data
        relevant_elements = []

        for field in fields[:3]:
            if field in first_item and first_item[field]:
                value = str(first_item[field])[:50]

                # Search for elements containing this text
                for elem in soup.find_all(string=re.compile(re.escape(value[:20]), re.I)):
                    parent = elem.parent
                    # Go up to find a reasonable container
                    for _ in range(5):
                        if parent and parent.name in ['article', 'div', 'li', 'tr', 'section']:
                            if parent not in relevant_elements:
                                relevant_elements.append(parent)
                            break
                        parent = parent.parent if parent else None

        if relevant_elements:
            # Return the parent of the first few elements (likely the container)
            container = relevant_elements[0].parent
            if container:
                return str(container)[:10000]

        return str(soup)[:10000]

    def _clean_code(self, code: str) -> str:
        """Clean up LLM-generated code"""
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)

        # Remove any leading/trailing whitespace
        code = code.strip()

        return code

    def _validate_code(self, code: str, html: str, extracted_items: List[Dict], fields: List[str]) -> bool:
        """Validate that generated code produces similar results"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            from .sandbox import safe_exec
            namespace = safe_exec(code, {'soup': soup, 'BeautifulSoup': BeautifulSoup})

            if 'extract_data' not in namespace:
                return False

            result = namespace['extract_data'](soup)

            if not result or not isinstance(result, list):
                return False

            # Filter out navigation/footer links and invalid items
            filtered_result = self._filter_invalid_items(result, extracted_items, fields)

            if not filtered_result:
                return False

            # Check if result is similar to extracted_items
            # Allow some variance (code might find more/fewer items)
            if len(filtered_result) < len(extracted_items) * 0.5:
                return False

            # Check if extracted items match expected items (not just count)
            # Compare first few items to see if they're similar
            match_count = 0
            for expected_item in extracted_items[:5]:
                for extracted_item in filtered_result[:10]:
                    if self._items_match(expected_item, extracted_item, fields):
                        match_count += 1
                        break

            # At least 60% of expected items should match
            if match_count < len(extracted_items[:5]) * 0.6:
                logger.warning(f"Pattern validation: Only {match_count}/{len(extracted_items[:5])} items matched expected items")
                return False

            # Check if fields are present (more lenient - fields are optional)
            if filtered_result:
                first_result = filtered_result[0]
                if isinstance(first_result, dict):
                    present_fields = [f for f in fields if f in first_result and first_result[f] is not None]
                    # Only require 30% of fields to be present (fields are optional)
                    # Also accept if at least 1 field is present
                    if len(present_fields) < max(1, len(fields) * 0.3):
                        return False

            return True

        except Exception as e:
            logger.warning(f"Code validation failed: {e}")
            return False

    def _filter_invalid_items(self, items: List[Dict], expected_items: List[Dict], fields: List[str]) -> List[Dict]:
        """Filter out navigation links, footer links, and other invalid items"""
        filtered = []

        # Common patterns for invalid items (navigation/footer links)
        invalid_patterns = [
            r'^/$',  # Root path
            r'^/#',  # Hash links
            r'/categories\?',  # Category query params
            r'/products/.*/reviews',  # Review pages
            r'/products/.*\?filter=',  # Filtered pages
            r'\?ref=footer',  # Footer links
            r'^mailto:',  # Email links
            r'^https://(x\.com|twitter\.com|linkedin\.com)',  # Social links
            r'/legal',  # Legal pages
            r'/about',  # About pages
            r'/sponsor',  # Sponsor pages
            r'/newsletters',  # Newsletter pages
            r'/apps',  # Apps pages
            r'/p/',  # Post pages (not products)
        ]

        for item in items:
            if not isinstance(item, dict):
                continue

            # Check if item looks like a navigation/footer link
            url = item.get('url', '') or item.get('link', '') or ''
            title = item.get('title', '') or item.get('name', '') or ''

            # Skip if URL matches invalid patterns
            is_invalid = False
            for pattern in invalid_patterns:
                if re.search(pattern, str(url), re.I):
                    is_invalid = True
                    break

            # Skip if title is just a path (likely navigation)
            if not is_invalid and title:
                if title.startswith('/') and len(title) < 50:
                    # Check if it's a simple path (not a product name)
                    if not any(char.isupper() for char in title) and '/' in title:
                        is_invalid = True

            # Skip if item has no meaningful content
            if not is_invalid:
                has_content = False
                for field in fields:
                    value = item.get(field)
                    if value and str(value).strip() and str(value) not in ['/', '#', 'null', 'None']:
                        # Check if value looks like actual content (not just a path)
                        if len(str(value)) > 3 and not str(value).startswith('/'):
                            has_content = True
                            break

                if not has_content:
                    is_invalid = True

            if not is_invalid:
                filtered.append(item)

        return filtered

    def _items_match(self, item1: Dict, item2: Dict, fields: List[str]) -> bool:
        """Check if two items match (similar content)"""
        matches = 0
        total_checked = 0

        for field in fields:
            val1 = item1.get(field)
            val2 = item2.get(field)

            if val1 and val2:
                total_checked += 1
                # Normalize values for comparison
                val1_str = str(val1).lower().strip()
                val2_str = str(val2).lower().strip()

                # Check if values are similar (exact match or substring)
                if val1_str == val2_str or val1_str in val2_str or val2_str in val1_str:
                    matches += 1

        # Items match if at least 50% of compared fields match
        if total_checked == 0:
            return False
        return matches / total_checked >= 0.5

    def _extract_item_selector(self, code: str) -> Optional[str]:
        """Extract the item selector from generated code"""
        # Look for .select() or .find_all() calls
        patterns = [
            r"\.select\(['\"]([^'\"]+)['\"]\)",
            r"\.find_all\(['\"]([^'\"]+)['\"]\)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, code)
            if matches:
                return matches[0]

        return None

    def _extract_field_extractors(self, code: str, fields: List[str]) -> List[FieldExtractor]:
        """Extract field extractors from generated code"""
        extractors = []

        for field in fields:
            # Look for patterns like: item['field'] = ... or 'field': ...
            patterns = [
                rf"['\"]?{field}['\"]?\s*[=:]\s*([^\n,}}]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, code, re.I)
                if match:
                    extraction_expr = match.group(1).strip()

                    # Try to extract selector from expression
                    selector_match = re.search(r"\.select_one\(['\"]([^'\"]+)['\"]\)", extraction_expr)
                    attr_match = re.search(r"\.get\(['\"]([^'\"]+)['\"]\)", extraction_expr)

                    extractor = FieldExtractor(
                        field_name=field,
                        selector=selector_match.group(1) if selector_match else None,
                        attribute=attr_match.group(1) if attr_match else None,
                    )
                    extractors.append(extractor)
                    break
            else:
                # Default extractor
                extractors.append(FieldExtractor(field_name=field))

        return extractors

    async def _generate_attribute_pattern(
        self,
        html: str,
        extracted_items: List[Dict],
        fields: List[str],
        url: str,
        structure_hash: str
    ) -> Optional[ExtractionPattern]:
        """
        Generate pattern for custom elements with data attributes.

        This handles cases like:
        - <shreddit-post author="user" score="100">
        - <div data-title="..." data-price="...">
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        if not extracted_items:
            return None

        first_item = extracted_items[0]

        # Find elements with data attributes matching our values
        for elem in soup.find_all(True):
            attrs = elem.attrs
            if not attrs:
                continue

            matches = 0
            field_to_attr = {}

            for field in fields:
                if field not in first_item or not first_item[field]:
                    continue

                value = str(first_item[field])[:50].lower()

                # Check each attribute
                for attr_name, attr_value in attrs.items():
                    if isinstance(attr_value, str) and value in attr_value.lower():
                        matches += 1
                        field_to_attr[field] = attr_name
                        break

            if matches >= len(fields) * 0.5:
                # Found a good match!
                item_selector = elem.name
                if elem.get('class'):
                    item_selector += '.' + '.'.join(elem.get('class'))

                field_extractors = [
                    FieldExtractor(
                        field_name=field,
                        attribute=field_to_attr.get(field, f"data-{field}"),
                        extract_text=False
                    )
                    for field in fields
                ]

                return ExtractionPattern(
                    pattern_id="",
                    domain="",
                    structure_hash=structure_hash,
                    pattern_type=PatternType.ATTRIBUTE_MAP,
                    item_selector=item_selector,
                    field_extractors=field_extractors,
                    learned_via="attribute_detection"
                )

        return None


class PatternExecutor:
    """
    Executes extraction patterns to extract data.

    This is the fast path - no LLM needed, just deterministic extraction.
    """

    def execute(self, pattern: ExtractionPattern, html: str) -> List[Dict[str, Any]]:
        """
        Execute an extraction pattern on HTML.

        Args:
            pattern: The extraction pattern to use
            html: HTML content to extract from

        Returns:
            List of extracted items (filtered to exclude navigation/footer links)
        """
        logger.info(f"⚡ Executing {pattern.pattern_type.value} pattern (no LLM)")

        items = []
        if pattern.pattern_type == PatternType.JSON_EMBEDDED:
            items = self._execute_json_embedded(pattern, html)

        elif pattern.pattern_type == PatternType.CSS_SELECTORS:
            items = self._execute_css_selectors(pattern, html)

        elif pattern.pattern_type == PatternType.ATTRIBUTE_MAP:
            items = self._execute_attribute_map(pattern, html)

        elif pattern.pattern_type == PatternType.JSON_API:
            items = self._execute_json_api(pattern)

        else:
            logger.warning(f"Unknown pattern type: {pattern.pattern_type}")
            return []

        # Filter out invalid items (navigation/footer links) for all pattern types
        filtered_items = [item for item in items if self._is_valid_item(item, pattern.expected_fields)]

        if len(filtered_items) < len(items):
            logger.info(f"   Filtered out {len(items) - len(filtered_items)} invalid items (navigation/footer links)")

        return filtered_items

    def _execute_json_embedded(self, pattern: ExtractionPattern, html: str) -> List[Dict[str, Any]]:
        """Execute JSON embedded pattern"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        items_path = pattern.json_items_path or ""

        # Find the JSON source
        if "__NEXT_DATA__" in items_path:
            script = soup.find('script', id='__NEXT_DATA__')
            if script:
                try:
                    data = json.loads(script.string)
                    # Navigate to items
                    path_parts = items_path.replace("__NEXT_DATA__.", "").split(".")
                    for part in path_parts:
                        if part and part != ".":
                            if "[" in part:
                                key = part.split("[")[0]
                                data = data.get(key, [])
                            else:
                                data = data.get(part, {})

                    if isinstance(data, list):
                        return self._extract_fields_from_json(data, pattern.expected_fields)
                except Exception as e:
                    logger.error(f"JSON embedded extraction failed: {e}")

        # Try JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    return self._extract_fields_from_json(data, pattern.expected_fields)
                elif isinstance(data, dict):
                    if "@graph" in data:
                        return self._extract_fields_from_json(data["@graph"], pattern.expected_fields)
            except Exception:
                continue

        return []

    def _execute_css_selectors(self, pattern: ExtractionPattern, html: str) -> List[Dict[str, Any]]:
        """Execute CSS selector pattern"""
        from bs4 import BeautifulSoup

        # If we have generated code, execute it
        if pattern.extraction_code:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                from .sandbox import safe_exec
                namespace = safe_exec(pattern.extraction_code, {'soup': soup, 'BeautifulSoup': BeautifulSoup})

                if 'extract_data' in namespace:
                    return namespace['extract_data'](soup)
            except Exception as e:
                logger.error(f"Code execution failed: {e}")

        # Fallback to using field extractors
        if pattern.item_selector and pattern.field_extractors:
            soup = BeautifulSoup(html, 'html.parser')
            items = []

            for elem in soup.select(pattern.item_selector):
                item = {}
                for extractor in pattern.field_extractors:
                    value = self._extract_field(elem, extractor)
                    item[extractor.field_name] = value
                items.append(item)

            return items

        return []

    def _execute_attribute_map(self, pattern: ExtractionPattern, html: str) -> List[Dict[str, Any]]:
        """Execute attribute map pattern"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        if not pattern.item_selector:
            return []

        items = []
        for elem in soup.select(pattern.item_selector):
            # Skip navigation/footer elements
            parent = elem.parent
            skip = False
            for _ in range(5):
                if parent:
                    if parent.name in ['nav', 'footer', 'header', 'aside']:
                        skip = True
                        break
                    parent = parent.parent
                else:
                    break

            if skip:
                continue

            # Extract item
            item = {}
            for extractor in pattern.field_extractors:
                if extractor.attribute:
                    value = elem.get(extractor.attribute)
                else:
                    value = elem.get(f"data-{extractor.field_name}") or elem.get(extractor.field_name)

                item[extractor.field_name] = value
            items.append(item)

        return items

    def _is_valid_item(self, item: Dict, fields: List[str]) -> bool:
        """Check if item is valid (not a navigation/footer link)"""
        url = item.get('url', '') or item.get('link', '') or ''
        title = item.get('title', '') or item.get('name', '') or ''

        # Invalid patterns
        invalid_patterns = [
            r'^/$', r'^/#', r'/categories\?', r'/products/.*/reviews',
            r'\?ref=footer', r'^mailto:', r'/legal', r'/about',
            r'/sponsor', r'/newsletters', r'/apps', r'/p/'
        ]

        # Check URL
        for pattern in invalid_patterns:
            if re.search(pattern, str(url), re.I):
                return False

        # Check if title is just a path
        if title and title.startswith('/') and len(title) < 50:
            if not any(char.isupper() for char in title) and '/' in title:
                return False

        # Must have at least one meaningful field
        for field in fields:
            value = item.get(field)
            if value and str(value).strip() and str(value) not in ['/', '#', 'null', 'None']:
                if len(str(value)) > 3 and not str(value).startswith('/'):
                    return True

        return False

    def _execute_json_api(self, pattern: ExtractionPattern) -> List[Dict[str, Any]]:
        """Execute JSON API pattern (direct endpoint)"""
        import requests

        if not pattern.json_endpoint:
            return []

        try:
            response = requests.request(
                method=pattern.json_method,
                url=pattern.json_endpoint,
                headers=pattern.json_headers or {},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                return self._extract_fields_from_json(data, pattern.expected_fields)
            elif isinstance(data, dict) and pattern.json_items_path:
                # Navigate to items
                path_parts = pattern.json_items_path.split(".")
                for part in path_parts:
                    if part:
                        data = data.get(part, {})

                if isinstance(data, list):
                    return self._extract_fields_from_json(data, pattern.expected_fields)

        except Exception as e:
            logger.error(f"JSON API extraction failed: {e}")

        return []

    def _extract_field(self, elem, extractor: FieldExtractor) -> Any:
        """Extract a single field from an element"""
        value = None

        if extractor.selector:
            sub_elem = elem.select_one(extractor.selector)
            if sub_elem:
                if extractor.attribute:
                    value = sub_elem.get(extractor.attribute)
                elif extractor.extract_text:
                    value = sub_elem.get_text(strip=extractor.strip)

        elif extractor.attribute:
            value = elem.get(extractor.attribute)

        elif extractor.extract_text:
            value = elem.get_text(strip=extractor.strip)

        if value is None:
            value = extractor.default_value

        return value

    def _extract_fields_from_json(self, items: List[Dict], fields: List[str]) -> List[Dict[str, Any]]:
        """Extract specified fields from JSON items"""
        result = []

        for item in items:
            if not isinstance(item, dict):
                continue

            extracted = {}
            for field in fields:
                # Try exact match
                if field in item:
                    extracted[field] = item[field]
                else:
                    # Try case-insensitive match
                    for key, value in item.items():
                        if key.lower() == field.lower():
                            extracted[field] = value
                            break

            if extracted:
                result.append(extracted)

        return result

