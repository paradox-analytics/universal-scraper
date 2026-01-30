"""
Incremental Field Extractor - Extracts new fields using existing pattern structure

This module implements the hybrid approach:
1. Try hardcoded patterns first (instant, free)
2. Use pattern-guided LLM (small targeted call)
3. Fall back to full extraction if needed
"""

import logging
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from .extraction_pattern import ExtractionPattern

logger = logging.getLogger(__name__)


class IncrementalFieldExtractor:
    """
    Extracts new fields incrementally using existing pattern structure.
    
    Uses hybrid approach:
    1. Hardcoded pattern matching (fast, free)
    2. Pattern-guided LLM (smart, cheap)
    3. Full extraction fallback (last resort)
    """
    
    # Hardcoded patterns for common fields
    HARDCODED_PATTERNS = {
        "url": [
            ("selector", "a[href]"),
            ("selector", ".url"),
            ("selector", "[data-url]"),
            ("selector", ".link"),
            ("attribute", "href"),
            ("attribute", "data-url"),
        ],
        "image": [
            ("selector", "img[src]"),
            ("selector", ".image"),
            ("selector", "[data-image]"),
            ("attribute", "src"),
            ("attribute", "data-image"),
        ],
        "description": [
            ("selector", ".description"),
            ("selector", "[data-description]"),
            ("selector", "p.description"),
            ("selector", ".desc"),
        ],
        "price": [
            ("selector", ".price"),
            ("selector", "[data-price]"),
            ("selector", ".cost"),
            ("attribute", "data-price"),
        ],
        "rating": [
            ("selector", ".rating"),
            ("selector", "[data-rating]"),
            ("selector", ".stars"),
            ("attribute", "data-rating"),
        ],
        "author": [
            ("selector", ".author"),
            ("selector", "[data-author]"),
            ("selector", ".byline"),
            ("attribute", "data-author"),
        ],
        "date": [
            ("selector", ".date"),
            ("selector", "[data-date]"),
            ("selector", ".published"),
            ("attribute", "data-date"),
            ("attribute", "datetime"),
        ],
    }
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name
    
    async def extract_fields(
        self,
        pattern: ExtractionPattern,
        html: str,
        existing_fields: List[str],
        new_fields: List[str],
        existing_items: List[Dict[str, Any]],
        html_elements: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract new fields using existing pattern structure.
        
        Args:
            pattern: Existing extraction pattern
            html: HTML content
            existing_fields: Fields already extracted by pattern
            new_fields: Fields to extract incrementally
            existing_items: Items with existing fields already extracted
        
        Returns:
            Dict with 'items' (list of dicts with new fields) and 'source' (extraction method)
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # CRITICAL: Re-execute the pattern to get the exact same items and ensure we're working with the right elements
        # This is the key - we need to use the pattern's extraction logic to find items, not our own selector logic
        from .extraction_pattern import PatternExecutor
        executor = PatternExecutor()
        
        # Re-execute pattern to verify it still works and get the count
        try:
            reexecuted_items = executor.execute(pattern, html, url)
            logger.info(f"🔍 Re-executed pattern: found {len(reexecuted_items)} items (original had {len(existing_items)})")
            
            if len(reexecuted_items) != len(existing_items):
                logger.warning(f"⚠️ Pattern re-execution mismatch: expected {len(existing_items)}, got {len(reexecuted_items)}")
                # Use the re-executed count as source of truth
                if len(reexecuted_items) < len(existing_items):
                    logger.warning(f"⚠️ Pattern found fewer items - some items may have been removed from page")
        except Exception as e:
            logger.error(f"❌ Failed to re-execute pattern: {e}")
            # Fall back to finding elements manually
            reexecuted_items = None
        
        # Use provided HTML elements if available (ensures we use same elements as pattern)
        if html_elements is not None and len(html_elements) == len(existing_items):
            items_elements = html_elements
            logger.info(f"✅ Using {len(items_elements)} HTML elements provided by pattern")
        else:
            # Find items using pattern's container/item selectors
            # Try multiple methods to ensure we find all items
            items_elements = self._find_items(soup, pattern)
            
            # If we found fewer items than the pattern extracted, try alternative methods
            if len(items_elements) != len(existing_items):
                logger.info(f"🔍 Found {len(items_elements)} HTML elements, but pattern extracted {len(existing_items)} items")
                
                # Try using pattern executor's logic directly
                if reexecuted_items and len(reexecuted_items) == len(existing_items):
                    # Re-find items using the same method executor uses
                    items_elements = self._find_items_using_executor_logic(soup, pattern)
                    logger.info(f"🔍 Re-found {len(items_elements)} items using executor logic")
        
        if not items_elements:
            logger.warning("⚠️ Could not find items using pattern selectors")
            # Return empty dicts for all existing items (preserve count)
            return {
                "items": [{} for _ in existing_items],
                "source": "failed"
            }
        
        # CRITICAL: Ensure we extract fields for ALL existing_items
        # Use existing_items count as source of truth (pattern found these)
        logger.info(f"📊 Pattern extracted {len(existing_items)} items, found {len(items_elements)} HTML elements")
        
        extracted_items = []
        
        # Extract new fields for each existing item
        for i, existing_item in enumerate(existing_items):
            new_item_data = {}
            
            # Get corresponding HTML element if available
            item_elem = items_elements[i] if i < len(items_elements) else None
            
            if item_elem is None:
                # If we can't find the element, still create entry with None values
                # This preserves the item count even if we can't find the HTML element
                logger.debug(f"⚠️ Could not find HTML element for item {i}/{len(existing_items)}, using None for new fields")
                for field in new_fields:
                    new_item_data[field] = None
            else:
                # Extract new fields from this element
                for field in new_fields:
                    # Try hardcoded patterns first
                    value = self._try_hardcoded_pattern(field, item_elem, pattern)
                    
                    if value is None:
                        # Try pattern-guided extraction
                        value = await self._try_pattern_guided_extraction(
                            field, item_elem, pattern, existing_fields, existing_item
                        )
                    
                    new_item_data[field] = value
            
            extracted_items.append(new_item_data)
        
        # Log if we found fewer elements than items
        if len(items_elements) < len(existing_items):
            logger.warning(
                f"⚠️ MISMATCH: Found {len(items_elements)} HTML elements but pattern extracted {len(existing_items)} items. "
                f"Items {len(items_elements)}-{len(existing_items)-1} will have None for new fields. "
                f"Pattern item_selector: {pattern.item_selector}, container_selector: {pattern.container_selector}"
            )
        elif len(items_elements) > len(existing_items):
            logger.warning(
                f"⚠️ MISMATCH: Found {len(items_elements)} HTML elements but pattern extracted {len(existing_items)} items. "
                f"Will only extract fields for first {len(existing_items)} elements."
            )
        
        source = "hardcoded" if all(
            self._try_hardcoded_pattern(f, items_elements[0] if items_elements else None, pattern) is not None
            for f in new_fields
        ) else "pattern_guided_llm"
        
        return {
            "items": extracted_items,
            "source": source
        }
    
    def _find_items(self, soup: BeautifulSoup, pattern: ExtractionPattern) -> List[Any]:
        """
        Find item elements using pattern's selectors.
        Uses the same logic as PatternExecutor to ensure consistency.
        """
        # Use the same logic as PatternExecutor._execute_css_selectors
        if pattern.item_selector:
            items = soup.select(pattern.item_selector)
            logger.debug(f"Found {len(items)} items using item_selector: {pattern.item_selector}")
            
            # If we found way more items than expected, the selector might be too broad
            # Try to filter to items that actually contain the expected fields
            if len(items) > 100:  # Suspiciously many items
                logger.warning(f"⚠️ Found {len(items)} items with selector '{pattern.item_selector}' - selector might be too broad")
                # Try to filter to items that have content matching expected fields
                # This is a heuristic - look for items that have text content (not just links)
                filtered_items = []
                for item in items:
                    # Skip if it's just a link with no other content
                    if item.name == 'a' and not item.find_all(['div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        continue
                    # Skip if it's a navigation/footer link (common patterns)
                    item_text = item.get_text().strip().lower()
                    if any(nav_word in item_text for nav_word in ['categories', 'footer', 'navigation', 'menu', 'legal', 'privacy', 'terms']):
                        continue
                    filtered_items.append(item)
                
                if len(filtered_items) > 0:
                    logger.info(f"🔍 Filtered {len(items)} items down to {len(filtered_items)} likely product items")
                    return filtered_items
            
            return items
        
        # If we have extraction code, try to extract item selector from it
        if pattern.extraction_code:
            # Try to find the item selector from the code
            import re
            # Look for .select() calls in the code
            select_matches = re.findall(r"\.select\(['\"]([^'\"]+)['\"]\)", pattern.extraction_code)
            if select_matches:
                # Use the first selector found (usually the item selector)
                item_selector = select_matches[0]
                items = soup.select(item_selector)
                logger.debug(f"Found {len(items)} items using selector from code: {item_selector}")
                return items
        
        # Fallback to container selector
        if pattern.container_selector:
            container = soup.select_one(pattern.container_selector)
            if container:
                # Try to find repeating elements - look for elements with same tag/class
                # This is a heuristic - ideally we'd use the pattern's item_selector
                items = container.find_all(True, recursive=True)
                # Filter to likely item elements (have common item-like classes)
                item_keywords = ['item', 'card', 'post', 'product', 'entry', 'article']
                filtered_items = [
                    elem for elem in items
                    if any(keyword in str(elem.get('class', [])).lower() for keyword in item_keywords)
                    or any(keyword in elem.name.lower() for keyword in ['article', 'div', 'li'])
                ]
                if filtered_items:
                    logger.debug(f"Found {len(filtered_items)} items using container heuristic")
                    return filtered_items[:100]  # Limit but higher than before
        
        logger.warning(f"Could not find items using pattern selectors (item_selector={pattern.item_selector}, container_selector={pattern.container_selector})")
        return []
    
    def _find_items_using_executor_logic(self, soup: BeautifulSoup, pattern: ExtractionPattern) -> List[Any]:
        """
        Find items using the exact same logic as PatternExecutor.
        This ensures we find the same items the pattern found.
        """
        # Use the same logic as PatternExecutor._execute_css_selectors
        if pattern.item_selector:
            return soup.select(pattern.item_selector)
        
        # If we have extraction code, try to execute it to find items
        if pattern.extraction_code:
            try:
                # Try to extract just the item finding part
                import re
                # Look for the main loop that finds items
                # Usually something like: for item in soup.select(...)
                loop_match = re.search(r"for\s+\w+\s+in\s+soup\.select\(['\"]([^'\"]+)['\"]\)", pattern.extraction_code)
                if loop_match:
                    item_selector = loop_match.group(1)
                    return soup.select(item_selector)
            except Exception as e:
                logger.debug(f"Could not extract item selector from code: {e}")
        
        return []
    
    def _try_hardcoded_pattern(
        self,
        field: str,
        item_elem: Any,
        pattern: ExtractionPattern
    ) -> Optional[Any]:
        """Try hardcoded patterns for common fields"""
        if field.lower() not in self.HARDCODED_PATTERNS:
            return None
        
        patterns = self.HARDCODED_PATTERNS[field.lower()]
        
        for pattern_type, pattern_value in patterns:
            try:
                if pattern_type == "selector":
                    elem = item_elem.select_one(pattern_value)
                    if elem:
                        if field.lower() == "url" and elem.name == "a":
                            return elem.get("href", "").strip()
                        return elem.get_text(strip=True)
                elif pattern_type == "attribute":
                    value = item_elem.get(pattern_value)
                    if value:
                        return value.strip()
            except Exception as e:
                logger.debug(f"Hardcoded pattern failed for {field}: {e}")
                continue
        
        return None
    
    async def _try_pattern_guided_extraction(
        self,
        field: str,
        item_elem: Any,
        pattern: ExtractionPattern,
        existing_fields: List[str],
        existing_item: Dict[str, Any]
    ) -> Optional[Any]:
        """Use pattern-guided LLM to extract new field"""
        if not self.api_key:
            logger.warning("No API key for pattern-guided extraction")
            return None
        
        # Get sample of existing field extractors
        existing_extractors = []
        for fe in pattern.field_extractors[:3]:  # Sample first 3
            existing_extractors.append({
                "field": fe.field_name,
                "selector": fe.selector or fe.xpath or fe.attribute,
                "type": "css" if fe.selector else ("xpath" if fe.xpath else "attribute")
            })
        
        # Get HTML sample around this item
        item_html = str(item_elem)[:2000]  # Limit size
        
        # Build prompt
        prompt = f"""Extract the '{field}' field from this HTML element.

Existing extraction pattern:
- Container selector: {pattern.item_selector or pattern.container_selector}
- Existing fields extracted: {existing_fields}
- Sample existing extractors: {existing_extractors[:2]}

HTML element:
{item_html}

Extract ONLY the '{field}' field value. Return just the value, no explanation.
If not found, return null."""

        try:
            import litellm
            response = await litellm.acompletion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a web scraping assistant. Extract field values from HTML."},
                    {"role": "user", "content": prompt}
                ],
                api_key=self.api_key,
                max_tokens=100  # Small response
            )
            
            value = response.choices[0].message.content.strip()
            
            # Clean up response (remove quotes, null handling)
            if value.lower() in ["null", "none", "n/a", ""]:
                return None
            
            # Remove quotes if present
            value = value.strip('"\'')
            
            return value if value else None
            
        except Exception as e:
            logger.warning(f"Pattern-guided extraction failed for {field}: {e}")
            return None

