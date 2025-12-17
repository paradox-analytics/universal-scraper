"""
Pagination Strategy Executor

Executes pagination strategies determined by LLM analyzer.
Deterministic execution based on LLM instructions (no additional LLM calls).
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PaginationExecutor:
    """
    Executes pagination strategies deterministically.
    No LLM calls - uses cached strategy from analyzer.
    """
    
    def __init__(self, browser_fetcher=None):
        self.browser_fetcher = browser_fetcher
    
    async def execute_strategy(
        self,
        strategy,  # PaginationStrategy object
        url: str,
        html: str,
        fetch_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute pagination strategy and return all items.
        
        Args:
            strategy: PaginationStrategy from analyzer
            url: Page URL
            html: Initial HTML content
            fetch_result: Result from initial fetch (may contain captured_json)
        
        Returns:
            List of all extracted items
        """
        
        logger.info(f" Executing {strategy.type} pagination strategy")
        
        try:
            if strategy.type == "preloaded_json":
                return self._execute_preloaded_json(html, fetch_result, strategy.strategy)
            
            elif strategy.type == "api_load_more":
                return await self._execute_api_load_more(url, strategy.strategy)
            
            elif strategy.type == "infinite_scroll":
                return await self._execute_infinite_scroll(url, strategy.strategy)
            
            elif strategy.type == "url_based":
                return await self._execute_url_based(url, strategy.strategy)
            
            elif strategy.type == "none":
                # Single page, extract from current data
                return self._execute_single_page(html, fetch_result)
            
            else:
                logger.warning(f" Unknown pagination type: {strategy.type}")
                return []
        
        except Exception as e:
            logger.error(f" Failed to execute pagination strategy: {e}")
            # Fallback: try to extract from current data
            return self._execute_single_page(html, fetch_result)
    
    def _execute_preloaded_json(
        self,
        html: str,
        fetch_result: Dict[str, Any],
        strategy_details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract ALL items from preloaded JSON in HTML.
        FAST - no additional requests needed!
        """
        
        logger.info(" Extracting from preloaded JSON...")
        
        data_location = strategy_details.get('data_location', 'script')
        json_path = strategy_details.get('json_path', 'auto_detect')
        
        # First, check if we already have captured JSON
        captured_json = fetch_result.get('captured_json', [])
        if captured_json:
            logger.info(f" Using {len(captured_json)} captured JSON blobs")
            items = self._extract_items_from_json_list(captured_json)
            if items:
                return items
        
        # Extract JSON from HTML
        soup = BeautifulSoup(html, 'html.parser')
        json_data = None
        
        # Try to find JSON by location
        if data_location and data_location != 'script':
            script = soup.select_one(data_location)
            if script and script.string:
                try:
                    json_data = json.loads(script.string)
                    logger.info(f" Found JSON at {data_location}")
                except json.JSONDecodeError:
                    pass
        
        # Fallback: Search all scripts
        if not json_data:
            json_data = self._find_json_in_scripts(soup)
        
        if not json_data:
            logger.warning(" No JSON data found in HTML")
            return []
        
        # Extract items using path or auto-detection
        if json_path and json_path != 'auto_detect':
            items = self._get_nested_value(json_data, json_path)
            if items and isinstance(items, list):
                logger.info(f" Extracted {len(items)} items from path: {json_path}")
                return items
        
        # Auto-detect items
        items = self._find_items_recursively(json_data)
        if items:
            logger.info(f" Auto-detected {len(items)} items in JSON")
        
        return items
    
    async def _execute_api_load_more(
        self,
        url: str,
        strategy_details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute Load More button clicking with API interception"""
        
        if not self.browser_fetcher:
            logger.error(" Browser fetcher required for Load More pagination")
            return []
        
        button_selector = strategy_details.get('button_selector')
        api_keywords = strategy_details.get('api_url_keywords', ['load', 'more', 'menu', 'products'])
        
        if not button_selector:
            logger.error(" No button selector provided in strategy")
            return []
        
        logger.info(f" Clicking Load More button: {button_selector}")
        
        # Use browser fetcher's Load More method
        fetch_result = await self.browser_fetcher.fetch(
            url,
            click_load_more=button_selector
        )
        
        # Extract items from all captured JSON
        captured_json = fetch_result.get('captured_json', [])
        items = self._extract_items_from_json_list(captured_json)
        
        logger.info(f" Extracted {len(items)} items via Load More")
        return items
    
    async def _execute_infinite_scroll(
        self,
        url: str,
        strategy_details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute infinite scroll with API monitoring"""
        
        if not self.browser_fetcher:
            logger.error(" Browser fetcher required for infinite scroll")
            return []
        
        logger.info(" Executing infinite scroll...")
        
        fetch_result = await self.browser_fetcher.fetch(
            url,
            scroll_to_bottom=True
        )
        
        # Extract items from captured JSON
        captured_json = fetch_result.get('captured_json', [])
        items = self._extract_items_from_json_list(captured_json)
        
        logger.info(f" Extracted {len(items)} items via infinite scroll")
        return items
    
    async def _execute_url_based(
        self,
        url: str,
        strategy_details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute URL-based pagination (traditional page numbers)"""
        
        logger.info(" Executing URL-based pagination...")
        
        # This would require crawler integration
        # For now, return empty and let crawler handle it
        logger.warning(" URL-based pagination requires crawler integration")
        return []
    
    def _execute_single_page(
        self,
        html: str,
        fetch_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract from single page (no pagination)"""
        
        logger.info(" Extracting from single page...")
        
        # Try captured JSON first
        captured_json = fetch_result.get('captured_json', [])
        if captured_json:
            items = self._extract_items_from_json_list(captured_json)
            if items:
                return items
        
        # Fallback: search HTML for JSON
        soup = BeautifulSoup(html, 'html.parser')
        json_data = self._find_json_in_scripts(soup)
        
        if json_data:
            items = self._find_items_recursively(json_data)
            if items:
                return items
        
        logger.warning(" No items found on single page")
        return []
    
    def _extract_items_from_json_list(self, json_list: List[Dict]) -> List[Dict]:
        """Extract items from a list of JSON blobs"""
        
        all_items = []
        
        for json_blob in json_list:
            items = self._find_items_recursively(json_blob)
            if items:
                all_items.extend(items)
        
        # Remove duplicates based on item content
        unique_items = []
        seen = set()
        
        for item in all_items:
            # Create a simple hash of the item
            item_str = json.dumps(item, sort_keys=True)
            item_hash = hash(item_str)
            
            if item_hash not in seen:
                seen.add(item_hash)
                unique_items.append(item)
        
        return unique_items
    
    def _find_json_in_scripts(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Find JSON data in script tags"""
        
        # Priority 1: Scripts with known patterns
        patterns = [
            (r'__NEXT_DATA__["\']?\s*type=["\']application/json["\']?\s*>([^<]+)</script>', 'Next.js data'),
            (r'window\.__INITIAL_STATE__\s*=\s*({.+?});', 'Initial state'),
            (r'window\.__DATA__\s*=\s*({.+?});', 'Window data'),
        ]
        
        html_str = str(soup)
        
        for pattern, desc in patterns:
            matches = re.findall(pattern, html_str, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        data = json.loads(match)
                        logger.info(f" Found {desc}")
                        return data
                    except json.JSONDecodeError:
                        continue
        
        # Priority 2: Scripts with type="application/json"
        json_scripts = soup.find_all('script', type='application/json')
        for script in json_scripts:
            if script.string:
                try:
                    data = json.loads(script.string)
                    return data
                except json.JSONDecodeError:
                    continue
        
        # Priority 3: Any script with JSON-like content
        all_scripts = soup.find_all('script')
        for script in all_scripts:
            if script.string and len(script.string) > 100:
                # Look for JSON patterns
                if 'props' in script.string or 'data' in script.string:
                    try:
                        # Try to extract JSON object
                        json_match = re.search(r'({.+})', script.string, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group(1))
                            return data
                    except (json.JSONDecodeError, AttributeError):
                        continue
        
        return None
    
    def _find_items_recursively(self, data: Any, max_depth: int = 10, current_depth: int = 0) -> List[Dict]:
        """
        Recursively search for item arrays in JSON structure.
        Returns the largest array that looks like product/item data.
        """
        
        if current_depth > max_depth:
            return []
        
        found_arrays = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Look for keys that suggest item arrays
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in ['items', 'products', 'menu', 'results', 'data', 'list']):
                    if isinstance(value, list) and len(value) > 0:
                        if self._looks_like_items(value):
                            found_arrays.append(value)
                
                # Recurse into nested structures
                nested_items = self._find_items_recursively(value, max_depth, current_depth + 1)
                if nested_items:
                    found_arrays.append(nested_items)
        
        elif isinstance(data, list):
            # Check if this list itself looks like items
            if self._looks_like_items(data):
                found_arrays.append(data)
            
            # Recurse into list items
            for item in data:
                nested_items = self._find_items_recursively(item, max_depth, current_depth + 1)
                if nested_items:
                    found_arrays.append(nested_items)
        
        # Return the largest array found
        if found_arrays:
            return max(found_arrays, key=len)
        
        return []
    
    def _looks_like_items(self, array: List) -> bool:
        """Check if array looks like product/item data"""
        
        if not array or not isinstance(array, list):
            return False
        
        if len(array) == 0:
            return False
        
        # Sample first few items
        sample_size = min(3, len(array))
        for i in range(sample_size):
            item = array[i]
            
            if not isinstance(item, dict):
                return False
            
            # Check for product-like fields
            item_keys_lower = [k.lower() for k in item.keys()]
            product_indicators = ['name', 'title', 'price', 'id', 'url', 'product', 'item']
            
            if not any(indicator in ' '.join(item_keys_lower) for indicator in product_indicators):
                return False
        
        return True
    
    def _get_nested_value(self, obj: Dict, path: str) -> Any:
        """Get value from nested object using dot notation"""
        
        if not path or path == 'auto_detect':
            return None
        
        keys = path.split('.')
        value = obj
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value

