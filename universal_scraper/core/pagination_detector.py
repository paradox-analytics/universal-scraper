"""
Fast Pattern-Based Pagination Detection
Handles 90% of cases instantly before falling back to LLM
"""

import re
import logging
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urljoin

logger = logging.getLogger(__name__)


class FastPaginationDetector:
    """
    Fast, deterministic pagination detection using patterns.
    Falls back to LLM only when patterns don't match.
    
    Detection priority:
    1. URL parameters (?page=N, ?p=N) - 70% of sites
    2. Path-based (/page/N, /p/N) - 15% of sites
    3. Next/Prev links - 5% of sites
    4. Load More buttons - 5% of sites
    5. Infinite scroll - 3% of sites
    6. LLM fallback - 2% of sites (complex cases)
    """
    
    # URL parameter patterns
    URL_PARAM_PATTERNS = [
        r'[?&]page=(\d+)',
        r'[?&]p=(\d+)',
        r'[?&]pg=(\d+)',
        r'[?&]paged=(\d+)',
        r'[?&]pageNum=(\d+)',
        r'[?&]pageNumber=(\d+)',
        r'[?&]offset=(\d+)',
        r'[?&]start=(\d+)',
    ]
    
    # Path-based patterns
    PATH_PATTERNS = [
        r'/page/(\d+)',
        r'/p/(\d+)',
        r'/pg/(\d+)',
        r'/(\d+)/?$',  # Ending with number
    ]
    
    def detect(self, url: str, html: str, current_page_items: int = 0) -> Optional[Dict[str, Any]]:
        """
        Fast pattern-based detection.
        Returns strategy dict or None if no pattern found.
        """
        logger.info(f" Fast pagination detection for: {url}")
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Priority 1: URL parameter pagination (FASTEST, MOST COMMON)
        url_param_result = self._detect_url_params(url, soup)
        if url_param_result:
            logger.info(f" Detected URL parameter pagination: {url_param_result['param_name']}")
            return url_param_result
        
        # Priority 2: Path-based pagination
        path_result = self._detect_path_pagination(url, soup)
        if path_result:
            logger.info(f" Detected path-based pagination")
            return path_result
        
        # Priority 3: Next/Previous links
        link_result = self._detect_next_links(url, soup)
        if link_result:
            logger.info(f" Detected link-based pagination")
            return link_result
        
        # Priority 4: Load More buttons
        load_more_result = self._detect_load_more(soup, current_page_items)
        if load_more_result:
            logger.info(f" Detected Load More button pagination")
            return load_more_result
        
        # Priority 5: Infinite scroll indicators
        scroll_result = self._detect_infinite_scroll(soup, current_page_items)
        if scroll_result:
            logger.info(f" Detected infinite scroll pagination")
            return scroll_result
        
        logger.info("ℹ  No standard pagination pattern detected, may need LLM analysis")
        return None
    
    def _detect_url_params(self, url: str, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Detect URL parameter-based pagination (?page=N)"""
        
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Check if current URL has a page parameter
        param_name = None
        current_page = 1
        
        for pattern in self.URL_PARAM_PATTERNS:
            match = re.search(pattern, url)
            if match:
                param_name = pattern.split('=')[0].replace('[?&]', '')
                current_page = int(match.group(1))
                break
        
        if not param_name:
            # Check if pagination links exist with page params
            for link in soup.find_all('a', href=True):
                href = link['href']
                for pattern in self.URL_PARAM_PATTERNS:
                    if re.search(pattern, href):
                        param_name = pattern.split('=')[0].replace('[?&]', '')
                        break
                if param_name:
                    break
        
        if not param_name:
            return None
        
        # Try to find max page number
        max_page = self._extract_max_page(soup, current_page)
        
        # Build base URL without page param
        base_url = url.split('?')[0]
        other_params = '&'.join([f"{k}={v[0]}" for k, v in query_params.items() 
                                 if k not in ['page', 'p', 'pg', 'paged', 'pageNum', 'pageNumber']])
        
        return {
            'type': 'url_param',
            'param_name': param_name,
            'current_page': current_page,
            'max_page': max_page,
            'base_url': base_url,
            'other_params': other_params,
            'confidence': 'high',
            'strategy': 'enumerate_urls',
            'reasoning': f'Found {param_name} parameter in URL or pagination links'
        }
    
    def _detect_path_pagination(self, url: str, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Detect path-based pagination (/page/N)"""
        
        current_page = 1
        pattern_type = None
        
        for pattern in self.PATH_PATTERNS:
            match = re.search(pattern, url)
            if match:
                current_page = int(match.group(1))
                pattern_type = pattern
                break
        
        if not pattern_type:
            # Check pagination links
            for link in soup.find_all('a', href=True):
                href = link['href']
                for pattern in self.PATH_PATTERNS:
                    if re.search(pattern, href):
                        pattern_type = pattern
                        break
                if pattern_type:
                    break
        
        if not pattern_type:
            return None
        
        max_page = self._extract_max_page(soup, current_page)
        
        return {
            'type': 'path_based',
            'pattern': pattern_type,
            'current_page': current_page,
            'max_page': max_page,
            'confidence': 'high',
            'strategy': 'enumerate_urls',
            'reasoning': 'Found path-based pagination pattern'
        }
    
    def _detect_next_links(self, url: str, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Detect Next/Previous link-based pagination"""
        
        # Look for common next link patterns
        next_selectors = [
            'a[rel="next"]',
            'a.next',
            'a.pagination-next',
            'a[aria-label*="next" i]',
            'a:contains("Next")',
            'a:contains("→")',
            'a:contains("›")',
        ]
        
        next_link = None
        for selector in next_selectors:
            try:
                links = soup.select(selector)
                if links:
                    next_link = links[0].get('href')
                    break
            except:
                pass
        
        # Also check for text-based next links
        if not next_link:
            for link in soup.find_all('a', href=True):
                text = link.get_text().strip().lower()
                if text in ['next', 'next page', '→', '›', '»']:
                    next_link = link['href']
                    break
        
        if not next_link:
            return None
        
        # Estimate max pages if possible
        max_page = self._extract_max_page(soup, 1)
        
        return {
            'type': 'link_based',
            'next_link': urljoin(url, next_link),
            'max_page': max_page,
            'confidence': 'medium',
            'strategy': 'follow_links',
            'reasoning': 'Found "Next" link in pagination'
        }
    
    def _detect_load_more(self, soup: BeautifulSoup, current_items: int) -> Optional[Dict[str, Any]]:
        """Detect Load More button pagination"""
        
        # Common Load More button patterns
        load_more_patterns = [
            ('button[aria-label*="load more" i]', 'button'),
            ('button:contains("Load More")', 'button'),
            ('button:contains("Show More")', 'button'),
            ('a:contains("Load More")', 'link'),
            ('[data-load-more]', 'element'),
            ('.load-more', 'element'),
        ]
        
        selector = None
        for pattern, _ in load_more_patterns:
            try:
                elements = soup.select(pattern)
                if elements:
                    # Try to get a unique selector
                    elem = elements[0]
                    if elem.get('aria-label'):
                        selector = f"[aria-label='{elem['aria-label']}']"
                    elif elem.get('class'):
                        selector = f".{'.'.join(elem['class'])}"
                    else:
                        selector = elem.name
                    break
            except:
                pass
        
        if not selector:
            return None
        
        return {
            'type': 'load_more',
            'button_selector': selector,
            'current_items': current_items,
            'confidence': 'medium',
            'strategy': 'click_button',
            'reasoning': 'Found "Load More" button in page'
        }
    
    def _detect_infinite_scroll(self, soup: BeautifulSoup, current_items: int) -> Optional[Dict[str, Any]]:
        """Detect infinite scroll patterns"""
        
        # Look for indicators of infinite scroll
        indicators = [
            'data-infinite-scroll',
            'data-scroll',
            'infinite-scroll',
            'lazy-load',
        ]
        
        has_indicator = False
        for indicator in indicators:
            if soup.find(attrs={indicator: True}):
                has_indicator = True
                break
            if soup.find(class_=re.compile(indicator, re.I)):
                has_indicator = True
                break
        
        # Also check if there are many similar items (suggests listing page)
        if not has_indicator and current_items > 10:
            # Heuristic: If we have many items and no other pagination, might be infinite scroll
            has_indicator = True
        
        if not has_indicator:
            return None
        
        return {
            'type': 'infinite_scroll',
            'current_items': current_items,
            'confidence': 'low',
            'strategy': 'scroll',
            'reasoning': 'Found infinite scroll indicators or many items without pagination'
        }
    
    def _extract_max_page(self, soup: BeautifulSoup, current_page: int) -> Optional[int]:
        """
        Try to extract the maximum page number from pagination widgets.
        Uses multiple strategies, prioritizing embedded JSON data.
        """
        max_page = None
        import json
        
        # Strategy 0: Check __NEXT_DATA__ for pagination info (HIGHEST PRIORITY for Next.js sites)
        nextjs_script = soup.find('script', id='__NEXT_DATA__')
        logger.info(f" Checking for Next.js data: script tag found = {nextjs_script is not None}")
        if nextjs_script and nextjs_script.string:
            logger.info(f" Next.js script content found ({len(nextjs_script.string)} chars)")
            try:
                nextjs_data = json.loads(nextjs_script.string)
                
                # Common paths for pagination in Next.js
                paths_to_check = [
                    'props.pageProps.menuData.totalPages',
                    'props.pageProps.menuData.pagination.totalPages',
                    'props.pageProps.menuData.pageCount',
                    'props.pageProps.pagination.totalPages',
                    'props.pageProps.totalPages',
                    'props.pageProps.pageCount',
                    'props.pageProps.menuData.total_pages',
                ]
                
                for path in paths_to_check:
                    try:
                        parts = path.split('.')
                        value = nextjs_data
                        for part in parts:
                            value = value[part]
                        if isinstance(value, (int, float)):
                            max_page = int(value)
                            logger.info(f" Found max_page={max_page} in Next.js data at {path}")
                            break
                    except (KeyError, TypeError):
                        continue
                
                # Also check for total items and items per page to calculate pages
                if not max_page:
                    try:
                        menu_data = nextjs_data.get('props', {}).get('pageProps', {}).get('menuData', {})
                        items_per_page = len(menu_data.get('menuItems', []))
                        
                        logger.info(f" Next.js menuData keys: {list(menu_data.keys()) if menu_data else 'None'}")
                        
                        # UNIVERSAL FIELD DETECTION: Use semantic search instead of hardcoded list
                        total_items = self._find_total_count_universal(menu_data, items_per_page)
                        
                        logger.info(f" totalItems={total_items}, itemsPerPage={items_per_page}")
                        
                        if total_items and items_per_page and items_per_page > 0:
                            max_page = (total_items + items_per_page - 1) // items_per_page  # Ceiling division
                            logger.info(f" Calculated max_page={max_page} from totalItems={total_items}, itemsPerPage={items_per_page}")
                        else:
                            logger.warning(f" Cannot calculate max_page: totalItems={total_items}, itemsPerPage={items_per_page}")
                    except Exception as e:
                        logger.warning(f" Failed to calculate pages from menuData: {e}")
            except json.JSONDecodeError as e:
                logger.warning(f" Failed to parse Next.js JSON: {e}")
        
        if max_page:
            logger.info(f" Returning max_page={max_page} from Next.js data")
            return max_page
        
        logger.info(" No max_page found in Next.js data, trying HTML pagination widgets...")
        
        # Strategy 1: Find pagination container and extract numbers
        pagination_containers = soup.find_all(class_=re.compile(r'paginat', re.I))
        pagination_containers.extend(soup.find_all(attrs={'role': 'navigation'}))
        
        for container in pagination_containers:
            # Look for all numbers in links
            for link in container.find_all('a', href=True):
                text = link.get_text().strip()
                if text.isdigit():
                    page_num = int(text)
                    if max_page is None or page_num > max_page:
                        max_page = page_num
        
        # Strategy 2: Look for "Page X of Y" text
        page_info_patterns = [
            r'page\s+\d+\s+of\s+(\d+)',
            r'(\d+)\s+pages?',
            r'showing\s+\d+\s+of\s+(\d+)',
        ]
        
        text = soup.get_text()
        for pattern in page_info_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                found_max = int(match.group(1))
                if max_page is None or found_max > max_page:
                    max_page = found_max
        
        # Strategy 3: Look in JSON-LD or other JSON script tags
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    # Look for numberOfPages or similar
                    if 'numberOfPages' in data:
                        max_page = int(data['numberOfPages'])
                    elif 'pagination' in data and 'total' in data['pagination']:
                        max_page = int(data['pagination']['total'])
            except:
                pass
        
        if max_page:
            logger.info(f" Final max_page from all strategies: {max_page}")
        else:
            logger.warning(f" Could not determine max_page from any strategy")
        
        return max_page
    
    def _find_total_count_universal(self, data: dict, items_per_page: int) -> Optional[int]:
        """
        UNIVERSAL field detection for total item count.
        Uses semantic search + heuristics instead of hardcoded field names.
        
        Args:
            data: The data object to search
            items_per_page: Number of items per page (for validation)
        
        Returns:
            Total count if found, None otherwise
        """
        if not data or not isinstance(data, dict):
            return None
        
        # Strategy 1: Common exact field names (fast path - covers 90% of cases)
        common_names = [
            'totalItems', 'total_items',
            'totalCount', 'total_count', 
            'itemCount', 'item_count',
            'totalProducts', 'total_products',
            'totalResults', 'total_results',
            'totalMenuItemCount', 'total_menu_item_count',
            'count', 'total', 'numItems', 'num_items',
            'total_entries', 'totalEntries'
        ]
        
        for name in common_names:
            value = data.get(name)
            if isinstance(value, (int, float)) and value > 0:
                logger.info(f" Found total count via exact match: {name}={int(value)}")
                return int(value)
        
        # Strategy 2: Semantic/fuzzy search - find keys containing relevant keywords
        # This catches variants like "totalAvailableProducts", "item_count_total", etc.
        keywords = ['total', 'count', 'items', 'products', 'results', 'num', 'entries']
        
        for key, value in data.items():
            if not isinstance(value, (int, float)) or value <= 0:
                continue
            
            key_lower = key.lower()
            
            # Check if key contains any relevant keywords
            if any(keyword in key_lower for keyword in keywords):
                # Heuristic: Value should be larger than items_per_page (otherwise it's probably something else)
                if value > items_per_page or items_per_page == 0:
                    logger.info(f" Found total count via semantic search: {key}={int(value)}")
                    return int(value)
        
        # Strategy 3: Look in nested pagination/meta objects
        # Some APIs nest the total count: { pagination: { total: 100 } }
        nested_keys = ['pagination', 'paging', 'meta', 'pageInfo', 'page_info', 'pageMeta']
        
        for key in nested_keys:
            if key in data and isinstance(data[key], dict):
                result = self._find_total_count_universal(data[key], items_per_page)  # Recursive
                if result:
                    logger.info(f" Found total count in nested object: {key}.total={result}")
                    return result
        
        logger.debug(f" Could not find total count field in data with keys: {list(data.keys())}")
        return None

