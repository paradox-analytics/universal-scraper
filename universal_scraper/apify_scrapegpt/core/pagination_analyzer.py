"""
LLM-Based Universal Pagination Analyzer

Analyzes web pages to detect pagination patterns and provides deterministic extraction strategies.
Uses LLM once per domain (cached), then executes strategies deterministically.
"""

import json
import logging
import hashlib
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PaginationStrategy:
    """Container for pagination strategy returned by LLM"""
    
    def __init__(self, data: Dict[str, Any]):
        self.type = data.get('pagination_type', 'none')
        self.strategy = data.get('strategy', {})
        self.confidence = data.get('confidence', 'low')
        self.reasoning = data.get('reasoning', '')
        # Store original data for dict-like access
        self._data = data
    
    def is_high_confidence(self) -> bool:
        return self.confidence == 'high'
    
    def needs_browser(self) -> bool:
        """Determine if this strategy requires browser automation"""
        return self.type in ['api_load_more', 'infinite_scroll', 'graphql']
    
    def can_use_static_html(self) -> bool:
        """Determine if strategy can work with static HTML"""
        return self.type in ['preloaded_json', 'url_based', 'none']
    
    # Dict-like access methods for compatibility
    def __getitem__(self, key: str):
        """Support dict-style access like strategy['type']"""
        if key == 'type':
            return self.type
        elif key == 'strategy':
            return self.strategy
        elif key == 'confidence':
            return self.confidence
        elif key == 'reasoning':
            return self.reasoning
        return self._data.get(key)
    
    def __setitem__(self, key: str, value):
        """Support dict-style assignment like strategy['key'] = value"""
        if key == 'type':
            self.type = value
        elif key == 'strategy':
            self.strategy = value
        elif key == 'confidence':
            self.confidence = value
        elif key == 'reasoning':
            self.reasoning = value
        # Always store in _data as well
        self._data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator"""
        return key in ['type', 'strategy', 'confidence', 'reasoning'] or key in self._data
    
    def __bool__(self) -> bool:
        """Make object truthy (for 'if pagination_strategy' checks)"""
        return True
    
    def get(self, key: str, default=None):
        """Support .get() method"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self):
        """Support .keys() method"""
        return ['type', 'strategy', 'confidence', 'reasoning'] + list(self._data.keys())


class LLMPaginationAnalyzer:
    """
    Uses LLM to analyze pagination patterns once per domain, caches the strategy.
    
    Benefits:
    - Universal: Works with ANY pagination pattern
    - Cost-efficient: One LLM call per domain (cached)
    - Self-adapting: Handles new patterns automatically
    - Explainable: LLM provides reasoning
    """
    
    def __init__(self, openai_api_key: str, cache_dir: str = ".pagination_cache"):
        self.openai_api_key = openai_api_key
        self.cache_dir = cache_dir
        self.memory_cache = {}  # In-memory cache for current session
        
        # Use litellm (same as ai_generator) - universal interface
        try:
            import litellm
            self.llm = litellm
            # Set API key in environment
            import os
            os.environ['OPENAI_API_KEY'] = openai_api_key
            logger.info(" LLM Pagination Analyzer initialized")
        except ImportError:
            logger.error(" litellm not installed")
            self.llm = None
        except Exception as e:
            logger.error(f" Failed to initialize LLM: {e}")
            self.llm = None
    
    async def analyze_pagination_strategy(
        self, 
        url: str, 
        html: str,
        user_hints: Optional[Dict[str, Any]] = None
    ) -> PaginationStrategy:
        """
        Analyze page and return pagination strategy.
        
        Args:
            url: Page URL
            html: Page HTML content
            user_hints: Optional user-provided hints (load_more_selector, etc.)
        
        Returns:
            PaginationStrategy object with extraction instructions
        """
        
        if not self.llm:
            logger.warning(" LLM not available, using fallback strategy")
            return self._create_fallback_strategy(user_hints)
        
        # Check cache first
        cache_key = self._get_cache_key(url)
        cached_strategy = self._load_from_cache(cache_key)
        
        if cached_strategy:
            logger.info(f" Using cached pagination strategy for {urlparse(url).netloc}")
            return PaginationStrategy(cached_strategy)
        
        # Analyze with LLM
        logger.info(f" Analyzing pagination strategy with LLM for {urlparse(url).netloc}")
        
        try:
            strategy_data = await self._analyze_with_llm(url, html, user_hints)
            
            # Validate and cache
            if self._validate_strategy(strategy_data):
                self._save_to_cache(cache_key, strategy_data)
                strategy = PaginationStrategy(strategy_data)
                
                logger.info(f" LLM detected: {strategy.type} (confidence: {strategy.confidence})")
                logger.info(f" Reasoning: {strategy.reasoning}")
                
                return strategy
            else:
                logger.warning(" Invalid strategy from LLM, using fallback")
                return self._create_fallback_strategy(user_hints)
        
        except Exception as e:
            logger.error(f" LLM analysis failed: {e}")
            return self._create_fallback_strategy(user_hints)
    
    async def _analyze_with_llm(
        self, 
        url: str, 
        html: str, 
        user_hints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call LLM to analyze page structure"""
        
        # Prepare HTML sample for LLM
        html_sample = self._prepare_html_for_llm(html, url)
        
        # Build hints section if provided
        hints_text = ""
        if user_hints:
            hints_text = "\n\nUSER PROVIDED HINTS:\n"
            if user_hints.get('scroll_to_bottom'):
                hints_text += "- User expects infinite scroll\n"
            if user_hints.get('click_load_more'):
                hints_text += f"- Load More button selector: {user_hints['click_load_more']}\n"
            if user_hints.get('wait_for_selector'):
                hints_text += f"- Wait for selector: {user_hints['wait_for_selector']}\n"
        
        # Construct prompt
        prompt = f"""You are analyzing a web page to determine how to extract ALL items, including paginated data.

URL: {url}

HTML STRUCTURE (key parts):
{html_sample}
{hints_text}

TASK: Analyze this page and return a JSON strategy for extracting all data efficiently.

PAGINATION PATTERNS TO DETECT:

1. **preloaded_json**: All data already embedded in initial HTML as JSON
   - Look for: <script id="__NEXT_DATA__">, window.__STATE__, etc.
   - Best for: Next.js, React SSR sites
   - Advantage: NO additional requests needed!
   
2. **api_load_more**: "Load More" button triggers API calls
   - Look for: Buttons with text like "Load More", "Show More", "Ver más"
   - Browser needed: Yes (to click and intercept)
   
3. **infinite_scroll**: Scrolling triggers new data loading
   - Look for: Long pages that load content on scroll
   - Browser needed: Yes
   
4. **url_based**: Traditional page number links
   - Look for: <a href="?page=2">, pagination controls
   - Browser needed: No (can crawl URLs)
   
5. **none**: All data visible, no pagination
   - Single page with all items

RESPONSE FORMAT (valid JSON only):
{{
  "pagination_type": "<one of: preloaded_json|api_load_more|infinite_scroll|url_based|none>",
  "strategy": {{
    // Type-specific extraction details (see below)
  }},
  "confidence": "<high|medium|low>",
  "reasoning": "Brief explanation (1-2 sentences)"
}}

STRATEGY DETAILS BY TYPE:

For "preloaded_json":
{{
  "data_location": "CSS selector or script ID",
  "json_path": "dot.notation.path.to.items",
  "total_items_path": "path.to.total.count" // optional
}}

For "api_load_more":
{{
  "button_selector": "CSS selector for button",
  "api_url_keywords": ["menu", "products", "items"],
  "response_items_path": "path.to.items.in.response"
}}

For "infinite_scroll":
{{
  "scroll_container": "CSS selector or 'window'",
  "api_url_keywords": ["load", "scroll", "more"]
}}

For "url_based":
{{
  "pagination_selector": "CSS selector for page links",
  "url_pattern": "?page={{page}}" // or similar
}}

For "none":
{{
  "extraction_method": "static_html"
}}

Return ONLY the JSON, no other text."""

        # Call LLM using litellm (same as ai_generator)
        response = self.llm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Clean markdown code blocks
        if response_text.startswith('```'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        strategy_data = json.loads(response_text)
        return strategy_data
    
    def _prepare_html_for_llm(self, html: str, url: str) -> str:
        """Extract key HTML parts for LLM analysis (reduce token count)"""
        
        soup = BeautifulSoup(html, 'html.parser')
        sample_parts = []
        
        # 1. JSON script tags (highest priority)
        json_scripts = soup.find_all('script', id=True)
        for script in json_scripts[:3]:
            if script.string and len(script.string) < 2000:
                sample_parts.append(f"<script id='{script.get('id')}'>{script.string[:500]}...</script>")
        
        # 2. Scripts with window.__ patterns
        for script in soup.find_all('script')[:5]:
            if script.string and any(pattern in script.string for pattern in ['window.__', '__NEXT_DATA__', 'props']):
                sample_parts.append(f"<script>{script.string[:500]}...</script>")
        
        # 3. Pagination-related elements
        pagination_keywords = ['pagination', 'page', 'load', 'more', 'next', 'previous']
        for keyword in pagination_keywords:
            elements = soup.find_all(class_=lambda x: x and keyword in x.lower())
            for elem in elements[:2]:
                sample_parts.append(str(elem)[:300])
        
        # 4. Buttons that might load more
        buttons = soup.find_all('button')
        for button in buttons[:5]:
            button_text = button.get_text().strip().lower()
            if any(word in button_text for word in ['load', 'more', 'show', 'ver', 'más']):
                sample_parts.append(str(button)[:200])
        
        # 5. Navigation elements
        nav_elements = soup.find_all('nav')
        for nav in nav_elements[:2]:
            sample_parts.append(str(nav)[:300])
        
        # 6. Sample of main content
        body = soup.find('body')
        if body:
            sample_parts.append(f"<body>{str(body)[:1000]}...</body>")
        
        return "\n\n".join(sample_parts)
    
    def _validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        """Validate strategy structure"""
        required_keys = ['pagination_type', 'strategy', 'confidence', 'reasoning']
        
        if not all(key in strategy for key in required_keys):
            logger.warning(f" Strategy missing required keys: {required_keys}")
            return False
        
        valid_types = ['preloaded_json', 'api_load_more', 'infinite_scroll', 'url_based', 'none']
        if strategy['pagination_type'] not in valid_types:
            logger.warning(f" Invalid pagination type: {strategy['pagination_type']}")
            return False
        
        return True
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from domain"""
        domain = urlparse(url).netloc
        return hashlib.md5(domain.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load strategy from cache"""
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        import os
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.memory_cache[cache_key] = data
                    return data
            except Exception as e:
                logger.warning(f" Failed to load cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, strategy: Dict[str, Any]):
        """Save strategy to cache"""
        
        # Save to memory
        self.memory_cache[cache_key] = strategy
        
        # Save to disk
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(strategy, f, indent=2)
            logger.info(f" Cached strategy for future use")
        except Exception as e:
            logger.warning(f" Failed to save cache: {e}")
    
    def _create_fallback_strategy(self, user_hints: Optional[Dict[str, Any]]) -> PaginationStrategy:
        """Create fallback strategy based on user hints or defaults"""
        
        # If user provided hints, use them
        if user_hints:
            if user_hints.get('click_load_more'):
                return PaginationStrategy({
                    "pagination_type": "api_load_more",
                    "strategy": {
                        "button_selector": user_hints['click_load_more'],
                        "api_url_keywords": ["menu", "products", "items", "load", "more"],
                        "response_items_path": "auto_detect"
                    },
                    "confidence": "medium",
                    "reasoning": "Using user-provided Load More selector"
                })
            
            elif user_hints.get('scroll_to_bottom'):
                return PaginationStrategy({
                    "pagination_type": "infinite_scroll",
                    "strategy": {
                        "scroll_container": "window",
                        "api_url_keywords": ["scroll", "load", "more"]
                    },
                    "confidence": "medium",
                    "reasoning": "Using user-provided infinite scroll hint"
                })
        
        # Default: Assume preloaded JSON (most common for modern sites)
        return PaginationStrategy({
            "pagination_type": "preloaded_json",
            "strategy": {
                "data_location": "script",
                "json_path": "auto_detect"
            },
            "confidence": "low",
            "reasoning": "Fallback strategy - will search for embedded JSON"
        })

