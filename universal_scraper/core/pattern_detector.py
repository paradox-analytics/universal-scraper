"""
Pattern Detector - Determines the best extraction strategy for a page
Uses LLM to detect patterns in HTML structure (not hardcoded!)
"""

import logging
from typing import Dict, Any, Literal
from bs4 import BeautifulSoup
import litellm

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Detects extraction patterns using LLM analysis
    Cached per domain for efficiency
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize pattern detector
        
        Args:
            api_key: API key for LLM
            model: Model to use for pattern detection
        """
        self.api_key = api_key
        self.model = model
        self.pattern_cache = {}  # Cache by domain
    
    def detect_pattern(
        self, 
        html: str, 
        url: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Detect the extraction pattern for a page
        
        Returns dict with:
        - strategy: 'attributes' or 'nested_elements'
        - confidence: 0.0-1.0
        - reasoning: Explanation
        - details: Pattern-specific details
        
        Args:
            html: Cleaned HTML
            url: Source URL (for domain caching)
            use_cache: Whether to use cached pattern
        """
        from urllib.parse import urlparse
        
        domain = urlparse(url).netloc
        
        # Check cache
        if use_cache and domain in self.pattern_cache:
            logger.info(f" Using cached pattern for {domain}")
            return self.pattern_cache[domain]
        
        logger.info(f" Detecting extraction pattern for {domain}...")
        
        # Smart sampling - find actual content, not just headers
        sample = self._smart_sample(html)
        
        # Build prompt for LLM
        prompt = f"""Analyze this HTML and find REPEATING CONTENT ELEMENTS (posts, products, items).

HTML SAMPLE:
```html
{sample}
```

TASK: Find the REPEATING element that contains the data to extract.

Look for elements that:
1. REPEAT multiple times (3+ instances)
2. Contain actual content (titles, text, data)
3. NOT containers/wrappers (ignore single instances)

Then determine extraction strategy:
A) HTML ATTRIBUTES - Data in element attributes
   Examples: <shreddit-post author="user" score="42">, <product-card price="99.99">
   
B) NESTED ELEMENTS - Data in child elements
   Examples: <div class="post"><h3>Title</h3><span>Author</span></div>

IMPORTANT: Look for the REPEATING element, not parent containers!

Respond with JSON:
{{
  "strategy": "attributes" or "nested_elements",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "element_name": "exact-element-name" (the REPEATING element),
  "key_attributes": ["attr1", "attr2"] (if attributes) or null,
  "repeat_count": number of times element appears
}}"""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert web scraping analyzer. Analyze HTML structure and recommend extraction strategies."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            import json
            import re
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = json.loads(content)
            
            # Cache result
            if use_cache:
                self.pattern_cache[domain] = result
            
            logger.info(f" Detected pattern: {result['strategy']} (confidence: {result['confidence']:.2f})")
            logger.info(f"   Reasoning: {result['reasoning']}")
            
            return result
            
        except Exception as e:
            logger.warning(f" Pattern detection failed: {e}")
            # Fallback to heuristic
            return self._heuristic_detection(html)
    
    def _smart_sample(self, html: str, sample_size: int = 12000) -> str:
        """
        Smart HTML sampling - find actual content, not headers
        Uses same logic as AI generator
        """
        content_markers = [
            '<shreddit-post',  # Reddit
            '<article',        # Articles
            'class="post',     # Generic posts
            'class="product',  # Products
            'class="item',     # List items
            'data-testid="post',
            '<main',           # Main content
        ]
        
        # Find earliest content marker
        earliest_pos = len(html)
        for marker in content_markers:
            pos = html.lower().find(marker.lower())
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        
        # Sample from content area with some context before
        if earliest_pos < len(html):
            start_pos = max(0, earliest_pos - 500)
            return html[start_pos:start_pos + sample_size]
        else:
            # Fallback: first sample_size chars
            return html[:sample_size]
    
    def _heuristic_detection(self, html: str) -> Dict[str, Any]:
        """
        Fallback heuristic-based pattern detection (no LLM)
        Fast but less accurate
        """
        soup = BeautifulSoup(html[:10000], 'html.parser')
        
        # Look for custom elements (contain hyphen)
        custom_elements = soup.find_all(lambda tag: '-' in tag.name)
        
        if len(custom_elements) > 3:
            # Found multiple custom elements - likely attribute-based
            sample_elem = custom_elements[0]
            attrs = list(sample_elem.attrs.keys())
            
            # Check if it has many attributes
            if len(attrs) > 5:
                logger.info(" Heuristic: Detected attribute-based pattern (custom elements with many attrs)")
                return {
                    'strategy': 'attributes',
                    'confidence': 0.7,
                    'reasoning': f'Found custom elements ({custom_elements[0].name}) with many attributes',
                    'element_name': custom_elements[0].name,
                    'key_attributes': attrs[:10]
                }
        
        # Default: nested elements
        logger.info(" Heuristic: Defaulting to nested elements pattern")
        return {
            'strategy': 'nested_elements',
            'confidence': 0.6,
            'reasoning': 'No strong indicators of attribute-based storage found',
            'element_name': None,
            'key_attributes': None
        }

