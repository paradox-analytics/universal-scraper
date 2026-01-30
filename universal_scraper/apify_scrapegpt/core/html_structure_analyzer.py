"""
HTML Structure Analyzer - Inspired by ScrapeGraphAI's html_analyzer_node
Analyzes HTML structure before code generation to improve extraction quality

Now enhanced with DOM pattern detection for faster, LLM-free analysis when possible
"""

import logging
from typing import Dict, Any, Optional
import litellm
from .dom_pattern_detector import DOMPatternDetector
from .smart_html_sampler import SmartHTMLSampler

logger = logging.getLogger(__name__)


class HTMLStructureAnalyzer:
    """
    Analyzes HTML structure to guide code generation
    Inspired by ScrapeGraphAI's HTML Analyzer Node
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize HTML structure analyzer
        
        Args:
            api_key: API key for LLM
            model: Model to use for analysis
        """
        self.api_key = api_key
        self.model = model
        self.analysis_cache = {}  # Cache by URL
        self.dom_detector = DOMPatternDetector()
        self.html_sampler = SmartHTMLSampler(max_chars=12000)
    
    def _analyze_element_frequency(self, html: str) -> list:
        """
        Analyze frequency of custom elements (tags with hyphens)
        Prioritizes data container elements over utility elements
        
        Args:
            html: Full HTML content
            
        Returns:
            List of (element_name, count) tuples, sorted by relevance then count
        """
        import re
        from collections import Counter
        
        # Find all custom elements (tags with hyphens)
        custom_elements = re.findall(r'<([a-z]+-[a-z-]+)', html, re.IGNORECASE)
        
        if not custom_elements:
            return []
        
        # Count frequency
        freq = Counter(custom_elements)
        
        # Utility element patterns (low priority)
        utility_patterns = [
            'loader', 'spinner', 'skeleton',
            'tooltip', 'popover', 'modal',
            'tracker', 'analytics', 'pixel',
            'progress', 'indicator',
            'menu', 'dropdown', 'button',
            'icon', 'svg', 'img',
            'ads', 'banner', 'promo',
            'hovercard', 'overlay'
        ]
        
        # Data container patterns (high priority)
        container_patterns = [
            'post', 'article', 'story', 'thread',
            'card', 'item', 'entry', 'record',
            'product', 'listing', 'offer',
            'comment', 'reply', 'message',
            'tile', 'box', 'cell'
        ]
        
        # Score each element
        scored_elements = []
        for elem, count in freq.items():
            score = count  # Base score is frequency
            
            elem_lower = elem.lower()
            
            # Check if it's PRIMARILY a data container or utility element
            # Priority: check what the element name ENDS with or is the MAIN component
            
            # Split by hyphens to get components
            components = elem_lower.split('-')
            
            # Check last component (most specific)
            last_component = components[-1] if components else ''
            
            # Check if any component (but prioritize last) matches data containers
            is_container = False
            for pattern in container_patterns:
                if pattern == last_component or pattern in elem_lower:
                    is_container = True
                    break
            
            # Check if last component is a utility element (these should be filtered out)
            is_utility = any(pattern == last_component or (pattern in last_component and last_component != 'post') for pattern in utility_patterns)
            
            if is_container and not is_utility:
                score *= 10  # Strong boost for data containers
            elif is_utility and not is_container:
                score *= 0.1  # Strong penalty for pure utility elements
            # Mixed elements (like shreddit-post-overflow-menu) get base score
            
            scored_elements.append((elem, count, score))
        
        # Sort by score (descending), then by count
        scored_elements.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Return as (element, count) tuples
        return [(elem, count) for elem, count, _ in scored_elements]
    
    def _smart_sample_html(self, html: str, sample_size: int = 15000) -> str:
        """
        Intelligently sample HTML to include relevant content areas
        
        Looks for common content markers and samples around them,
        rather than just taking the first N characters.
        
        Args:
            html: Full HTML content
            sample_size: Target sample size in characters
            
        Returns:
            Sampled HTML containing relevant content
        """
        import re
        
        # Content markers to look for (in priority order)
        content_markers = [
            # Custom elements (highest priority)
            (r'<[a-z]+-[a-z-]+', 'custom element'),
            # Semantic elements
            (r'<article', 'article tag'),
            (r'<main', 'main tag'),
            (r'<section', 'section tag'),
            # Common content classes
            (r'class="post', 'post class'),
            (r'class="item', 'item class'),
            (r'class="product', 'product class'),
            (r'class="card', 'card class'),
            # Data attributes
            (r'data-testid="post', 'post testid'),
            (r'data-type="', 'data-type attribute'),
        ]
        
        # Try to find content marker
        earliest_pos = len(html)
        found_marker = None
        
        for pattern, name in content_markers:
            match = re.search(pattern, html, re.IGNORECASE)
            if match and match.start() < earliest_pos:
                earliest_pos = match.start()
                found_marker = name
        
        if found_marker:
            logger.debug(f"   Smart sampling: Found {found_marker} at position {earliest_pos}")
            # Sample from shortly before the marker to include context
            start = max(0, earliest_pos - 1000)
            end = min(len(html), start + sample_size)
            return html[start:end]
        else:
            # No markers found, fall back to first N chars
            logger.debug(f"   Smart sampling: No content markers found, using first {sample_size} chars")
            return html[:sample_size]
    
    def analyze(
        self,
        html: str,
        url: str,
        context: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze HTML structure to guide extraction
        
        Returns structure analysis with:
        - element_type: 'custom_elements' or 'standard_elements'
        - repeating_element: Name of repeating element
        - data_location: 'attributes' or 'nested_elements'
        - key_selectors: List of important selectors
        - extraction_strategy: Recommended approach
        - confidence: 0.0-1.0
        
        Args:
            html: HTML content to analyze
            url: Source URL (for caching)
            context: User's extraction goal
            use_cache: Whether to use cached analysis
        """
        from urllib.parse import urlparse
        import hashlib
        
        domain = urlparse(url).netloc
        
        # Create cache key from domain + structure sample
        # This ensures cache invalidation when structure changes
        structure_sample = html[:5000]  # First 5K chars as structure indicator
        cache_key_str = f"{domain}_{structure_sample}"
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
        
        # Check cache with structure-aware key
        if use_cache and cache_key in self.analysis_cache:
            logger.info(f" Using cached HTML analysis for {domain}")
            return self.analysis_cache[cache_key]
        
        logger.info(f" Analyzing HTML structure for {domain}...")
        
        # PHASE 1: DOM Pattern Detection (fast, no LLM)
        logger.info("   Phase 1: DOM pattern detection (no LLM)...")
        dom_patterns = self.dom_detector.detect_patterns(html)
        
        # If DOM detection has high confidence, return immediately!
        if dom_patterns['confidence'] >= 0.85:
            logger.info(f"    High-confidence DOM pattern detected (confidence={dom_patterns['confidence']:.2f})")
            logger.info("    Skipping LLM call (saving time & cost)")
            
            best_pattern = dom_patterns['best_pattern']
            
            # Convert DOM pattern to structure analysis format
            analysis = {
                'element_type': 'custom_elements' if best_pattern['type'] == 'custom_component' else 'standard_elements',
                'repeating_element': best_pattern['selector'],
                'data_location': 'attributes' if best_pattern['type'] in ['custom_component', 'data_attributes'] else 'nested_elements',
                'key_selectors': [best_pattern['selector']],
                'extraction_strategy': best_pattern.get('extraction_hint', 'Use CSS selectors'),
                'confidence': dom_patterns['confidence'],
                'pattern_count': best_pattern['count'],
                'source': 'dom_pattern_detection'  # Mark as non-LLM
            }
            
            # Cache and return
            self.analysis_cache[cache_key] = analysis
            return analysis
        
        # PHASE 2: LLM Analysis (fallback for lower confidence)
        logger.info(f"   Phase 2: LLM analysis (DOM confidence={dom_patterns['confidence']:.2f}, need LLM)")
        
        # Pre-analyze: Find most common custom elements (legacy, kept for backward compat)
        custom_element_freq = self._analyze_element_frequency(html)
        freq_hint = ""
        if custom_element_freq:
            freq_hint = f"\n\n**Element Frequency Analysis** (most common first):\n"
            for elem, count in custom_element_freq[:5]:  # Top 5
                freq_hint += f"  - <{elem}>: {count} occurrences\n"
            logger.debug(f"   Found {len(custom_element_freq)} unique custom elements")
        
        # Smart content sampling - use new smart sampler
        content_sample = self.html_sampler.sample_html(html, dom_patterns)
        
        # If DOM patterns found, add them to the hint
        if dom_patterns['best_pattern']:
            bp = dom_patterns['best_pattern']
            freq_hint += f"\n**DOM Pattern Detected** (confidence={dom_patterns['confidence']:.2f}):\n"
            freq_hint += f"  - Pattern: {bp['selector']} ({bp['count']} occurrences)\n"
            freq_hint += f"  - Type: {bp['type']}\n"
            freq_hint += f"  - Hint: {bp.get('extraction_hint', 'N/A')}\n"
        
        # Build analysis prompt (inspired by ScrapeGraphAI)
        context_str = f"\n\n**User's Goal**: {context}" if context else ""
        
        prompt = f"""Analyze this HTML structure to guide data extraction code generation.

**Source URL**: {url}{context_str}{freq_hint}

**HTML Sample**:
```html
{content_sample}
```

**CRITICAL INSTRUCTIONS - READ CAREFULLY**:

 **PRIORITY 1: Use the Element Frequency Analysis**
   - LOOK at the "Element Frequency Analysis" above
   - The element with the HIGHEST count is likely the repeating data container
   - Elements with 1-2 occurrences are usually containers/wrappers (SKIP THESE)
   - Elements with 10+ occurrences are usually the data items (USE THESE)
   - Example: If <shreddit-post> appears 62 times and <shreddit-app> appears 1 time,
     choose <shreddit-post> as the repeating element

 **PRIORITY 2: Check for Custom Web Components**
   - Custom elements have HYPHENS in tag names: <shreddit-post>, <product-card>, <my-component>
   - These are MOST IMPORTANT - they store data in HTML attributes
   - If you see ANY custom elements with high frequency, analyze those first
   - Example: <shreddit-post author="user123" post-title="Hello" score="42">
   
 **PRIORITY 3: Never return generic "div" or "span" as repeating element**
   - "div" and "span" are TOO GENERIC
   - Look for specific classes: <div class="post-card">, <article>, <li class="item">
   - If you must use div/span, include the specific class: "div.post-card", not just "div"

 **PRIORITY 4: Determine where data ACTUALLY lives**
   - For custom elements (<tag-with-hyphen>): Data is in ATTRIBUTES (e.g., post-title="...")
   - For standard elements: Data might be in nested text OR data-* attributes
   - Check BOTH: elem.get('attribute-name') AND elem.select_one('.class').text

**Analysis Steps**:
1. **Scan for custom elements** (tags with hyphens):
   - Run: re.findall(r'<([a-z]+-[a-z-]+)', html)
   - If found → return those as repeating_element
   
2. **If no custom elements, look for semantic tags**:
   - <article>, <section>, <li>, etc.
   - These are better than generic <div>
   
3. **As LAST RESORT, use div with specific class**:
   - Return "div.post-card" not "div"
   - Return "li.item" not "li"

4. **Identify actual data location**:
   - Custom elements → "attributes" (data is in elem.get('attr'))
   - Standard elements with nested text → "nested_elements"
   - Mix of both → "mixed"

5. **Extract field mappings**:
   - For attributes: list attribute names ("post-title", "author", "score")
   - For nested: list CSS selectors (".title", ".author", "span.score")

**Response Format** (strict JSON):
{{
  "repeating_element": "shreddit-post" or "article" or "div.post-card" (NEVER just "div"),
  "element_type": "custom_elements" or "standard_elements",
  "data_location": "attributes" or "nested_elements" or "mixed",
  "key_selectors": {{
    "element_selector": "How to find repeating elements (CSS or lambda)",
    "field_mappings": {{
      "field_name": "attribute-name OR css-selector"
    }}
  }},
  "extraction_strategy": "Step-by-step: 1) Find elements via..., 2) Extract data from...",
  "sample_element": "First 500 chars of ONE repeating element",
  "confidence": 0.0-1.0,
  "reasoning": "Why this is the repeating element and where data lives"
}}

**EXAMPLES**:

Example 1 (Reddit - Custom Elements):
{{
  "repeating_element": "shreddit-post",
  "element_type": "custom_elements",
  "data_location": "attributes",
  "key_selectors": {{
    "element_selector": "lambda tag: tag.name == 'shreddit-post'",
    "field_mappings": {{
      "title": "post-title",
      "author": "author",
      "score": "score",
      "comments": "comment-count"
    }}
  }},
  "extraction_strategy": "1) Find all <shreddit-post> elements, 2) Extract data from attributes using elem.get('post-title'), etc.",
  "confidence": 0.95
}}

Example 2 (Hacker News - Standard Elements):
{{
  "repeating_element": "tr.athing",
  "element_type": "standard_elements",
  "data_location": "nested_elements",
  "key_selectors": {{
    "element_selector": "tr.athing",
    "field_mappings": {{
      "title": ".titleline > a",
      "points": ".score",
      "author": ".hnuser"
    }}
  }},
  "extraction_strategy": "1) Find tr.athing elements, 2) Extract text from nested elements using CSS selectors",
  "confidence": 0.90
}}

**Remember**: If you see <any-tag-with-hyphen>, that's a custom element and data is in attributes!"""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HTML structure analyzer specializing in identifying custom web components and data extraction patterns. Be extremely precise and specific - the code generator depends on your accuracy."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1500  # Increased for detailed field mappings
            )
            
            import json
            import re
            
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                analysis = json.loads(json_match.group(0))
            else:
                analysis = json.loads(content)
            
            # Cache the analysis with structure-aware key
            if use_cache:
                self.analysis_cache[cache_key] = analysis
            
            logger.info(f" HTML Structure Analysis:")
            logger.info(f"   Repeating Element: {analysis.get('repeating_element')}")
            logger.info(f"   Element Type: {analysis.get('element_type')}")
            logger.info(f"   Data Location: {analysis.get('data_location')}")
            logger.info(f"   Confidence: {analysis.get('confidence', 0):.2f}")
            
            return analysis
            
        except Exception as e:
            logger.warning(f" HTML structure analysis failed: {e}")
            # Return basic analysis
            return {
                'repeating_element': 'unknown',
                'element_type': 'standard_elements',
                'data_location': 'nested_elements',
                'key_selectors': {},
                'extraction_strategy': 'Use standard CSS selectors',
                'confidence': 0.3,
                'reasoning': f'Analysis failed: {str(e)}'
            }

