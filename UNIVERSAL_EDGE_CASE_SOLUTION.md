# Universal Edge Case Solution

## Problem Statement

**Current Issue**: System fails on websites with non-standard HTML patterns:
- Custom web components (Reddit's `<shreddit-post>`)
- Attribute-based data storage
- Shadow DOM
- Framework-specific patterns

**Impact**: Falls back to expensive LLM direct extraction ($0.10/page) instead of efficient code generation ($0.01/page)

---

## Root Cause Analysis

### Layer 1: HTML Cleaning
**Problem**: Custom elements may be stripped or minified
```python
# Current: Aggressive cleaning
self.html_cleaner.clean(html)  # May remove <shreddit-post>
```

**Solution**: Preserve custom elements
```python
# Detect and preserve custom elements before cleaning
custom_elements = self._detect_custom_elements(html)
cleaned = self.html_cleaner.clean(html, preserve_tags=custom_elements)
```

### Layer 2: Structure Analysis  
**Problem**: LLM doesn't see or recognize custom elements in cleaned HTML
```
Current output:
  Repeating Element: div           ← Generic!
  Data Location: nested_elements   ← Wrong!
```

**Solution**: Enhanced prompt + raw HTML sample
```python
# Pass BOTH cleaned and raw HTML snippet
structure_analysis = self.html_structure_analyzer.analyze(
    cleaned_html=cleaned_html,
    raw_sample=raw_html[:20000],  # First 20KB of raw HTML
    wait_for_selector=wait_for_selector  # Hint about important elements
)
```

### Layer 3: AI Code Generation
**Problem**: Examples don't emphasize attribute extraction enough
```python
# Current: Mostly nested element examples
examples = [
    "title = post.find('h2').text",
    "price = item.find('span', class_='price').text"
]
```

**Solution**: Prioritize attribute examples + inspect actual HTML
```python
# NEW: Attribute-first examples
examples = [
    "# Try attributes FIRST (modern sites)",
    "title = post.get('post-title') or post.get('data-title')",
    "score = post.get('score') or post.get('data-score')",
    "# Fall back to nested elements if needed",
    "title = title or post.find('h2').text if post.find('h2') else None"
]
```

### Layer 4: Smart Sampling
**Problem**: May not show the actual data-bearing elements to LLM
```python
# Current: Generic sampling
sample = html[:12000]  # First 12KB
```

**Solution**: Targeted sampling around wait_for_selector
```python
# NEW: Sample around the actual content
if wait_for_selector:
    # Find the selector in HTML and sample around it
    sample = self._sample_around_selector(html, wait_for_selector, size=15000)
else:
    sample = self._smart_sample(html)  # Existing logic
```

---

## Implementation Plan

### Phase 1: Immediate Fixes (High Impact, Low Risk)

#### Fix 1: Preserve Custom Elements in Cleaner
**File**: `universal_scraper/core/html_cleaner.py`

```python
def _detect_custom_elements(self, html: str) -> Set[str]:
    """Detect custom elements (tags with hyphens)"""
    import re
    pattern = r'<([a-z]+-[a-z-]+)'
    custom_tags = set(re.findall(pattern, html.lower()))
    return custom_tags

def clean(self, html: str, preserve_tags: Set[str] = None) -> str:
    """Clean HTML while preserving important custom elements"""
    preserve_tags = preserve_tags or set()
    
    # Add to KEEP_TAGS temporarily
    original_keep = self.KEEP_TAGS.copy()
    self.KEEP_TAGS.update(preserve_tags)
    
    try:
        cleaned = self._clean_implementation(html)
        return cleaned
    finally:
        self.KEEP_TAGS = original_keep
```

#### Fix 2: Enhanced Structure Analysis Prompt
**File**: `universal_scraper/core/html_structure_analyzer.py`

```python
def analyze(
    self, 
    cleaned_html: str, 
    url: str, 
    context: Optional[str] = None,
    raw_sample: Optional[str] = None,  # NEW
    wait_for_selector: Optional[str] = None  # NEW
) -> Dict[str, Any]:
    """Analyze HTML structure with hints"""
    
    # Use raw sample if provided (better for custom elements)
    analysis_html = raw_sample if raw_sample else cleaned_html
    
    prompt = f"""Analyze this HTML to identify repeating data elements.

IMPORTANT: Modern websites often use:
1. Custom web components (tags with hyphens like <product-card>, <shreddit-post>)
2. Data in HTML ATTRIBUTES (post-title="...", data-score="...")
3. Shadow DOM and framework patterns

{f"HINT: The page waits for selector '{wait_for_selector}' - this likely contains the data." if wait_for_selector else ""}

HTML Sample:
{analysis_html[:15000]}

Return JSON:
{{
    "repeating_element": "exact tag name or selector",
    "element_type": "custom_element|standard_element", 
    "data_location": "attributes|nested_elements|mixed",
    "key_attributes": ["list", "of", "data", "attributes"],  # NEW
    "confidence": 0.0-1.0,
    "reasoning": "why you chose this"
}}
"""
```

#### Fix 3: Attribute-First AI Prompt
**File**: `universal_scraper/core/ai_generator.py`

```python
def _build_prompt(self, ...):
    """Build prompt with attribute-first examples"""
    
    # Detect if we're dealing with custom elements
    has_custom_elements = bool(re.search(r'<[a-z]+-[a-z-]+', cleaned_content))
    
    if has_custom_elements or (structure_analysis and structure_analysis.get('element_type') == 'custom_element'):
        # Use attribute-first strategy
        examples = """
## PRIORITY 1: Check HTML Attributes (Modern Sites)

Custom elements often store data in attributes, not nested text:

```python
# Reddit example: <shreddit-post post-title="..." author="..." score="42">
for post in soup.find_all('shreddit-post'):
    item = {
        'title': post.get('post-title'),  # Direct attribute access
        'author': post.get('author'),
        'score': int(post.get('score', 0)),
        'comments': int(post.get('comment-count', 0))
    }
    items.append(item)
```

## PRIORITY 2: Try Nested Elements (Traditional Sites)

```python
for article in soup.find_all('article'):
    item = {
        'title': article.find('h2').text if article.find('h2') else None,
        'price': article.find('span', class_='price').text
    }
```

## BEST PRACTICE: Try Both (Robust)

```python
# Try attribute first, fall back to nested
title = post.get('post-title') or post.get('data-title')
if not title and post.find('h2'):
    title = post.find('h2').text
```
"""
    else:
        # Use traditional nested-first strategy
        examples = """... existing examples ..."""
```

#### Fix 4: Targeted Sampling
**File**: `universal_scraper/core/ai_generator.py`

```python
def _sample_around_selector(self, html: str, selector: str, size: int = 15000) -> str:
    """Sample HTML around the wait_for_selector (where data likely is)"""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html, 'lxml')
    
    # Find the selector (could be tag name, class, or ID)
    target = None
    if selector.startswith('.'):
        target = soup.find(class_=selector[1:])
    elif selector.startswith('#'):
        target = soup.find(id=selector[1:])
    else:
        target = soup.find(selector)
    
    if not target:
        # Fallback to smart sample
        return self._smart_sample(html, size)
    
    # Get parent context + siblings
    parent = target.parent
    if parent:
        # Get a good chunk around the target
        sample_html = str(parent)[:size]
        return sample_html
    
    return str(target)[:size]
```

---

### Phase 2: Pattern Library (Medium Priority)

Create a pattern library for known difficult sites:

**File**: `universal_scraper/core/site_patterns.py`

```python
SITE_PATTERNS = {
    'reddit.com': {
        'type': 'custom_elements_with_attributes',
        'repeating_element': 'shreddit-post',
        'data_location': 'attributes',
        'field_mappings': {
            'title': ['post-title', 'data-post-title'],
            'author': ['author', 'data-author'],
            'upvotes': ['score', 'data-score'],
            'comments': ['comment-count', 'data-comment-count']
        },
        'wait_for': 'shreddit-post'
    },
    'amazon.com': {
        'type': 'mixed',
        'repeating_element': 'div[data-component-type="s-search-result"]',
        'data_location': 'mixed',
        'field_mappings': {
            'title': {'selector': 'h2 a span', 'type': 'text'},
            'price': {'selector': 'span.a-price-whole', 'type': 'text'},
            'asin': {'selector': None, 'type': 'attribute', 'attr': 'data-asin'}
        }
    },
    # ... more patterns
}

def get_site_pattern(url: str) -> Optional[Dict]:
    """Get known pattern for a site"""
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    
    for pattern_domain, pattern in SITE_PATTERNS.items():
        if pattern_domain in domain:
            return pattern
    return None
```

Use in scraper:

```python
# In UniversalScraper.scrape()
site_pattern = get_site_pattern(url)

if site_pattern:
    logger.info(f"🎯 Using known pattern for {urlparse(url).netloc}")
    # Apply pattern-specific optimizations
    wait_for_selector = wait_for_selector or site_pattern.get('wait_for')
    # Pass hints to structure analyzer
```

---

### Phase 3: Automated Pattern Learning (Future)

When extraction succeeds, save the pattern:

```python
def _save_successful_pattern(self, url: str, extraction_code: str, items_count: int):
    """Learn from successful extractions"""
    if items_count > 5:  # Only save if we got good results
        pattern = self._extract_pattern_from_code(extraction_code)
        self.pattern_cache.save(url, pattern)
```

---

## Testing Strategy

### Test Matrix

| Site Type | Example | Key Challenge | Solution |
|-----------|---------|---------------|----------|
| Custom Elements + Attributes | Reddit | `<shreddit-post post-title="...">` | Preserve custom tags, attribute-first extraction |
| Shadow DOM | YouTube | Encapsulated content | Detect shadow roots, use JS injection |
| Heavy JavaScript | SPA sites | No SSR, needs rendering | Browser mode + wait strategies |
| Anti-bot | Amazon, eBay | Blocks scrapers | Proxy + anti-detection |
| Complex nesting | News sites | Deep DOM trees | Better sampling around selectors |
| Mixed patterns | E-commerce | Some attributes, some nested | Try both strategies |

### Regression Tests

```python
# Test custom elements
def test_reddit_custom_elements():
    result = scraper.scrape("https://www.reddit.com/r/webscraping/")
    assert len(result) > 5
    assert result[0]['title']  # Not None
    assert result[0]['author']
    
# Test attribute extraction
def test_attribute_based_extraction():
    html = '<div data-title="Test" data-price="$99"></div>'
    # Should extract from attributes
    
# Test mixed patterns
def test_mixed_extraction():
    html = '<product id="123"><h2>Title</h2></product>'
    # Should get id from attribute, title from nested element
```

---

## Expected Improvements

### Before (Current State)
- Reddit: Falls back to LLM ($0.10/page, 60-80s)
- Success rate: ~60% on custom element sites
- Code generation fails 3/3 iterations

### After (With Fixes)
- Reddit: Code generation succeeds ($0.01/page, 5-10s)
- Success rate: >90% on custom element sites
- First iteration success increases to >70%

### Cost Savings
- Per Reddit page: $0.10 → $0.01 (10x cheaper)
- Per 1000 pages: $100 → $10 (saves $90)
- Speed: 60s → 10s (6x faster)

---

## Implementation Priority

### 🔴 Critical (Do First)
1. ✅ Fix 1: Preserve Custom Elements - **30 min**
2. ✅ Fix 3: Attribute-First Prompt - **45 min**
3. ✅ Fix 4: Targeted Sampling - **30 min**

### 🟡 High Priority  
4. ✅ Fix 2: Enhanced Structure Analysis - **1 hour**
5. ✅ Reddit-specific pattern - **20 min**

### 🟢 Medium Priority
6. ⏳ Pattern library for top 20 sites - **4 hours**
7. ⏳ Automated pattern learning - **8 hours**

---

## Success Metrics

Track these metrics to measure improvement:

```python
metrics = {
    'code_generation_success_rate': 0.85,  # Target: >90%
    'first_iteration_success_rate': 0.60,  # Target: >70%
    'llm_fallback_rate': 0.15,  # Target: <10%
    'avg_extraction_cost': 0.02,  # Target: <$0.02
    'avg_extraction_time': 12,  # Target: <15s
    'custom_element_success_rate': 0.70  # Target: >90%
}
```

---

## Conclusion

The fundamental issue is that the system was designed for "traditional" HTML but modern web has evolved to:
- Custom web components
- Attribute-based data storage
- Framework-specific patterns
- Heavy JavaScript rendering

**Solution**: Multi-layer defensive strategy:
1. **Preserve** custom elements during cleaning
2. **Detect** custom elements in structure analysis
3. **Prioritize** attribute extraction in AI prompts
4. **Sample** around actual content (wait_for_selector)
5. **Learn** from successful patterns (pattern library)

This makes the system **truly universal** by handling both traditional HTML patterns AND modern web component patterns.

**Next Step**: Implement the 5 critical fixes (estimated 3-4 hours total).







