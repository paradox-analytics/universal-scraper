# Quality Analysis & Improvement Plan

## Current Quality by Site

| Site | Quality | Items | Issue Analysis |
|------|---------|-------|----------------|
| **Quotes to Scrape** | 100% ✅ | 10 | Perfect - simple structure |
| **Hacker News** | 95% ✅ | 13 | Near-perfect |
| **Books to Scrape** | 75% 🔶 | 20 | Missing: ratings in CSS classes |
| **Stack Overflow** | 61% 🔶 | 15 | Missing: votes/answers (numeric labels) |
| **Product Hunt** | 57% 🔶 | 42 | Missing: upvotes, descriptions |
| **GitHub Trending** | 39% ❌ | 1 | Regression - pattern fallback failed |

---

## Root Cause Analysis

### 1. Books to Scrape (75% → target 100%)

**Problem**: Ratings are stored in CSS classes, not text
```html
<p class="star-rating Three">  <!-- "Three" = 3 stars -->
```

**Current Status**: We extract CSS class data but LLM isn't using it effectively

**Fix**: Enhance `get_metadata_summary()` to format CSS ratings more explicitly:
```
RATINGS FOUND: star-rating-Three means 3 stars
```

### 2. Stack Overflow (61% → target 90%)

**Problem**: Votes/answers/views are labeled numbers scattered across the page
```html
<span class="s-post-summary--stats-item-number">5</span>
<span class="s-post-summary--stats-item-unit">answers</span>
```

**Current Status**: We extract labeled numbers but they're not associated with specific items

**Fix**: 
1. Better context association - link numbers to their parent items
2. Pass extracted labeled numbers with item context

### 3. Product Hunt (57% → target 85%)

**Problem**: Complex React-rendered structure with data in multiple locations:
- Upvotes in buttons
- Descriptions truncated
- Tags in nested components

**Current Status**: HTML cleaning removes some UI elements with data

**Fix**:
1. Less aggressive cleaning for known product listing patterns
2. Extract upvote buttons before cleaning
3. Use data attributes more effectively

### 4. GitHub Trending (39% → target 95%)

**Problem**: Pattern-based extraction fallback failing; only 1 item extracted

**Root Cause**: The iterative refinement loop isn't being used (large page goes to chunking path)

**Fix**:
1. For pages with chunking, apply refinement after chunk merging
2. Or: use a higher chunk size threshold to trigger single-pass for medium pages

---

## Improvement Strategy

### Phase 1: Quick Wins (Immediate)

#### 1.1 Apply Refinement to Chunked Pages

Currently refinement only applies to small pages. Apply it post-chunking:

```python
# After chunking and deduplication
if quality < 0.7:
    # Refine the merged results
    items = await self._refine_extraction(html, fields, items, ...)
```

#### 1.2 Better CSS Class Formatting

Make CSS data more explicit for LLM:

```python
# Instead of: "star-rating: ['One', 'Two', 'Three']"
# Format as: "RATINGS: One=1 star, Two=2 stars, Three=3 stars"
```

#### 1.3 Associate Labeled Numbers with Items

```python
# Instead of: "votes: ['5', '10', '15']"  
# Format as: "Item 1 has 5 votes, Item 2 has 10 votes..."
```

### Phase 2: Structural Improvements

#### 2.1 Item-Aware Extraction

Instead of extracting all data then hoping LLM associates it:

```python
# Find item containers first
items = soup.find_all('.product-card')  
for item in items:
    # Extract data specific to this item
    title = item.find('.title')
    price = item.find('.price')
    rating = item.find_class('star-rating')
```

#### 2.2 Two-Pass Extraction

1. **First pass**: Extract items with high-confidence fields
2. **Second pass**: For each item, targeted extraction of missing fields

### Phase 3: Site-Specific Patterns (if needed)

For stubborn sites, maintain a pattern registry:

```python
SITE_PATTERNS = {
    'github.com': {
        'item_selector': 'article.Box-row',
        'rating_location': 'css_class',
    },
    'stackoverflow.com': {
        'votes_selector': '.s-post-summary--stats-item-number',
    }
}
```

---

## Recommended Implementation Order

1. **[HIGH] Refinement for chunked pages** - GitHub will benefit most
2. **[HIGH] Better CSS rating formatting** - Books will hit 95%+
3. **[MED] Item-context for labeled numbers** - Stack Overflow improvement
4. **[MED] Less aggressive cleaning for product pages** - Product Hunt improvement
5. **[LOW] Site-specific patterns** - Last resort for edge cases

---

## Expected Quality After Fixes

| Site | Current | After Phase 1 | After Phase 2 |
|------|---------|---------------|---------------|
| Quotes | 100% | 100% | 100% |
| Hacker News | 95% | 95% | 98% |
| Books | 75% | 90% | 95% |
| Stack Overflow | 61% | 75% | 85% |
| Product Hunt | 57% | 70% | 80% |
| GitHub | 39% | 80% | 95% |

**Overall**: 71% → **85%** → **92%**


