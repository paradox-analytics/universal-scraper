# 🎯 Crawl Targeting Guide - Control What Gets Scraped

## The Problem

When you crawl a website, the crawler discovers **ALL** links:
- Navigation links (`/products`, `/news`, `/about`)
- Category pages (`/category/electronics`)
- Detail pages (`/product/123`)
- Utility pages (`/login`, `/cart`, `/search`)

**But you only want to scrape certain pages!**

---

## The Solution: URL Pattern Filtering

Use `follow_patterns` and `ignore_patterns` in `CrawlConfig` to control which URLs the crawler follows.

---

## Method 1: Follow Patterns (Whitelist Approach)

**Only follow URLs that match these patterns:**

```python
from universal_scraper.crawler import CrawlConfig, UniversalCrawler

config = CrawlConfig(
    max_depth=3,
    max_pages=100,
    
    # ✅ ONLY follow URLs containing these strings
    follow_patterns=[
        '/dispensary-info/',  # Dispensary detail pages
        '/dispensaries/'      # Dispensary listing pages
    ]
)

crawler = UniversalCrawler(config=config)
results = crawler.crawl(["https://www.leafly.com/dispensaries/nevada"])

# Result: Only dispensary URLs are crawled, navigation is ignored!
```

### Example Output:
```
✅ Followed: https://leafly.com/dispensaries/nevada
✅ Followed: https://leafly.com/dispensary-info/mammoth-holistics
✅ Followed: https://leafly.com/dispensary-info/mammoth-holistics/menu
❌ Ignored: https://leafly.com/products  (doesn't match pattern)
❌ Ignored: https://leafly.com/news      (doesn't match pattern)
❌ Ignored: https://leafly.com/strains   (doesn't match pattern)
```

---

## Method 2: Ignore Patterns (Blacklist Approach)

**Follow all URLs EXCEPT those that match these patterns:**

```python
config = CrawlConfig(
    max_depth=3,
    max_pages=100,
    
    # ❌ IGNORE URLs containing these strings
    ignore_patterns=[
        '/products',
        '/strains',
        '/news',
        '/brands',
        '/learn',
        '/auth',
        '/api',
        '.jpg',
        '.png',
        '.pdf'
    ]
)

crawler = UniversalCrawler(config=config)
results = crawler.crawl(["https://example.com"])
```

---

## Method 3: Combined (Recommended)

**Use both for precise control:**

```python
config = CrawlConfig(
    max_depth=3,
    max_pages=100,
    
    # ✅ Only follow these patterns
    follow_patterns=[
        '/product/',
        '/category/',
        '/item/'
    ],
    
    # ❌ But ignore these specific ones
    ignore_patterns=[
        '/category/archived',  # Ignore archived categories
        '.pdf',                # Ignore PDF files
        '?sort=',              # Ignore sort variations
        '/product/preview'     # Ignore preview pages
    ]
)

# Logic: URL must match follow_patterns AND not match ignore_patterns
```

---

## Real-World Examples

### Example 1: E-commerce Site

**Goal:** Only scrape product pages, ignore navigation

```python
config = CrawlConfig(
    follow_patterns=[
        '/product/',      # Product detail pages
        '/category/',     # Category listings
        '/collection/'    # Product collections
    ],
    ignore_patterns=[
        '/cart',
        '/checkout',
        '/account',
        '/auth',
        '/search',
        '?page='          # Ignore pagination URLs (handle separately)
    ]
)
```

### Example 2: News Site

**Goal:** Only scrape articles from 2024

```python
config = CrawlConfig(
    follow_patterns=[
        '/2024/',         # Only 2024 articles
        '/article/',      # Article pages
        '/post/'          # Blog posts
    ],
    ignore_patterns=[
        '/author/',       # Ignore author pages
        '/tag/',          # Ignore tag pages
        '/category/',     # Ignore category pages
        '/comments'       # Ignore comment sections
    ]
)
```

### Example 3: Leafly Dispensaries

**Goal:** Only scrape dispensary data, ignore everything else

```python
config = CrawlConfig(
    follow_patterns=[
        '/dispensaries/',     # Listing pages
        '/dispensary-info/'   # Detail pages (info + menu)
    ],
    ignore_patterns=[
        '/products',
        '/strains',
        '/news',
        '/brands',
        '/doctors',
        '/learn',
        '/cannabis-101'
    ]
)

crawler = UniversalCrawler(config=config)
results = crawler.crawl(["https://www.leafly.com/dispensaries/nevada"])

# Result: Only dispensary pages crawled!
```

### Example 4: Real Estate Site

**Goal:** Only scrape property listings

```python
config = CrawlConfig(
    follow_patterns=[
        '/property/',
        '/listing/',
        '/for-sale/'
    ],
    ignore_patterns=[
        '/agent/',
        '/about',
        '/contact',
        '/blog',
        '/mortgage-calculator'
    ]
)
```

---

## How Pattern Matching Works

### String Containment (Default)

Patterns are checked using **string containment** (substring matching):

```python
follow_patterns=['/product/']

# ✅ Matches
'https://shop.com/product/123'        # Contains '/product/'
'https://shop.com/category/product/abc' # Contains '/product/'

# ❌ Does NOT match
'https://shop.com/products'            # No trailing slash
'https://shop.com/cart'                # Doesn't contain '/product/'
```

### Tips for Effective Patterns

1. **Be specific with slashes:**
   - `/product/` - Matches product detail pages
   - `/products` - Matches products listing page
   
2. **Use URL structure:**
   - `/2024/` - Only 2024 content
   - `/en/` - Only English content
   - `/us/` - Only US content

3. **Ignore file extensions:**
   - `.pdf`, `.jpg`, `.png`, `.css`, `.js`

4. **Ignore utility pages:**
   - `/cart`, `/checkout`, `/login`, `/auth`

5. **Ignore query parameters:**
   - `?sort=`, `?filter=`, `?utm_`

---

## Page Type Classification

After crawling, pages are classified by type:

```python
results = crawler.crawl([start_url])

for crawled_url in results.urls:
    print(f"{crawled_url.page_type}: {crawled_url.url}")

# Output:
# PageType.LISTING: https://example.com/category/electronics
# PageType.DETAIL: https://example.com/product/123
# PageType.DETAIL: https://example.com/product/456
```

**Page Types:**
- `LISTING` - Pages with multiple items (categories, search results)
- `DETAIL` - Individual item pages (products, articles)
- `NAVIGATION` - Nav pages (homepage, category indexes)
- `SEARCH_REQUIRED` - Pages requiring search interaction
- `UNKNOWN` - Could not classify

---

## Combining with Scraping

### Full Pipeline Example

```python
from universal_scraper import UniversalWorkflow
from universal_scraper.crawler import CrawlConfig

# Step 1: Configure targeted crawling
crawl_config = CrawlConfig(
    max_depth=2,
    max_pages=100,
    follow_patterns=['/product/'],
    ignore_patterns=['/cart', '/auth']
)

# Step 2: Run full pipeline
workflow = UniversalWorkflow(
    mode="full_pipeline",
    openai_api_key="your-key",
    crawl_config=crawl_config
)

# Step 3: Scrape all discovered product pages
results = workflow.run(
    start_urls=["https://shop.com/category/electronics"],
    fields=["name", "price", "rating", "description"]
)

# Result: Only product pages scraped with consistent data!
```

---

## Advanced: Dynamic Patterns

### URL Pattern Detection

The crawler can auto-detect URL patterns:

```python
config = CrawlConfig(
    max_depth=2,
    follow_patterns=[],  # Empty = follow all
    ignore_patterns=[]   # Empty = ignore nothing
)

# Crawler will discover ALL patterns and classify them
results = crawler.crawl([start_url])

# Then you can filter by page type
product_urls = [
    url for url in results.urls 
    if url.page_type == PageType.DETAIL
]
```

### Regex Patterns (Future Feature)

```python
# Coming soon: Regex support
config = CrawlConfig(
    follow_patterns=[
        r'/product/\d+',           # Product IDs
        r'/category/[a-z\-]+',     # Category slugs
        r'/\d{4}/\d{2}/\d{2}/'     # Date-based URLs
    ],
    use_regex=True
)
```

---

## Performance Tips

### 1. Be Specific

**❌ Bad (crawls too much):**
```python
follow_patterns=['/']  # Matches EVERYTHING
```

**✅ Good (targeted):**
```python
follow_patterns=['/product/', '/category/']
```

### 2. Use Ignore Patterns for Speed

Ignoring unnecessary pages speeds up crawling:

```python
ignore_patterns=[
    '.pdf', '.jpg', '.png',  # Files
    '/api/', '/auth/',       # Utility endpoints
    '?utm_', '?ref='         # Tracking parameters
]
```

### 3. Limit Depth

```python
config = CrawlConfig(
    max_depth=2,  # Don't go too deep
    follow_patterns=['/product/']
)
```

### 4. Use Pagination Handling

Instead of crawling `?page=2`, `?page=3`, etc.:

```python
config = CrawlConfig(
    handle_pagination=True,  # Auto-detects pagination
    ignore_patterns=['?page=']  # Ignore manual page links
)
```

---

## Testing Your Patterns

### Quick Test Script

```python
from universal_scraper.crawler import CrawlConfig, UniversalCrawler

config = CrawlConfig(
    max_depth=1,
    max_pages=10,  # Limit for testing
    follow_patterns=['/your-pattern/'],
    ignore_patterns=['/ignore-this/']
)

crawler = UniversalCrawler(config=config)
results = crawler.crawl(["https://your-site.com"])

print(f"Discovered: {results.total_discovered}")
print(f"Crawled: {results.total_crawled}")

for url in results.urls:
    print(f"  • {url.url}")
```

---

## Common Mistakes

### ❌ Mistake 1: No Patterns

```python
config = CrawlConfig()  # No patterns!
# Result: Crawls EVERYTHING (slow, wasteful)
```

### ❌ Mistake 2: Too Broad

```python
follow_patterns=['/']  # Matches everything
# Result: Same as having no patterns
```

### ❌ Mistake 3: Conflicting Patterns

```python
follow_patterns=['/product/']
ignore_patterns=['/product/']  # Conflicts!
# Result: Nothing matches
```

### ❌ Mistake 4: Missing Slashes

```python
follow_patterns=['product']  # Too broad!
# Matches: /products, /product-info, /myproduct, etc.
```

### ✅ Correct Version

```python
follow_patterns=['/product/']  # Specific!
# Only matches: /product/123, /product/abc, etc.
```

---

## Summary

### Quick Reference

| Goal | Configuration |
|------|---------------|
| Only follow specific URLs | Use `follow_patterns` |
| Ignore specific URLs | Use `ignore_patterns` |
| Precise control | Use both together |
| Speed optimization | Add file extensions to `ignore_patterns` |
| Depth control | Set `max_depth` |

### Example Configuration

```python
config = CrawlConfig(
    # Targeting
    follow_patterns=['/product/', '/category/'],
    ignore_patterns=['.pdf', '/cart', '/auth'],
    
    # Performance
    max_depth=2,
    max_pages=100,
    
    # Features
    handle_pagination=True,
    discover_apis=True
)
```

---

**Now run the test:**

```bash
python3 test_leafly_targeted.py
```

This will show exactly how pattern filtering works on Leafly!

---

**Last Updated:** November 7, 2025








