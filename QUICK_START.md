# 🚀 Universal Scraper - Quick Start Guide

## What Is This?

A **universal web scraping system** that works on **ANY website** - no custom code needed!

- ✅ **Crawls** entire websites automatically
- ✅ **Scrapes** data from any page structure  
- ✅ **Handles** static HTML and JavaScript sites
- ✅ **Maintains** stable schemas even when sites change
- ✅ **Deploys** to Apify in one command

---

## 30-Second Example

```python
from universal_scraper import UniversalWorkflow

# Create workflow
workflow = UniversalWorkflow(
    mode="full_pipeline",
    openai_api_key="YOUR_KEY_HERE"
)

# Scrape entire site
results = workflow.run(
    start_urls=["https://example.com"],
    fields=["title", "price", "description"]
)

# Done! Results contain all discovered pages with extracted data
print(f"Scraped {len(results)} items")
```

---

## Installation

```bash
# Clone repo
git clone https://github.com/yourusername/universal-scraper.git
cd universal-scraper

# Install dependencies
pip install -r requirements.txt

# Install Playwright (for JavaScript sites)
playwright install chromium
```

---

## Usage Modes

### Mode 1: Scrape Single URL
```python
from universal_scraper import UniversalScraper

scraper = UniversalScraper(api_key="YOUR_OPENAI_KEY")

result = scraper.scrape(
    url="https://example.com/product/123",
    fields=["name", "price", "rating"]
)

print(result['data'])
# [{'name': 'Product Name', 'price': 29.99, 'rating': 4.5}]
```

### Mode 2: Crawl Entire Site
```python
from universal_scraper.crawler import UniversalCrawler, CrawlConfig

config = CrawlConfig(
    max_depth=3,
    max_pages=1000,
    handle_pagination=True
)

crawler = UniversalCrawler(config=config)

results = crawler.crawl(["https://example.com"])

print(f"Discovered {results.total_discovered} URLs")
print(f"Crawled {results.total_crawled} pages")
```

### Mode 3: Full Pipeline (Crawl + Scrape)
```python
from universal_scraper import UniversalWorkflow

workflow = UniversalWorkflow(
    mode="full_pipeline",
    openai_api_key="YOUR_KEY"
)

results = workflow.run(
    start_urls=["https://example.com/category"],
    fields=["title", "price", "description"]
)

# Automatically:
# 1. Discovers all product pages
# 2. Scrapes data from each
# 3. Returns consistent results
```

---

## Real-World Examples

### Example 1: E-commerce Store
```python
workflow = UniversalWorkflow(
    mode="full_pipeline",
    openai_api_key=OPENAI_KEY
)

results = workflow.run(
    start_urls=["https://shop.example.com/electronics"],
    fields=["name", "price", "rating", "stock"]
)

# Discovers:
# - All category pages
# - All product pages  
# - Handles pagination automatically
# - Extracts consistent data
```

### Example 2: News Site
```python
workflow = UniversalWorkflow(mode="full_pipeline")

results = workflow.run(
    start_urls=["https://news.example.com/archive/2024"],
    fields=["headline", "author", "date", "content"]
)

# Discovers:
# - All archive pages
# - All article links
# - Extracts full content
```

### Example 3: Directory Listing
```python
workflow = UniversalWorkflow(
    mode="full_pipeline",
    fetch_mode="browser"  # JS-heavy site
)

results = workflow.run(
    start_urls=["https://directory.example.com/city/restaurants"],
    fields=["name", "address", "rating", "cuisine"]
)

# Handles:
# - JavaScript rendering
# - Pagination
# - Dynamic content
```

---

## Advanced Features

### Schema Stability
```python
from universal_scraper.core import SchemaDefinition, FieldDefinition

# Define expected schema
schema = SchemaDefinition(
    name="products",
    version="1.0",
    fields=[
        FieldDefinition("name", "string", required=True),
        FieldDefinition("price", "number", required=True),
        FieldDefinition("rating", "number", required=False)
    ]
)

workflow = UniversalWorkflow(
    mode="full_pipeline",
    schema=schema,
    strict_schema=False  # Auto-map field variations
)

# Even if website changes field names,
# schema manager will adapt automatically!
```

### Auto-Schema Generation
```python
from universal_scraper.core import infer_schema_from_scrape

# First scrape (no schema)
results = scraper.scrape(url, fields=["title", "price"])

# Auto-generate schema
schema = infer_schema_from_scrape(
    url=url,
    scraped_data=results['data'],
    schema_name="my_products"
)

# Save for future use
schema.save("schemas/my_products_v1.json")

# Future scrapes use stable schema
scraper = UniversalScraper(schema=schema)
```

### JavaScript Sites
```python
# Hybrid mode (auto-detects JS need)
workflow = UniversalWorkflow(
    fetch_mode="hybrid"  # Tries static first, browser fallback
)

# Or force browser mode
workflow = UniversalWorkflow(
    fetch_mode="browser",  # Always use browser
    headless=True
)
```

---

## Configuration

### Crawl Configuration
```python
from universal_scraper.crawler import CrawlConfig

config = CrawlConfig(
    mode="smart",                    # 'smart', 'links_only', 'api_only'
    max_depth=3,                     # How deep to crawl
    max_pages=1000,                  # Max pages to visit
    max_items=10000,                 # Max items to extract
    handle_pagination=True,          # Auto-detect pagination
    discover_apis=True,              # Intercept API calls
    enable_search_discovery=False,   # Enable search enumeration
    respect_robots_txt=True,         # Respect robots.txt
    rate_limit="10/minute",          # Rate limiting
    timeout_minutes=60               # Max crawl time
)
```

### Workflow Configuration
```python
from universal_scraper import UniversalWorkflow, WorkflowConfig

config = WorkflowConfig(
    mode="full_pipeline",            # 'crawl_only', 'scrape_only', 'full_pipeline'
    openai_api_key="...",            # Your OpenAI key
    fetch_mode="hybrid",             # 'hybrid', 'static', 'browser'
    crawl_config=CrawlConfig(...),   # Crawler settings
    schema=SchemaDefinition(...),    # Output schema
    strict_schema=False,             # Fail on schema errors
    enable_cache=True,               # Cache results
    headless=True,                   # Browser headless mode
    proxy_config=None                # Optional proxy
)

workflow = UniversalWorkflow(config=config)
```

---

## Testing

### Test on Leafly (JavaScript site)
```bash
python3 test_universal_leafly.py
```

### Test Full Crawl
```bash
python3 test_end_to_end_crawl.py
```

### Test Schema Stability
```bash
python3 test_schema_stability.py
```

---

## Deployment

### Local Use
```python
# Already done! Just import and use
from universal_scraper import UniversalWorkflow
```

### Apify Deployment
```bash
# One-command deployment
./deploy_to_apify.sh

# Then run via Apify API or UI
```

---

## Performance Tips

### 1. Use Hybrid Mode (Default)
```python
# Fastest - tries static first, browser only if needed
workflow = UniversalWorkflow(fetch_mode="hybrid")
```

### 2. Enable Caching
```python
# Caches discovered APIs and results
workflow = UniversalWorkflow(enable_cache=True)
```

### 3. Limit Crawl Depth
```python
# Don't go too deep unnecessarily
config = CrawlConfig(max_depth=2)  # Usually enough
```

### 4. Use Pagination Detection
```python
# Discovers pages automatically (no manual URL building)
config = CrawlConfig(handle_pagination=True)
```

---

## Troubleshooting

### "No data extracted"
**Solution:** Site likely uses JavaScript
```python
workflow = UniversalWorkflow(fetch_mode="browser")
```

### "Timeout error"
**Solution:** Increase timeout
```python
fetcher = HybridFetcher(timeout=60000)  # 60 seconds
```

### "Different field names"
**Solution:** Use schema with auto-mapping
```python
schema = SchemaDefinition(...)
workflow = UniversalWorkflow(schema=schema, strict_schema=False)
```

### "Too slow"
**Solution:** Enable caching and limit depth
```python
config = CrawlConfig(
    max_depth=2,
    max_pages=500
)
workflow = UniversalWorkflow(config=config, enable_cache=True)
```

---

## Architecture

```
                 ORCHESTRATOR
                 (Coordinates)
                      │
        ┌─────────────┴─────────────┐
        │                           │
    CRAWLER                     SCRAPER
  (URL Discovery)            (Data Extraction)
        │                           │
        └───────────┬───────────────┘
                    │
              HYBRID FETCHER
           (Static + Browser + API)
```

**Modules:**
- **Core:** Scraping engine (JSON-first, AI-powered)
- **Crawler:** URL discovery (links, pagination, APIs, search)
- **Orchestrator:** Integration layer (combines crawler + scraper)
- **Apify:** Deployment (single unified actor)

---

## What Makes It Universal?

### ❌ What We DON'T Do
- ❌ No hardcoded websites
- ❌ No site-specific logic
- ❌ No custom selectors

### ✅ What We DO
- ✅ Universal pattern detection
- ✅ Adaptive fetching (static/JS)
- ✅ Generic page classification
- ✅ AI-powered data extraction

### Result
**One system works on EVERY website:**
- E-commerce (Amazon, eBay)
- News (NYTimes, HN)
- Forums (Reddit, SO)
- Directories (Yelp, Leafly)
- Government (county records)
- And MORE!

---

## Next Steps

1. **Try It Out**
   ```bash
   python3 test_universal_leafly.py
   ```

2. **Read Documentation**
   - `FINAL_SUMMARY.md` - Complete overview
   - `INTEGRATION_COMPLETE.md` - Technical details
   - `MODULAR_ARCHITECTURE.md` - Module breakdown
   - `FETCHER_INTEGRATION.md` - Fetching strategy

3. **Deploy to Apify**
   ```bash
   ./deploy_to_apify.sh
   ```

4. **Start Scraping!**
   ```python
   workflow = UniversalWorkflow(mode="full_pipeline")
   results = workflow.run(
       start_urls=["https://your-site.com"],
       fields=["your", "fields", "here"]
   )
   ```

---

## Support

**Documentation:**
- `README.md` - Project overview
- `FINAL_SUMMARY.md` - Complete implementation
- `QUICK_START.md` - This file

**Examples:**
- `test_universal_leafly.py` - Single URL scraping
- `test_end_to_end_crawl.py` - Full crawling
- `test_schema_stability.py` - Schema management

**Questions?**
Check the documentation files or review the test scripts for examples.

---

**Status:** ✅ **PRODUCTION READY**

Start scraping any website in minutes!

---

**Last Updated:** November 7, 2025








