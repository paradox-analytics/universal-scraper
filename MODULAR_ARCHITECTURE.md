# Universal Web Scraper - Modular Architecture

## Overview

The Universal Web Scraper is built as **separate modules within a unified product**, providing maximum flexibility while maintaining ease of use.

```
┌─────────────────────────────────────────────────────────────────┐
│                   UNIVERSAL WEB SCRAPER                          │
│                    (Unified Product)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
      ┌──────────┐     ┌──────────┐    ┌──────────┐
      │ CRAWLER  │     │ SCRAPER  │    │ORCHESTRATOR│
      │ Module   │     │ Module   │    │  Module  │
      └──────────┘     └──────────┘    └──────────┘
            │                 │                 │
            │                 │                 │
     ┌──────┴──────┐   ┌──────┴──────┐        │
     │ Sub-Modules │   │ Sub-Modules │        │
     └─────────────┘   └─────────────┘        │
                                               │
                     Coordinates Both ─────────┘
```

---

## Module Structure

### Module 1: CRAWLER (URL Discovery)

**Location**: `universal_scraper/crawler/`

**Purpose**: Discover URLs across websites using multiple strategies

**Sub-Modules**:
```
crawler/
├── crawler.py              # Main orchestration
├── link_discovery.py       # Traditional HTML link extraction
├── api_discovery.py        # Network request interception
├── search_discovery.py     # Query enumeration (A, AA, AB...)
├── page_classifier.py      # Page type detection
└── pagination_handler.py   # Pagination discovery
```

**Capabilities**:
- ✅ **Link-Based Discovery**: Extract links from HTML
- ✅ **API-Based Discovery**: Intercept network requests
- ✅ **Search-Based Discovery**: Query enumeration for search-required sites
- ✅ **Pagination Handling**: All pagination types
- ✅ **Page Classification**: Listing vs Detail vs Search

**Usage (Standalone)**:
```python
from universal_scraper.crawler import UniversalCrawler, CrawlConfig

config = CrawlConfig(
    mode='smart',
    max_depth=3,
    handle_pagination=True,
    discover_apis=True,
    enable_search_discovery=True
)

crawler = UniversalCrawler(config)
result = crawler.crawl(['https://example.com'])

print(f"Discovered {len(result.urls)} URLs")
```

---

### Module 2: SCRAPER (Data Extraction)

**Location**: `universal_scraper/core/`

**Purpose**: Extract structured data from URLs

**Sub-Modules**:
```
core/
├── scraper.py              # Main orchestration
├── html_fetcher.py         # Static HTML fetching
├── browser_fetcher.py      # JavaScript rendering
├── hybrid_fetcher.py       # Smart fetching (static + browser)
├── json_detector.py        # JSON source detection
├── html_cleaner.py         # HTML simplification
├── structural_hash.py      # Page structure fingerprinting
├── code_cache.py           # Extraction code caching
├── ai_generator.py         # AI code generation
├── schema_manager.py       # Schema enforcement
└── schema_inference.py     # Auto schema generation
```

**Capabilities**:
- ✅ **JSON-First**: Prioritize embedded JSON/APIs
- ✅ **JavaScript Rendering**: Full SPA support
- ✅ **AI Code Generation**: BeautifulSoup code generation
- ✅ **Schema Management**: Stable output schemas
- ✅ **Auto Schema Generation**: Learn from data

**Usage (Standalone)**:
```python
from universal_scraper.core import UniversalScraper

scraper = UniversalScraper(api_key='your-api-key')
result = scraper.scrape(
    'https://example.com/product',
    fields=['name', 'price', 'description']
)

print(f"Extracted {len(result['data'])} items")
```

---

### Module 3: ORCHESTRATOR (Workflow Coordination)

**Location**: `universal_scraper/orchestrator/`

**Purpose**: Coordinate crawler and scraper for complete workflows

**Components**:
```
orchestrator/
└── workflow.py             # Workflow coordination
```

**Workflow Modes**:
1. **CRAWL_ONLY**: Just discover URLs
2. **SCRAPE_ONLY**: Extract from provided URLs
3. **CRAWL_THEN_SCRAPE**: Discover all, then scrape all
4. **STREAM_SCRAPE**: Scrape as URLs discovered
5. **FULL_AUTO**: Auto-detect and execute

**Usage**:
```python
from universal_scraper.orchestrator import UniversalWorkflow, WorkflowMode

workflow = UniversalWorkflow()
result = workflow.execute(
    start_urls=['https://example.com'],
    fields=['name', 'price']
)

print(f"Extracted {result['total_items']} items from {len(result['urls_discovered'])} pages")
```

---

## Unified Apify Actor

**Location**: `universal_scraper/apify/actor_v2.py`

**Purpose**: Single integrated product for Apify platform

### Single Input Schema

All modules are configured through one unified INPUT_SCHEMA:

```json
{
  "mode": "full_auto",
  "startUrls": ["https://example.com"],
  "fields": ["name", "price"],
  "crawlConfig": {
    "maxDepth": 3,
    "handlePagination": true,
    "discoverApis": true
  },
  "searchConfig": {
    "strategy": "auto",
    "maxDepth": 4
  },
  "schemaConfig": {
    "useSchema": true,
    "schemaType": "auto"
  }
}
```

### Apify UI Sections

The INPUT_SCHEMA is organized into logical sections in the Apify UI:

1. **Execution Mode** - Choose workflow
2. **Input URLs** - Start URLs or specific URLs
3. **Data Extraction** - Fields to extract
4. **Crawl Settings** - Depth, pagination, patterns
5. **Search Discovery** - Query enumeration config
6. **Output Schema** - Schema management
7. **API Configuration** - API keys
8. **Browser Settings** - JavaScript rendering
9. **Proxy Settings** - Proxy configuration
10. **Output Settings** - Format preferences
11. **Advanced Settings** - Debug, concurrency

---

## How Modules Work Together

### Example 1: E-commerce Site Crawl

**Goal**: Scrape all products from an e-commerce category

**Flow**:
```
1. User Input (Apify UI):
   - Mode: crawl_then_scrape
   - Start URL: https://shop.com/category/electronics
   - Fields: [name, price, rating]

2. Orchestrator receives input

3. CRAWLER Module:
   - Link Discoverer: Finds product links
   - Pagination Handler: Discovers pages 1-10
   - Page Classifier: Identifies product detail pages
   → Discovers 200 product URLs

4. SCRAPER Module:
   - For each URL:
     - Hybrid Fetcher: Static fetch (fast)
     - JSON Detector: Finds embedded product data
     - Schema Manager: Normalizes to stable schema
   → Extracts 200 products

5. Output: 200 products with stable schema
```

### Example 2: County Assessor Database

**Goal**: Extract all property records from search-only database

**Flow**:
```
1. User Input:
   - Mode: full_auto
   - Start URL: https://county-assessor.gov/search
   - Fields: [owner, address, value]
   - searchConfig: { strategy: "alphabetic" }

2. Orchestrator detects search-required page

3. CRAWLER Module:
   - Page Classifier: Detects SEARCH_REQUIRED
   - Search Discoverer: Activates
     - Tries: A, AA, AB, AC... (recursive)
     - Detects 100 result limit
     - Goes deeper when capped
   → Discovers 5,000 property URLs

4. SCRAPER Module:
   - Extracts data from each property page
   → 5,000 property records

5. Output: Complete dataset via search enumeration
```

### Example 3: Leafly Dispensary Crawl

**Goal**: Scrape all Nevada dispensaries and their menus

**Flow**:
```
1. User Input:
   - Mode: crawl_then_scrape
   - Start URL: https://leafly.com/dispensaries/nevada
   - Fields: [name, address, products]
   - crawlConfig: { discoverApis: true }

2. CRAWLER Module:
   - API Discoverer: Intercepts /api/dispensaries
   - Discovers: 500 dispensary URLs
   - For each dispensary:
     - Discovers: /menu link
   → 500 dispensaries + 500 menu pages

3. SCRAPER Module:
   - Hybrid Fetcher: Renders JavaScript
   - JSON Detector: Extracts __NEXT_DATA__
   - Schema Manager: Normalizes products
   → 500 dispensaries + 10,000 products

4. Output: Hierarchical data (dispensaries → products)
```

---

## Benefits of Modular Architecture

### 1. Independent Usage

Each module can be used alone:

```python
# Use only crawler
from universal_scraper.crawler import UniversalCrawler
urls = crawler.crawl(['https://example.com'])

# Use only scraper
from universal_scraper.core import UniversalScraper
data = scraper.scrape('https://example.com/product', fields)

# Use orchestrator (both together)
from universal_scraper.orchestrator import UniversalWorkflow
result = workflow.execute(start_urls, fields)
```

### 2. Easy Testing

Test modules in isolation:

```python
# Test crawler alone
def test_crawler():
    crawler = UniversalCrawler()
    result = crawler.crawl(['https://test.com'])
    assert len(result.urls) > 0

# Test scraper alone
def test_scraper():
    scraper = UniversalScraper()
    result = scraper.scrape('https://test.com', ['title'])
    assert len(result['data']) > 0
```

### 3. Independent Scaling

Scale modules independently in production:

```python
# High crawl volume, low scraping
crawler_workers = 10
scraper_workers = 2

# High scraping volume, low crawling
crawler_workers = 2
scraper_workers = 10
```

### 4. Flexible Workflows

Mix and match as needed:

```python
# Option 1: Crawl separately, scrape later
urls = crawler.crawl(start_urls)
save_urls(urls)  # Save for later

# Later...
urls = load_urls()
data = scraper.scrape_batch(urls)

# Option 2: Integrated
result = workflow.execute(start_urls, fields)
```

---

## Adding New Sub-Modules

The architecture makes it easy to add new discovery strategies:

### Example: Add Sitemap Discovery

```python
# 1. Create new sub-module
# universal_scraper/crawler/sitemap_discovery.py

class SitemapDiscoverer:
    def discover(self, url: str) -> List[str]:
        # Parse sitemap.xml
        # Return all URLs
        pass

# 2. Integrate into crawler
# universal_scraper/crawler/crawler.py

from .sitemap_discovery import SitemapDiscoverer

class UniversalCrawler:
    def __init__(self, config):
        # ... existing code ...
        self.sitemap_discoverer = SitemapDiscoverer()
    
    def _crawl_url(self, url_info):
        # ... existing code ...
        if is_sitemap(url):
            urls = self.sitemap_discoverer.discover(url)
            # Add to queue

# 3. Use immediately
config = CrawlConfig(discover_sitemaps=True)
crawler = UniversalCrawler(config)
```

No changes needed to scraper or orchestrator!

---

## Configuration Flow

### From Apify UI to Modules

```
Apify INPUT_SCHEMA
        │
        ▼
actor_v2.py (parse_input)
        │
        ├─> WorkflowConfig (orchestrator)
        ├─> CrawlConfig (crawler)
        ├─> Schema (scraper)
        └─> API keys, proxy, etc.
        │
        ▼
UniversalWorkflow (orchestrator)
        │
        ├─> UniversalCrawler (if crawling)
        │     ├─> LinkDiscoverer
        │     ├─> APIDiscoverer
        │     └─> SearchDiscoverer
        │
        └─> UniversalScraper (if scraping)
              ├─> HybridFetcher
              ├─> JSONDetector
              └─> SchemaManager
```

---

## Summary

### Modular but Integrated

✅ **Separate Modules**: Crawler, Scraper, Orchestrator  
✅ **Independent Sub-Modules**: Link discovery, API discovery, etc.  
✅ **Unified Product**: Single Apify Actor  
✅ **Single UI**: One INPUT_SCHEMA for all modules  
✅ **Flexible Usage**: Use modules independently or together  
✅ **Easy Testing**: Test each module in isolation  
✅ **Scalable**: Scale modules independently  
✅ **Extensible**: Add new sub-modules easily  

### Best of Both Worlds

**Separation where it matters** (modularity, testing, scaling)  
**Integration where it helps** (unified product, single UI, coordinated workflows)

This architecture provides the flexibility of separate products with the convenience of a unified solution.








