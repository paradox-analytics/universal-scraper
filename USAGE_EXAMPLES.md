# Universal Web Scraper - Usage Examples

## Quick Start Examples

### Example 1: Simple Product Scraping

**Goal**: Extract products from a single page

```json
{
  "mode": "scrape_only",
  "urls": [
    {"url": "https://example.com/products"}
  ],
  "fields": ["name", "price", "description"]
}
```

**What happens**:
- Scraper module extracts data
- No crawling
- Fast and simple

---

### Example 2: E-commerce Category Crawl

**Goal**: Scrape all products in a category

```json
{
  "mode": "crawl_then_scrape",
  "startUrls": [
    {"url": "https://shop.com/category/electronics"}
  ],
  "fields": ["name", "price", "rating", "reviews"],
  "crawlConfig": {
    "maxDepth": 2,
    "handlePagination": true,
    "followPatterns": ["/product/"]
  }
}
```

**What happens**:
1. Crawler discovers product links
2. Handles pagination automatically
3. Scraper extracts from all products
4. Returns all products with stable schema

---

### Example 3: Leafly All Nevada Dispensaries

**Goal**: Extract all dispensaries and their menus

```json
{
  "mode": "crawl_then_scrape",
  "startUrls": [
    {"url": "https://www.leafly.com/dispensaries/nevada"}
  ],
  "fields": ["name", "address", "phone", "rating", "products"],
  "crawlConfig": {
    "maxDepth": 3,
    "handlePagination": true,
    "discoverApis": true,
    "followPatterns": ["/dispensary-info/"]
  },
  "schemaConfig": {
    "useSchema": true,
    "schemaType": "auto"
  }
}
```

**What happens**:
1. Crawler discovers all dispensary pages
2. API discovery captures menu endpoints
3. For each dispensary:
   - Extracts info page
   - Discovers menu link
   - Extracts products
4. Schema ensures stable output
5. Returns hierarchical data

---

### Example 4: County Assessor Search Database

**Goal**: Extract all properties via search enumeration

```json
{
  "mode": "full_auto",
  "startUrls": [
    {"url": "https://county-assessor.gov/search"}
  ],
  "fields": ["owner_name", "property_address", "assessed_value"],
  "crawlConfig": {
    "enableSearchDiscovery": true
  },
  "searchConfig": {
    "strategy": "alphabetic",
    "maxDepth": 4,
    "resultLimit": 100
  }
}
```

**What happens**:
1. Detects search-required page
2. Search discoverer activates
3. Queries: A, AA, AB... recursively
4. Handles 100-result limit
5. Extracts all ~5,000 properties

---

### Example 5: News Site with Pagination

**Goal**: Scrape all articles with pagination

```json
{
  "mode": "crawl_then_scrape",
  "startUrls": [
    {"url": "https://news-site.com/articles"}
  ],
  "fields": ["title", "author", "date", "content"],
  "crawlConfig": {
    "maxDepth": 1,
    "handlePagination": true,
    "maxPages": 50
  }
}
```

**What happens**:
1. Crawler detects pagination (pages 1-50)
2. Discovers all article links
3. Scraper extracts from each article
4. Returns all articles

---

### Example 6: JavaScript SPA with APIs

**Goal**: Scrape modern React site

```json
{
  "mode": "scrape_only",
  "urls": [
    {"url": "https://modern-spa.com/products"}
  ],
  "fields": ["name", "price", "stock"],
  "crawlConfig": {
    "discoverApis": true
  },
  "browserConfig": {
    "headless": true,
    "captureApiRequests": true,
    "waitForNetworkIdle": true
  }
}
```

**What happens**:
1. Browser renders JavaScript
2. Captures API calls
3. Extracts data from __NEXT_DATA__ or APIs
4. JSON-first extraction
5. Fast and reliable

---

### Example 7: Multiple Sites with Same Schema

**Goal**: Scrape multiple e-commerce sites with consistent output

```json
{
  "mode": "scrape_only",
  "urls": [
    {"url": "https://shop1.com/product/123"},
    {"url": "https://shop2.com/item/456"},
    {"url": "https://shop3.com/p/789"}
  ],
  "fields": ["name", "price", "description"],
  "schemaConfig": {
    "useSchema": true,
    "schemaType": "ecommerce",
    "strictSchema": false
  }
}
```

**What happens**:
1. Scraper uses e-commerce schema
2. Maps different field names to standard output
3. All sites return same schema:
   ```json
   {
     "product_name": "...",
     "price_usd": 29.99,
     "product_description": "..."
   }
   ```

---

## Advanced Examples

### Example 8: Custom Crawl Patterns

**Goal**: Only crawl specific URL patterns

```json
{
  "mode": "crawl_then_scrape",
  "startUrls": [
    {"url": "https://example.com"}
  ],
  "fields": ["title", "content"],
  "crawlConfig": {
    "maxDepth": 5,
    "followPatterns": [
      "/category/",
      "/product/",
      "/item/"
    ],
    "ignorePatterns": [
      "/login",
      "/cart",
      "/checkout",
      "/admin"
    ]
  }
}
```

---

### Example 9: Auto Schema Generation

**Goal**: Let system learn optimal schema

```json
{
  "mode": "crawl_then_scrape",
  "startUrls": [
    {"url": "https://new-site.com/products"}
  ],
  "fields": ["name", "price", "brand", "description"],
  "schemaConfig": {
    "useSchema": true,
    "schemaType": "auto"
  }
}
```

**What happens**:
1. First scrape learns data structure
2. Auto-generates optimal schema
3. All subsequent scrapes use stable schema
4. Output remains consistent even if site changes

---

### Example 10: Streaming Mode (Coming Soon)

**Goal**: Get results as they're discovered

```json
{
  "mode": "stream_scrape",
  "startUrls": [
    {"url": "https://example.com/listings"}
  ],
  "fields": ["title", "price"]
}
```

**What happens**:
1. Crawls and scrapes simultaneously
2. Streams results to dataset in real-time
3. Lower memory usage
4. Faster time-to-first-result

---

## Configuration Patterns

### Pattern 1: Depth-Limited Crawl

```json
{
  "crawlConfig": {
    "maxDepth": 2
  }
}
```

- `maxDepth: 0` - Only start URLs
- `maxDepth: 1` - Start URLs + links on those pages
- `maxDepth: 2` - Two levels deep
- `maxDepth: 3` - Three levels deep

### Pattern 2: URL Filtering

```json
{
  "crawlConfig": {
    "followPatterns": ["/product/", "/item/"],
    "ignorePatterns": ["/auth/", "/api/"]
  }
}
```

### Pattern 3: Search Enumeration

```json
{
  "searchConfig": {
    "strategy": "alphabetic",  // A, AA, AB...
    "maxDepth": 4,             // How deep to permute
    "resultLimit": 100         // When to subdivide
  }
}
```

### Pattern 4: Schema Control

```json
{
  "schemaConfig": {
    "useSchema": true,
    "schemaType": "auto",     // or "ecommerce" or "custom"
    "strictSchema": false     // Warn vs fail on missing fields
  }
}
```

---

## Output Examples

### Standard Output

```json
{
  "data": [
    {
      "name": "Product 1",
      "price": 29.99,
      "description": "..."
    },
    {
      "name": "Product 2",
      "price": 39.99,
      "description": "..."
    }
  ],
  "total_items": 2,
  "mode": "scrape_only",
  "workflow_metadata": {
    "duration_seconds": 5.2,
    "start_time": "2024-01-15T10:30:00"
  }
}
```

### With Crawl Metadata

```json
{
  "data": [...],
  "total_items": 150,
  "mode": "crawl_then_scrape",
  "urls_discovered": ["url1", "url2", ...],
  "crawl_metadata": {
    "total_discovered": 150,
    "total_crawled": 150,
    "crawl_tree": {
      "depth_0": 1,
      "depth_1": 10,
      "depth_2": 139
    }
  },
  "scrape_metadata": {
    "successful": 150,
    "failed": 0
  }
}
```

### With Schema Quality

```json
{
  "data": [...],
  "schema_quality": {
    "status": "healthy",
    "success_rate": 94.7,
    "field_coverage": {
      "name": 100.0,
      "price": 94.7,
      "brand": 89.3
    }
  }
}
```

---

## Common Use Cases

### Use Case 1: E-commerce Monitoring

```json
{
  "mode": "crawl_then_scrape",
  "startUrls": [{"url": "https://competitor.com/products"}],
  "fields": ["name", "price", "stock"],
  "schemaConfig": {"useSchema": true, "schemaType": "ecommerce"}
}
```

**Benefit**: Stable schema for price comparison

### Use Case 2: Real Estate Listings

```json
{
  "mode": "crawl_then_scrape",
  "startUrls": [{"url": "https://realestate.com/city/listings"}],
  "fields": ["address", "price", "bedrooms", "sqft"],
  "crawlConfig": {"handlePagination": true, "maxPages": 100}
}
```

**Benefit**: Complete market data with pagination

### Use Case 3: Government Database

```json
{
  "mode": "full_auto",
  "startUrls": [{"url": "https://county-records.gov/search"}],
  "fields": ["record_id", "owner", "value"],
  "searchConfig": {"strategy": "alphabetic", "maxDepth": 4}
}
```

**Benefit**: Extract from search-only interfaces

### Use Case 4: News Aggregation

```json
{
  "mode": "crawl_then_scrape",
  "startUrls": [
    {"url": "https://news1.com"},
    {"url": "https://news2.com"}
  ],
  "fields": ["title", "author", "date", "content"],
  "crawlConfig": {"maxDepth": 2}
}
```

**Benefit**: Aggregate from multiple sources

---

## Tips & Best Practices

### Tip 1: Start Simple

Begin with `scrape_only` on a single URL to test:

```json
{
  "mode": "scrape_only",
  "urls": [{"url": "https://example.com"}],
  "fields": ["title"]
}
```

Then expand to crawling once you know it works.

### Tip 2: Use Schema for Production

Always use schema in production:

```json
{
  "schemaConfig": {
    "useSchema": true,
    "schemaType": "auto"
  }
}
```

This ensures stable output even when sites change.

### Tip 3: Limit Depth Initially

Start with low depth to avoid runaway crawls:

```json
{
  "crawlConfig": {
    "maxDepth": 1,
    "maxPages": 100
  }
}
```

Increase gradually as needed.

### Tip 4: Use Patterns for Control

Control exactly what gets crawled:

```json
{
  "crawlConfig": {
    "followPatterns": ["/product/"],
    "ignorePatterns": ["/cart", "/login"]
  }
}
```

### Tip 5: Enable Debug for Troubleshooting

When things don't work as expected:

```json
{
  "debugMode": true
}
```

This provides detailed logs for diagnosing issues.

---

## Summary

The unified Apify Actor supports:

✅ **5 Workflow Modes** - From simple scraping to complex crawling  
✅ **3 Discovery Strategies** - Links, APIs, Search enumeration  
✅ **Auto Pagination** - All pagination types handled  
✅ **Schema Management** - Stable output schemas  
✅ **JavaScript Support** - Full SPA compatibility  
✅ **JSON-First** - Prioritize APIs and embedded JSON  
✅ **Flexible Configuration** - Fine-grained control when needed  
✅ **Auto-Detection** - Works out of the box  

**One product, any website!** 🚀








