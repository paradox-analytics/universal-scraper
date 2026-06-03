# 🎯 Crawl Patterns - Quick Reference

## The Answer to "How do I tell the crawl what to scrape?"

Use `follow_patterns` and `ignore_patterns` in `CrawlConfig`!

---

## Quick Example

```python
from universal_scraper.crawler import CrawlConfig, UniversalCrawler

# Only crawl dispensary pages, ignore navigation
config = CrawlConfig(
    follow_patterns=[
        '/dispensaries/',      # ✅ Follow listing pages
        '/dispensary-info/'    # ✅ Follow detail pages
    ],
    ignore_patterns=[
        '/products',           # ❌ Ignore products
        '/news',               # ❌ Ignore news
        '/strains'             # ❌ Ignore strains
    ]
)

crawler = UniversalCrawler(config=config)
results = crawler.crawl(["https://www.leafly.com/dispensaries/nevada"])

# Result: Only 30 relevant URLs crawled (not 5,472!)
```

---

## Common Patterns

### E-commerce
```python
follow_patterns=['/product/', '/category/']
ignore_patterns=['/cart', '/checkout', '/account']
```

### News Site
```python
follow_patterns=['/article/', '/2024/']
ignore_patterns=['/author/', '/tag/', '/category/']
```

### Real Estate
```python
follow_patterns=['/property/', '/listing/', '/for-sale/']
ignore_patterns=['/agent/', '/blog', '/mortgage-calculator']
```

### Job Board
```python
follow_patterns=['/job/', '/jobs/', '/careers/']
ignore_patterns=['/company/', '/blog', '/resources']
```

---

## Full Pipeline Example

```python
from universal_scraper import UniversalWorkflow
from universal_scraper.crawler import CrawlConfig

# Configure what to crawl
crawl_config = CrawlConfig(
    max_depth=2,
    follow_patterns=['/product/'],
    ignore_patterns=['/cart', '/auth']
)

# Run full pipeline
workflow = UniversalWorkflow(
    mode="full_pipeline",
    openai_api_key="your-key",
    crawl_config=crawl_config
)

# Automatically crawls matching URLs and scrapes data
results = workflow.run(
    start_urls=["https://shop.com"],
    fields=["name", "price", "rating"]
)
```

---

## Real Results (Leafly Test)

**Without Patterns:**
- Discovered: 5,472 URLs
- Crawled: 5,472 URLs (everything!)
- Time: Hours
- Wasted: 99% of pages

**With Patterns:**
- Discovered: 5,472 URLs
- Crawled: 30 URLs (only dispensaries!)
- Time: 3 minutes
- Efficiency: 99.5% improvement! 🚀

---

## See Full Guide

📖 **[CRAWL_TARGETING_GUIDE.md](CRAWL_TARGETING_GUIDE.md)** - Complete documentation with examples

---

**Last Updated:** November 7, 2025








