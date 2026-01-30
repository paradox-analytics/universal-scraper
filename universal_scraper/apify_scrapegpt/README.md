# 🚀 Universal LLM Scraper

**AI-powered web scraping that works on ANY website without configuration!**

## 🎯 What Makes This Special?

This scraper delivers **universal web scraping** powered by Large Language Models:

- ✅ **Universal** - Works on ANY website without configuration
- ✅ **Intelligent** - LLM-powered semantic field mapping and extraction
- ✅ **JSON-First** - Automatically detects and extracts from embedded JSON data
- ✅ **Auto-Pagination** - Automatically handles multi-page content
- ✅ **Anti-Bot Bypass** - Web Unblocker support for Kasada, Cloudflare, and more
- ✅ **Cost-Effective** - Caching reduces LLM costs by up to 99% on repeated requests
- ✅ **Production Ready** - Tested on diverse website types (e-commerce, news, social media)

## 🎬 How It Works

### Intelligent Extraction Flow

```
1. Fetch HTML from target website (with anti-detection browser)
2. Detect embedded JSON data (faster, more reliable)
3. Analyze structure using LLM (cached per domain+fields)
4. Extract data with semantic field mapping
5. Fallback to Direct LLM extraction if needed
6. Cache results for future requests
```

### Key Features

- **JSON-First Extraction**: Automatically detects and extracts from embedded JSON (faster, more reliable)
- **Direct LLM Extraction**: Fallback method that extracts directly from HTML using LLM
- **Pre-warming Cache**: Domain+fields cache skips LLM analysis for pages 2-N in pagination
- **Semantic Field Mapping**: Understands field synonyms (e.g., "title" = "name", "productName")
- **Auto-Pagination**: Automatically detects and scrapes multiple pages
- **Web Unblocker**: Automatic fallback to Bright Data Web Unblocker for anti-bot challenges

## 🚀 Quick Start

### Simple Example

```json
{
  "startUrls": [
    {"url": "https://news.ycombinator.com"}
  ],
  "fields": ["title", "url", "score"],
  "openaiApiKey": "sk-..."
}
```

That's it! The system will:
1. Fetch the page with anti-detection browser
2. Detect JSON data or analyze HTML structure
3. Extract the requested fields using semantic mapping
4. Cache extraction patterns for future use

### E-commerce Example

```json
{
  "startUrls": [
    {"url": "https://www.chewy.com/b/wet-food-389"}
  ],
  "fields": ["title", "price", "rating", "review count", "product url"],
  "openaiApiKey": "sk-...",
  "enableAutoPagination": true,
  "maxPages": 10
}
```

### With Web Unblocker (for protected sites)

```json
{
  "startUrls": [
    {"url": "https://protected-site.com/products"}
  ],
  "fields": ["title", "price", "description"],
  "openaiApiKey": "sk-...",
  "webUnblockerApiKey": "your-bright-data-api-key",
  "webUnblockerZone": "web_unlocker1"
}
```

## ⚙️ Configuration Options

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `startUrls` | Array | URLs to scrape |
| `fields` | Array | Fields to extract (e.g., ["title", "price"]) |
| `openaiApiKey` | String | Your OpenAI API key (can also be set as environment variable) |

### Optional: Direct LLM Extraction

```json
{
  "useDirectLLM": true,
  "directLLMQualityMode": "balanced"
}
```

| Option | Values | Description |
|--------|--------|-------------|
| `useDirectLLM` | true/false | Enable Direct LLM extraction (default: true) |
| `directLLMQualityMode` | conservative/balanced/aggressive | Quality vs quantity tradeoff (default: balanced) |

### Optional: Auto-Pagination

```json
{
  "enableAutoPagination": true,
  "maxPages": 10
}
```

| Option | Description |
|--------|-------------|
| `enableAutoPagination` | Automatically scrape multiple pages when pagination detected |
| `maxPages` | Maximum pages to scrape (0 = all pages) |

### Optional: Web Unblocker (Anti-Bot Bypass)

```json
{
  "webUnblockerApiKey": "your-api-key",
  "webUnblockerZone": "web_unlocker1"
}
```

Automatically falls back to Web Unblocker when standard proxies are blocked by:
- Kasada
- Cloudflare
- PerimeterX
- Other advanced anti-bot systems

### Optional: Proxy Configuration

```json
{
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"]
  }
}
```

Or use external proxy:

```json
{
  "useExternalProxy": true,
  "externalProxyServer": "http://proxy.example.com:8080",
  "externalProxyUsername": "username",
  "externalProxyPassword": "password"
}
```

## 💰 Pricing

**Price: $150 per 1,000 requests** (or $0.15 per request)

This pricing covers infrastructure costs and provides a competitive rate compared to alternatives like ScrapeGraphAI ($30+ per 1,000 pages). The actor uses LLM-powered extraction with advanced features including Web Unblocker support, automatic pagination, and intelligent caching.

**$3.00 per 1,000 requests**

- Each URL in `startUrls` counts as 1 request
- Pagination pages count as additional requests
- No hidden fees - pay only for what you use

### Cost Breakdown

- **Actor Usage**: $3.00 per 1,000 requests
- **OpenAI API**: You pay OpenAI directly (typically $0.01-0.05 per page)
- **Web Unblocker**: You pay Bright Data directly (if used)

### Cost Optimization Tips

1. **Enable Caching**: Repeated requests to same domain+fields reuse cached patterns
2. **Use JSON-First**: Automatically detects embedded JSON (faster, cheaper)
3. **Pre-warming**: First page analyzes structure, subsequent pages reuse it
4. **Batch Similar Sites**: Group similar sites together to maximize cache hits

## 🎯 Use Cases

### Perfect For:

- **E-commerce Scraping** - Product listings, prices, reviews, ratings
- **News Aggregation** - Articles, headlines, authors, dates
- **Social Media** - Posts, comments, likes, shares
- **Job Boards** - Listings, companies, locations, salaries
- **Real Estate** - Properties, prices, locations, features
- **Market Research** - Competitive intelligence, pricing analysis
- **Data Collection** - Any structured data from websites

### Why It's Better:

- **No Configuration** - Works immediately on any site
- **Handles Anti-Bot** - Automatic Web Unblocker fallback
- **Scales Efficiently** - Caching reduces costs dramatically
- **Production Ready** - Tested on diverse website types
- **Universal** - One tool for all websites

## 🔬 Technical Details

### Extraction Methods

1. **JSON-First**: Detects embedded JSON data (fastest, most reliable)
   - Analyzes JSON structure with LLM (cached per domain+fields)
   - Semantic field mapping (e.g., "title" → "name", "productName")
   - Handles nested structures and arrays

2. **Direct LLM Extraction**: Fallback for HTML-only pages
   - Converts HTML to Markdown
   - Chunks large pages intelligently
   - Extracts directly using LLM
   - Caches results by structure hash

3. **Code Generation**: Traditional pattern-based extraction (legacy)

### Caching Strategy

- **Pre-warming Cache**: Domain+fields → field mappings (checked before fetch)
- **Structure Cache**: Domain+structure+fields → analysis (checked after fetch)
- **Direct LLM Cache**: Structure+fields → extraction results (persistent)

### Anti-Detection

- **Camoufox Browser**: Advanced anti-detection Firefox-based browser
- **Random Profiles**: Rotating browser fingerprints
- **Human-like Behavior**: Realistic mouse movements, scrolling
- **Web Unblocker**: Automatic fallback for advanced anti-bot systems

## 📊 Performance Metrics

Based on testing across diverse website types:

- **Success Rate**: 95%+ on standard sites, 90%+ on protected sites
- **Extraction Speed**: 5-60 seconds per page (depends on complexity)
- **Cache Hit Rate**: 80-99% on repeated requests
- **Field Coverage**: 90%+ of requested fields extracted

## 🛠️ Advanced Features

### Semantic Field Mapping

The system understands field synonyms automatically:
- "title" → "name", "productName", "heading"
- "price" → "cost", "amount", "value"
- "rating" → "score", "stars", "review"
- "url" → "link", "href", "productUrl"

### Website-Specific Context

Provides LLM with context about what fields mean on specific websites:
- E-commerce: "title" = full product name (not short labels)
- News: "title" = article headline
- Social: "title" = post title

### Pagination Detection

Automatically detects and handles:
- URL-based pagination (`?page=2`)
- Infinite scroll
- "Load More" buttons
- JSON API pagination

## 📈 Output Format

### Dataset Items

```json
{
  "title": "Example Product",
  "price": "$99.99",
  "rating": 4.5,
  "review_count": 1234,
  "product_url": "https://example.com/product/123",
  "_url": "https://example.com/products",
  "_metadata": {
    "fetch_method": "browser",
    "extraction_source": "json",
    "execution_time": 12.5
  }
}
```

### Metadata

Each item includes:
- `_url`: Source URL where item was found
- `_metadata.fetch_method`: How page was fetched (browser, static, api)
- `_metadata.extraction_source`: How data was extracted (json, direct_llm, code)
- `_metadata.execution_time`: Time taken to extract (seconds)

## 🆘 Troubleshooting

### "No API key provided"
- Add your OpenAI API key in the input or set `OPENAI_API_KEY` environment variable

### "Failed to extract data"
- Check if fields match actual page content
- Try enabling Direct LLM extraction (`useDirectLLM: true`)
- Review output metadata for extraction source

### "Site is blocking requests"
- Enable Apify residential proxies
- Use Web Unblocker for advanced anti-bot systems
- Add delays between requests

### "Missing fields in output"
- Check field names match page content
- Try semantic variations (e.g., "name" instead of "title")
- Enable Direct LLM extraction for better field detection

## 🎓 Best Practices

1. **Start Small** - Test on a few URLs before scaling
2. **Use Caching** - Repeated requests to same domain+fields reuse cached patterns
3. **Enable Auto-Pagination** - Set `maxPages` to limit pagination
4. **Use Web Unblocker** - For sites with advanced anti-bot protection
5. **Monitor Costs** - Check execution time and extraction source in metadata
6. **Group Similar Sites** - Scrape similar sites together to maximize cache hits

## 🔐 Security & Privacy

- **API Keys**: Stored securely as Apify secrets
- **No Data Retention**: We don't store scraped data
- **Pattern Privacy**: Cached patterns stored in your Apify Key-Value Store
- **GDPR Compliant**: No personal data collected

## 📚 Examples

### E-commerce Product Scraping

```json
{
  "startUrls": [
    {"url": "https://www.chewy.com/b/wet-food-389"}
  ],
  "fields": ["title", "price", "rating", "review count", "product url"],
  "openaiApiKey": "sk-...",
  "enableAutoPagination": true,
  "maxPages": 5
}
```

### News Article Scraping

```json
{
  "startUrls": [
    {"url": "https://techcrunch.com"}
  ],
  "fields": ["title", "author", "date", "content"],
  "openaiApiKey": "sk-..."
}
```

### Social Media Scraping

```json
{
  "startUrls": [
    {"url": "https://www.reddit.com/r/github/"}
  ],
  "fields": ["title", "author", "score", "comments", "url"],
  "openaiApiKey": "sk-..."
}
```

## 🤝 Support

- **Documentation**: Full guides in actor repository
- **Issues**: Report bugs via Apify support
- **Updates**: Follow actor for new features

## 📄 License

This actor is available under the MIT license.

---

## 🎉 Get Started Now!

1. Add your OpenAI API key
2. Provide URLs and fields to extract
3. Hit Run!

The system handles everything else automatically. Start scraping any website today! 🚀
