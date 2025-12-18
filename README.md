# Universal Web Scraper

A powerful, AI-driven universal web scraper that can extract structured data from any website. Built for cost efficiency and speed by prioritizing JSON detection, using intelligent HTML cleaning, and leveraging code caching.

## Key Features

- **JSON-First Architecture**: Automatically detects and extracts JSON data before resorting to HTML parsing
- **Smart HTML Cleaning**: Reduces HTML size by ~98% while preserving structure
- **Code Caching**: Generates extraction code once, reuses for similar pages
- **Multi-Provider AI**: Supports OpenAI, Gemini, Claude, and 100+ models via LiteLLM
- **Residential Proxies**: Built-in support for proxy rotation and anti-blocking
- **Apify Ready**: Deployable to Apify platform with one command
- **Cost Optimized**: Uses LLMs only for understanding structure, not extraction

## Architecture

```
URL → HTML Fetcher → Smart Cleaner → JSON Detector → Structural Hash → Code Cache
                                           ↓                    ↓            ↓
                                    JSON Extractor      Cache Miss    Cache Hit
                                           ↓                    ↓            ↓
                                    Structured Data ← AI Code Gen ← Cached Code
```

### Components

1. **HTML Fetcher**: CloudScraper-based fetcher with anti-bot protection and proxy support
2. **JSON Detector**: Scans for JSON endpoints, GraphQL APIs, and embedded JSON-LD
3. **Smart HTML Cleaner**: Removes 98% of HTML while keeping structure
4. **Structural Hash**: Generates fingerprint of page structure for cache matching
5. **Code Cache**: Stores and reuses extraction code for similar pages
6. **AI Code Generator**: Creates BeautifulSoup extraction code using LLMs
7. **Data Extractor**: Executes generated code and returns structured data

## Quick Start

### Installation

```bash
cd universal-scraper
pip install -r requirements.txt
```

### Basic Usage

```python
from universal_scraper import UniversalScraper

# Initialize with OpenAI API key
scraper = UniversalScraper(
    api_key="your-openai-api-key",
    model_name="gpt-4o-mini"  # Or gemini-2.5-flash, claude-3-haiku, etc.
)

# Scrape any URL
result = scraper.scrape(
    url="https://example.com/products",
    fields=["product_name", "price", "rating", "availability"]
)

print(f"Extracted {len(result['data'])} items")
print(result['data'])
```

### Command Line

```bash
# Single URL
python -m universal_scraper.cli \
    --url "https://example.com/products" \
    --fields product_name price rating \
    --output products.json

# Multiple URLs
python -m universal_scraper.cli \
    --urls urls.txt \
    --fields product_name price \
    --output-dir results/
```

## 📊 Configuration

### Proxy Support

```python
scraper = UniversalScraper(
    api_key="your-api-key",
    proxy_config={
        "server": "http://proxy.brightdata.com:22225",
        "username": "customer-user-zone-residential",
        "password": "your-password"
    }
)
```

### AI Model Selection

```python
# OpenAI (default)
scraper = UniversalScraper(api_key="sk-...", model_name="gpt-4o-mini")

# Gemini
scraper = UniversalScraper(api_key="AIza...", model_name="gemini-2.5-flash")

# Claude
scraper = UniversalScraper(api_key="sk-ant-...", model_name="claude-3-haiku-20240307")

# Any LiteLLM model
scraper = UniversalScraper(api_key="...", model_name="llama-2-70b-chat")
```

### Caching

```python
scraper = UniversalScraper(
    api_key="your-api-key",
    cache_dir="./cache",  # Directory for code cache
    cache_ttl=86400,      # Cache TTL in seconds (24 hours)
    enable_cache=True     # Enable/disable caching
)
```

## Use Cases

### E-commerce Product Scraping

```python
result = scraper.scrape(
    url="https://shop.com/products",
    fields=[
        "product_name",
        "product_price",
        "product_rating",
        "product_reviews_count",
        "product_availability",
        "product_image_url"
    ]
)
```

### Job Listings

```python
result = scraper.scrape(
    url="https://jobs.com/listings",
    fields=[
        "job_title",
        "company_name",
        "location",
        "salary_range",
        "job_description",
        "apply_url"
    ]
)
```

### Real Estate

```python
result = scraper.scrape(
    url="https://realestate.com/listings",
    fields=[
        "property_address",
        "price",
        "bedrooms",
        "bathrooms",
        "square_feet",
        "listing_agent",
        "property_images"
    ]
)
```

## Apify Deployment

### Deploy to Apify

```bash
cd universal-scraper
./deploy_to_apify.sh
```

### Apify Input Schema

```json
{
  "urls": ["https://example.com/page1", "https://example.com/page2"],
  "fields": ["field1", "field2", "field3"],
  "proxyType": "residential",
  "aiModel": "gpt-4o-mini",
  "outputFormat": "json"
}
```

## Advanced Features

### JSON Detection Priority

The scraper automatically detects JSON in the following order:

1. **JSON-LD Scripts**: Structured data in `<script type="application/ld+json">`
2. **GraphQL Endpoints**: Detects and queries GraphQL APIs
3. **XHR/Fetch Requests**: Monitors network traffic for JSON endpoints
4. **Embedded JSON**: Finds JSON in JavaScript variables

### Smart HTML Cleaning

Removes unnecessary elements while preserving structure:

- Scripts and styles
- Ads and analytics
- Inline SVG images
- Navigation elements
- Empty divs
- Non-essential attributes
- Detects and samples repeating structures (keeps 2, removes rest)

### Structural Hashing

Generates a hash of the page structure for intelligent caching:

```python
# Pages with same structure reuse cached extraction code
hash1 = scraper.get_structural_hash(url1)  # First time: generates code
hash2 = scraper.get_structural_hash(url2)  # Same structure: reuses code
```

## Project Structure

```
universal-scraper/
├── README.md
├── requirements.txt
├── setup.py
├── universal_scraper/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── scraper.py          # Main scraper class
│   │   ├── html_fetcher.py     # CloudScraper + proxy support
│   │   ├── html_cleaner.py     # Smart HTML cleaning
│   │   ├── json_detector.py    # JSON detection priority
│   │   ├── structural_hash.py  # Page structure fingerprinting
│   │   ├── code_cache.py       # Caching system
│   │   └── ai_generator.py     # Multi-provider AI code generation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── anti_blocking.py    # Anti-blocking utilities
│   │   └── proxy_manager.py    # Proxy rotation
│   ├── cli.py                  # Command-line interface
│   └── apify/
│       ├── __init__.py
│       ├── actor.py            # Apify actor main
│       ├── Dockerfile
│       └── INPUT_SCHEMA.json
├── examples/
│   ├── basic_usage.py
│   ├── batch_scraping.py
│   └── custom_fields.py
└── tests/
    ├── test_scraper.py
    ├── test_json_detector.py
    └── test_html_cleaner.py
```

## Examples

See the `examples/` directory for comprehensive examples:

- `basic_usage.py`: Simple single-URL scraping
- `batch_scraping.py`: Scraping multiple URLs efficiently
- `custom_fields.py`: Advanced field extraction
- `with_proxies.py`: Using residential proxies
- `cache_management.py`: Managing the code cache

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

