# Universal Web Scraper

A powerful, AI-driven universal web scraper that can extract structured data from any website. Built for cost efficiency and speed by prioritizing JSON detection, using intelligent HTML cleaning, and leveraging code caching.

## Key Features

- **JSON-First Architecture**: Automatically detects and extracts JSON data before resorting to HTML parsing
- **Smart HTML Cleaning**: Reduces HTML size by ~98% while preserving structure
- **Code Caching**: Generates extraction code once, reuses for similar pages
- **Multi-Provider AI**: Supports OpenAI, Gemini, Claude, and 100+ models via LiteLLM
- **Residential Proxies**: Built-in support for proxy rotation and anti-blocking
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

### Docker (Recommended)

```bash
# Clone and configure
git clone https://github.com/paradox-analytics/universal-scraper.git
cd universal-scraper
cp .env.example .env  # Edit with your API keys

# Start API + Redis
docker compose up -d

# API is now running at http://localhost:8080
curl http://localhost:8080/health
```

### Manual Installation

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
│   │   ├── proxy_manager.py    # Proxy rotation
│   └── cli.py                  # Command-line interface
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

## 🔧 Configuration Reference

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM integration | Yes (or other provider) |
| `ANTHROPIC_API_KEY` | Anthropic key if using Claude models | No |
| `GEMINI_API_KEY` | Google Gemini key | No |
| `PROXY_URL` | Default proxy URL (e.g., `http://user:pass@host:port`) | No |
| `HEADLESS` | Run browser in headless mode (`true`/`false`) | No (default: true) |

### UniversalScraper Options

```python
scraper = UniversalScraper(
    api_key="...",              # API Key for LLM
    model_name="gpt-4o",        # Model to use
    headless=True,              # Run browser invisibly
    proxy_config={...},         # Proxy settings
    cache_dir="./cache",        # Custom cache location
    enable_cache=True,          # Enable structural caching
    extraction_context="..."    # Optional hint for extractor
)
```

## ❓ Troubleshooting

### Common Issues

**1. 403/407 Forbidden Errors**
- **Cause**: The site is blocking the request or proxy authentication failed.
- **Solution**: 
  - Ensure you are using high-quality residential proxies.
  - Verify your proxy credentials in `proxy_config`.
  - Try enabling `web_unblocker=True` (if configured) or switching proxy providers.

**2. "No JSON found"**
- **Cause**: The page renders content via complex JavaScript that doesn't expose data in standard JSON blobs.
- **Solution**: 
  - The scraper will automatically fallback to LLM-based HTML extraction.
  - Ensure the page is fully loaded (increase `timeout` if needed).

**3. Slow Performance**
- **Cause**: Browser launching or LLM latency.
- **Solution**: 
  - Enable caching (`enable_cache=True`) to skip LLM generation for known structures.
  - Reuse the `scraper` instance instead of creating a new one for every request.

## API Documentation

Interactive API docs are available when the server is running:

- **Swagger UI**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **ReDoc**: [http://localhost:8080/redoc](http://localhost:8080/redoc)

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test module
pytest tests/test_json_detector.py -v

# Run with coverage
pytest --cov=universal_scraper --cov-report=html
```

## CI/CD

This project uses GitHub Actions for continuous integration:

- **Python lint + tests** — ruff linting, import validation, pytest suite
- **Frontend lint + build** — ESLint, TypeScript type checking, Vite build
- **Docker build** — validates the Dockerfile builds successfully

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

