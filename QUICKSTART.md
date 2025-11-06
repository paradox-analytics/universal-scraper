# Universal Scraper - Quick Start Guide

Get started with Universal Scraper in 5 minutes!

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/universal-scraper.git
cd universal-scraper
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Set Up API Key

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```bash
# For OpenAI (recommended for beginners)
OPENAI_API_KEY=sk-your-key-here

# OR for Gemini (free tier available)
GEMINI_API_KEY=your-key-here

# OR for Claude
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## 📖 Your First Scrape

### Option 1: Python Code

Create `my_first_scrape.py`:

```python
from universal_scraper import UniversalScraper
import os

# Initialize scraper
scraper = UniversalScraper(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name='gpt-4o-mini'  # Cheapest option
)

# Scrape a URL
result = scraper.scrape(
    url='https://books.toscrape.com/',
    fields=['title', 'price', 'rating']
)

# Print results
print(f"✅ Extracted {len(result['data'])} items")
for item in result['data'][:5]:
    print(item)

scraper.close()
```

Run it:

```bash
python my_first_scrape.py
```

### Option 2: Command Line

```bash
universal-scraper \
  --url "https://books.toscrape.com/" \
  --fields title price rating \
  --output results.json
```

### Option 3: Run Example

```bash
python examples/basic_usage.py
```

## 🎯 Common Use Cases

### E-commerce Product Scraping

```python
scraper.scrape(
    url='https://shop.example.com/products',
    fields=['product_name', 'price', 'rating', 'availability', 'image_url']
)
```

### Job Listings

```python
scraper.scrape(
    url='https://jobs.example.com/listings',
    fields=['job_title', 'company', 'location', 'salary', 'description']
)
```

### News Articles

```python
scraper.scrape(
    url='https://news.example.com/',
    fields=['title', 'author', 'date', 'content', 'category']
)
```

## 🌍 Using Proxies (Optional)

For sites that block scrapers, add proxy configuration:

```python
scraper = UniversalScraper(
    api_key='your-key',
    proxy_config={
        'server': 'http://proxy.brightdata.com:22225',
        'username': 'your-username-zone-residential',
        'password': 'your-password'
    }
)
```

## 🔧 Configuration Options

### AI Models

**Cheap & Fast (Recommended):**
- `gpt-4o-mini` (OpenAI) - $0.15/1M tokens
- `gemini-2.0-flash-exp` (Google) - Free tier available
- `claude-3-haiku-20240307` (Anthropic) - $0.25/1M tokens

**Powerful:**
- `gpt-4o` (OpenAI)
- `gemini-1.5-pro` (Google)
- `claude-3-sonnet-20240229` (Anthropic)

### Caching

Enable caching to save costs on repeated scrapes:

```python
scraper = UniversalScraper(
    api_key='your-key',
    enable_cache=True,  # Default
    cache_dir='./cache',
    cache_ttl=86400  # 24 hours
)
```

## 📊 Batch Scraping

Scrape multiple URLs efficiently:

```python
urls = [
    'https://example.com/page1',
    'https://example.com/page2',
    'https://example.com/page3'
]

results = scraper.scrape_multiple(urls, fields=['title', 'content'])

# Cache automatically reuses code for similar pages!
```

## ☁️ Deploy to Apify

Deploy to Apify for cloud scraping:

```bash
# Install Apify CLI
npm install -g apify-cli

# Login to Apify
apify login

# Deploy
./deploy_to_apify.sh
```

Then run your scraper on Apify platform with a simple UI!

## 🐛 Troubleshooting

### "No API key found"
Make sure your `.env` file has the correct API key:
```bash
export OPENAI_API_KEY=sk-your-key-here  # Or add to .env
```

### "Failed to fetch URL"
- Check if the URL is accessible
- Try adding a proxy configuration
- Some sites require session warming (enabled by default)

### "No data extracted"
- Try more specific field names (e.g., 'product_title' instead of 'title')
- Check if the page has the data you're looking for
- Enable debug logging: `log_level=logging.DEBUG`

### "Too expensive"
- Use cheaper models (`gpt-4o-mini`, `gemini-flash`)
- Enable caching (enabled by default)
- Similar pages reuse code automatically

## 📚 Next Steps

1. **Read Examples**: Check out `examples/` directory
2. **Explore API**: See `README.md` for full API reference
3. **Join Community**: Open issues, contribute on GitHub
4. **Deploy to Cloud**: Try Apify deployment
5. **Advanced Features**: Custom extraction, batch processing, cache management

## 💡 Tips

1. **Cache Saves Money**: First page generates code, similar pages reuse it (90%+ cost savings)
2. **JSON First**: Scraper automatically detects JSON sources (faster & cheaper)
3. **Batch Processing**: Scraping multiple similar URLs? Cache reuse makes it much faster
4. **Proxy Rotation**: Use residential proxies for production scraping
5. **Field Names Matter**: More specific = better results

## 🎓 Learning Resources

- **Basic Usage**: `examples/basic_usage.py`
- **Batch Scraping**: `examples/batch_scraping.py`
- **With Proxies**: `examples/with_proxies.py`
- **Cache Management**: `examples/cache_management.py`
- **E-commerce**: `examples/ecommerce_scraping.py`

## 🤝 Need Help?

- **Documentation**: `README.md`
- **Issues**: Open GitHub issue
- **Examples**: Check `examples/` directory
- **Contributing**: See `CONTRIBUTING.md`

---

Happy scraping! 🎉

