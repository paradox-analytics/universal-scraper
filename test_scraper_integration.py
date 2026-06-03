
import sys
import os
import asyncio
import logging
from universal_scraper.core.scraper import UniversalScraper

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_integration():
    print("=" * 80)
    print("🧪 Testing UniversalScraper Integration (Product Hunt)")
    print("=" * 80)

    # Mock HTML fetcher to return our local file
    class MockFetcher:
        async def fetch(self, url, **kwargs):
            with open("product_hunt_raw_debug.html", 'r', encoding='utf-8') as f:
                html = f.read()
            return {'html': html, 'captured_json': []}

    api_key = os.getenv("OPENAI_API_KEY", "sk-mock-key")
    scraper = UniversalScraper(api_key=api_key, fetch_mode="static")
    scraper.html_fetcher = MockFetcher() # Inject mock fetcher

    # Test extraction
    print("\n🚀 Starting scrape...")
    result = await scraper.scrape(
        url="https://www.producthunt.com/",
        fields=[], # Auto-extract
        force_html=False
    )

    print(f"\n✅ Scrape complete!")
    print(f"Source: {result.get('source')}")
    data = result.get('data', [])
    print(f"Items extracted: {len(data)}")

    if data:
        print("\n🔍 Sample Item:")
        import json
        print(json.dumps(data[0], indent=2)[:500] + "...")
        
        # Check if we got Product Hunt posts
        posts = [item for item in data if item.get('__typename') == 'Post' or item.get('name')]
        print(f"\n📦 Posts found: {len(posts)}")
    else:
        print("❌ No data extracted.")

if __name__ == "__main__":
    asyncio.run(test_integration())
