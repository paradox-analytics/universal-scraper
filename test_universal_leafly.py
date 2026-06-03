"""
Test Universal Scraper with Leafly (JavaScript-heavy site)
Demonstrates JSON-forward architecture with browser support
"""

import sys
import os

# Add the package to path
sys.path.insert(0, '/Users/jevon_williams/Dev/universal-scraper')

from universal_scraper.core.scraper import UniversalScraper

# Set API key
os.environ['OPENAI_API_KEY'] = "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"

print("=" * 80)
print("🧪 UNIVERSAL SCRAPER - LEAFLY TEST")
print("=" * 80)
print()
print("This test demonstrates the JSON-forward architecture:")
print("  1. Tries static HTML first (fast)")
print("  2. Detects JavaScript is required")
print("  3. Falls back to Camoufox browser")
print("  4. Captures API requests during page load")
print("  5. Caches discovered APIs for future use")
print()
print("=" * 80)
print()

# Initialize scraper with hybrid mode (default)
print("🚀 Initializing Universal Scraper...")
scraper = UniversalScraper(
    model_name="gpt-4o-mini",
    enable_cache=True,
    fetch_mode="hybrid",  # Auto-detect: static → browser
    headless=True
)

# Test URL
url = "https://www.leafly.com/dispensary-info/mammoth-holistics/menu"

# Fields to extract
fields = [
    "product_name",
    "product_type",
    "price",
    "thc_content",
    "cbd_content",
    "brand",
    "strain_type"
]

print()
print("=" * 80)
print("📍 TARGET")
print("=" * 80)
print(f"URL: {url}")
print(f"Fields: {', '.join(fields)}")
print()

try:
    print("=" * 80)
    print("🕷️ SCRAPING")
    print("=" * 80)
    
    # Scrape
    result = scraper.scrape(url, fields)
    
    print()
    print("=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"Items extracted: {len(result['data'])}")
    print(f"Source: {result['source']}")
    print(f"Fetch method: {result['metadata'].get('fetch_method', 'N/A')}")
    print(f"Execution time: {result['metadata']['execution_time']:.2f}s")
    print()
    
    # Show hybrid fetcher stats if available
    if hasattr(scraper.html_fetcher, 'get_stats'):
        stats = scraper.html_fetcher.get_stats()
        print("🔀 HYBRID FETCHER STATS:")
        print(f"  API cache hits: {stats.get('api_cache_hits', 0)}")
        print(f"  Static HTML success: {stats.get('static_html_success', 0)}")
        print(f"  Browser fallback: {stats.get('browser_fallback', 0)}")
        print(f"  APIs discovered: {stats.get('apis_discovered', 0)}")
        print()
    
    # Show API cache stats if available
    if hasattr(scraper.html_fetcher, 'get_api_cache_stats'):
        api_stats = scraper.html_fetcher.get_api_cache_stats()
        if api_stats:
            print("💾 API CACHE STATS:")
            print(f"  Total domains: {api_stats.get('total_domains', 0)}")
            print(f"  Total APIs: {api_stats.get('total_apis', 0)}")
            if api_stats.get('domains'):
                print(f"  Cached domains: {', '.join(api_stats['domains'])}")
            print()
    
    if result['data']:
        print("=" * 80)
        print("📦 SAMPLE DATA (First 3 items)")
        print("=" * 80)
        for i, item in enumerate(result['data'][:3], 1):
            print(f"\n{i}. {item}")
        
        if len(result['data']) > 3:
            print(f"\n... and {len(result['data']) - 3} more items")
    else:
        print("⚠️ No data extracted!")
        print()
        print("This might mean:")
        print("1. Camoufox is not installed (run: pip install 'camoufox[geoip]')")
        print("2. The site has additional protection")
        print("3. The selectors need adjustment")

except ImportError as e:
    print()
    print("=" * 80)
    print("❌ INSTALLATION REQUIRED")
    print("=" * 80)
    print(f"Error: {e}")
    print()
    print("To enable browser support, install Camoufox:")
    print()
    print("  pip install 'camoufox[geoip]' playwright")
    print("  playwright install chromium")
    print()
    print("After installation, run this test again.")
    print()

except Exception as e:
    print()
    print("=" * 80)
    print("❌ ERROR")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    scraper.close()

print()
print("=" * 80)
print("✅ TEST COMPLETE")
print("=" * 80)
print()
print("KEY FEATURES DEMONSTRATED:")
print("  ✓ Hybrid fetching (static → browser)")
print("  ✓ JavaScript detection")
print("  ✓ API request capture")
print("  ✓ API caching for future runs")
print("  ✓ Universal architecture")
print()
print("NEXT RUN:")
print("  - If APIs were discovered, they'll be cached")
print("  - Future scrapes of this domain will be faster")
print("  - Optionally call APIs directly (bypass browser)")
print()

