"""
Proxy Usage Example
Using residential proxies for better reliability
"""

import os
from universal_scraper import UniversalScraper

API_KEY = os.getenv('OPENAI_API_KEY')

# Proxy configuration (example with BrightData)
PROXY_CONFIG = {
    'server': 'http://brd.superproxy.io:22225',
    'username': 'your-username-zone-residential',
    'password': 'your-password'
}

def main():
    # Initialize scraper with proxy
    scraper = UniversalScraper(
        api_key=API_KEY,
        model_name='gpt-4o-mini',
        proxy_config=PROXY_CONFIG,  # Enable proxy
        enable_warming=True  # Warm session for better success
    )
    
    # Scrape URL (will use proxy)
    result = scraper.scrape(
        url='https://example.com/products',
        fields=['product_name', 'price', 'rating']
    )
    
    print(f"\n✅ Extracted {len(result['data'])} items with proxy")
    print(f"⏱️  Time: {result['metadata']['execution_time']:.2f}s")
    
    # Display results
    for i, item in enumerate(result['data'][:3], 1):
        print(f"\nItem {i}: {item}")
    
    scraper.close()


if __name__ == '__main__':
    # Note: Replace PROXY_CONFIG with your actual proxy credentials
    print("⚠️  Update PROXY_CONFIG with your proxy credentials before running")
    # main()

