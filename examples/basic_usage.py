"""
Basic Usage Example
Simple single-URL scraping with Universal Scraper
"""

import os
from universal_scraper import UniversalScraper

# Set your API key (or use environment variable)
API_KEY = os.getenv('OPENAI_API_KEY')  # or GEMINI_API_KEY, ANTHROPIC_API_KEY

def main():
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=API_KEY,
        model_name='gpt-4o-mini',  # Fast and cheap
        enable_cache=True
    )
    
    # Scrape a single URL
    result = scraper.scrape(
        url='https://books.toscrape.com/',
        fields=['title', 'price', 'rating', 'availability']
    )
    
    # Print results
    print(f"\n✅ Extracted {len(result['data'])} items")
    print(f"⏱️  Time: {result['metadata']['execution_time']:.2f}s")
    print(f"📊 Source: {result['source']}")
    
    # Display first few items
    for i, item in enumerate(result['data'][:5], 1):
        print(f"\nItem {i}:")
        for field, value in item.items():
            print(f"  {field}: {value}")
    
    # Close scraper
    scraper.close()


if __name__ == '__main__':
    main()

