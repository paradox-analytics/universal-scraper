"""
Batch Scraping Example
Scraping multiple URLs efficiently with caching
"""

import os
import json
from universal_scraper import UniversalScraper

API_KEY = os.getenv('OPENAI_API_KEY')

def main():
    # URLs to scrape (similar structure = code reuse!)
    urls = [
        'https://books.toscrape.com/catalogue/category/books/mystery_3/index.html',
        'https://books.toscrape.com/catalogue/category/books/science-fiction_16/index.html',
        'https://books.toscrape.com/catalogue/category/books/fantasy_19/index.html',
    ]
    
    # Fields to extract
    fields = ['title', 'price', 'rating', 'availability']
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=API_KEY,
        model_name='gpt-4o-mini',
        enable_cache=True  # Important for batch jobs!
    )
    
    # Scrape all URLs
    results = scraper.scrape_multiple(urls, fields)
    
    # Analyze results
    total_items = sum(len(r['data']) for r in results)
    cache_hits = sum(1 for r in results if r['metadata'].get('code_cached'))
    
    print(f"\n📊 Batch Scraping Results:")
    print(f"   URLs processed: {len(urls)}")
    print(f"   Total items: {total_items}")
    print(f"   Cache hits: {cache_hits}/{len(urls)}")
    
    # Show cache savings
    cache_stats = scraper.get_cache_stats()
    print(f"\n💾 Cache Stats:")
    print(f"   Entries: {cache_stats.get('size', 0)}")
    
    # Save all results to JSON
    with open('batch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to batch_results.json")
    
    scraper.close()


if __name__ == '__main__':
    main()

