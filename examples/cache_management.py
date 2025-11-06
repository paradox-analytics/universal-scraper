"""
Cache Management Example
Managing the code cache for optimal performance
"""

import os
from universal_scraper import UniversalScraper

API_KEY = os.getenv('OPENAI_API_KEY')

def main():
    scraper = UniversalScraper(
        api_key=API_KEY,
        cache_dir='./my_cache',
        cache_ttl=3600,  # 1 hour TTL
        enable_cache=True
    )
    
    # First scrape - generates code
    print("🔄 First scrape (will generate code)...")
    result1 = scraper.scrape(
        url='https://books.toscrape.com/',
        fields=['title', 'price']
    )
    print(f"   Cached: {result1['metadata']['code_cached']}")
    
    # Second scrape - uses cache
    print("\n🔄 Second scrape (should use cache)...")
    result2 = scraper.scrape(
        url='https://books.toscrape.com/catalogue/page-2.html',
        fields=['title', 'price']
    )
    print(f"   Cached: {result2['metadata']['code_cached']}")
    
    # Get cache statistics
    cache_stats = scraper.get_cache_stats()
    print(f"\n💾 Cache Statistics:")
    print(f"   Enabled: {cache_stats['enabled']}")
    print(f"   Entries: {cache_stats['size']}")
    print(f"   Directory: {cache_stats['directory']}")
    
    # Export cache (for sharing or backup)
    print("\n📤 Exporting cache...")
    scraper.export_cache('cache_backup.json')
    print("   ✅ Cache exported to cache_backup.json")
    
    # Clear cache if needed
    print("\n🗑️  Clearing cache...")
    scraper.clear_cache()
    print("   ✅ Cache cleared")
    
    # Import cache back
    print("\n📥 Importing cache...")
    scraper.import_cache('cache_backup.json')
    print("   ✅ Cache imported")
    
    scraper.close()


if __name__ == '__main__':
    main()

