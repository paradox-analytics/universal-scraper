#!/usr/bin/env python3
"""
Simple local test - directly test the core functionality
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_crawler():
    """Test the crawler directly"""
    print("🧪 Testing Universal Crawler Locally")
    print("=" * 60)
    
    from universal_scraper.crawler import UniversalCrawler, CrawlConfig
    from universal_scraper.core import HybridFetcher
    
    # Create configuration
    config = CrawlConfig(
        max_depth=1,
        max_pages=3,
        follow_patterns=["/dispensary-info/"],
        ignore_patterns=["/products", "/strains", "/news", "?filter"],
        handle_pagination=False,
        discover_apis=False
    )
    
    # Create fetcher and crawler
    fetcher = HybridFetcher(
        headless=True,
        browser_timeout=30000
    )
    
    crawler = UniversalCrawler(
        config=config,
        fetcher=fetcher
    )
    
    # Test URL
    start_url = "https://www.leafly.com/dispensaries/nevada"
    
    print(f"\n🌐 Crawling: {start_url}")
    print(f"📊 Max Pages: {config.max_pages}")
    print(f"🔍 Follow Patterns: {config.follow_patterns}")
    print(f"🚫 Ignore Patterns: {config.ignore_patterns}")
    print()
    
    try:
        # Run the crawl
        results = await crawler.crawl([start_url])
        
        # Print results
        print(f"\n✅ Crawl complete!")
        print(f"📊 Total URLs discovered: {results.total_discovered}")
        print(f"🕷️  URLs crawled: {results.total_crawled}")
        print(f"⏱️  Duration: {results.duration:.2f}s")
        
        if results.urls:
            print(f"\n📋 Discovered URLs:")
            for i, url_data in enumerate(results.urls[:10], 1):
                url = url_data if isinstance(url_data, str) else url_data.get('url', url_data)
                print(f"  {i}. {url}")
            
            if len(results.urls) > 10:
                print(f"  ... and {len(results.urls) - 10} more")
        
        # Close the fetcher
        await fetcher.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await fetcher.close()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_crawler())








