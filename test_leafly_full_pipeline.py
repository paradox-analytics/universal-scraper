#!/usr/bin/env python3
"""
Full Pipeline Test: Crawl Leafly Nevada → Scrape Discovered URLs
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from universal_scraper.crawler import UniversalCrawler, CrawlConfig
from universal_scraper import UniversalScraper
from universal_scraper.core import HybridFetcher

def print_banner(title):
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80 + "\n")

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved to: {filename}")

def test_crawl_leafly_nevada():
    """
    Step 1: Crawl Leafly Nevada to discover all URLs
    """
    print_banner("STEP 1: Crawl Leafly Nevada")
    
    print("🌐 Starting URL: https://www.leafly.com/dispensaries/nevada")
    print("📊 Configuration:")
    print("   - Max Depth: 2")
    print("   - Max Pages: 25 (limiting for testing)")
    print("   - Handle Pagination: Yes")
    print("   - Discover APIs: Yes")
    print("\n⏳ Crawling... (this may take a moment)\n")
    
    # Configure crawler
    config = CrawlConfig(
        mode='smart',
        max_depth=2,
        max_pages=25,  # Limit for testing
        handle_pagination=True,
        discover_apis=True,
        enable_search_discovery=False,
        respect_robots_txt=False
    )
    
    # Create fetcher with longer timeout for JS-heavy sites
    fetcher = HybridFetcher(
        enable_cache=True,
        enable_warming=False,
        browser_timeout=60000,  # 60 seconds
        headless=True
    )
    
    # Create crawler
    crawler = UniversalCrawler(config=config, fetcher=fetcher)
    
    # Crawl
    start_time = datetime.now()
    results = crawler.crawl(["https://www.leafly.com/dispensaries/nevada"])
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Display results
    print(f"\n✅ Crawl Complete!")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Total Discovered: {results.total_discovered}")
    print(f"   Total Crawled: {results.total_crawled}")
    
    # Group URLs by type
    urls_by_type = {}
    for crawled in results.urls:
        page_type = str(crawled.page_type) if crawled.page_type else 'unknown'
        if page_type not in urls_by_type:
            urls_by_type[page_type] = []
        urls_by_type[page_type].append({
            'url': crawled.url,
            'depth': crawled.depth,
            'discovered_via': crawled.discovered_via
        })
    
    print(f"\n📊 URLs by Type:")
    for page_type, urls in urls_by_type.items():
        print(f"\n   {page_type.upper()} ({len(urls)} URLs):")
        for i, url_info in enumerate(urls[:5], 1):
            print(f"      {i}. {url_info['url']}")
            print(f"         Depth: {url_info['depth']}, Via: {url_info['discovered_via']}")
        if len(urls) > 5:
            print(f"      ... and {len(urls) - 5} more")
    
    # Save full results
    crawl_data = {
        'start_url': 'https://www.leafly.com/dispensaries/nevada',
        'duration_seconds': duration,
        'total_discovered': results.total_discovered,
        'total_crawled': results.total_crawled,
        'urls_by_type': urls_by_type
    }
    save_json(crawl_data, 'leafly_nevada_crawl_results.json')
    
    # Return URLs for scraping
    all_urls = []
    for crawled in results.urls:
        all_urls.append(crawled.url)
    
    return all_urls, fetcher

def test_scrape_urls(urls, fetcher):
    """
    Step 2: Scrape the first few discovered URLs
    """
    print_banner("STEP 2: Scrape Discovered URLs")
    
    # Get OpenAI API key
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        # Try to get from PARSERA-PROJECT
        parsera_path = os.path.expanduser('~/Dev/PARSERA-PROJECT/.env')
        if os.path.exists(parsera_path):
            with open(parsera_path) as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        openai_key = line.split('=', 1)[1].strip().strip('"\'')
                        break
    
    if not openai_key:
        print("⚠️  No OpenAI API key found. Skipping scraping.")
        print("   Set OPENAI_API_KEY environment variable to enable scraping.")
        return []
    
    # Initialize scraper
    scraper = UniversalScraper(
        api_key=openai_key,
        fetch_mode="hybrid",
        enable_cache=True,
        headless=True
    )
    
    # Scrape first 3 URLs
    urls_to_scrape = urls[:3]
    
    print(f"📋 Scraping {len(urls_to_scrape)} URLs:")
    for i, url in enumerate(urls_to_scrape, 1):
        print(f"   {i}. {url}")
    
    scraped_results = []
    
    for i, url in enumerate(urls_to_scrape, 1):
        print(f"\n{'─'*80}")
        print(f"🔍 Scraping URL {i}/{len(urls_to_scrape)}")
        print(f"   {url}")
        print(f"{'─'*80}")
        
        try:
            # Determine fields based on URL type
            if '/dispensary-info/' in url:
                if '/menu' in url:
                    # Menu page - extract products
                    fields = ["name", "brand", "category", "thc", "cbd", "price", "type"]
                    print("   Type: Dispensary Menu Page")
                else:
                    # Info page - extract dispensary info
                    fields = ["name", "address", "phone", "rating", "hours", "website"]
                    print("   Type: Dispensary Info Page")
            elif '/dispensaries/' in url:
                # Listing page - extract dispensary listings
                fields = ["name", "address", "city", "rating", "distance"]
                print("   Type: Dispensary Listing Page")
            else:
                # Generic
                fields = ["name", "title", "description", "content"]
                print("   Type: Generic Page")
            
            print(f"   Fields: {', '.join(fields)}")
            print("\n   ⏳ Scraping...")
            
            result = scraper.scrape(url=url, fields=fields)
            
            print(f"\n   ✅ Success!")
            print(f"      Items Extracted: {len(result['data'])}")
            print(f"      Execution Time: {result['metadata']['execution_time']:.2f}s")
            print(f"      Source: {result['source']}")
            
            if result['data']:
                print(f"\n   📊 Sample Data (first 2 items):")
                for j, item in enumerate(result['data'][:2], 1):
                    print(f"\n      Item {j}:")
                    for key, value in list(item.items())[:5]:  # Show first 5 fields
                        value_str = str(value)[:60]  # Truncate long values
                        print(f"         {key}: {value_str}")
                    if len(item) > 5:
                        print(f"         ... and {len(item) - 5} more fields")
                
                if len(result['data']) > 2:
                    print(f"\n      ... and {len(result['data']) - 2} more items")
            
            scraped_results.append({
                'url': url,
                'success': True,
                'items_count': len(result['data']),
                'execution_time': result['metadata']['execution_time'],
                'source': result['source'],
                'data': result['data']
            })
            
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            scraped_results.append({
                'url': url,
                'success': False,
                'error': str(e)
            })
    
    # Save scraping results
    print(f"\n{'='*80}")
    save_json(scraped_results, 'leafly_nevada_scrape_results.json')
    
    return scraped_results

def main():
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "LEAFLY NEVADA - FULL PIPELINE TEST" + " "*24 + "║")
    print("║" + " "*18 + "Crawl → Discover URLs → Scrape Data" + " "*24 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Step 1: Crawl
        urls, fetcher = test_crawl_leafly_nevada()
        
        if not urls:
            print("\n⚠️  No URLs discovered. Cannot proceed to scraping.")
            return 1
        
        # Step 2: Scrape
        scraped_results = test_scrape_urls(urls, fetcher)
        
        # Final summary
        print_banner("🎉 FINAL SUMMARY")
        
        print("✅ CRAWL RESULTS:")
        print(f"   Total URLs Discovered: {len(urls)}")
        print(f"   Results saved to: leafly_nevada_crawl_results.json")
        
        print("\n✅ SCRAPE RESULTS:")
        successful = sum(1 for r in scraped_results if r.get('success'))
        print(f"   URLs Scraped: {len(scraped_results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(scraped_results) - successful}")
        
        total_items = sum(r.get('items_count', 0) for r in scraped_results)
        print(f"   Total Items Extracted: {total_items}")
        print(f"   Results saved to: leafly_nevada_scrape_results.json")
        
        print("\n" + "="*80)
        print("✅ FULL PIPELINE TEST COMPLETE!")
        print("="*80)
        
        print("\n📂 Output Files:")
        print("   • leafly_nevada_crawl_results.json   - All discovered URLs")
        print("   • leafly_nevada_scrape_results.json  - Scraped data from first 3 URLs")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

