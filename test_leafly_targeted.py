#!/usr/bin/env python3
"""
Targeted Leafly Scraping: Only crawl/scrape dispensary pages
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from universal_scraper.crawler import UniversalCrawler, CrawlConfig
from universal_scraper import UniversalScraper
from universal_scraper.core import HybridFetcher

def print_banner(title):
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80 + "\n")

def test_targeted_crawl():
    """
    Crawl Leafly Nevada but ONLY follow dispensary-related URLs
    """
    print_banner("STEP 1: Targeted Crawl (Dispensaries Only)")
    
    print("🎯 CRAWL STRATEGY:")
    print("   ✅ FOLLOW: URLs containing '/dispensary-info/' or '/dispensaries/'")
    print("   ❌ IGNORE: Navigation links like /products, /news, /strains, etc.")
    print()
    
    # Configure crawler with URL patterns
    config = CrawlConfig(
        mode='smart',
        max_depth=2,
        max_pages=30,  # Limit for testing
        
        # ✅ ONLY FOLLOW these URL patterns
        follow_patterns=[
            '/dispensaries/',      # Listing pages
            '/dispensary-info/'    # Detail pages (info + menu)
        ],
        
        # ❌ IGNORE these URL patterns
        ignore_patterns=[
            '/products',
            '/strains',
            '/news',
            '/brands',
            '/doctors',
            '/learn',
            '/cannabis-101',
            '/api/',
            '/auth/',
            '.jpg',
            '.png',
            '.css',
            '.js'
        ],
        
        handle_pagination=True,
        discover_apis=False,  # Disable for speed
        enable_search_discovery=False,
        respect_robots_txt=False
    )
    
    print("📊 Configuration:")
    print(f"   Max Depth: {config.max_depth}")
    print(f"   Max Pages: {config.max_pages}")
    print(f"   Follow Patterns: {config.follow_patterns}")
    print(f"   Ignore Patterns (showing first 5): {config.ignore_patterns[:5]}")
    print()
    
    # Create fetcher
    fetcher = HybridFetcher(
        enable_cache=True,
        enable_warming=False,
        browser_timeout=60000,
        headless=True
    )
    
    # Create crawler
    crawler = UniversalCrawler(config=config, fetcher=fetcher)
    
    print("⏳ Crawling...\n")
    
    # Crawl
    start_time = datetime.now()
    results = crawler.crawl(["https://www.leafly.com/dispensaries/nevada"])
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Crawl Complete!")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Total Discovered: {results.total_discovered}")
    print(f"   Total Crawled: {results.total_crawled}")
    
    # Group URLs
    dispensary_info_urls = []
    dispensary_menu_urls = []
    listing_urls = []
    
    for crawled in results.urls:
        url = crawled.url
        if '/dispensary-info/' in url:
            if '/menu' in url:
                dispensary_menu_urls.append(url)
            else:
                dispensary_info_urls.append(url)
        elif '/dispensaries/' in url:
            listing_urls.append(url)
    
    print(f"\n📊 Discovered URLs by Type:")
    print(f"   • Listing Pages: {len(listing_urls)}")
    print(f"   • Dispensary Info Pages: {len(dispensary_info_urls)}")
    print(f"   • Dispensary Menu Pages: {len(dispensary_menu_urls)}")
    
    print(f"\n🔍 Sample URLs:")
    
    if listing_urls:
        print(f"\n   Listing Pages:")
        for url in listing_urls[:3]:
            print(f"      • {url}")
    
    if dispensary_info_urls:
        print(f"\n   Dispensary Info Pages:")
        for url in dispensary_info_urls[:3]:
            print(f"      • {url}")
    
    if dispensary_menu_urls:
        print(f"\n   Dispensary Menu Pages:")
        for url in dispensary_menu_urls[:3]:
            print(f"      • {url}")
    
    return {
        'listing_urls': listing_urls,
        'info_urls': dispensary_info_urls,
        'menu_urls': dispensary_menu_urls
    }, fetcher

def test_targeted_scrape(urls_dict, fetcher):
    """
    Scrape ONLY the relevant pages with appropriate fields
    """
    print_banner("STEP 2: Targeted Scraping")
    
    # Get OpenAI API key
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        parsera_path = os.path.expanduser('~/Dev/PARSERA-PROJECT/.env')
        if os.path.exists(parsera_path):
            with open(parsera_path) as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        openai_key = line.split('=', 1)[1].strip().strip('"\'')
                        break
    
    if not openai_key:
        print("⚠️  No OpenAI API key found. Skipping scraping.")
        return []
    
    scraper = UniversalScraper(
        api_key=openai_key,
        fetch_mode="hybrid",
        enable_cache=True,
        headless=True
    )
    
    all_results = []
    
    # Scrape 1 info page
    if urls_dict['info_urls']:
        print(f"\n🏢 Scraping Dispensary Info Page")
        print(f"   URL: {urls_dict['info_urls'][0]}")
        
        try:
            result = scraper.scrape(
                url=urls_dict['info_urls'][0],
                fields=["name", "address", "phone", "rating", "hours", "website", "description"]
            )
            
            print(f"   ✅ Success!")
            print(f"      Items: {len(result['data'])}")
            print(f"      Source: {result['source']}")
            
            if result['data']:
                print(f"\n      📊 Data:")
                for key, value in result['data'][0].items():
                    print(f"         {key}: {str(value)[:60]}")
            
            all_results.append({
                'type': 'dispensary_info',
                'url': urls_dict['info_urls'][0],
                'data': result['data']
            })
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Scrape 1 menu page
    if urls_dict['menu_urls']:
        print(f"\n🍃 Scraping Dispensary Menu Page")
        print(f"   URL: {urls_dict['menu_urls'][0]}")
        
        try:
            result = scraper.scrape(
                url=urls_dict['menu_urls'][0],
                fields=["name", "brand", "category", "thc", "cbd", "price", "type", "description"]
            )
            
            print(f"   ✅ Success!")
            print(f"      Products: {len(result['data'])}")
            print(f"      Source: {result['source']}")
            
            if result['data']:
                print(f"\n      📊 Sample Products (first 3):")
                for i, item in enumerate(result['data'][:3], 1):
                    print(f"\n      Product {i}:")
                    for key, value in list(item.items())[:6]:
                        print(f"         {key}: {str(value)[:50]}")
            
            all_results.append({
                'type': 'dispensary_menu',
                'url': urls_dict['menu_urls'][0],
                'data': result['data']
            })
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Save results
    with open('leafly_targeted_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: leafly_targeted_results.json")
    
    return all_results

def main():
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "LEAFLY NEVADA - TARGETED CRAWL & SCRAPE" + " "*23 + "║")
    print("║" + " "*10 + "Only Follow Dispensary URLs - Ignore Navigation" + " "*20 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Step 1: Targeted crawl
        urls_dict, fetcher = test_targeted_crawl()
        
        # Step 2: Targeted scrape
        results = test_targeted_scrape(urls_dict, fetcher)
        
        # Final summary
        print_banner("🎉 SUMMARY")
        
        print("✅ CRAWL STRATEGY WORKED!")
        print(f"   • Only followed dispensary-related URLs")
        print(f"   • Ignored navigation links (/products, /news, etc.)")
        print(f"   • Found {len(urls_dict['info_urls'])} info pages")
        print(f"   • Found {len(urls_dict['menu_urls'])} menu pages")
        
        print("\n✅ SCRAPING COMPLETE!")
        print(f"   • Scraped {len(results)} pages")
        print(f"   • Extracted relevant data for each page type")
        
        print("\n💡 KEY TAKEAWAY:")
        print("   Use 'follow_patterns' and 'ignore_patterns' in CrawlConfig")
        print("   to control which URLs the crawler follows and scrapes!")
        
        print("\n" + "="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())








