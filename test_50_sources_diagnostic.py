"""
Aggressive A/B Test: 50 Diverse Sources
Goal: Identify failure patterns and categorize edge cases
"""
import asyncio
import time
import json
from universal_scraper import UniversalScraper

# 50 DIVERSE sources across different categories
TEST_SOURCES = [
    # === NEWS (10) ===
    {"name": "BBC News", "url": "https://www.bbc.com/news", "fields": ["headline", "summary"], "category": "news"},
    {"name": "The Guardian", "url": "https://www.theguardian.com/us", "fields": ["headline", "author"], "category": "news"},
    {"name": "CNN", "url": "https://www.cnn.com/", "fields": ["headline", "category"], "category": "news"},
    {"name": "NPR", "url": "https://www.npr.org/sections/news/", "fields": ["headline", "description"], "category": "news"},
    {"name": "Reuters", "url": "https://www.reuters.com/", "fields": ["headline", "time"], "category": "news"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/", "fields": ["headline", "section"], "category": "news"},
    {"name": "WSJ", "url": "https://www.wsj.com/", "fields": ["headline", "author"], "category": "news"},
    {"name": "NYTimes", "url": "https://www.nytimes.com/", "fields": ["headline", "section"], "category": "news"},
    {"name": "Washington Post", "url": "https://www.washingtonpost.com/", "fields": ["headline", "author"], "category": "news"},
    {"name": "Bloomberg", "url": "https://www.bloomberg.com/", "fields": ["headline", "time"], "category": "news"},
    
    # === E-COMMERCE (10) ===
    {"name": "Amazon", "url": "https://www.amazon.com/s?k=laptop", "fields": ["product_name", "price"], "category": "ecommerce"},
    {"name": "eBay", "url": "https://www.ebay.com/sch/i.html?_nkw=laptop", "fields": ["product_name", "price"], "category": "ecommerce"},
    {"name": "Etsy", "url": "https://www.etsy.com/search?q=art", "fields": ["product_name", "price"], "category": "ecommerce"},
    {"name": "Walmart", "url": "https://www.walmart.com/search?q=laptop", "fields": ["product_name", "price"], "category": "ecommerce"},
    {"name": "Target", "url": "https://www.target.com/s?searchTerm=laptop", "fields": ["product_name", "price"], "category": "ecommerce"},
    {"name": "Best Buy", "url": "https://www.bestbuy.com/site/searchpage.jsp?st=laptop", "fields": ["product_name", "price"], "category": "ecommerce"},
    {"name": "Newegg", "url": "https://www.newegg.com/p/pl?d=laptop", "fields": ["product_name", "price"], "category": "ecommerce"},
    {"name": "Shopify (Allbirds)", "url": "https://www.allbirds.com/collections/mens", "fields": ["product_name", "price"], "category": "ecommerce"},
    {"name": "AliExpress", "url": "https://www.aliexpress.com/wholesale?SearchText=laptop", "fields": ["product_name", "price"], "category": "ecommerce"},
    {"name": "Wayfair", "url": "https://www.wayfair.com/furniture/sb0/sofas-c414059.html", "fields": ["product_name", "price"], "category": "ecommerce"},
    
    # === SOCIAL/FORUMS (10) ===
    {"name": "Reddit", "url": "https://www.reddit.com/r/programming/", "fields": ["title", "author"], "category": "social"},
    {"name": "Hacker News", "url": "https://news.ycombinator.com/", "fields": ["title", "points"], "category": "social"},
    {"name": "Stack Overflow", "url": "https://stackoverflow.com/questions", "fields": ["title", "votes"], "category": "social"},
    {"name": "Twitter/X", "url": "https://twitter.com/search?q=AI", "fields": ["tweet_text", "author"], "category": "social"},
    {"name": "LinkedIn Posts", "url": "https://www.linkedin.com/feed/", "fields": ["post_text", "author"], "category": "social"},
    {"name": "Quora", "url": "https://www.quora.com/", "fields": ["question", "answers"], "category": "social"},
    {"name": "Medium", "url": "https://medium.com/tag/technology", "fields": ["title", "author"], "category": "social"},
    {"name": "Dev.to", "url": "https://dev.to/", "fields": ["title", "author"], "category": "social"},
    {"name": "Product Hunt", "url": "https://www.producthunt.com/", "fields": ["product_name", "tagline"], "category": "social"},
    {"name": "Lobsters", "url": "https://lobste.rs/", "fields": ["title", "points"], "category": "social"},
    
    # === JOBS (5) ===
    {"name": "Indeed", "url": "https://www.indeed.com/jobs?q=software+engineer", "fields": ["job_title", "company"], "category": "jobs"},
    {"name": "LinkedIn Jobs", "url": "https://www.linkedin.com/jobs/search/?keywords=software%20engineer", "fields": ["job_title", "company"], "category": "jobs"},
    {"name": "Glassdoor", "url": "https://www.glassdoor.com/Job/jobs.htm?sc.keyword=software+engineer", "fields": ["job_title", "company"], "category": "jobs"},
    {"name": "AngelList", "url": "https://wellfound.com/jobs", "fields": ["job_title", "company"], "category": "jobs"},
    {"name": "RemoteOK", "url": "https://remoteok.com/", "fields": ["job_title", "company"], "category": "jobs"},
    
    # === REAL ESTATE (3) ===
    {"name": "Zillow", "url": "https://www.zillow.com/homes/San-Francisco,-CA_rb/", "fields": ["address", "price"], "category": "real_estate"},
    {"name": "Redfin", "url": "https://www.redfin.com/city/17151/CA/San-Francisco", "fields": ["address", "price"], "category": "real_estate"},
    {"name": "Realtor.com", "url": "https://www.realtor.com/realestateandhomes-search/San-Francisco_CA", "fields": ["address", "price"], "category": "real_estate"},
    
    # === TRAVEL (3) ===
    {"name": "Booking.com", "url": "https://www.booking.com/searchresults.html?ss=New+York", "fields": ["hotel_name", "price"], "category": "travel"},
    {"name": "Airbnb", "url": "https://www.airbnb.com/s/New-York/homes", "fields": ["listing_name", "price"], "category": "travel"},
    {"name": "TripAdvisor", "url": "https://www.tripadvisor.com/Hotels-g60763-New_York_City_New_York-Hotels.html", "fields": ["hotel_name", "rating"], "category": "travel"},
    
    # === MEDIA (4) ===
    {"name": "YouTube Trending", "url": "https://www.youtube.com/feed/trending", "fields": ["video_title", "views"], "category": "media"},
    {"name": "Spotify", "url": "https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF", "fields": ["song_title", "artist"], "category": "media"},
    {"name": "IMDb", "url": "https://www.imdb.com/chart/top/", "fields": ["movie_title", "rating"], "category": "media"},
    {"name": "Rotten Tomatoes", "url": "https://www.rottentomatoes.com/browse/movies_in_theaters/", "fields": ["movie_title", "score"], "category": "media"},
    
    # === TECH (5) ===
    {"name": "GitHub Trending", "url": "https://github.com/trending", "fields": ["repository", "stars"], "category": "tech"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/", "fields": ["title", "author"], "category": "tech"},
    {"name": "Ars Technica", "url": "https://arstechnica.com/", "fields": ["title", "author"], "category": "tech"},
    {"name": "The Verge", "url": "https://www.theverge.com/", "fields": ["title", "author"], "category": "tech"},
    {"name": "Wired", "url": "https://www.wired.com/", "fields": ["title", "author"], "category": "tech"},
]

# Failure categorization
FAILURE_CATEGORIES = {
    "anti_blocking": "Bot detection, CAPTCHA, or access denied",
    "html_structure": "DOM detection failed or wrong elements selected",
    "json_extraction": "JSON found but quality check failed",
    "custom_elements": "Custom web components (shadow DOM, React, etc)",
    "authentication": "Login/auth required",
    "empty_page": "Page loaded but no content found",
    "timeout": "Page took too long to load",
    "code_generation": "LLM failed to generate working code",
    "unknown": "Unknown failure"
}

async def categorize_failure(site, result, error):
    """Categorize the failure type based on logs and errors"""
    categories = []
    
    # Check error messages
    error_str = str(error).lower() if error else ""
    
    if "access denied" in error_str or "captcha" in error_str or "blocked" in error_str:
        categories.append("anti_blocking")
    elif "timeout" in error_str or "timed out" in error_str:
        categories.append("timeout")
    elif "authentication" in error_str or "login required" in error_str:
        categories.append("authentication")
    elif "code generation failed" in error_str:
        categories.append("code_generation")
    elif result.get('source') == 'json' and len(result.get('data', [])) == 0:
        categories.append("json_extraction")
    elif len(result.get('data', [])) == 0:
        # Check if it's HTML structure or empty page
        if "no pattern" in error_str or "dom" in error_str:
            categories.append("html_structure")
        else:
            categories.append("empty_page")
    
    if not categories:
        categories.append("unknown")
    
    return categories

async def test_50_sources():
    """Run aggressive A/B test on 50 diverse sources"""
    
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║              AGGRESSIVE A/B TEST: 50 DIVERSE SOURCES                      ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("🎯 Goal: Test universality and categorize failure patterns")
    print("🎯 One page per source, no pagination")
    print("🎯 Success criteria: >5 items, >70% quality")
    print()
    
    scraper = UniversalScraper(
        api_key='$OPENAI_API_KEY',
        use_camoufox=True,
        enable_auto_pagination=False
    )
    
    results = []
    failure_breakdown = {cat: [] for cat in FAILURE_CATEGORIES.keys()}
    category_stats = {}
    
    total_start = time.time()
    
    for i, site in enumerate(TEST_SOURCES, 1):
        print(f"[{i}/50] Testing {site['name']} ({site['category']})...", end=" ", flush=True)
        
        start = time.time()
        try:
            result = await scraper.scrape(
                url=site['url'],
                fields=site['fields']
            )
            
            elapsed = time.time() - start
            items = result.get('data', [])
            
            # Calculate quality
            if items:
                total_fields = len(items) * len(site['fields'])
                filled_fields = sum(
                    1 for item in items 
                    for v in item.values() 
                    if v is not None and v != ''
                )
                quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0
            else:
                quality = 0
            
            # Determine status
            if len(items) >= 5 and quality >= 70:
                status = "✅ PASS"
                failure_cats = []
            elif len(items) >= 5 and quality >= 50:
                status = "⚠️  PARTIAL"
                failure_cats = ["low_quality"]
            else:
                status = "❌ FAIL"
                failure_cats = await categorize_failure(site, result, None)
            
            result_data = {
                'name': site['name'],
                'category': site['category'],
                'url': site['url'],
                'fields': site['fields'],
                'items': len(items),
                'quality': quality,
                'time': elapsed,
                'status': status,
                'source': result.get('source', 'unknown'),
                'failure_categories': failure_cats,
                'sample_item': items[0] if items else None
            }
            
            results.append(result_data)
            
            # Track failures
            for cat in failure_cats:
                failure_breakdown[cat].append(site['name'])
            
            # Track category stats
            if site['category'] not in category_stats:
                category_stats[site['category']] = {'pass': 0, 'partial': 0, 'fail': 0}
            
            if status == "✅ PASS":
                category_stats[site['category']]['pass'] += 1
            elif status == "⚠️  PARTIAL":
                category_stats[site['category']]['partial'] += 1
            else:
                category_stats[site['category']]['fail'] += 1
            
            print(f"{status} ({len(items)} items, {quality:.0f}%, {elapsed:.1f}s)")
            
        except Exception as e:
            elapsed = time.time() - start
            failure_cats = await categorize_failure(site, {}, e)
            
            result_data = {
                'name': site['name'],
                'category': site['category'],
                'url': site['url'],
                'fields': site['fields'],
                'items': 0,
                'quality': 0,
                'time': elapsed,
                'status': "❌ ERROR",
                'source': 'error',
                'failure_categories': failure_cats,
                'error': str(e)[:100],
                'sample_item': None
            }
            
            results.append(result_data)
            
            # Track failures
            for cat in failure_cats:
                failure_breakdown[cat].append(site['name'])
            
            # Track category stats
            if site['category'] not in category_stats:
                category_stats[site['category']] = {'pass': 0, 'partial': 0, 'fail': 0}
            category_stats[site['category']]['fail'] += 1
            
            print(f"❌ ERROR ({elapsed:.1f}s): {str(e)[:50]}")
    
    await scraper.close()
    
    total_time = time.time() - total_start
    
    # === SUMMARY ===
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                          TEST SUMMARY                                      ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    
    # Overall stats
    passed = sum(1 for r in results if r['status'] == "✅ PASS")
    partial = sum(1 for r in results if r['status'] == "⚠️  PARTIAL")
    failed = sum(1 for r in results if r['status'] in ["❌ FAIL", "❌ ERROR"])
    
    print(f"\n📊 Overall Results:")
    print(f"   ✅ Passed: {passed}/50 ({passed*2}%)")
    print(f"   ⚠️  Partial: {partial}/50 ({partial*2}%)")
    print(f"   ❌ Failed: {failed}/50 ({failed*2}%)")
    print(f"   ⏱️  Total Time: {total_time:.1f}s ({total_time/50:.1f}s avg)")
    
    # Category breakdown
    print(f"\n📂 Results by Category:")
    for cat, stats in sorted(category_stats.items()):
        total = stats['pass'] + stats['partial'] + stats['fail']
        print(f"   {cat.upper()}: {stats['pass']}/{total} pass, {stats['partial']}/{total} partial, {stats['fail']}/{total} fail")
    
    # Failure analysis
    print(f"\n🔍 Failure Breakdown:")
    for cat, sites in sorted(failure_breakdown.items(), key=lambda x: len(x[1]), reverse=True):
        if sites:
            print(f"   {cat.upper()}: {len(sites)} sites")
            print(f"      {', '.join(sites[:5])}{' ...' if len(sites) > 5 else ''}")
    
    # Save detailed results
    with open('test_50_sources_results.json', 'w') as f:
        json.dump({
            'summary': {
                'passed': passed,
                'partial': partial,
                'failed': failed,
                'total_time': total_time
            },
            'category_stats': category_stats,
            'failure_breakdown': {k: v for k, v in failure_breakdown.items() if v},
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: test_50_sources_results.json")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if failure_breakdown['anti_blocking']:
        print(f"   1. Anti-blocking is a major issue ({len(failure_breakdown['anti_blocking'])} sites)")
        print(f"      → Need better proxy rotation + Camoufox enhancements")
    
    if failure_breakdown['code_generation']:
        print(f"   2. Code generation failing ({len(failure_breakdown['code_generation'])} sites)")
        print(f"      → LLM prompts need refinement OR switch to direct extraction")
    
    if failure_breakdown['html_structure']:
        print(f"   3. HTML structure detection failing ({len(failure_breakdown['html_structure'])} sites)")
        print(f"      → DOM pattern detector needs improvement")
    
    if failure_breakdown['json_extraction']:
        print(f"   4. JSON quality issues ({len(failure_breakdown['json_extraction'])} sites)")
        print(f"      → JSON validator too strict OR wrong JSON sources selected")
    
    if passed < 35:  # <70% success
        print(f"\n   ⚠️  CRITICAL: System is NOT universal (<70% success rate)")
        print(f"   → Consider: LLM-per-request OR data warehouse for learned patterns")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_50_sources())





