"""
Test the scraper on 10 COMPLETELY NEW websites we've never tested before.
This will prove whether the system is truly universal or requires per-site refinement.
"""
import asyncio
import time
from universal_scraper import UniversalScraper

# 10 BRAND NEW websites we've never tested
TEST_SITES = [
    {
        "name": "The Guardian",
        "url": "https://www.theguardian.com/us",
        "fields": ["headline", "author"]
    },
    {
        "name": "Shopify Store (Allbirds)",
        "url": "https://www.allbirds.com/collections/mens",
        "fields": ["product_name", "price"]
    },
    {
        "name": "ArXiv (Research Papers)",
        "url": "https://arxiv.org/list/cs.AI/recent",
        "fields": ["title", "authors"]
    },
    {
        "name": "Booking.com",
        "url": "https://www.booking.com/searchresults.html?ss=New+York",
        "fields": ["hotel_name", "price"]
    },
    {
        "name": "Goodreads",
        "url": "https://www.goodreads.com/list/show/1.Best_Books_Ever",
        "fields": ["book_title", "author"]
    },
    {
        "name": "DeviantArt",
        "url": "https://www.deviantart.com/",
        "fields": ["artwork_title", "artist"]
    },
    {
        "name": "RealEstate (Zillow Listings)",
        "url": "https://www.zillow.com/homes/San-Francisco,-CA_rb/",
        "fields": ["address", "price"]
    },
    {
        "name": "Job Board (Indeed)",
        "url": "https://www.indeed.com/jobs?q=software+engineer&l=San+Francisco",
        "fields": ["job_title", "company"]
    },
    {
        "name": "LinkedIn Jobs",
        "url": "https://www.linkedin.com/jobs/search/?keywords=software%20engineer",
        "fields": ["job_title", "company"]
    },
    {
        "name": "Steam Games",
        "url": "https://store.steampowered.com/search/?sort_by=Released_DESC",
        "fields": ["game_title", "price"]
    }
]

async def test_universal_scraping():
    """Test if the system is TRULY universal."""
    
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║          UNIVERSAL SCRAPING TEST - 10 BRAND NEW WEBSITES                  ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("🎯 Goal: Extract data from sites we've NEVER tested before")
    print("🎯 Success criteria: >70% quality, >5 items per site")
    print("🎯 NO site-specific code or refinement allowed")
    print()
    
    scraper = UniversalScraper(
        api_key='$OPENAI_API_KEY',
        use_camoufox=True,  # Best anti-detection
        enable_auto_pagination=False
    )
    
    results = []
    total_start = time.time()
    
    for i, site in enumerate(TEST_SITES, 1):
        print(f"{'='*80}")
        print(f"🔍 Testing {i}/10: {site['name']}")
        print(f"{'='*80}")
        
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
            elif len(items) >= 5 and quality >= 50:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ FAIL"
            
            results.append({
                'name': site['name'],
                'items': len(items),
                'quality': quality,
                'time': elapsed,
                'status': status,
                'source': result.get('source', 'unknown')
            })
            
            print(f"📊 Results ({elapsed:.1f}s):")
            print(f"   Items: {len(items)}")
            print(f"   Quality: {quality:.0f}%")
            print(f"   Source: {result.get('source', 'unknown')}")
            if items:
                print(f"   Sample: {items[0]}")
            print(f"{status}")
            print()
            
        except Exception as e:
            elapsed = time.time() - start
            results.append({
                'name': site['name'],
                'items': 0,
                'quality': 0,
                'time': elapsed,
                'status': "❌ ERROR",
                'source': 'error'
            })
            print(f"❌ ERROR ({elapsed:.1f}s): {e}")
            print()
    
    await scraper.close()
    
    total_time = time.time() - total_start
    
    # Summary
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                        UNIVERSAL TEST RESULTS                              ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"{'Site':<25} {'Items':<8} {'Quality':<10} {'Time':<10} {'Status':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25} {r['items']:<8} {r['quality']:.0f}%{'':<7} {r['time']:.1f}s{'':<6} {r['status']:<10}")
    print("-" * 80)
    
    # Calculate success rate
    passed = sum(1 for r in results if r['status'] == "✅ PASS")
    partial = sum(1 for r in results if r['status'] == "⚠️  PARTIAL")
    failed = sum(1 for r in results if r['status'] in ["❌ FAIL", "❌ ERROR"])
    
    total_items = sum(r['items'] for r in results)
    avg_quality = sum(r['quality'] for r in results) / len(results) if results else 0
    
    print()
    print(f"✅ Passed: {passed}/10 ({passed*10}%)")
    print(f"⚠️  Partial: {partial}/10 ({partial*10}%)")
    print(f"❌ Failed: {failed}/10 ({failed*10}%)")
    print(f"📦 Total Items: {total_items}")
    print(f"📊 Avg Quality: {avg_quality:.0f}%")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    print()
    
    if passed >= 7:
        print("🎉 TRULY UNIVERSAL - System works on new sites without refinement!")
    elif passed + partial >= 7:
        print("⚠️  PARTIALLY UNIVERSAL - Some sites work, but refinement needed")
    else:
        print("❌ NOT UNIVERSAL - System requires per-site refinement")
        print()
        print("💡 Recommendation: Need a fundamentally different approach")

if __name__ == "__main__":
    asyncio.run(test_universal_scraping())





