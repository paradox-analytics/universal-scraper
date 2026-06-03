"""
Generate CSV samples - SINGLE PAGE ONLY (no pagination)
"""
import asyncio
import csv
import os
from universal_scraper.core.scraper import UniversalScraper
from bs4 import BeautifulSoup

async def save_to_csv(data, filename, fields=None):
    """Save extracted data to CSV"""
    if not data:
        print(f"   ⚠️  No data to save for {filename}")
        return
    
    # Get all unique keys if fields not specified
    if not fields:
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        fields = sorted(list(all_keys))[:10]  # Limit to 10 fields for readability
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        
        for item in data[:100]:  # Limit to 100 rows
            if isinstance(item, dict):
                row = {k: str(v)[:200] if v else '' for k, v in item.items() if k in fields}
                writer.writerow(row)
    
    print(f"   ✅ Saved {min(len(data), 100)} items to {filename}")

async def test_and_save(name, url, context, csv_filename):
    """Test a site and save results to CSV - SINGLE PAGE ONLY"""
    print(f"\n{'='*80}")
    print(f"🧪 {name}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"CSV: {csv_filename}")
    print(f"Mode: SINGLE PAGE (pagination disabled)\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,  # DISABLE pagination
        extraction_context=context,
        enable_context_validation=True,
    )
    
    # Monkey-patch to force single page
    original_detect = scraper.fast_pagination_detector.detect if hasattr(scraper, 'fast_pagination_detector') else None
    if original_detect:
        async def no_pagination(html, url):
            return {'type': 'none', 'confidence': 'high'}
        scraper.fast_pagination_detector.detect = no_pagination
    
    print("⏱️  Scraping (single page only)...")
    result = await scraper.scrape(url, fields=[])
    
    if result and 'data' in result:
        data = result['data']
        metadata = result.get('metadata', {})
        source = metadata.get('extraction_source', 'unknown')
        
        print(f"\n📊 Results:")
        print(f"   Items: {len(data)}")
        print(f"   Source: {source}")
        
        if len(data) > 0:
            await save_to_csv(data, csv_filename)
            
            # Show sample
            print(f"\n📝 Sample (first item):")
            if isinstance(data[0], dict):
                for key, value in list(data[0].items())[:5]:
                    value_str = str(value)[:60] if value else 'None'
                    print(f"      {key}: {value_str}")
        else:
            print(f"   ⚠️  No items extracted")
            
            # For Metacritic, analyze HTML structure
            if name == "Metacritic":
                await analyze_metacritic(scraper, url)
    
    return result

async def analyze_metacritic(scraper, url):
    """Analyze Metacritic's HTML to find the issue"""
    print(f"\n🔬 METACRITIC ANALYSIS")
    print(f"{'='*80}")
    
    fetch_result = await scraper.html_fetcher.fetch(url)
    html = fetch_result['html']
    soup = BeautifulSoup(html, 'html.parser')
    
    # Look for game containers
    patterns = [
        ('div[class*="browse"]', 'Browse containers'),
        ('div[class*="product"]', 'Product containers'),
        ('div[class*="game"]', 'Game containers'),
        ('div[class*="item"]', 'Item containers'),
        ('article', 'Article tags'),
    ]
    
    print("\n🔍 Searching for game listing patterns:")
    for selector, description in patterns:
        try:
            elements = soup.select(selector)
            if elements:
                print(f"\n✅ {description}: {len(elements)} found")
                print(f"   Selector: {selector}")
                
                # Sample first element
                first = elements[0]
                classes = first.get('class', [])
                print(f"   Classes: {' '.join(classes) if classes else 'none'}")
                
                # Look for title
                title = (first.select_one('h3') or 
                        first.select_one('[class*="title"]') or 
                        first.select_one('a'))
                if title:
                    text = title.get_text(strip=True)[:60]
                    print(f"   Sample text: {text}...")
        except:
            pass
    
    # Save HTML sample
    cleaned = scraper.html_cleaner.clean(html)
    with open('metacritic_debug.html', 'w', encoding='utf-8') as f:
        f.write(cleaned['html'][:50000])
    print(f"\n✅ Saved HTML to: metacritic_debug.html")
    print(f"{'='*80}\n")

async def main():
    print("\n" + "="*80)
    print("📊 CSV SAMPLE GENERATOR - SINGLE PAGE ONLY")
    print("="*80)
    print("This will scrape ONE page from each site (no pagination)")
    print("="*80 + "\n")
    
    tests = [
        {
            'name': 'Reddit',
            'url': 'https://www.reddit.com/r/webscraping/',
            'context': 'Extract Reddit posts with title, author, upvotes',
            'csv': 'reddit_sample.csv'
        },
        {
            'name': 'Apify',
            'url': 'https://apify.com/',
            'context': 'Extract Actors/scrapers with name, description',
            'csv': 'apify_sample.csv'
        },
        {
            'name': 'Metacritic',
            'url': 'https://www.metacritic.com/browse/game/all/all/current-year/',
            'context': 'Extract games with title, platform, score',
            'csv': 'metacritic_sample.csv'
        },
        {
            'name': 'eBay',
            'url': 'https://www.ebay.com/b/Apple-Laptops/111422/bn_320025',
            'context': 'Extract laptops with title, price',
            'csv': 'ebay_sample.csv'
        }
    ]
    
    results = []
    for test in tests:
        try:
            result = await test_and_save(
                test['name'],
                test['url'],
                test['context'],
                test['csv']
            )
            results.append({
                'name': test['name'],
                'csv': test['csv'],
                'items': len(result.get('data', [])) if result else 0
            })
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append({'name': test['name'], 'csv': test['csv'], 'items': 0})
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 CSV GENERATION COMPLETE")
    print(f"{'='*80}\n")
    
    for r in results:
        status = "✅" if r['items'] > 0 else "❌"
        print(f"{status} {r['name']}: {r['items']} items → {r['csv']}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())








