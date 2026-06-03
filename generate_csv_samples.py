"""
Generate CSV samples from all test sites and analyze Metacritic issue
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
        fields = sorted(list(all_keys))
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        
        for item in data:
            if isinstance(item, dict):
                # Only write fields that exist
                row = {k: v for k, v in item.items() if k in fields}
                writer.writerow(row)
    
    print(f"   ✅ Saved {len(data)} items to {filename}")

async def test_and_save(name, url, context, csv_filename):
    """Test a site and save results to CSV"""
    print(f"\n{'='*80}")
    print(f"🧪 {name}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"CSV: {csv_filename}\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="browser",
        enable_llm_pagination=False,
        extraction_context=context,
        enable_context_validation=True,
    )
    
    print("⏱️  Scraping...")
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
            
            # For failed extractions, save the HTML structure for analysis
            if name == "Metacritic":
                print(f"\n🔍 Analyzing Metacritic HTML structure...")
                await analyze_metacritic_html(scraper, url)
    
    return result

async def analyze_metacritic_html(scraper, url):
    """Analyze Metacritic's actual HTML structure to find correct selectors"""
    print(f"\n{'='*80}")
    print("🔬 METACRITIC HTML ANALYSIS")
    print(f"{'='*80}")
    
    # Fetch HTML
    fetch_result = await scraper.html_fetcher.fetch(url)
    html = fetch_result['html']
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Save cleaned HTML sample for review
    cleaned = scraper.html_cleaner.clean(html)
    cleaned_html = cleaned['html']
    
    with open('metacritic_html_sample.html', 'w', encoding='utf-8') as f:
        # Save first 50KB for review
        f.write(cleaned_html[:50000])
    
    print(f"✅ Saved HTML sample to: metacritic_html_sample.html")
    
    # Look for game listing patterns
    print(f"\n🔍 Searching for game listing containers...")
    
    # Try various common patterns
    patterns = [
        ('div.c-finderProductCard', 'Product cards'),
        ('div[class*="product"]', 'Any div with "product" in class'),
        ('div[class*="game"]', 'Any div with "game" in class'),
        ('div[class*="item"]', 'Any div with "item" in class'),
        ('article', 'Article tags'),
        ('li[class*="browse"]', 'List items with "browse"'),
        ('div.c-gameCard', 'Game cards'),
        ('div.c-productCard', 'Product cards (alternate)'),
    ]
    
    for selector, description in patterns:
        try:
            elements = soup.select(selector)
            if elements:
                print(f"\n✅ Found {len(elements)} matches: {description}")
                print(f"   Selector: {selector}")
                
                # Show first element's structure
                if len(elements) > 0:
                    first = elements[0]
                    print(f"   First element classes: {first.get('class', [])}")
                    
                    # Look for title
                    title_candidates = [
                        first.select_one('h3'),
                        first.select_one('[class*="title"]'),
                        first.select_one('a[class*="title"]'),
                        first.find('a')
                    ]
                    
                    for candidate in title_candidates:
                        if candidate and candidate.get_text(strip=True):
                            print(f"   Title found: {candidate.get_text(strip=True)[:60]}...")
                            break
        except Exception as e:
            pass
    
    print(f"\n{'='*80}\n")

async def main():
    print("\n" + "="*80)
    print("📊 GENERATING CSV SAMPLES - Phase 1 + 2 Test Results")
    print("="*80)
    print("This will re-scrape all sites and generate CSV files for review")
    print("="*80 + "\n")
    
    tests = [
        {
            'name': 'Reddit r/webscraping',
            'url': 'https://www.reddit.com/r/webscraping/',
            'context': 'Extract Reddit posts with title, author, upvotes, comments count',
            'csv': 'reddit_sample.csv'
        },
        {
            'name': 'Apify Homepage',
            'url': 'https://apify.com/',
            'context': 'Extract featured Actors/scrapers with name, description, author',
            'csv': 'apify_sample.csv'
        },
        {
            'name': 'Metacritic',
            'url': 'https://www.metacritic.com/browse/game/all/all/current-year/',
            'context': 'Extract video game listings with title, platform, release date, and Metascore rating',
            'csv': 'metacritic_sample.csv'
        },
        {
            'name': 'eBay',
            'url': 'https://www.ebay.com/b/Apple-Laptops/111422/bn_320025',
            'context': 'Extract laptop listings with title, price, condition',
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
                'success': result and result.get('data') and len(result['data']) > 0
            })
        except Exception as e:
            print(f"\n❌ Error with {test['name']}: {e}")
            results.append({
                'name': test['name'],
                'csv': test['csv'],
                'success': False
            })
    
    # Final summary
    print(f"\n{'='*80}")
    print("📊 CSV GENERATION COMPLETE")
    print(f"{'='*80}\n")
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} {r['name']}: {r['csv']}")
    
    print(f"\n{'='*80}\n")
    print("📁 Files generated:")
    print("   - reddit_sample.csv")
    print("   - apify_sample.csv")
    print("   - metacritic_sample.csv (may be empty)")
    print("   - ebay_sample.csv")
    print("   - metacritic_html_sample.html (for debugging)")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())








