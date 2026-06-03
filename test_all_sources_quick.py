"""Quick test of all sources with improved custom element detection"""
import asyncio
import os
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper

# Test sources
TEST_SOURCES = [
    {
        'name': 'Reddit',
        'url': 'https://www.reddit.com/r/webscraping/',
        'context': 'Extract Reddit posts with title, author, upvotes, comments',
        'fields': ['title', 'author', 'upvotes', 'comments_count'],
        'wait_for': 'shreddit-post'
    },
    {
        'name': 'eBay',
        'url': 'https://www.ebay.com/sch/i.html?_nkw=laptop',
        'context': 'Extract eBay product listings with name, price, condition, shipping',
        'fields': ['name', 'price', 'condition', 'shipping'],
        'wait_for': None
    },
    {
        'name': 'Metacritic',
        'url': 'https://www.metacritic.com/browse/game/',
        'context': 'Extract game listings with title, platform, release date, metascore',
        'fields': ['title', 'platform', 'release_date', 'metascore'],
        'wait_for': None
    },
    {
        'name': 'Hacker News',
        'url': 'https://news.ycombinator.com/',
        'context': 'Extract HN posts with title, points, author, comments',
        'fields': ['title', 'points', 'author', 'comments'],
        'wait_for': None
    },
    {
        'name': 'GitHub Trending',
        'url': 'https://github.com/trending',
        'context': 'Extract trending repos with name, description, stars, language',
        'fields': ['name', 'description', 'stars', 'language'],
        'wait_for': None
    }
]


async def test_source(source):
    """Test a single source"""
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {source['name']}")
    print(f"   URL: {source['url']}")
    print(f"{'='*80}\n")
    
    scraper = None
    try:
        # Create scraper
        scraper = UniversalScraper(
            api_key=os.environ.get('OPENAI_API_KEY'),
            model_name="gpt-4o-mini",
            extraction_context=source['context'],
            fetch_mode="browser",
            headless=True,
            enable_llm_pagination=False
        )
        
        # Disable pagination explicitly
        if hasattr(scraper, 'fast_pagination_detector') and scraper.fast_pagination_detector:
            scraper.fast_pagination_detector.detect = lambda url, html, current_items: None
        if hasattr(scraper, 'pagination_analyzer') and scraper.pagination_analyzer:
            scraper.pagination_analyzer.analyze_pagination_strategy = lambda url, html, user_hints: None
        
        # Scrape
        result = await scraper.scrape(
            source['url'],
            fields=source['fields'],
            wait_for_selector=source.get('wait_for')
        )
        
        # Extract data
        if isinstance(result, dict) and 'data' in result:
            items = result['data']
            metadata = result.get('metadata', {})
        elif isinstance(result, list):
            items = result
            metadata = {}
        else:
            items = []
            metadata = {}
        
        # Print results
        print(f"\n✅ RESULTS:")
        print(f"   • Items extracted: {len(items)}")
        print(f"   • Extraction source: {metadata.get('extraction_source', 'unknown')}")
        print(f"   • Execution time: {metadata.get('execution_time', 0):.1f}s")
        
        if items:
            # Show sample
            print(f"\n📋 Sample (first 2 items):")
            for i, item in enumerate(items[:2], 1):
                print(f"\n   Item {i}:")
                for key, value in item.items():
                    if not key.startswith('_'):
                        value_str = str(value) if value else 'N/A'
                        value_str = value_str[:80]
                        if value and len(str(value)) > 80:
                            value_str += "..."
                        print(f"     • {key}: {value_str}")
            
            # Quality metrics
            complete = sum(1 for item in items if all(item.get(f) for f in source['fields']))
            completeness = (complete / len(items)) * 100 if items else 0
            print(f"\n📈 Quality:")
            print(f"   • Complete items: {complete}/{len(items)} ({completeness:.0f}%)")
        
        return items
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    finally:
        if scraper:
            scraper.close()


def save_csv(items, source_name):
    """Save items to CSV"""
    if not items:
        return
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / f"{source_name.lower().replace(' ', '_')}.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if items:
            # Get all unique keys
            keys = set()
            for item in items:
                keys.update(k for k in item.keys() if not k.startswith('_'))
            
            writer = csv.DictWriter(f, fieldnames=sorted(keys))
            writer.writeheader()
            
            for item in items:
                # Filter out metadata fields
                clean_item = {k: v for k, v in item.items() if not k.startswith('_')}
                writer.writerow(clean_item)
    
    print(f"   💾 Saved to: {csv_path}")


async def main():
    """Test all sources"""
    print(f"\n{'#'*80}")
    print("🚀 TESTING ALL SOURCES WITH IMPROVED CUSTOM ELEMENT DETECTION")
    print(f"{'#'*80}")
    
    results = {}
    
    for source in TEST_SOURCES:
        items = await test_source(source)
        results[source['name']] = items
        
        if items:
            save_csv(items, source['name'])
        
        print()  # Spacing
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    for name, items in results.items():
        status = "✅" if items else "❌"
        count = len(items) if items else 0
        print(f"   {status} {name:20s} → {count:3d} items")
    
    total_items = sum(len(items) for items in results.values() if items)
    successful = sum(1 for items in results.values() if items)
    
    print(f"\n   Total: {successful}/{len(TEST_SOURCES)} sources successful, {total_items} items extracted")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())







