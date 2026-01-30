"""Debug script to understand why eBay and GitHub are failing"""
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper import UniversalScraper
from bs4 import BeautifulSoup


async def debug_source(name, url, context, fields, wait_for=None):
    """Debug a single source"""
    print(f"\n{'='*80}")
    print(f"🔍 DEBUGGING: {name}")
    print(f"{'='*80}\n")
    
    scraper = UniversalScraper(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="gpt-4o-mini",
        extraction_context=context,
        fetch_mode="browser",
        headless=True,
        enable_llm_pagination=False
    )
    
    try:
        # Fetch the page
        print("📥 Fetching page...")
        from universal_scraper.core.hybrid_fetcher import HybridFetcher
        
        fetcher = HybridFetcher(
            enable_cache=False,
            headless=True
        )
        
        result = await fetcher.fetch(url, wait_for_selector=wait_for)
        
        # Extract HTML from result
        if isinstance(result, dict):
            html = result.get('html', '')
        elif isinstance(result, str):
            html = result
        else:
            html = str(result)
        
        print(f"✅ Fetched: {len(html)} bytes\n")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check for custom elements
        import re
        custom_elements = set(re.findall(r'<([a-z]+-[a-z-]+)', html))
        if custom_elements:
            print(f"🚨 Custom elements found: {', '.join(list(custom_elements)[:10])}\n")
        else:
            print("ℹ️  No custom elements found\n")
        
        # Look for common container patterns
        print("🔍 Looking for data containers...\n")
        
        patterns = [
            ('Articles', 'article'),
            ('List items (li)', 'li'),
            ('Divs with class containing "item"', soup.find_all('div', class_=re.compile(r'item', re.I))),
            ('Divs with class containing "product"', soup.find_all('div', class_=re.compile(r'product', re.I))),
            ('Divs with class containing "card"', soup.find_all('div', class_=re.compile(r'card', re.I))),
            ('Sections', 'section'),
        ]
        
        for pattern_name, selector in patterns:
            if isinstance(selector, str):
                elements = soup.find_all(selector)
            else:
                elements = selector
            
            if elements:
                print(f"   ✅ {pattern_name}: {len(elements)} found")
                if len(elements) > 0:
                    # Show first element
                    first = str(elements[0])[:500]
                    print(f"      Sample: {first}...\n")
        
        # Check for specific classes/IDs
        print("\n🔍 Analyzing HTML structure...\n")
        
        # Get all unique classes
        all_classes = set()
        for tag in soup.find_all(class_=True):
            if isinstance(tag.get('class'), list):
                all_classes.update(tag['class'])
        
        # Filter for likely data container classes
        likely_containers = [c for c in all_classes if any(
            keyword in c.lower() 
            for keyword in ['item', 'product', 'card', 'list', 'result', 'post', 'article', 'repo', 'entry']
        )]
        
        if likely_containers:
            print(f"   📦 Likely container classes ({len(likely_containers)}):")
            for cls in sorted(likely_containers)[:20]:
                elements = soup.find_all(class_=cls)
                print(f"      • .{cls}: {len(elements)} elements")
        
        # Save HTML sample
        output_dir = Path(__file__).parent / "debug_output"
        output_dir.mkdir(exist_ok=True)
        
        sample_path = output_dir / f"{name.lower().replace(' ', '_')}_sample.html"
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(html[:50000])  # First 50K chars
        
        print(f"\n💾 Saved HTML sample to: {sample_path}")
        
        await fetcher.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        scraper.close()


async def main():
    """Debug all failed sources"""
    
    sources = [
        {
            'name': 'eBay',
            'url': 'https://www.ebay.com/sch/i.html?_nkw=laptop',
            'context': 'Extract eBay product listings with name, price, condition, shipping',
            'fields': ['name', 'price', 'condition', 'shipping'],
            'wait_for': None
        },
        {
            'name': 'GitHub Trending',
            'url': 'https://github.com/trending',
            'context': 'Extract trending repos with name, description, stars, language',
            'fields': ['name', 'description', 'stars', 'language'],
            'wait_for': None
        },
        {
            'name': 'Metacritic',
            'url': 'https://www.metacritic.com/browse/game/',
            'context': 'Extract game listings with title, platform, release date, metascore',
            'fields': ['title', 'platform', 'release_date', 'metascore'],
            'wait_for': None
        }
    ]
    
    for source in sources:
        await debug_source(**source)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())

