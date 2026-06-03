import asyncio
import re
from bs4 import BeautifulSoup
from universal_scraper.core.hybrid_fetcher import HybridFetcher

async def debug_html(url: str, site_name: str, search_term: str):
    """Debug HTML content to find where data actually is"""
    print(f"\n{'='*80}")
    print(f"🔍 DEBUGGING HTML: {site_name}")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Searching for: '{search_term}'\n")
    
    # Fetch page
    print("📥 Fetching page...")
    fetcher = HybridFetcher(
        force_mode='browser',
        proxy_config=None,
        browser_timeout=30000,
        enable_cache=False
    )
    
    fetch_result = await fetcher.fetch(url)
    html = fetch_result['html']
    
    print(f"✅ Page loaded: {len(html):,} bytes\n")
    
    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Check 1: Is search term in HTML text?
    print(f"🔍 CHECK 1: Is '{search_term}' in raw HTML?")
    if search_term.lower() in html.lower():
        print(f"   ✅ YES - Found in HTML!")
        # Find context
        matches = re.finditer(re.escape(search_term), html, re.IGNORECASE)
        count = 0
        for match in matches:
            count += 1
            start = max(0, match.start() - 100)
            end = min(len(html), match.end() + 100)
            context = html[start:end]
            print(f"\n   Match {count} context:")
            print(f"   ...{context}...")
            if count >= 2:  # Show first 2 matches
                break
    else:
        print(f"   ❌ NO - Not found in HTML")
    
    print()
    
    # Check 2: Look for script tags with JSON
    print("🔍 CHECK 2: Embedded JSON in <script> tags")
    scripts = soup.find_all('script', type='application/json')
    print(f"   Found {len(scripts)} <script type='application/json'> tags")
    
    for i, script in enumerate(scripts[:5], 1):  # Show first 5
        content = script.string or ""
        print(f"\n   Script {i}:")
        print(f"      ID: {script.get('id', 'N/A')}")
        print(f"      Size: {len(content):,} bytes")
        if search_term.lower() in content.lower():
            print(f"      ✅ Contains '{search_term}'!")
            # Show snippet
            idx = content.lower().find(search_term.lower())
            snippet_start = max(0, idx - 150)
            snippet_end = min(len(content), idx + 150)
            print(f"      Context: ...{content[snippet_start:snippet_end]}...")
        else:
            print(f"      ❌ Does not contain '{search_term}'")
            # Show first 200 chars
            print(f"      Preview: {content[:200]}...")
    
    print()
    
    # Check 3: Look for data attributes
    print("🔍 CHECK 3: Data attributes containing search term")
    elements_with_data = soup.find_all(attrs={"data-testid": True})
    print(f"   Found {len(elements_with_data)} elements with data-testid")
    
    # Check if any contain our search term
    matching_elements = []
    for elem in elements_with_data:
        if search_term.lower() in str(elem).lower():
            matching_elements.append(elem)
    
    if matching_elements:
        print(f"   ✅ Found {len(matching_elements)} elements containing '{search_term}'")
        for i, elem in enumerate(matching_elements[:3], 1):  # Show first 3
            print(f"\n   Element {i}:")
            print(f"      Tag: {elem.name}")
            print(f"      Attributes: {elem.attrs}")
            print(f"      Text preview: {elem.get_text(strip=True)[:200]}...")
    else:
        print(f"   ❌ No elements with data-testid contain '{search_term}'")
    
    print()
    
    # Check 4: Look in visible text
    print("🔍 CHECK 4: Visible text on page")
    visible_text = soup.get_text(separator=' ', strip=True)
    if search_term.lower() in visible_text.lower():
        print(f"   ✅ '{search_term}' appears in visible text")
        # Find parent elements
        text_elements = soup.find_all(string=re.compile(re.escape(search_term), re.IGNORECASE))
        print(f"   Found in {len(text_elements)} text node(s)")
        
        for i, text_elem in enumerate(text_elements[:3], 1):  # Show first 3
            parent = text_elem.parent
            print(f"\n   Occurrence {i}:")
            print(f"      Parent tag: {parent.name}")
            print(f"      Parent classes: {parent.get('class', [])}")
            print(f"      Full text: {text_elem.strip()[:200]}...")
            
            # Show siblings (other data might be nearby)
            print(f"      Siblings: {len(list(parent.find_all())) if parent else 0} child elements")
    else:
        print(f"   ❌ '{search_term}' NOT in visible text")
    
    print(f"\n{'='*80}\n")

async def main():
    print("\n" + "="*80)
    print("🔬 HTML CONTENT DEBUG")
    print("="*80)
    print("Looking for actual data location in HTML\n")
    
    tests = [
        {
            'site': 'Apify Homepage',
            'url': 'https://apify.com/',
            'search': 'TikTok Scraper'  # We know this Actor should be on the homepage
        },
        {
            'site': 'Reddit r/webscraping',
            'url': 'https://www.reddit.com/r/webscraping/',
            'search': 'webscraping'  # Should appear in post titles/text
        }
    ]
    
    for test in tests:
        await debug_html(test['url'], test['site'], test['search'])
    
    print("="*80)
    print("🏁 HTML DEBUG COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())








