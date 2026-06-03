#!/usr/bin/env python3
"""
Test field discovery for Product Hunt
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from universal_scraper.core.field_discovery import FieldDiscovery
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_fetcher import HTMLFetcher

async def test_field_discovery():
    """Test field discovery for Product Hunt"""
    
    url = "https://www.producthunt.com/categories/vibe-coding"
    
    print(f"Testing field discovery for: {url}")
    print("-" * 80)
    
    # Fetch HTML
    print("Step 1: Fetching HTML...")
    html_fetcher = HTMLFetcher()
    try:
        fetch_result = html_fetcher.fetch(url)
        html = fetch_result.get('html', '')
        # If HTML is too small, likely JS-rendered, use browser
        if len(html) < 50000:
            print(f"Static HTML too small ({len(html):,} bytes), using browser...")
            raise Exception("HTML too small, need browser")
    except Exception as e:
        print(f"Static fetch failed or insufficient: {e}, trying browser...")
        from universal_scraper.core.hybrid_fetcher import HybridFetcher
        hybrid_fetcher = HybridFetcher(
            proxy_config=None,
            force_mode="browser"  # Force browser mode for JS-rendered pages
        )
        fetch_result = await hybrid_fetcher.fetch(url, scroll_to_bottom=False)
        html = fetch_result.get('html', '')
    
    if not html or len(html) < 100:
        print("ERROR: Failed to fetch HTML")
        return
    
    print(f"✅ Fetched {len(html):,} bytes of HTML")
    print("-" * 80)
    
    # Test field discovery (without LLM first)
    print("\nStep 2: Discovering fields (JSON + HTML structure analysis)...")
    field_discovery = FieldDiscovery()
    result = await field_discovery.discover_fields(html, url, use_llm=False)
    
    print(f"\n{'='*80}")
    print("RESULTS (JSON + HTML Analysis)")
    print(f"{'='*80}")
    print(f"Fields found: {len(result['fields'])}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Source: {result['source']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"\nSuggested fields:")
    for i, field in enumerate(result['fields'], 1):
        print(f"  {i}. {field}")
    
    # Always test with LLM (if API key available) for comparison
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"\n{'-'*80}")
        print("Step 3: Discovering fields with LLM (more accurate, analyzes HTML structure)...")
        # Force LLM discovery by passing use_llm=True
        # But first check if we can find better JSON sources
        from bs4 import BeautifulSoup
        import json
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check __NEXT_DATA__
        next_data = soup.find('script', id='__NEXT_DATA__')
        if next_data:
            try:
                next_json = json.loads(next_data.string)
                print("  Found __NEXT_DATA__, analyzing structure...")
                # Try to find product/item arrays
                def find_item_arrays(obj, path=""):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if isinstance(value, list) and len(value) > 0:
                                first = value[0]
                                if isinstance(first, dict) and len(first) > 3:
                                    print(f"    Found array at {path}.{key} with {len(value)} items")
                                    print(f"    Sample keys: {list(first.keys())[:10]}")
                            find_item_arrays(value, f"{path}.{key}" if path else key)
                    elif isinstance(obj, list) and len(obj) > 0:
                        first = obj[0]
                        if isinstance(first, dict) and len(first) > 3:
                            print(f"    Found array at {path} with {len(obj)} items")
                            print(f"    Sample keys: {list(first.keys())[:10]}")
                
                find_item_arrays(next_json)
            except Exception as e:
                print(f"    Failed to parse __NEXT_DATA__: {e}")
        
        field_discovery_llm = FieldDiscovery(api_key=api_key)
        result_llm = await field_discovery_llm.discover_fields(html, url, use_llm=True)
        
        print(f"\n{'='*80}")
        print("RESULTS (LLM Analysis)")
        print(f"{'='*80}")
        print(f"Fields found: {len(result_llm['fields'])}")
        print(f"Confidence: {result_llm['confidence']:.1%}")
        print(f"Source: {result_llm['source']}")
        print(f"Reasoning: {result_llm['reasoning']}")
        print(f"\nSuggested fields:")
        for i, field in enumerate(result_llm['fields'], 1):
            print(f"  {i}. {field}")
    else:
        print("\n⚠️  OPENAI_API_KEY not set, skipping LLM-based discovery")
    
    print(f"\n{'='*80}")
    print("✅ Field discovery test completed!")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(test_field_discovery())

