#!/usr/bin/env python3
"""
Quick Product Hunt JSON Debug - Extract and analyze __NEXT_DATA__
"""
import asyncio
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.hybrid_fetcher import HybridFetcher


async def main():
    print("=" * 80)
    print("🔍 PRODUCT HUNT JSON DEBUG - Capturing __NEXT_DATA__")
    print("=" * 80)
    print()
    
    fetcher = HybridFetcher(
        proxy_config=None,
        headless=True,
        use_camoufox=True,
        enable_cache=False,
        force_mode='browser'  # Force browser to ensure we get JS-rendered content
    )
    
    try:
        url = "https://www.producthunt.com/"
        print(f"📥 Fetching: {url}")
        
        result = await fetcher.fetch(url)  # Will use browser due to force_mode
        html = result['html']
        
        print(f"✅ Fetched {len(html):,} bytes")
        print()
        
        # Save raw HTML for inspection
        with open("product_hunt_raw_debug.html", 'w', encoding='utf-8') as f:
            f.write(html)
        print("💾 Saved raw HTML to: product_hunt_raw_debug.html")
        print()
        
        # Try multiple patterns for __NEXT_DATA__
        print("🔍 Extracting __NEXT_DATA__...")
        
        patterns = [
            r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            r'<script id="__NEXT_DATA__"[^>]*>\s*(.*?)\s*</script>',
            r'__NEXT_DATA__["\']?\s*=\s*({.*?});',
            r'self\.__NEXT_DATA__\s*=\s*({.*?});'
        ]
        
        next_data_match = None
        for i, pattern in enumerate(patterns, 1):
            match = re.search(pattern, html, re.DOTALL)
            if match:
                print(f"✅ Found with pattern {i}")
                next_data_match = match
                break
        
        if next_data_match:
            json_str = next_data_match.group(1)
            print(f"✅ Found __NEXT_DATA__: {len(json_str):,} characters")
            
            try:
                data = json.loads(json_str)
                
                # Save to file
                output_file = "product_hunt_next_data_debug.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"💾 Saved to: {output_file}")
                print()
                
                # Analyze structure
                print("📊 Structure Analysis:")
                print(f"   Top-level keys: {list(data.keys())}")
                print()
                
                # Look for props
                if 'props' in data:
                    props = data['props']
                    print(f"   props keys: {list(props.keys()) if isinstance(props, dict) else type(props)}")
                    
                    if isinstance(props, dict) and 'pageProps' in props:
                        page_props = props['pageProps']
                        print(f"   pageProps keys: {list(page_props.keys()) if isinstance(page_props, dict) else type(page_props)}")
                        print()
                        
                        # Look for arrays
                        print("🔍 Looking for item arrays in pageProps:")
                        for key, value in page_props.items():
                            if isinstance(value, list) and len(value) > 0:
                                print(f"   • {key}: {len(value)} items")
                                if isinstance(value[0], dict):
                                    print(f"      Sample keys: {list(value[0].keys())[:10]}")
                            elif isinstance(value, dict):
                                # Check nested dicts for arrays
                                for nested_key, nested_value in value.items():
                                    if isinstance(nested_value, list) and len(nested_value) > 3:
                                        print(f"   • {key}.{nested_key}: {len(nested_value)} items")
                                        if nested_value and isinstance(nested_value[0], dict):
                                            print(f"      Sample keys: {list(nested_value[0].keys())[:10]}")
                
                # Look for apollo/redux state
                print()
                print("🔍 Looking for other state keys:")
                for key in data.keys():
                    if 'state' in key.lower() or 'apollo' in key.lower() or 'redux' in key.lower():
                        print(f"   • Found: {key}")
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
        else:
            print("❌ __NEXT_DATA__ not found")
            
            # Check for other Next.js indicators
            if '__NEXT_' in html:
                print("   ⚠️  Found __NEXT_ references, but not __NEXT_DATA__")
            
    finally:
        await fetcher.close()
    
    print()
    print("=" * 80)
    print("✅ Debug complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
