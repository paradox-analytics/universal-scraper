#!/usr/bin/env python3
"""
Test Product Hunt field discovery with universal system
"""
import asyncio
import sys
import os
sys.path.insert(0, '.')

from universal_scraper.core.field_discovery import FieldDiscovery
from universal_scraper.core.hybrid_fetcher import HybridFetcher

async def test_field_discovery():
    # API key
    api_key = "REDACTED_OPENAI_KEY_1"
    
    # Bright Data proxy config
    proxy_config = {
        'server': 'brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    url = "https://www.producthunt.com/categories/vibe-coding"
    
    print(f"Testing Product Hunt field discovery (UNIVERSAL - no hardcoded logic)...")
    print(f"URL: {url}")
    print(f"Using Bright Data proxy: {proxy_config['server']}")
    print("-" * 80)
    
    try:
        # Step 1: Fetch HTML with proxy
        print("\n📥 Fetching HTML...")
        hybrid_fetcher = HybridFetcher(
            proxy_config=proxy_config,
            headless=True,
            browser_timeout=120000,
            force_mode='browser'  # Use browser to get full content
        )
        
        fetch_result = await hybrid_fetcher.fetch(url)
        html = fetch_result.get('html', '')
        
        if not html:
            print("❌ Failed to fetch HTML")
            return 1
        
        print(f"✅ Fetched {len(html)} bytes of HTML")
        
        # Step 2: Test field discovery
        print("\n🔍 Testing universal field discovery...")
        field_discovery = FieldDiscovery(api_key=api_key, model_name="gpt-4o-mini")
        
        # Use LLM-based discovery (the universal approach)
        result = await field_discovery.discover_fields(
            html=html,
            url=url,
            use_llm=True  # Force LLM to analyze content
        )
        
        if result:
            print(f"\n✅ Field discovery completed!")
            print(f"Source: {result.get('source', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.1%}")
            print(f"Reasoning: {result.get('reasoning', 'N/A')}")
            
            fields = result.get('fields', [])
            print(f"\n📋 Suggested fields ({len(fields)} total):")
            for i, field in enumerate(fields, 1):
                print(f"  {i}. {field}")
            
            # Check if it correctly identified Product Hunt fields (not e-commerce)
            ecommerce_fields = ['price', 'condition', 'seller name', 'shipping cost']
            producthunt_fields = ['maker', 'creator', 'upvotes', 'votes', 'tagline']
            
            found_ecommerce = [f for f in fields if any(ef in f.lower() for ef in ecommerce_fields)]
            found_producthunt = [f for f in fields if any(pf in f.lower() for pf in producthunt_fields)]
            
            print(f"\n📊 Analysis:")
            if found_ecommerce:
                print(f"  ⚠️  Found e-commerce fields: {found_ecommerce}")
                print(f"     This suggests the LLM may have misclassified Product Hunt as e-commerce")
            else:
                print(f"  ✅ No e-commerce fields found (good - Product Hunt is not e-commerce)")
            
            if found_producthunt:
                print(f"  ✅ Found Product Hunt fields: {found_producthunt}")
                print(f"     This suggests the LLM correctly identified it as a product discovery platform")
            else:
                print(f"  ⚠️  No Product Hunt-specific fields found")
            
            return 0 if found_producthunt and not found_ecommerce else 1
        else:
            print("\n❌ Field discovery returned no results")
            return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(test_field_discovery())
    sys.exit(exit_code)




