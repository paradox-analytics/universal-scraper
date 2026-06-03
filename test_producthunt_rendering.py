#!/usr/bin/env python3
"""
Test Product Hunt rendering with Bright Data proxy to verify JavaScript execution.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.hybrid_fetcher import HybridFetcher

async def test_producthunt():
    """Test Product Hunt with Bright Data proxy"""
    print("🧪 Testing Product Hunt rendering with Bright Data proxy...\n")
    
    url = "https://www.producthunt.com/categories/vibe-coding"
    
    # Bright Data proxy config
    proxy_config = {
        'server': 'brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    }
    
    print(f"📡 URL: {url}")
    print(f"🔐 Proxy: {proxy_config['server']}\n")
    
    fetcher = HybridFetcher(
        proxy_config=proxy_config,
        headless=True,
        browser_timeout=90000,
        force_mode='browser'  # Force browser mode for JS rendering
    )
    
    try:
        result = await fetcher.fetch(url)
        html = result.get('html', '')
        
        print(f"✅ Fetched {len(html)} bytes\n")
        
        # Check for JavaScript-rendered content indicators
        checks = {
            'Product Hunt title': 'producthunt' in html.lower()[:1000],
            'React/Next.js content': '__NEXT_DATA__' in html or 'react' in html.lower()[:500],
            'Vibe Coding category': 'vibe-coding' in html.lower()[:2000],
            'Product cards': 'product' in html.lower()[:2000] and ('card' in html.lower()[:2000] or 'item' in html.lower()[:2000]),
            'JavaScript executed': len(html) > 100000,  # Large HTML usually means JS rendered
        }
        
        print("📊 Content Checks:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}: {passed}")
        
        # Show HTML preview
        print(f"\n📄 HTML Preview (first 500 chars):")
        print(html[:500])
        print("...")
        
        # Check if it's just a shell
        if len(html) < 50000:
            print("\n⚠️  Warning: HTML seems small, might be static shell")
        else:
            print(f"\n✅ HTML size looks good ({len(html)} bytes)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_producthunt())



