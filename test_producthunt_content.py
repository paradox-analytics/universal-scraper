#!/usr/bin/env python3
"""
Check if Product Hunt HTML contains actual rendered content.
"""
import asyncio
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.hybrid_fetcher import HybridFetcher

async def test():
    url = "https://www.producthunt.com/categories/vibe-coding"
    
    proxy_config = {
        'server': 'brd.superproxy.io:33335',
        'username': 'brd-customer-hl_803e8195-zone-residential_proxy2',
        'password': 'rs2mvj79xi2t'
    }
    
    fetcher = HybridFetcher(
        proxy_config=proxy_config,
        headless=True,
        browser_timeout=90000,
        force_mode='browser'
    )
    
    result = await fetcher.fetch(url)
    html = result.get('html', '')
    
    print(f"HTML Length: {len(html)} bytes\n")
    
    # Check for various Product Hunt indicators
    indicators = {
        'producthunt.com': html.lower().count('producthunt.com'),
        'vibe-coding': html.lower().count('vibe-coding'),
        '__NEXT_DATA__': '__NEXT_DATA__' in html,
        'product': html.lower().count('product'),
        'category': html.lower().count('category'),
        'title': html.lower().count('title'),
        'description': html.lower().count('description'),
        'author': html.lower().count('author'),
        'date': html.lower().count('date'),
    }
    
    print("Content Indicators:")
    for key, count in indicators.items():
        print(f"  {key}: {count}")
    
    # Check for JSON data
    if '__NEXT_DATA__' in html:
        print("\n✅ Found __NEXT_DATA__ (Next.js hydration data)")
        # Extract JSON
        match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if match:
            json_str = match.group(1)
            print(f"   JSON length: {len(json_str)} bytes")
            if 'vibe-coding' in json_str.lower():
                print("   ✅ Contains 'vibe-coding' in JSON data")
    
    # Show sample of HTML at different positions
    print("\n📄 HTML Samples:")
    print(f"  First 300 chars: {html[:300]}")
    print(f"\n  Middle (10000-10300): {html[10000:10300] if len(html) > 10000 else 'N/A'}")
    print(f"\n  Last 300 chars: {html[-300:]}")
    
    # Check if it's mostly empty/loading
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
    if body_match:
        body_content = body_match.group(1)
        # Remove script and style tags
        body_clean = re.sub(r'<script[^>]*>.*?</script>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
        body_clean = re.sub(r'<style[^>]*>.*?</style>', '', body_clean, flags=re.DOTALL | re.IGNORECASE)
        body_text = re.sub(r'<[^>]+>', '', body_clean)
        body_text = ' '.join(body_text.split())
        
        print(f"\n📝 Body text length (cleaned): {len(body_text)} chars")
        print(f"   Sample: {body_text[:200]}...")
        
        if len(body_text) < 100:
            print("   ⚠️  Body seems mostly empty - JavaScript may not have rendered")
        else:
            print("   ✅ Body contains text content")

if __name__ == "__main__":
    asyncio.run(test())



