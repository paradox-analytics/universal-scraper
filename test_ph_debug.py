#!/usr/bin/env python3
"""
Debug Product Hunt page structure
"""
import asyncio
from playwright.async_api import async_playwright

async def debug_page():
    print("Debugging Product Hunt page structure...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("\n1. Loading page...")
        await page.goto("https://www.producthunt.com/categories/vibe-coding", wait_until='domcontentloaded')
        print("   DOM content loaded")
        
        print("\n2. Waiting 5 seconds for JS to execute...")
        await asyncio.sleep(5)
        
        # Check what's actually in the HTML
        html = await page.content()
        print(f"\n3. HTML Analysis:")
        print(f"   Size: {len(html):,} bytes")
        print(f"   Contains '__next': {'__next' in html}")
        print(f"   Contains '__NEXT_DATA__': {'__NEXT_DATA__' in html}")
        print(f"   Contains 'next/script': {'next/script' in html}")
        print(f"   Contains 'Product Hunt': {'Product Hunt' in html}")
        
        # Check for Next.js indicators
        next_indicators = await page.evaluate("""
            () => {
                return {
                    hasNextDataEl: !!document.querySelector('#__NEXT_DATA__'),
                    hasNextEl: !!document.querySelector('#__next'),
                    hasRootEl: !!document.querySelector('#root'),
                    hasAppEl: !!document.querySelector('#__nuxt'),
                    hasNextDataWindow: typeof window.__NEXT_DATA__ !== 'undefined',
                    hasReact: typeof window.React !== 'undefined',
                    hasNextConfig: typeof window.__NEXT_CONFIG__ !== 'undefined',
                    bodyClasses: document.body.className,
                    htmlAttrs: document.documentElement.getAttribute('lang'),
                    firstDivId: document.querySelector('body > div')?.id || 'none',
                    scriptTags: Array.from(document.querySelectorAll('script')).length,
                    scriptSrcs: Array.from(document.querySelectorAll('script[src]')).slice(0, 5).map(s => s.src)
                };
            }
        """)
        
        print(f"\n4. Page indicators:")
        for key, value in next_indicators.items():
            print(f"   {key}: {value}")
        
        # Get body text
        body_text = await page.evaluate('document.body.innerText')
        print(f"\n5. Body text:")
        print(f"   Length: {len(body_text):,} chars")
        print(f"   First 500 chars:\n   {body_text[:500]}")
        
        # Save HTML for inspection
        with open('/tmp/product_hunt_debug.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"\n6. HTML saved to: /tmp/product_hunt_debug.html")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_page())




