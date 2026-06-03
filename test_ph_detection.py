#!/usr/bin/env python3
"""
Test Product Hunt framework detection
"""
import asyncio
from playwright.async_api import async_playwright

async def test_framework_detection():
    print("Testing Product Hunt framework detection...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.goto("https://www.producthunt.com/categories/vibe-coding", wait_until='domcontentloaded')
        await asyncio.sleep(2)  # Wait a bit for initial load
        
        # Test the CURRENT detection logic
        framework_detected = await page.evaluate(
            """
            () => {
                // Quick framework detection (no waiting)
                return !!(
                    document.getElementById('__NEXT_DATA__') ||
                    document.getElementById('__next') ||
                    document.getElementById('root') ||
                    document.querySelector('[data-reactroot]') ||
                    document.querySelector('[data-vue-app]') ||
                    document.querySelector('[ng-app]') ||
                    window.__NEXT_DATA__ ||
                    window.React ||
                    window.Vue ||
                    window.angular
                );
            }
            """
        )
        
        print(f"Framework detected (CURRENT logic): {framework_detected}")
        
        # Check individual indicators
        has_next_data_el = await page.evaluate('!!document.getElementById("__NEXT_DATA__")')
        has_next_el = await page.evaluate('!!document.getElementById("__next")')
        has_root_el = await page.evaluate('!!document.getElementById("root")')
        has_next_data_window = await page.evaluate('!!window.__NEXT_DATA__')
        has_react_window = await page.evaluate('!!window.React')
        
        print(f"\nIndividual checks:")
        print(f"  #__NEXT_DATA__ element: {has_next_data_el}")
        print(f"  #__next element: {has_next_el}")
        print(f"  #root element: {has_root_el}")
        print(f"  window.__NEXT_DATA__: {has_next_data_window}")
        print(f"  window.React: {has_react_window}")
        
        # Get HTML size
        html = await page.content()
        print(f"\nHTML size: {len(html):,} bytes")
        print(f"Contains '__next': {'__next' in html}")
        print(f"Contains '__NEXT_DATA__': {'__NEXT_DATA__' in html}")
        print(f"Contains 'Product Hunt': {'Product Hunt' in html}")
        
        # Check if content is loaded
        body_text = await page.evaluate('document.body.innerText')
        print(f"\nBody text length: {len(body_text):,} chars")
        print(f"Contains 'Lovable': {'Lovable' in body_text}")
        print(f"Contains 'v0 by Vercel': {'v0 by Vercel' in body_text}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_framework_detection())




