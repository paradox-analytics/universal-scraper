#!/usr/bin/env python3
"""
Debug Stack Overflow Code Generation

See what code the AI is actually generating and why it's failing
"""

import asyncio
import os
from universal_scraper import UniversalScraper
from universal_scraper.core.camoufox_fetcher import CamoufoxFetcher
from universal_scraper.core.anti_detection import AntiDetectionManager
from bs4 import BeautifulSoup

async def main():
    print("="*80)
    print("Debugging Stack Overflow Code Generation")
    print("="*80)
    
    # Step 1: Fetch the HTML
    print("\n1. Fetching HTML with Camoufox...")
    fetcher = CamoufoxFetcher()
    result = await fetcher.fetch('https://stackoverflow.com/questions?tab=newest')
    html = result['html']
    print(f"   Fetched {len(html)} bytes")
    
    # Step 2: Check if the selector actually exists
    print("\n2. Checking if selector exists in HTML...")
    soup = BeautifulSoup(html, 'lxml')
    
    # Try the escaped selector
    try:
        items = soup.select('li.h\\:bg-black-150')
        print(f"   ✅ Escaped selector 'li.h\\\\:bg-black-150': {len(items)} items")
    except Exception as e:
        print(f"   ❌ Escaped selector failed: {e}")
    
    # Try finding ANY li elements
    all_li = soup.find_all('li', limit=10)
    print(f"\n   Found {len(all_li)} <li> elements (showing first 10)")
    
    for i, li in enumerate(all_li[:5], 1):
        classes = li.get('class', [])
        print(f"   {i}. Classes: {classes}")
        # Show first 100 chars
        text = li.get_text(strip=True)[:100]
        print(f"      Text: {text}...")
    
    # Step 3: Try to manually extract a question
    print("\n3. Manually extracting questions...")
    
    # Stack Overflow uses div.s-post-summary, not li.h:bg-black-150
    questions = soup.select('div.s-post-summary')
    print(f"   Found {len(questions)} questions using 'div.s-post-summary'")
    
    if questions:
        print("\n   First question structure:")
        q = questions[0]
        
        # Title
        title_elem = q.select_one('.s-post-summary--content-title a')
        title = title_elem.get_text(strip=True) if title_elem else None
        print(f"   Title: {title}")
        
        # Votes
        votes_elem = q.select_one('.s-post-summary--stats-item__emphasized')
        votes = votes_elem.get_text(strip=True) if votes_elem else None
        print(f"   Votes: {votes}")
        
        # Answers
        answers_elem = q.select_one('.s-post-summary--stats-item:nth-of-type(2) .s-post-summary--stats-item-number')
        answers = answers_elem.get_text(strip=True) if answers_elem else None
        print(f"   Answers: {answers}")
        
        # Views
        views_elem = q.select_one('.s-post-summary--stats-item:nth-of-type(3) .s-post-summary--stats-item-number')
        views = views_elem.get_text(strip=True) if views_elem else None
        print(f"   Views: {views}")
    
    # Step 4: Check what the DOM pattern detector is finding
    print("\n4. What DOM pattern detector is finding...")
    
    # Find elements with h:bg-black-150 class
    elements_with_colon = soup.find_all(class_=lambda c: c and any(':' in cls for cls in (c if isinstance(c, list) else [c])))
    print(f"   Found {len(elements_with_colon)} elements with ':' in class names")
    
    if elements_with_colon:
        print(f"\n   First 5 elements with colon classes:")
        for i, elem in enumerate(elements_with_colon[:5], 1):
            classes = elem.get('class', [])
            print(f"   {i}. Tag: {elem.name}, Classes: {classes}")
    
    await fetcher.close()
    
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    if not items and questions:
        print("\n❌ PROBLEM IDENTIFIED:")
        print("   • DOM pattern detector is finding 'li.h:bg-black-150'")
        print("   • But Stack Overflow actually uses 'div.s-post-summary'")
        print("   • The li elements with colons are likely UI elements, not data")
        print("\n✅ SOLUTION:")
        print("   • Need to improve DOM pattern detector to distinguish data vs UI")
        print("   • Should prioritize elements with meaningful content over frequency")
        print("   • Or adjust scoring to penalize UI-related class names")

if __name__ == '__main__':
    asyncio.run(main())

