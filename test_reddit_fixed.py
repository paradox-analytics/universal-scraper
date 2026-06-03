#!/usr/bin/env python3
"""
Test Reddit with manually fixed extraction code
"""
import asyncio
import os
import sys
import json

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.hybrid_fetcher import HybridFetcher
from bs4 import BeautifulSoup

async def main():
    print("\n" + "="*80)
    print("🔍 REDDIT TEST - With Fixed Extraction")
    print("="*80 + "\n")
    
    url = "https://www.reddit.com/r/webscraping/"
    
    print(f"🧪 Testing: {url}\n")
    
    # Fetch HTML
    print("📥 Fetching HTML with wait for posts...")
    fetcher = HybridFetcher(headless=True)
    result = await fetcher.fetch(url, wait_for_selector="shreddit-post")
    html = result['html']
    print(f"✅ Fetched {len(html)} bytes")
    
    # Parse with BeautifulSoup
    print("\n📝 Parsing HTML...")
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract from shreddit-post elements using attributes
    posts = soup.find_all('shreddit-post')
    print(f"✅ Found {len(posts)} shreddit-post elements")
    
    items = []
    for post in posts:
        item = {
            'title': post.get('post-title'),
            'author': post.get('author'),
            'upvotes': post.get('score'),
            'comments_count': post.get('comment-count'),
            'permalink': post.get('permalink'),
            'created': post.get('created-timestamp'),
            'subreddit': post.get('subreddit-name')
        }
        items.append(item)
    
    print(f"\n✅ Extracted {len(items)} posts")
    
    if items:
        print(f"\n📊 First 3 posts:\n")
        for i, item in enumerate(items[:3], 1):
            print(f"{i}. {item['title']}")
            print(f"   Author: {item['author']}, Upvotes: {item['upvotes']}, Comments: {item['comments_count']}")
            print()
        
        # Show full first item
        print(f"📝 Full first item:")
        print(json.dumps(items[0], indent=2))
    
    print("\n" + "="*80 + "\n")
    
    await fetcher.close()

if __name__ == "__main__":
    asyncio.run(main())







