#!/usr/bin/env python3
"""
MINIMAL TEST - Just verify the JSON analyzer's select_best_source method works
NO browser, NO actual scraping - just test the JSON selection logic
Should take < 5 seconds
"""
import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universal_scraper.core.json_analyzer import LLMJsonAnalyzer
from universal_scraper.core.context_manager import ContextManager, ExtractionContext

async def main():
    print("\n" + "="*80)
    print("🔬 MINIMAL TEST - JSON Source Selection Logic Only")
    print("="*80)
    print("Purpose: Verify select_best_source() can pick Reddit posts over config")
    print("Expected time: < 5 seconds")
    print("="*80 + "\n")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: No OPENAI_API_KEY")
        return
    
    print("✅ API key found")
    
    # Create mock JSON sources (simulating what Reddit returns)
    mock_sources = {
        "config_data": [
            {
                "ACCOUNT_MANAGER_ORIGIN": "https://www.reddit.com",
                "APPLE_SSO_CLIENT_ID": "com.reddit.RedditAppleSSO",
                "USE_DEBUG": False
            }
        ],
        "posts_data": [
            {
                "title": "Best web scraping tools in 2024?",
                "author": "user123",
                "upvotes": 45,
                "comments": 12,
                "subreddit": "webscraping"
            },
            {
                "title": "How to handle JavaScript rendering",
                "author": "scraper_pro",
                "upvotes": 89,
                "comments": 23,
                "subreddit": "webscraping"
            }
        ],
        "tracking_data": [
            {
                "event": "page_view",
                "timestamp": 1234567890,
                "session_id": "abc123"
            }
        ]
    }
    
    # Create context
    context = ExtractionContext(
        goal="Extract Reddit posts with title, author, upvotes, comments count",
        data_type="posts",
        fields=["title", "author", "upvotes", "comments"],
        raw_prompt="Extract Reddit posts with title, author, upvotes, comments count"
    )
    
    print(f"📋 Context: {context.goal}")
    print(f"📦 Testing with {len(mock_sources)} JSON sources")
    print(f"\n⏱️  Calling LLM to select best source...\n")
    
    # Test the JSON analyzer
    analyzer = LLMJsonAnalyzer(api_key=api_key)
    
    try:
        best_source = analyzer.select_best_source(
            json_sources=mock_sources,
            url="https://www.reddit.com/r/webscraping/",
            context=context
        )
        
        print("\n" + "="*80)
        print("📊 RESULTS")
        print("="*80)
        
        if best_source:
            print(f"✅ Selected source: {best_source}")
            
            print("\n" + "="*80)
            print("🔍 VALIDATION")
            print("="*80)
            
            if best_source == "posts_data":
                print("✅ SUCCESS: Correctly selected 'posts_data'!")
                print("   🎉 The JSON selection IS working!")
            elif best_source == "config_data":
                print("❌ FAIL: Selected 'config_data' (wrong)")
                print("   🐛 The JSON selection is NOT working")
            else:
                print(f"⚠️  UNEXPECTED: Selected '{best_source}'")
                if "post" in best_source.lower():
                    print("   (Still might be correct if it contains post data)")
        else:
            print("❌ FAIL: No source selected")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())

