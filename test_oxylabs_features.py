"""
Test Oxylabs-inspired Features on Production Sites

Demonstrates natural language field generation + scraping on:
- Hacker News
- Stack Overflow
- GitHub Trending
"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def test_with_natural_language():
    print("\n" + "="*80)
    print("🎯 TESTING OXYLABS FEATURES - Natural Language + Scraping")
    print("="*80 + "\n")
    
    api_key = os.environ['OPENAI_API_KEY']
    
    # Test data: (name, url, natural language prompt)
    sites = [
        {
            'name': 'Hacker News',
            'url': 'https://news.ycombinator.com/',
            'prompt': 'I want post titles, points, and comment counts'
        },
        {
            'name': 'Stack Overflow',
            'url': 'https://stackoverflow.com/questions?tab=newest',
            'prompt': 'Get question titles and vote counts'
        },
        {
            'name': 'GitHub Trending',
            'url': 'https://github.com/trending',
            'prompt': 'I need repository names, descriptions, and star counts'
        }
    ]
    
    results = []
    total_time = 0
    
    for site in sites:
        print(f"\n{'='*80}")
        print(f"🔍 Testing: {site['name']}")
        print(f"{'='*80}")
        print(f"URL: {site['url']}")
        print(f"Prompt: \"{site['prompt']}\"")
        print()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use the new scrape_from_prompt() convenience method!
            result = await UniversalScraper.scrape_from_prompt(
                url=site['url'],
                prompt=site['prompt'],
                api_key=api_key,
                use_camoufox=False,  # Fast test
                enable_auto_pagination=False
            )
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            total_time += duration
            
            items = result.get('data', [])
            quality = result.get('quality', 0)
            
            print(f"📊 Results ({duration:.1f}s):")
            print(f"   Items: {len(items)}")
            print(f"   Quality: {quality:.0f}%")
            print(f"\n   Sample Items:")
            for i, item in enumerate(items[:2], 1):
                print(f"   {i}. {item}")
            
            status = "✅" if quality >= 90 else "⚠️" if quality >= 50 else "❌"
            print(f"\n{status} Status: {'PRODUCTION READY' if quality >= 90 else 'NEEDS WORK'}")
            
            results.append({
                'site': site['name'],
                'items': len(items),
                'quality': quality,
                'time': duration,
                'status': status
            })
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append({
                'site': site['name'],
                'items': 0,
                'quality': 0,
                'time': asyncio.get_event_loop().time() - start_time,
                'status': '❌'
            })
    
    # Summary
    print("\n" + "="*80)
    print("📊 OXYLABS FEATURES TEST RESULTS")
    print("="*80)
    print(f"{'Site':<20} {'Items':<8} {'Quality':<10} {'Time':<10} {'Status':<10}")
    print("-" * 80)
    for res in results:
        print(f"{res['site']:<20} {res['items']:<8} {res['quality']:.0f}%{res['time']:<10.1f}s {res['status']:<10}")
    print("-" * 80)
    
    success_count = sum(1 for res in results if res['quality'] >= 90)
    print(f"\n✅ Success Rate: {success_count}/{len(results)} ({success_count/len(results):.0%})")
    print(f"📦 Total Items: {sum(r['items'] for r in results)}")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    print(f"⚡ Avg Time/Site: {total_time/len(results):.1f}s")
    
    if success_count == len(results):
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - Natural Language Feature Working!")
        print("="*80)
    elif success_count >= len(results) / 2:
        print("\n" + "="*80)
        print("✅ MOSTLY WORKING - Some sites need refinement")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️ NEEDS WORK - Multiple issues detected")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(test_with_natural_language())





