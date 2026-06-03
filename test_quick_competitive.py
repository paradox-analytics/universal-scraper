"""
Quick Competitive Analysis: Universal Scraper vs ScrapeGraphAI
Tests 6 representative sites (2 per category)
"""

import asyncio
import json
import time
import os
from typing import Dict, List, Any
from datetime import datetime

# Our scraper
from universal_scraper.core.scraper import UniversalScraper

# ScrapeGraphAI
try:
    from scrapegraphai.graphs import SmartScraperGraph
    SCRAPEGRAPH_AVAILABLE = True
except ImportError:
    SCRAPEGRAPH_AVAILABLE = False
    print("⚠️  ScrapeGraphAI not installed. Running Universal Scraper only.")


# Simplified test sites (2 per category = 6 total)
TEST_SITES = [
    # E-commerce
    {
        'site_name': 'Books to Scrape',
        'category': 'E-commerce',
        'url': 'https://books.toscrape.com/',
        'fields': ['title', 'price', 'rating', 'availability'],
        'extraction_goal': 'Extract all book listings with title, price, rating, and availability'
    },
    {
        'site_name': 'Quotes to Scrape',
        'category': 'Content',
        'url': 'https://quotes.toscrape.com/',
        'fields': ['text', 'author', 'tags'],
        'extraction_goal': 'Extract all quotes with text, author, and tags'
    },
    
    # News/Content
    {
        'site_name': 'Hacker News',
        'category': 'News',
        'url': 'https://news.ycombinator.com/',
        'fields': ['title', 'points', 'comments', 'author'],
        'extraction_goal': 'Extract all article listings with title, points, comments, and author'
    },
    {
        'site_name': 'GitHub Trending',
        'category': 'Directory',
        'url': 'https://github.com/trending',
        'fields': ['repository', 'description', 'stars', 'language'],
        'extraction_goal': 'Extract trending repositories with name, description, stars, and language'
    },
    
    # Other
    {
        'site_name': 'Stack Overflow',
        'category': 'Forum',
        'url': 'https://stackoverflow.com/questions',
        'fields': ['title', 'votes', 'answers', 'views'],
        'extraction_goal': 'Extract questions with title, votes, answers, and views'
    },
    {
        'site_name': 'Product Hunt',
        'category': 'Social',
        'url': 'https://www.producthunt.com/',
        'fields': ['name', 'tagline', 'votes'],
        'extraction_goal': 'Extract product listings with name, tagline, and votes'
    }
]


async def test_our_scraper(url: str, fields: List[str], api_key: str) -> Dict:
    """Test with our scraper"""
    scraper = UniversalScraper(
        api_key=api_key,
        fetch_mode="hybrid",
        enable_cache=True
    )
    
    start_time = time.time()
    try:
        result = await scraper.scrape(url=url, fields=fields)
        duration = time.time() - start_time
        
        await scraper.close()
        
        return {
            'success': True,
            'items': result['data'],
            'item_count': len(result['data']),
            'duration': duration,
            'source': result.get('source', 'unknown'),
            'error': None
        }
    except Exception as e:
        duration = time.time() - start_time
        await scraper.close()
        return {
            'success': False,
            'items': [],
            'item_count': 0,
            'duration': duration,
            'error': str(e)
        }


async def test_scrapegraph(url: str, extraction_goal: str, api_key: str) -> Dict:
    """Test with ScrapeGraphAI"""
    if not SCRAPEGRAPH_AVAILABLE:
        return {'error': 'Not installed', 'item_count': 0, 'duration': 0}
    
    start_time = time.time()
    try:
        graph_config = {
            "llm": {
                "api_key": api_key,
                "model": "openai/gpt-4o-mini",
            },
            "verbose": False,
            "headless": True
        }
        
        smart_scraper = SmartScraperGraph(
            prompt=extraction_goal,
            source=url,
            config=graph_config
        )
        
        result = smart_scraper.run()
        duration = time.time() - start_time
        
        # Normalize result
        items = []
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, list):
                    items = value
                    break
            if not items:
                items = [result]
        elif isinstance(result, list):
            items = result
        
        return {
            'success': True,
            'items': items,
            'item_count': len(items),
            'duration': duration,
            'error': None
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'items': [],
            'item_count': 0,
            'duration': duration,
            'error': str(e)
        }


def calculate_completeness(items: List[Dict], fields: List[str]) -> float:
    """Calculate field completeness"""
    if not items or not fields:
        return 0.0
    
    total = len(items) * len(fields)
    filled = sum(
        1 for item in items
        for field in fields
        if item.get(field) not in [None, '', []]
    )
    return (filled / total * 100) if total > 0 else 0.0


async def main():
    api_key = os.getenv('OPENAI_API_KEY') or "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"
    
    print(f"\n{'='*80}")
    print(f"🚀 Quick Competitive Analysis: Universal Scraper vs ScrapeGraphAI")
    print(f"{'='*80}")
    print(f"Testing {len(TEST_SITES)} sites across different categories\n")
    
    results = []
    
    for i, site in enumerate(TEST_SITES, 1):
        print(f"\n{'-'*80}")
        print(f"Test {i}/{len(TEST_SITES)}: {site['site_name']} ({site['category']})")
        print(f"{'-'*80}")
        
        # Test our scraper
        print(f"🔵 Testing Universal Scraper...")
        our_result = await test_our_scraper(site['url'], site['fields'], api_key)
        print(f"   Items: {our_result['item_count']}, Time: {our_result['duration']:.1f}s")
        
        # Test ScrapeGraphAI
        scrapegraph_result = {'item_count': 0, 'duration': 0}
        if SCRAPEGRAPH_AVAILABLE:
            print(f"🟢 Testing ScrapeGraphAI...")
            scrapegraph_result = await test_scrapegraph(site['url'], site['extraction_goal'], api_key)
            print(f"   Items: {scrapegraph_result['item_count']}, Time: {scrapegraph_result['duration']:.1f}s")
        
        # Calculate metrics
        our_completeness = calculate_completeness(our_result['items'], site['fields'])
        scrapegraph_completeness = calculate_completeness(scrapegraph_result.get('items', []), site['fields'])
        
        result = {
            'site_name': site['site_name'],
            'category': site['category'],
            'our_items': our_result['item_count'],
            'our_time': our_result['duration'],
            'our_completeness': our_completeness,
            'scrapegraph_items': scrapegraph_result['item_count'],
            'scrapegraph_time': scrapegraph_result['duration'],
            'scrapegraph_completeness': scrapegraph_completeness,
        }
        
        results.append(result)
        
        # Print comparison
        if SCRAPEGRAPH_AVAILABLE:
            item_delta = our_result['item_count'] - scrapegraph_result['item_count']
            time_delta = our_result['duration'] - scrapegraph_result['duration']
            qual_delta = our_completeness - scrapegraph_completeness
            
            print(f"\n📊 Comparison:")
            print(f"   Items: {item_delta:+d} ({(item_delta/max(scrapegraph_result['item_count'],1)*100):+.0f}%)")
            print(f"   Speed: {time_delta:+.1f}s ({(our_result['duration']/max(scrapegraph_result['duration'],1)):.2f}x)")
            print(f"   Quality: {qual_delta:+.1f}%")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    total_our_items = sum(r['our_items'] for r in results)
    total_scrapegraph_items = sum(r['scrapegraph_items'] for r in results)
    total_our_time = sum(r['our_time'] for r in results)
    total_scrapegraph_time = sum(r['scrapegraph_time'] for r in results)
    avg_our_completeness = sum(r['our_completeness'] for r in results) / len(results)
    avg_scrapegraph_completeness = sum(r['scrapegraph_completeness'] for r in results) / len(results) if SCRAPEGRAPH_AVAILABLE else 0
    
    print("📋 Results by Site:\n")
    for r in results:
        winner_items = '🔵' if r['our_items'] >= r['scrapegraph_items'] else '🟢'
        winner_speed = '🔵' if r['our_time'] <= r['scrapegraph_time'] else '🟢'
        winner_quality = '🔵' if r['our_completeness'] >= r['scrapegraph_completeness'] else '🟢'
        
        print(f"  {r['site_name']}:")
        print(f"    Items: {winner_items} {r['our_items']} vs {r['scrapegraph_items']}")
        print(f"    Speed: {winner_speed} {r['our_time']:.1f}s vs {r['scrapegraph_time']:.1f}s")
        print(f"    Quality: {winner_quality} {r['our_completeness']:.0f}% vs {r['scrapegraph_completeness']:.0f}%")
        print()
    
    print("🏆 Overall Statistics:\n")
    print(f"  Total Items:")
    print(f"    Universal Scraper: {total_our_items}")
    print(f"    ScrapeGraphAI: {total_scrapegraph_items}")
    if total_scrapegraph_items > 0:
        print(f"    Delta: {total_our_items - total_scrapegraph_items:+d} ({(total_our_items/total_scrapegraph_items-1)*100:+.0f}%)")
    print()
    print(f"  Total Time:")
    print(f"    Universal Scraper: {total_our_time:.1f}s")
    print(f"    ScrapeGraphAI: {total_scrapegraph_time:.1f}s")
    if total_scrapegraph_time > 0:
        print(f"    Ratio: {total_our_time/total_scrapegraph_time:.2f}x")
    print()
    print(f"  Average Completeness:")
    print(f"    Universal Scraper: {avg_our_completeness:.1f}%")
    print(f"    ScrapeGraphAI: {avg_scrapegraph_completeness:.1f}%")
    print(f"    Delta: {avg_our_completeness - avg_scrapegraph_completeness:+.1f}%")
    
    # Save results
    with open('quick_competitive_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_our_items': total_our_items,
                'total_scrapegraph_items': total_scrapegraph_items,
                'total_our_time': total_our_time,
                'total_scrapegraph_time': total_scrapegraph_time,
                'avg_our_completeness': avg_our_completeness,
                'avg_scrapegraph_completeness': avg_scrapegraph_completeness
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to quick_competitive_results.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())


