"""
Test Enhanced Reinforcement System
===================================

Testing improvements:
1. Lower quality threshold (70% instead of 50%)
2. Per-field quality tracking
3. Field-specific feedback to LLM
"""

import asyncio
import os
from universal_scraper import UniversalScraper

async def test_site(url, fields, name):
    """Test a single site"""
    
    print(f"\n{'='*75}")
    print(f"🔍 Testing: {name}")
    print(f"{'='*75}")
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False
    )
    
    try:
        result = await scraper.scrape(url=url, fields=fields)
        items = result.get('data', [])
        
        if not items:
            print(f"   ❌ 0 items extracted")
            return {'name': name, 'items': 0, 'quality': 0, 'status': '❌'}
        
        # Calculate overall quality
        total_fields = len(items) * len(fields)
        filled_fields = sum(
            1 for item in items 
            for v in item.values() 
            if v not in (None, '', [])
        )
        quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        
        # Calculate per-field quality
        field_quality = {}
        for field in fields:
            filled = sum(1 for item in items if item.get(field) not in (None, '', []))
            field_quality[field] = (filled / len(items)) * 100
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Items: {len(items)}")
        print(f"   Overall Quality: {quality:.0f}%")
        print(f"\n   Per-Field Quality:")
        for field, field_qual in sorted(field_quality.items(), key=lambda x: x[1]):
            status = "✅" if field_qual >= 80 else "⚠️" if field_qual >= 50 else "❌"
            print(f"     {status} {field}: {field_qual:.0f}%")
        
        print(f"\n   Sample Items:")
        for i, item in enumerate(items[:3], 1):
            print(f"   {i}. {item}")
        
        # Determine status
        if quality >= 80:
            status = "✅"
            print(f"\n✅ SUCCESS!")
        elif quality >= 50:
            status = "⚠️"
            print(f"\n⚠️ PARTIAL - Need to improve quality")
        else:
            status = "❌"
            print(f"\n❌ FAILED")
        
        return {
            'name': name,
            'items': len(items),
            'quality': quality,
            'field_quality': field_quality,
            'status': status
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'name': name, 'items': 0, 'quality': 0, 'status': '❌'}
    
    finally:
        await scraper.close()


async def main():
    """Test enhanced reinforcement on problem sites"""
    
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║     Testing Enhanced Reinforcement System                                 ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print("✅ Quality threshold: 70% (was 50%)")
    print("✅ Per-field quality tracking")
    print("✅ Field-specific LLM feedback")
    print()
    
    # Test sites with known null field issues
    tests = [
        {
            'name': 'Stack Overflow',
            'url': 'https://stackoverflow.com/questions?tab=newest',
            'fields': ['title', 'votes']
        },
        {
            'name': 'GitHub Trending',
            'url': 'https://github.com/trending',
            'fields': ['repository', 'description', 'stars']
        },
        {
            'name': 'Hacker News',
            'url': 'https://news.ycombinator.com/',
            'fields': ['title', 'points', 'comments']
        }
    ]
    
    results = []
    for test in tests:
        result = await test_site(test['url'], test['fields'], test['name'])
        results.append(result)
    
    # Summary
    print(f"\n{'='*75}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*75}")
    print(f"{'Site':<25} {'Items':<10} {'Quality':<12} {'Status':<10}")
    print("-" * 75)
    
    success = 0
    partial = 0
    failed = 0
    
    for r in results:
        print(f"{r['name']:<25} {r['items']:<10} {r['quality']:.0f}%{'':<9} {r['status']:<10}")
        if r['status'] == '✅':
            success += 1
        elif r['status'] == '⚠️':
            partial += 1
        else:
            failed += 1
    
    print("-" * 75)
    print(f"✅ Success: {success}/{len(results)}")
    if partial > 0:
        print(f"⚠️  Partial: {partial}/{len(results)}")
    if failed > 0:
        print(f"❌ Failed:  {failed}/{len(results)}")
    print("=" * 75)
    
    # Detailed field analysis
    print("\n📋 FIELD-LEVEL ANALYSIS:")
    print("-" * 75)
    for r in results:
        if 'field_quality' in r and r['field_quality']:
            print(f"\n{r['name']}:")
            for field, quality in sorted(r['field_quality'].items(), key=lambda x: x[1]):
                status = "✅" if quality >= 80 else "⚠️" if quality >= 50 else "❌"
                print(f"  {status} {field}: {quality:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())






