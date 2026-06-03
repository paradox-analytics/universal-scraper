#!/usr/bin/env python3
"""
Test GPT-4o vs GPT-4o-mini for Code Generation

Compares code generation quality on Stack Overflow (previously failing with mini)
"""

import asyncio
import os
import sys
from universal_scraper import UniversalScraper

async def test_with_model(model_name: str, label: str):
    """Test scraping with specified model"""
    print(f"\n{'='*80}")
    print(f"Testing: {label}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    
    scraper = UniversalScraper(
        api_key=os.environ['OPENAI_API_KEY'],
        model_name=model_name,
        use_camoufox=True,
        headless=True,
        enable_auto_pagination=False
    )
    
    try:
        result = await scraper.scrape(
            url='https://stackoverflow.com/questions?tab=newest',
            fields=['title', 'votes', 'answers', 'views']
        )
        
        items = result.get('data', [])
        
        # Calculate quality
        if items:
            total_fields = len(items) * 4
            filled_fields = sum(
                1 for item in items 
                for v in item.values() 
                if v is not None and v != ''
            )
            quality = (filled_fields / total_fields * 100) if total_fields > 0 else 0
        else:
            quality = 0
        
        print(f"\n📊 Results:")
        print(f"   Items Extracted: {len(items)}")
        print(f"   Quality: {quality:.0f}%")
        
        if items:
            print(f"\n   Sample Items:")
            for i, item in enumerate(items[:3], 1):
                null_count = sum(1 for v in item.values() if v is None or v == '')
                print(f"   {i}. {item}")
                if null_count > 0:
                    null_fields = [k for k, v in item.items() if v is None or v == '']
                    print(f"      ⚠️  Null fields: {', '.join(null_fields)}")
        
        return {
            'model': model_name,
            'label': label,
            'items': len(items),
            'quality': quality,
            'success': len(items) >= 10 and quality >= 70
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return {
            'model': model_name,
            'label': label,
            'items': 0,
            'quality': 0,
            'success': False,
            'error': str(e)
        }
    finally:
        await scraper.close()

async def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              GPT-4o vs GPT-4o-mini Code Generation Test                   ║
╚═══════════════════════════════════════════════════════════════════════════╝

Test Site: Stack Overflow (failing with GPT-4o-mini)
Challenge: CSS selectors with escaped colons (li.h\\:bg-black-150)
Expected: GPT-4o should generate more robust code with better error handling

Cost Comparison:
  • GPT-4o-mini: ~$0.005 per scrape (fast, cheap, but lower quality)
  • GPT-4o:      ~$0.05 per scrape (10x more, but significantly better)
    """)
    
    # Test 1: GPT-4o-mini (baseline)
    result_mini = await test_with_model('gpt-4o-mini', 'GPT-4o-mini (Baseline)')
    
    # Brief pause
    print("\n⏸️  Waiting 3s before next test...")
    await asyncio.sleep(3)
    
    # Test 2: GPT-4o (our new default for code generation)
    result_4o = await test_with_model('gpt-4o', 'GPT-4o (Code Generation)')
    
    # Summary
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n📊 Results:")
    print(f"   GPT-4o-mini: {result_mini['items']} items, {result_mini['quality']:.0f}% quality")
    print(f"   GPT-4o:      {result_4o['items']} items, {result_4o['quality']:.0f}% quality")
    
    if result_4o['items'] > result_mini['items']:
        improvement = result_4o['items'] - result_mini['items']
        print(f"\n   ✅ GPT-4o extracted {improvement} MORE items (+{improvement/max(result_mini['items'],1)*100:.0f}%)")
    
    if result_4o['quality'] > result_mini['quality']:
        improvement = result_4o['quality'] - result_mini['quality']
        print(f"   ✅ GPT-4o has {improvement:.0f}% HIGHER quality")
    
    print(f"\n💰 Cost Analysis:")
    print(f"   GPT-4o-mini: ~$0.005 per scrape")
    print(f"   GPT-4o:      ~$0.05 per scrape (10x more expensive)")
    
    if result_4o['success'] and not result_mini['success']:
        print(f"\n🎯 Recommendation: GPT-4o is WORTH IT")
        print(f"   • GPT-4o-mini: Failed (0 items or low quality)")
        print(f"   • GPT-4o: Success ({result_4o['items']} items, {result_4o['quality']:.0f}% quality)")
        print(f"   • 10x cost increase for WORKING solution")
    elif result_4o['success'] and result_mini['success']:
        quality_improvement = result_4o['quality'] - result_mini['quality']
        if quality_improvement > 20:
            print(f"\n🎯 Recommendation: GPT-4o is WORTH IT")
            print(f"   • {quality_improvement:.0f}% quality improvement")
            print(f"   • More robust code, better error handling")
        else:
            print(f"\n⚠️  Recommendation: GPT-4o-mini might be sufficient")
            print(f"   • Quality improvement: {quality_improvement:.0f}% (small)")
            print(f"   • Consider keeping GPT-4o-mini to save costs")
    else:
        print(f"\n⚠️  Both models failed - deeper debugging needed")
    
    print(f"\n{'='*80}\n")
    
    return result_4o['success']

if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)






