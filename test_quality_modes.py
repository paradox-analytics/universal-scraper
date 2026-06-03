#!/usr/bin/env python3
"""
Test DirectLLMExtractor with different quality modes
Compare conservative (like ScrapeGraphAI) vs balanced vs aggressive
"""
import asyncio
import os
import sys
from pathlib import Path
import json

script_dir = Path(__file__).parent.absolute()
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from universal_scraper.core.direct_llm_extractor import DirectLLMExtractor
from universal_scraper.core.hybrid_fetcher import HybridFetcher
from universal_scraper.core.html_cleaner import SmartHTMLCleaner


async def test_quality_modes():
    """Test all three quality modes on Amazon"""
    print("\n" + "="*100)
    print("🔬 QUALITY MODE COMPARISON TEST")
    print("="*100)
    print()
    print("Testing DirectLLMExtractor with three quality modes:")
    print("  • conservative: ≥70% fields filled (like ScrapeGraphAI)")
    print("  • balanced: ≥50% fields filled (default)")
    print("  • aggressive: ≥30% fields filled (maximum extraction)")
    print()
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Test URL
    url = "https://www.amazon.com/s?k=laptop"
    fields = ["product_title", "price", "rating"]
    
    # Fetch HTML once
    print("📥 Fetching HTML from Amazon...")
    fetcher = HybridFetcher(proxy_config=None, headless=True, use_camoufox=True, enable_cache=False)
    result = await fetcher.fetch(url)
    html = result['html']
    print(f"✅ Fetched {len(html):,} bytes via {result.get('fetch_method')}")
    print()
    
    # Clean HTML
    print("🧹 Cleaning HTML...")
    cleaner = SmartHTMLCleaner()
    cleaned_result = cleaner.clean(html)
    cleaned_html = cleaned_result['html']
    print(f"✅ Cleaned: {len(html):,} → {len(cleaned_html):,} bytes ({cleaned_result['reduction_percent']:.1f}% reduction)")
    print()
    
    # Test each quality mode
    modes = ['conservative', 'balanced', 'aggressive']
    results = {}
    
    for mode in modes:
        print("\n" + "="*100)
        print(f"🎯 TESTING: {mode.upper()} MODE")
        print("="*100)
        print()
        
        # Create extractor with this quality mode
        extractor = DirectLLMExtractor(
            api_key=api_key,
            quality_mode=mode
        )
        
        # Extract data
        print(f"🤖 Extracting with {mode} quality mode...")
        items = await extractor.extract(cleaned_html, fields)
        
        # Analyze results
        print()
        print("📊 RESULTS:")
        print(f"  Items extracted: {len(items)}")
        
        if items:
            # Calculate field completeness
            for field in fields:
                filled = sum(1 for item in items if item.get(field) and str(item.get(field)).strip())
                empty = len(items) - filled
                fill_rate = (filled / len(items)) * 100
                print(f"  {field}: {fill_rate:.1f}% filled ({filled}/{len(items)}, {empty} empty)")
            
            # Calculate average fill rate
            total_filled = sum(
                1 for item in items
                for field in fields
                if item.get(field) and str(item.get(field)).strip()
            )
            total_fields = len(items) * len(fields)
            avg_fill_rate = (total_filled / total_fields) * 100 if total_fields > 0 else 0
            print(f"  Average completeness: {avg_fill_rate:.1f}%")
            
            # Show sample items
            print()
            print("Sample items:")
            for i, item in enumerate(items[:3], 1):
                print(f"\nItem {i}:")
                for key, value in item.items():
                    value_str = str(value)[:80] if value else "None"
                    value_type = type(value).__name__
                    print(f"  • {key} ({value_type}): {value_str}")
            
            # Check data types
            print()
            print("Data type analysis:")
            type_counts = {}
            for item in items:
                for key, value in item.items():
                    value_type = type(value).__name__
                    type_counts.setdefault(key, {}).setdefault(value_type, 0)
                    type_counts[key][value_type] += 1
            
            for field in fields:
                if field in type_counts:
                    types_str = ", ".join(f"{t}: {c}" for t, c in type_counts[field].items())
                    print(f"  {field}: {types_str}")
            
            # Store results
            results[mode] = {
                'count': len(items),
                'avg_fill_rate': avg_fill_rate,
                'items': items[:5]  # Store first 5 for comparison
            }
        else:
            print("  ❌ No items extracted")
            results[mode] = {'count': 0, 'avg_fill_rate': 0, 'items': []}
        
        print()
    
    # Comparison summary
    print("\n" + "="*100)
    print("📊 QUALITY MODE COMPARISON SUMMARY")
    print("="*100)
    print()
    
    print("| Mode | Items | Avg Fill Rate | Quality × Quantity |")
    print("|------|-------|---------------|-------------------|")
    
    for mode in modes:
        r = results[mode]
        quality_quantity = (r['avg_fill_rate'] / 100) * r['count']
        print(f"| {mode:<12} | {r['count']:>5} | {r['avg_fill_rate']:>13.1f}% | {quality_quantity:>17.1f} |")
    
    print()
    print("Analysis:")
    print()
    
    # Determine which mode is best for different use cases
    conservative = results['conservative']
    balanced = results['balanced']
    aggressive = results['aggressive']
    
    print(f"✅ CONSERVATIVE mode (like ScrapeGraphAI):")
    print(f"   • {conservative['count']} items extracted")
    print(f"   • {conservative['avg_fill_rate']:.1f}% average completeness")
    print(f"   • Best for: Financial data, legal docs, precision analytics")
    print(f"   • Quality score: {(conservative['avg_fill_rate']/100) * conservative['count']:.1f}")
    print()
    
    print(f"⚖️  BALANCED mode (default):")
    print(f"   • {balanced['count']} items extracted")
    print(f"   • {balanced['avg_fill_rate']:.1f}% average completeness")
    print(f"   • Best for: General use, most applications")
    print(f"   • Quality score: {(balanced['avg_fill_rate']/100) * balanced['count']:.1f}")
    print()
    
    print(f"🚀 AGGRESSIVE mode (maximum extraction):")
    print(f"   • {aggressive['count']} items extracted")
    print(f"   • {aggressive['avg_fill_rate']:.1f}% average completeness")
    print(f"   • Best for: Market research, data aggregation, large-scale scraping")
    print(f"   • Quality score: {(aggressive['avg_fill_rate']/100) * aggressive['count']:.1f}")
    print()
    
    # Recommendation
    if balanced['count'] > conservative['count'] * 1.5 and balanced['avg_fill_rate'] > 70:
        print("🏆 RECOMMENDATION: Use BALANCED mode (best compromise)")
    elif conservative['avg_fill_rate'] > 90:
        print("🏆 RECOMMENDATION: Use CONSERVATIVE mode (highest quality)")
    else:
        print("🏆 RECOMMENDATION: Use AGGRESSIVE mode (maximum data collection)")
    
    print()
    
    # Save results to file
    output_file = "quality_modes_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            'url': url,
            'fields': fields,
            'results': {
                mode: {
                    'count': r['count'],
                    'avg_fill_rate': r['avg_fill_rate'],
                    'sample_items': r['items']
                }
                for mode, r in results.items()
            }
        }, f, indent=2)
    
    print(f"📁 Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    asyncio.run(test_quality_modes())



