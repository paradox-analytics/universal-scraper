#!/usr/bin/env python3
"""
Test Phase 1 Optimizations on Metacritic Movie Preview Page
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.scraper import UniversalScraper
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_metacritic():
    """Test Metacritic movie preview page"""
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    url = "https://www.metacritic.com/pictures/november-2025-movie-preview-wicked-hamnet-running-man/"
    fields = ["title", "metascore", "director", "release_date", "description"]
    
    print("="*80)
    print("🧪 PHASE 1 OPTIMIZATION TEST: Metacritic Movie Preview")
    print("="*80)
    print(f"   URL: {url}")
    print(f"   Fields: {', '.join(fields)}")
    print("="*80)
    print()
    
    scraper = UniversalScraper(
        api_key=api_key,
        use_direct_llm=True,
        enable_cache=True,
        browser_timeout=30000  # 30 second timeout
    )
    
    start_time = time.time()
    
    try:
        # Add timeout wrapper
        result = await asyncio.wait_for(
            scraper.scrape(url=url, fields=fields),
            timeout=120.0  # 2 minute overall timeout
        )
        
        execution_time = time.time() - start_time
        
        # Check results
        items = result.get('data', [])
        metadata = result.get('metadata', {})
        source = result.get('source', 'unknown')
        early_exit = metadata.get('early_exit', False)
        
        print(f"\n✅ Extraction Complete:")
        print(f"   Items extracted: {len(items)}")
        print(f"   Extraction source: {source}")
        print(f"   Execution time: {execution_time:.2f}s")
        print(f"   Early exit: {'✅ YES' if early_exit else '❌ NO'}")
        
        if early_exit:
            print(f"   ⚡ PHASE 1 OPTIMIZATION WORKING: Early exit triggered!")
            print(f"   Time saved: ~{execution_time * 0.2:.1f}-{execution_time * 0.3:.1f}s (estimated)")
        
        # Show sample items
        if items:
            print(f"\n   Sample items (first 5):")
            for i, item in enumerate(items[:5], 1):
                print(f"\n   {i}. Movie:")
                for field in fields:
                    value = item.get(field, 'N/A')
                    if value and value != 'N/A':
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"      {field}: {value}")
        else:
            print("\n   ⚠️  No items extracted")
        
        # Show metadata details
        print(f"\n   Metadata:")
        print(f"      Pagination detected: {metadata.get('pagination_detected', 'None')}")
        print(f"      Total pages scraped: {metadata.get('total_pages_scraped', 1)}")
        if 'direct_llm_quality' in metadata:
            print(f"      Direct LLM quality: {metadata['direct_llm_quality']:.1f}%")
        
        print("\n" + "="*80)
        
        # Save results to file
        output_file = "metacritic_test_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'url': url,
                'fields': fields,
                'items': items,
                'metadata': metadata,
                'source': source,
                'early_exit': early_exit,
                'execution_time': execution_time
            }, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {output_file}")
        print("="*80)
        
    except asyncio.TimeoutError:
        print(f"\n❌ Test timed out after 2 minutes")
        print("   This might indicate:")
        print("   - Website is blocking the scraper")
        print("   - Network issues")
        print("   - Page is taking too long to load")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_metacritic())

