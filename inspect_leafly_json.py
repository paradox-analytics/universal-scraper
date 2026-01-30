"""
Inspect captured JSON from Leafly to understand structure
"""

import asyncio
import os
import json
from universal_scraper.core.hybrid_fetcher import HybridFetcher

async def inspect():
    """Fetch and inspect JSON"""
    fetcher = HybridFetcher(use_camoufox=True, headless=True)
    
    url = "https://www.leafly.com/dispensary-info/seven-point/menu"
    print(f"Fetching: {url}\n")
    
    result = await fetcher.fetch(url)
    
    json_blobs = result.get('json_data', [])
    
    print(f"Captured {len(json_blobs)} JSON blobs\n")
    print("="*80)
    
    for i, blob in enumerate(json_blobs, 1):
        print(f"\n📦 BLOB {i}:")
        print("-" * 80)
        
        # Pretty print first 100 lines
        json_str = json.dumps(blob, indent=2)
        lines = json_str.split('\n')
        
        print(f"Total lines: {len(lines)}")
        print(f"First 100 lines:\n")
        print('\n'.join(lines[:100]))
        
        if len(lines) > 100:
            print(f"\n... ({len(lines) - 100} more lines)")
        
        print("\n" + "="*80)
        
        # Show keys at root level
        if isinstance(blob, dict):
            print(f"\nRoot keys: {list(blob.keys())}")
            
            # Find arrays recursively
            def find_arrays(obj, path="root", depth=0, max_depth=5):
                if depth > max_depth:
                    return []
                
                arrays = []
                if isinstance(obj, list) and len(obj) > 0:
                    sample = obj[0] if obj else None
                    arrays.append((path, len(obj), type(sample).__name__))
                    if isinstance(sample, dict):
                        arrays.append((f"{path}[0] keys", list(sample.keys())[:10], ""))
                elif isinstance(obj, dict):
                    for key, value in obj.items():
                        arrays.extend(find_arrays(value, f"{path}.{key}", depth + 1))
                
                return arrays
            
            arrays = find_arrays(blob)
            if arrays:
                print(f"\nArrays found:")
                for path, size, typ in arrays:
                    if isinstance(size, list):
                        print(f"  {path}: {size}")
                    else:
                        print(f"  {path}: {size} items of type {typ}")
        
        print("\n" + "="*80)
    
    await fetcher.close()

if __name__ == '__main__':
    asyncio.run(inspect())




