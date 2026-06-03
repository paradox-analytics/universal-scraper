#!/usr/bin/env python3
"""
Test Apify Actor locally using the Apify SDK (no deployment needed)

This uses the real Apify SDK in local mode, simulating the exact
same environment as the deployed actor without using credits.
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Ensure we can import the actor module
actor_dir = Path(__file__).parent / "universal_scraper" / "apify"
sys.path.insert(0, str(actor_dir))

# Set environment variables for local Apify SDK
os.environ['APIFY_LOCAL_STORAGE_DIR'] = './apify_storage_local'
os.environ['APIFY_TOKEN'] = 'dummy_token_for_local_testing'
os.environ['ACTOR_TEST_MODE'] = '1'  # Prevent actor.py from auto-executing

# Import Apify SDK
try:
    from apify import Actor
except ImportError:
    print("❌ Apify SDK not installed. Install with: pip install apify")
    sys.exit(1)

# Test input (same as Apify test)
TEST_INPUT = {
  "apiKeys": {
    "openaiApiKey": "REDACTED_OPENAI_KEY_1"
  },
  "browserConfig": {
    "useCamoufox": True,
    "headless": True,
    "waitForNetworkIdle": True
  },
  "crawlConfig": {
    "maxDepth": 0,
    "maxPages": 1,
    "handlePagination": False
  },
  "debugMode": True,
  "fields": [
    "title",
    "price",
    "condition",
    "shipping"
  ],
  "mode": "scrape_only",
  "proxyConfiguration": {
    "useApifyProxy": True,
    "apifyProxyGroups": [
      "RESIDENTIAL"
    ]
  },
  "urls": [
    {
      "url": "https://www.ebay.com/sch/i.html?_nkw=laptop"
    }
  ],
  "startUrls": [],
  "searchConfig": {
    "strategy": "auto",
    "maxDepth": 4,
    "resultLimit": 0
  },
  "schemaConfig": {
    "useSchema": False,
    "schemaType": "auto",
    "strictSchema": False
  },
  "outputFormat": "json",
  "maxConcurrency": 10
}


async def main():
    """Run the actor locally with Apify SDK"""
    print("="*80)
    print("🧪 LOCAL APIFY ACTOR TEST (using real Apify SDK)")
    print("="*80)
    print("📁 Local storage: ./apify_storage_local")
    print("🔑 Using dummy token (local mode)")
    print()
    
    # Initialize Actor in local mode
    async with Actor:
        print("✅ Actor initialized (local mode)")
        
        # Set the input (simulates Actor.get_input())
        await Actor.set_value('INPUT', TEST_INPUT)
        print(f"✅ Input set: {len(json.dumps(TEST_INPUT))} bytes")
        print()
        
        # Import the actor's parse and execute functions directly
        print("🚀 Running actor logic...\n")
        
        # Import actor module
        import actor as actor_module
        
        try:
            # Get input
            actor_input = await Actor.get_input() or {}
            print(f"📍 Got input: {bool(actor_input)}")
            
            # Parse configuration
            print("📍 Parsing config...")
            config = actor_module.parse_input(actor_input)
            print("📍 Config parsed")
            
            # Execute workflow
            print("📍 Executing workflow...")
            result = await actor_module.execute_workflow(config)
            print(f"📍 Workflow complete, items={len(result.get('data', []))}")
            
            # Save results
            print(f"📍 Pushing {len(result.get('data', []))} items to dataset...")
            await Actor.push_data(result.get('data', []))
            
            # Save metadata
            output_metadata = {
                'mode': result.get('mode'),
                'total_items': result.get('total_items', 0),
                'urls_discovered': result.get('crawl_metadata', {}).get('urls_discovered', 0),
                'workflow_metadata': result.get('workflow_metadata', {}),
                'crawl_metadata': result.get('crawl_metadata', {}),
                'scrape_metadata': result.get('scrape_metadata', {})
            }
            await Actor.set_value('OUTPUT_METADATA', output_metadata)
            
            print("\n" + "="*80)
            print("✅ ACTOR TEST COMPLETE")
            print("="*80)
            print(f"📊 Total items collected: {len(result.get('data', []))}")
            
            if result.get('data'):
                print("\n📋 First item sample:")
                print(json.dumps(result['data'][0], indent=2))
            
            print(f"\n💾 Metadata: {json.dumps(output_metadata, indent=2)}")
            
        except Exception as e:
            print(f"\n❌ LOCAL ACTOR TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    # Clean up previous local storage
    import shutil
    storage_dir = Path('./apify_storage_local')
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
        print("🧹 Cleaned previous local storage")
    
    asyncio.run(main())
