#!/usr/bin/env python3
"""
Test the Apify Actor V2 locally before deploying
"""

import asyncio
import os
import sys
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up mock Apify environment
os.environ['APIFY_IS_AT_HOME'] = 'false'
os.environ['APIFY_DEFAULT_DATASET_ID'] = 'test-dataset'
os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'] = 'test-kvs'
os.environ['APIFY_TOKEN'] = 'test-token'

# Set OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')

# Mock input - Reddit test
TEST_INPUT = {
    "mode": "scrape_only",
    "urls": [{"url": "https://www.reddit.com/r/webscraping/"}],
    "fields": ["title", "author", "upvotes", "comments"],
    "browserConfig": {
        "useCamoufox": True,
        "headless": True,
        "waitForNetworkIdle": True,
        "captureApiRequests": True
    },
    "proxyConfiguration": None,  # Disable proxy for local testing
    "apiKeys": {
        "openaiApiKey": OPENAI_API_KEY
    },
    "crawlConfig": {
        "maxDepth": 0,
        "maxPages": 1,
        "handlePagination": False,
        "discoverApis": True
    },
    "debugMode": True
}

async def main():
    """Run the actor locally"""
    print("🧪 Testing Actor V2 Locally")
    print("=" * 80)
    print(f"📋 Test Input:")
    print(json.dumps(TEST_INPUT, indent=2))
    print("=" * 80)
    
    # Import after setting env vars
    try:
        from universal_scraper.apify.actor_v2 import run_apify_actor, parse_input, execute_workflow
        
        # Mock Actor methods
        import unittest.mock as mock
        
        # Track pushed data
        pushed_data = []
        
        async def mock_get_input():
            return TEST_INPUT
        
        async def mock_push_data(data):
            pushed_data.append(data)
            print(f"📤 Pushed data: {len(data) if isinstance(data, list) else 1} items")
        
        async def mock_set_status_message(msg):
            print(f"📢 Status: {msg}")
        
        # Mock Actor class
        with mock.patch('universal_scraper.apify.actor_v2.Actor') as MockActor:
            # Setup mock
            MockActor.get_input = mock_get_input
            MockActor.push_data = mock_push_data
            MockActor.set_status_message = mock_set_status_message
            
            # Mock create_proxy_configuration to return None for local testing
            async def mock_create_proxy():
                return None
            MockActor.create_proxy_configuration = mock_create_proxy
            
            # Parse input
            print("\n🔍 Parsing input...")
            config = parse_input(TEST_INPUT)
            print(f"✅ Parsed config:")
            print(f"   Mode: {config['workflow_config'].mode.value}")
            print(f"   URLs: {len(config['urls'])}")
            print(f"   Fields: {config['fields']}")
            print(f"   Browser: {'Camoufox' if config['use_camoufox'] else 'Playwright'}")
            print(f"   Proxy: {'Enabled' if config['proxy_config'] else 'Disabled'}")
            
            # Execute workflow
            print("\n🚀 Executing workflow...")
            result = await execute_workflow(config)
            
            # Print results
            print("\n" + "=" * 80)
            print("✅ Actor completed successfully!")
            print("=" * 80)
            print(f"📊 Results:")
            print(f"   Total items: {result.get('total_items', 0)}")
            print(f"   Successful URLs: {result.get('scrape_metadata', {}).get('successful', 0)}")
            print(f"   Failed URLs: {result.get('scrape_metadata', {}).get('failed', 0)}")
            print(f"   Duration: {result.get('workflow_metadata', {}).get('duration_seconds', 0):.2f}s")
            
            if result.get('data'):
                print(f"\n📋 Sample items (first 3):")
                for i, item in enumerate(result['data'][:3], 1):
                    print(f"\n   Item {i}:")
                    for key, value in item.items():
                        print(f"      {key}: {value}")
            
            print("\n" + "=" * 80)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())







