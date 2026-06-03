#!/usr/bin/env python3
"""
Local test for Apify actor
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the apify directory to the path
sys.path.insert(0, str(Path(__file__).parent / "universal_scraper" / "apify"))

# Mock Apify Actor for local testing
class MockActor:
    """Mock Apify Actor for local testing"""
    
    def __init__(self):
        self.data = []
        self.values = {}
    
    async def __aenter__(self):
        print("📍 MockActor: Entering context")
        return self
    
    async def __aexit__(self, *args):
        print("📍 MockActor: Exiting context")
        return None
    
    async def get_input(self):
        """Return test input"""
        return TEST_INPUT
    
    async def push_data(self, item):
        """Store data"""
        self.data.append(item)
        print(f"📦 Pushed item: {json.dumps(item, indent=2)}")
    
    async def set_value(self, key, value):
        """Store value"""
        self.values[key] = value
        print(f"💾 Set {key}: {value}")
    
    @staticmethod
    async def create_proxy_configuration(**kwargs):
        """Mock proxy configuration"""
        print(f"🔗 Mock proxy configuration: {kwargs}")
        
        class MockProxyConfig:
            async def new_url(self):
                # Return None to simulate no external proxy access
                # (matching the user's Apify plan)
                return None
        
        return MockProxyConfig()


# Test input (user provided)
TEST_INPUT = {
    "apiKeys": {
        "openaiApiKey": "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"
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
    """Test the actor locally"""
    print("="*80)
    print("🧪 LOCAL ACTOR TEST")
    print("="*80)
    
    # Mock the Apify module BEFORE any imports
    from unittest.mock import MagicMock
    
    # Create mock Actor
    mock_actor = MockActor()
    
    # Inject mock into sys.modules BEFORE importing actor
    apify_mock = MagicMock()
    apify_mock.Actor = mock_actor
    sys.modules['apify'] = apify_mock
    
    # Prevent module-level execution by temporarily setting a flag
    os.environ['ACTOR_TEST_MODE'] = '1'
    
    # Now import the actor module (it will use our mock)
    import actor
    
    # Ensure APIFY_AVAILABLE is True
    actor.APIFY_AVAILABLE = True
    actor.Actor = mock_actor
    
    # Run the actor's main function
    print("\n🚀 Running actor.main()...\n")
    
    try:
        await actor.main()
        
        print("\n" + "="*80)
        print("✅ ACTOR TEST COMPLETE")
        print("="*80)
        print(f"📊 Total items collected: {len(mock_actor.data)}")
        print(f"💾 Values stored: {list(mock_actor.values.keys())}")
        
        if mock_actor.data:
            print(f"\n📋 First item sample:")
            print(json.dumps(mock_actor.data[0], indent=2))
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ACTOR TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Set environment variable for OpenAI
    os.environ['OPENAI_API_KEY'] = TEST_INPUT['apiKeys']['openaiApiKey']
    
    # Run the test
    asyncio.run(main())
