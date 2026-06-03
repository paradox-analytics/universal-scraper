#!/usr/bin/env python3
"""
Test Web Unblocker via API endpoint locally
"""
import asyncio
import sys
import os
from fastapi.testclient import TestClient

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_endpoint():
    """Test the proxy test endpoint with Web Unblocker"""
    print("=" * 80)
    print("Testing Web Unblocker API Endpoint (Local)")
    print("=" * 80)
    
    # Import the app
    try:
        from api.main import app
        client = TestClient(app)
    except Exception as e:
        print(f"❌ Failed to import app: {e}")
        return False
    
    # Get API key from environment or use placeholder
    api_key = os.getenv('WEB_UNBLOCKER_API_KEY', 'test-key-placeholder')
    zone = os.getenv('WEB_UNBLOCKER_ZONE', 'web_unlocker1')
    
    print(f"\nAPI Key: {api_key[:20]}...")
    print(f"Zone: {zone}")
    print()
    
    # Test request payload
    payload = {
        "provider": "web_unlocker",
        "webUnblocker": {
            "apiKey": api_key,
            "zone": zone
        }
    }
    
    print("1. Calling /api/v1/proxy/test endpoint...")
    print(f"   Payload: provider=web_unlocker, zone={zone}")
    
    try:
        response = client.post("/api/v1/proxy/test", json=payload)
        
        print(f"\n2. Response:")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data.get('success')}")
            print(f"   Message: {data.get('message', 'N/A')}")
            
            if data.get('success'):
                print("\n" + "=" * 80)
                print("✅ API endpoint test PASSED!")
                print("   The async fix is working correctly.")
                print("=" * 80)
                return True
            else:
                print("\n" + "=" * 80)
                print("⚠️  API endpoint responded but test failed")
                print(f"   Message: {data.get('message')}")
                print("=" * 80)
                return False
        else:
            print(f"   ❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Note: This tests the endpoint structure, not actual Web Unblocker connection
    # For full test, you need valid API credentials
    success = test_api_endpoint()
    sys.exit(0 if success else 1)




