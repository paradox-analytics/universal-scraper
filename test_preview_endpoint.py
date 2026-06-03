#!/usr/bin/env python3
"""
Test the preview endpoint to verify browser rendering works correctly.
"""
import asyncio
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.main import app
from fastapi.testclient import TestClient

def test_preview_endpoint():
    """Test the preview endpoint with Product Hunt URL"""
    print("🧪 Testing preview endpoint with Product Hunt URL...\n")
    
    client = TestClient(app)
    
    # Test request
    request_data = {
        "url": "https://www.producthunt.com/categories/vibe-coding",
        "browser_timeout": 90000,
        "proxy_config": None  # No proxy for this test
    }
    
    print(f"📤 Request: {json.dumps(request_data, indent=2)}\n")
    
    try:
        response = client.post("/api/v1/preview", json=request_data)
        
        print(f"📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            html_length = len(data.get('html', ''))
            print(f"✅ Success!")
            print(f"   HTML length: {html_length} bytes")
            print(f"   Method: {data.get('method', 'N/A')}")
            print(f"   Final URL: {data.get('final_url', 'N/A')}")
            print(f"   Fallback reason: {data.get('fallback_reason', 'None')}")
            
            # Check if HTML contains JavaScript-rendered content
            html_preview = data.get('html', '')[:500]
            if 'producthunt' in html_preview.lower() or 'vibe-coding' in html_preview.lower():
                print(f"   ✅ HTML appears to contain rendered content")
            else:
                print(f"   ⚠️  HTML preview: {html_preview[:200]}...")
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preview_endpoint()



