#!/usr/bin/env python3
"""
Test browser rendering in Cloud Run
"""
import requests
import json
import os
from getpass import getpass

# Cloud Run service URL
CLOUD_RUN_URL = os.getenv('CLOUD_RUN_URL', 'https://universal-scraper-api-968720932091.us-central1.run.app')
TEST_URL = "https://www.producthunt.com/categories/vibe-coding"

def test_preview_endpoint():
    """Test the preview endpoint in Cloud Run"""
    print("Testing Browser Rendering in Cloud Run")
    print("=" * 60)
    print(f"Cloud Run URL: {CLOUD_RUN_URL}")
    print(f"Test URL: {TEST_URL}")
    print()
    
    # Get API key from environment
    api_key = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️ No API key found. Set API_KEY or OPENAI_API_KEY environment variable.")
        print("   Testing without authentication (may fail if auth is required)...")
        api_key = ""
    
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': api_key
    }
    
    payload = {
        'url': TEST_URL,
        'browser_timeout': 60000
    }
    
    print("1. Calling preview endpoint...")
    try:
        response = requests.post(
            f"{CLOUD_RUN_URL}/api/v1/preview",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ❌ Error: {response.text}")
            return False
        
        data = response.json()
        
        print("\n2. Analyzing response...")
        print(f"   Success: {data.get('success')}")
        print(f"   Fetch Method: {data.get('fetch_method', 'unknown')}")
        print(f"   Browser Rendering Failed: {data.get('browser_rendering_failed', False)}")
        print(f"   Fallback Reason: {data.get('fallback_reason', 'N/A')}")
        
        html = data.get('html', '')
        html_size = len(html)
        print(f"   HTML Size: {html_size:,} bytes")
        
        # Check for Product Hunt specific content that only appears with JS rendering
        html_lower = html.lower()
        
        print("\n3. Checking for JS-rendered content...")
        
        # Product Hunt specific indicators that require JS
        indicators = {
            'product listings': ['producthunt', 'upvote', 'maker', 'hunter'],
            'react content': ['__next_data__', 'react', 'data-react'],
            'dynamic content': ['loading', 'hydrating', 'rendering']
        }
        
        found_indicators = []
        for category, terms in indicators.items():
            for term in terms:
                if term in html_lower:
                    found_indicators.append(f"{category}: {term}")
                    break
        
        if found_indicators:
            print(f"   ✅ Found JS indicators: {', '.join(found_indicators)}")
        else:
            print("   ⚠️ No clear JS-rendered content indicators found")
        
        # Check HTML size - Product Hunt with JS should be 500KB+
        if html_size > 500000:
            print(f"   ✅ Large HTML size ({html_size:,} bytes) suggests full rendering")
        elif html_size > 50000:
            print(f"   ⚠️ Moderate HTML size ({html_size:,} bytes) - may be partially rendered")
        else:
            print(f"   ❌ Small HTML size ({html_size:,} bytes) - likely static HTML only")
        
        # Check for common static HTML patterns
        if '<div id="__next">' in html or '<div id="root">' in html:
            # Check if these divs have content
            if html.count('product') > 10 or html.count('upvote') > 5:
                print("   ✅ React root divs contain substantial content")
            else:
                print("   ⚠️ React root divs found but may be empty/minimal")
        
        # Final verdict
        print("\n4. Verdict:")
        fetch_method = data.get('fetch_method', 'unknown')
        browser_failed = data.get('browser_rendering_failed', False)
        
        if fetch_method == 'browser' and not browser_failed:
            if html_size > 200000:
                print("   ✅ Browser rendering is working! Full JS-rendered content detected.")
                return True
            else:
                print("   ⚠️ Browser mode used but HTML seems small - may not be fully rendered")
                return False
        elif fetch_method == 'static' or fetch_method == 'static_fallback' or browser_failed:
            print("   ❌ Browser rendering failed - using static HTML fallback")
            print(f"   Reason: {data.get('fallback_reason', 'Unknown')}")
            return False
        else:
            print(f"   ⚠️ Unknown fetch method: {fetch_method}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_preview_endpoint()
    exit(0 if success else 1)

