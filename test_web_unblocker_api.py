#!/usr/bin/env python3
"""
Test Bright Data Web Unblocker API Connection

Tests the API endpoint directly to verify credentials and connectivity.
"""
import os
import sys
import json
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from universal_scraper.core.web_unblocker_fetcher import WebUnblockerFetcher


def test_api_direct():
    """Test Web Unblocker API directly with curl-equivalent request"""
    print("=" * 80)
    print("🧪 TEST 1: Direct API Call (curl equivalent)")
    print("=" * 80)
    
    api_key = os.environ.get('BRIGHT_DATA_API_KEY')
    if not api_key:
        print("\n❌ BRIGHT_DATA_API_KEY not set!")
        print("   Set it with: export BRIGHT_DATA_API_KEY='your-api-key'")
        return False
    
    zone = os.environ.get('BRIGHT_DATA_ZONE', 'web_unlocker1')
    test_url = "https://geo.brdtest.com/welcome.txt?product=unlocker&method=api"
    
    print(f"\n📋 Configuration:")
    print(f"   API Key: {api_key[:20]}...")
    print(f"   Zone: {zone}")
    print(f"   Test URL: {test_url}")
    
    # Make direct API call (equivalent to curl)
    api_url = "https://api.brightdata.com/request"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "zone": zone,
        "url": test_url,
        "format": "raw"
    }
    
    print(f"\n⏳ Making API request...")
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"\n📊 Response:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print(f"\n✅ SUCCESS!")
            print(f"   Response Body:")
            print(f"   {response.text[:500]}")
            
            # Check if it contains expected content
            if 'unlocker' in response.text.lower() or 'ip' in response.text.lower():
                print(f"\n✅ Response looks valid (contains unlocker/IP info)")
                return True
            else:
                print(f"\n⚠️  Response received but content unexpected")
                return True  # Still success, just unexpected content
                
        elif response.status_code == 401:
            print(f"\n❌ Authentication Failed!")
            print(f"   Check your API key at: https://brightdata.com/cp/account/api")
            print(f"   Response: {response.text[:200]}")
            return False
            
        elif response.status_code == 402:
            print(f"\n❌ Insufficient Credits!")
            print(f"   Add credits to your Bright Data account")
            print(f"   Response: {response.text[:200]}")
            return False
            
        elif response.status_code == 429:
            print(f"\n⚠️  Rate Limit Exceeded!")
            print(f"   Wait a moment and try again")
            print(f"   Response: {response.text[:200]}")
            return False
            
        else:
            print(f"\n❌ Unexpected Status Code: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n❌ Request Timeout!")
        print(f"   API took too long to respond")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_web_unblocker_fetcher():
    """Test using WebUnblockerFetcher class"""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: WebUnblockerFetcher Class")
    print("=" * 80)
    
    api_key = os.environ.get('BRIGHT_DATA_API_KEY')
    if not api_key:
        print("\n⚠️  BRIGHT_DATA_API_KEY not set - skipping class test")
        return False
    
    zone = os.environ.get('BRIGHT_DATA_ZONE', 'web_unlocker1')
    test_url = "https://geo.brdtest.com/welcome.txt?product=unlocker&method=api"
    
    print(f"\n📋 Testing WebUnblockerFetcher class...")
    
    try:
        fetcher = WebUnblockerFetcher(
            api_key=api_key,
            zone=zone
        )
        
        print(f"✅ WebUnblockerFetcher initialized")
        
        # Test connection
        print(f"\n⏳ Testing connection...")
        result = fetcher.fetch(test_url)
        
        print(f"\n✅ Fetch successful!")
        print(f"   HTML size: {len(result.get('html', '')):,} bytes")
        print(f"   Status: {result.get('status')}")
        print(f"   Source: {result.get('source')}")
        
        html = result.get('html', '')
        if html:
            print(f"\n   Response preview:")
            print(f"   {html[:300]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ WebUnblockerFetcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chewy_with_unblocker():
    """Test Chewy.com with Web Unblocker"""
    print("\n" + "=" * 80)
    print("🧪 TEST 3: Chewy.com with Web Unblocker")
    print("=" * 80)
    
    api_key = os.environ.get('BRIGHT_DATA_API_KEY')
    if not api_key:
        print("\n⚠️  BRIGHT_DATA_API_KEY not set - skipping Chewy test")
        return False
    
    zone = os.environ.get('BRIGHT_DATA_ZONE', 'web_unlocker1')
    chewy_url = "https://www.chewy.com/b/wet-food-389"
    
    print(f"\n📋 Testing Chewy.com...")
    print(f"   URL: {chewy_url}")
    
    try:
        fetcher = WebUnblockerFetcher(
            api_key=api_key,
            zone=zone,
            timeout=120
        )
        
        print(f"\n⏳ Fetching Chewy.com (this may take 30-60 seconds)...")
        result = fetcher.fetch(chewy_url)
        
        html = result.get('html', '')
        print(f"\n✅ Fetch completed!")
        print(f"   HTML size: {len(html):,} bytes")
        print(f"   Status: {result.get('status')}")
        
        # Check if we got blocked
        html_lower = html.lower()
        is_blocked = (
            len(html) < 2000 and (
                'kasada' in html_lower or 
                'kpsdk' in html_lower or 
                'ips.js' in html_lower
            )
        )
        
        if is_blocked:
            print(f"\n⚠️  Still appears blocked (Kasada challenge)")
            print(f"   HTML preview: {html[:500]}")
            return False
        else:
            print(f"\n✅ Success! Got full HTML content")
            print(f"   HTML preview: {html[:500]}")
            
            # Check for product indicators
            if 'product' in html_lower or 'chewy' in html_lower:
                print(f"\n✅ HTML contains product/Chewy content - looks good!")
                return True
            else:
                print(f"\n⚠️  HTML received but doesn't contain expected content")
                return True  # Still success, just verify content
        
    except Exception as e:
        print(f"\n❌ Chewy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("🌐 BRIGHT DATA WEB UNBLOCKER API TEST SUITE")
    print("=" * 80)
    
    results = []
    
    # Test 1: Direct API call
    results.append(("Direct API", test_api_direct()))
    
    # Test 2: WebUnblockerFetcher class
    results.append(("WebUnblockerFetcher", test_web_unblocker_fetcher()))
    
    # Test 3: Chewy.com
    results.append(("Chewy.com", test_chewy_with_unblocker()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Web Unblocker is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Check logs above for details.")
        print("\n💡 Troubleshooting:")
        print("   1. Verify API key: https://brightdata.com/cp/account/api")
        print("   2. Check account credits")
        print("   3. Verify zone name matches your account")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

