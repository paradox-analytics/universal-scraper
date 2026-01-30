#!/usr/bin/env python3
"""
Diagnostic: Check Premium Domain Configuration

Tests different domain formats to see which one works.
"""
import requests
import sys

def test_domain(domain):
    """Test a specific domain format"""
    proxy_url = "http://brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1:REDACTED_PROXY_PASS@brd.superproxy.io:33335"
    
    # Test URL
    test_url = f"https://{domain}/b/wet-food-389"
    
    try:
        response = requests.get(
            test_url,
            proxies={'http': proxy_url, 'https': proxy_url},
            timeout=30,
            verify=False
        )
        
        html = response.text
        is_premium_error = 'Premium permissions' in html or 'Premium domains' in html
        is_kasada = 'kasada' in html.lower() or 'kpsdk' in html.lower()
        is_success = len(html) > 10000 and not is_premium_error and not is_kasada
        
        return {
            'domain': domain,
            'status': response.status_code,
            'size': len(html),
            'premium_error': is_premium_error,
            'kasada': is_kasada,
            'success': is_success,
            'preview': html[:200]
        }
    except Exception as e:
        return {
            'domain': domain,
            'error': str(e),
            'success': False
        }

def main():
    print("=" * 80)
    print("🔍 DIAGNOSTIC: Testing Premium Domain Formats")
    print("=" * 80)
    
    # Test different domain formats
    domains_to_test = [
        'chewy.com',
        'www.chewy.com',
        'm.chewy.com',
    ]
    
    print(f"\n📋 Testing {len(domains_to_test)} domain formats...")
    
    results = []
    for domain in domains_to_test:
        print(f"\n⏳ Testing: {domain}")
        result = test_domain(domain)
        results.append(result)
        
        if result.get('success'):
            print(f"   ✅ SUCCESS! Size: {result['size']:,} bytes")
        elif result.get('premium_error'):
            print(f"   ⚠️  Premium permissions error")
        elif result.get('kasada'):
            print(f"   ⚠️  Kasada challenge detected")
        elif result.get('error'):
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ⚠️  Status: {result.get('status')}, Size: {result.get('size')} bytes")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.get('success')]
    premium_errors = [r for r in results if r.get('premium_error')]
    
    if successful:
        print(f"\n✅ Working domain(s):")
        for r in successful:
            print(f"   - {r['domain']} ({r['size']:,} bytes)")
    else:
        print(f"\n❌ No working domains found")
    
    if premium_errors:
        print(f"\n⚠️  Premium permissions needed for:")
        for r in premium_errors:
            print(f"   - {r['domain']}")
        print(f"\n💡 Action: Add these domains to Premium domains in Bright Data dashboard")
        print(f"   URL: https://brightdata.com/cp/zones/web_unlocker1/edit?id=REDACTED_CUSTOMER_ID")
    
    # Recommendations
    if not successful and premium_errors:
        print(f"\n💡 Recommendations:")
        print(f"   1. Verify Premium domains are enabled in dashboard")
        print(f"   2. Wait 5-15 minutes for changes to propagate")
        print(f"   3. Try adding both 'chewy.com' and 'www.chewy.com'")
        print(f"   4. Check if there's a 'Save' or 'Apply' button you need to click")
    
    return len(successful) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

