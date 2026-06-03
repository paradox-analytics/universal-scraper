#!/usr/bin/env python3
"""
Direct proxy test to isolate the 502 issue
Tests the Web Unblocker credentials directly with requests
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Try RESIDENTIAL proxy instead of Web Unblocker (which is rate-limited)
proxy_url = os.getenv('BRIGHT_DATA_PROXY_URL')  # residential_proxy2
print(f"🔐 Testing Residential Proxy (avoiding Web Unblocker rate limit): {proxy_url}")

# Configure proxy
proxies = {
    'http': proxy_url,
    'https': proxy_url
}

# Test URLs
test_urls = [
    ('Google', 'https://www.google.com'),
    ('Home Depot Product', 'https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591'),
]

for name, url in test_urls:
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"URL: {url[:80]}...")
    print(f"{'='*80}")
    
    try:
        response = requests.get(
            url,
            proxies=proxies,
            timeout=60,
            verify=False,  # Web Unblocker uses self-signed cert
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        )
        
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Content Length: {len(response.text)} bytes")
        print(f"📋 Headers: {dict(list(response.headers.items())[:5])}")
        
        if response.status_code == 200:
            print(f"✅ SUCCESS - Got valid response")
            # Check for GraphQL endpoints
            if 'graphql' in response.text.lower():
                print(f"🎯 GraphQL endpoints detected in HTML!")
        else:
            print(f"⚠️  Non-200 status")
            print(f"Preview: {response.text[:500]}")
            
    except requests.exceptions.ProxyError as e:
        print(f"❌ PROXY ERROR: {e}")
    except requests.exceptions.Timeout as e:
        print(f"❌ TIMEOUT: {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

print(f"\n{'='*80}")
print("Test Complete")
print(f"{'='*80}")
