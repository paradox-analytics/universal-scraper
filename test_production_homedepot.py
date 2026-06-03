#!/usr/bin/env python3
"""
Test the production API with Home Depot URL to debug the Playwright Sync API error
"""
import requests
import json

# Production API URL
API_URL = "https://universal-scraper-api-r3crozpq7q-uc.a.run.app"

# Home Depot product URL
url = "https://www.homedepot.com/p/GE-27-cu-ft-French-Door-Refrigerator-in-Fingerprint-Resistant-Stainless-with-Internal-Dispenser-ENERGY-STAR-GNE27JYMFS/320243591"

# Test the /scrape endpoint
print(f"🧪 Testing production API: {API_URL}")
print(f"📍 URL: {url}\n")

try:
    response = requests.post(
        f"{API_URL}/scrape",
        json={
            "url": url,
            "fields": ["title", "price", "brand"]
        },
        timeout=120
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Success!")
        print(f"Extracted items: {len(data.get('items', []))}")
        
        if 'metadata' in data and 'unblocker_log' in data['metadata']:
            print("\n📋 Unblocker Log:")
            for entry in data['metadata']['unblocker_log']:
                print(f"  {entry.get('message', entry)}")
        
        if data.get('items'):
            print("\n📦 First item:")
            print(json.dumps(data['items'][0], indent=2))
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"\n❌ Request failed: {e}")
