import requests
import json
import time

API_URL = "https://universal-scraper-api-968720932091.us-central1.run.app"

def test_smart_json_first():
    print("🚀 Testing Smart JSON-First Escalation...")
    
    # URL that might trigger escalation (needs JS for JSON or has JS indicators)
    # We'll use a product page from a site that uses React/Next.js
    url = "https://www.homedepot.com/p/Milwaukee-M18-18-Volt-Lithium-Ion-Cordless-Drill-Driver-Impact-Driver-Combo-Kit-2-Tool-with-Two-1-5Ah-Batteries-2691-22/100650378"
    fields = ["name", "price", "brand"]
    
    payload = {
        "url": url,
        "fields": fields,
        "use_cache": False
    }
    
    headers = {
        "X-API-Key": "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"
    }
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/scrape", json=payload, headers=headers)
    duration = time.time() - start_time
    
    print(f"⏱️ Request took {duration:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Request successful!")
        
        metadata = result.get("metadata", {})
        strategy = metadata.get("strategy", {})
        
        print(f"📊 Strategy Used: {json.dumps(strategy, indent=2)}")
        
        # Check if it escalated to browser
        if strategy.get("method") == "browser":
            print("🎯 SUCCESS: Escalated to browser as expected.")
        else:
            print(f"ℹ️ Used method: {strategy.get('method')}")
            
        print(f"📦 Data: {json.dumps(result.get('data', []), indent=2)[:500]}...")
    else:
        print(f"❌ Request failed with status {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_smart_json_first()
