import requests
import json
import os

API_BASE_URL = "https://universal-scraper-api-r3crozpq7q-uc.a.run.app"
API_KEY = "REDACTED_OPENAI_KEY_1"

def test_ph_extraction():
    url = f"{API_BASE_URL}/scrape"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    payload = {
        "url": "https://www.producthunt.com",
        "fields": ["name", "tagline", "votes"],
        "target": "products",
        "mode": "hybrid"
    }
    
    print(f"Testing extraction for: {payload['url']}")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("Success!")
        print(json.dumps(data.get("data", [])[:2], indent=2))
        
        # Check if we got actual data or Cloudflare challenge
        first_item = data.get("data", [])[0] if data.get("data") else {}
        if "verify you are human" in str(first_item).lower():
            print("❌ Still blocked by Cloudflare")
        else:
            print("✅ Successfully bypassed Cloudflare and extracted data!")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_ph_extraction()
