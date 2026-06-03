import asyncio
import json
from camoufox.async_api import AsyncCamoufox

# Minimal Query
graphql_query = """
query searchModel($keyword: String, $storeId: String) {
  searchModel(keyword: $keyword, storeId: $storeId) {
    products {
      itemId
      identifiers {
        brandName
        modelNumber
      }
      pricing(storeId: $storeId) {
        value
      }
    }
  }
}
"""

async def run_test():
    print(f"🚀 Testing Home Depot GraphQL API (Minimal Payload)...")
    
    config = {
        'humanize': True,
        'headless': True,
        'geoip': False
    }
    
    async with AsyncCamoufox(**config) as browser:
        page = await browser.new_page()
        
        # 1. Navigate to homepage
        await page.goto("https://www.homedepot.com/", wait_until="domcontentloaded", timeout=60000)
        
        # 2. Execute fetch
        payload = {
            "operationName": "searchModel",
            "variables": {
                "keyword": "hammer",
                "storeId": "121" # Atlanta
            },
            "query": graphql_query
        }
        
        result = await page.evaluate("""
            async (payload) => {
                const response = await fetch("https://apionline.homedepot.com/federation-gateway/graphql?opname=searchModel", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "x-experience-name": "general-merchandise",
                        "x-hd-dc": "origin"
                    },
                    body: JSON.stringify(payload)
                });
                
                if (!response.ok) {
                    return { error: response.status, text: await response.text() };
                }
                return await response.json();
            }
        """, payload)
        
        if isinstance(result, dict) and "error" in result:
            print(f"❌ API Request failed: HTTP {result['error']}")
            print(f"Response: {result.get('text', '')[:500]}")
        elif "errors" in result:
             print(f"❌ API returned errors:")
             print(json.dumps(result['errors'], indent=2))
        else:
            print("✅ Success!")
            print(json.dumps(result, indent=2)[:1000])

if __name__ == "__main__":
    asyncio.run(run_test())
