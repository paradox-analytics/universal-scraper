import asyncio
import json
from camoufox.async_api import AsyncCamoufox

async def run_test():
    print(f"🚀 Intercepting Home Depot API Requests (Broad Capture)...")
    
    config = {
        'humanize': True,
        'headless': True,
        'geoip': False
    }
    
    async with AsyncCamoufox(**config) as browser:
        page = await browser.new_page()
        
        # Capture ALL API requests
        captured_requests = []
        
        async def handle_request(request):
            url = request.url
            # Broaden filter: capture anything that looks like an API or search
            if ("graphql" in url or "apionline" in url or "/s/" in url or "search" in url) and ".js" not in url and ".css" not in url:
                print(f"🎯 Request: {url} [{request.method}]")
                
                req_data = {
                    'url': url,
                    'method': request.method,
                    'headers': request.headers
                }
                
                if request.method == "POST":
                    try:
                        req_data['data'] = request.post_data_json
                        # Check if it looks like search
                        if req_data['data'] and 'operationName' in req_data['data']:
                            print(f"   Op: {req_data['data']['operationName']}")
                    except:
                        pass
                
                captured_requests.append(req_data)

        page.on("request", handle_request)
        
        # 1. Navigate directly to search results
        print("Navigating to search results page...")
        await page.goto("https://www.homedepot.com/s/refrigerator", wait_until="domcontentloaded", timeout=60000)
        
        # Wait for network idle
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except:
            pass
            
        # 2. Inspect HTML for embedded data
        print("Inspecting HTML for embedded JSON...")
        content = await page.content()
        
        # Look for __NEXT_DATA__ or similar
        if "__NEXT_DATA__" in content:
            print("✅ Found __NEXT_DATA__ in HTML (SSR confirmed?)")
            # Extract it
            import re
            match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', content)
            if match:
                json_data = json.loads(match.group(1))
                print("   Extracted __NEXT_DATA__")
                # Check for products
                # Usually in props.pageProps.initialState...
                with open("homedepot_next_data.json", "w") as f:
                    json.dump(json_data, f, indent=2)
                print("   Saved to homedepot_next_data.json")
        
        # Look for Apollo state or similar
        if "__APOLLO_STATE__" in content:
             print("✅ Found __APOLLO_STATE__ in HTML")
             
        # 3. Analyze Captured Requests (same as before)
        print(f"\n📦 Captured {len(captured_requests)} API requests.")
        
        # Find searchModel or similar
        search_payload = None
        for req in captured_requests:
            data = req.get('data', {})
            if data and data.get('operationName') == 'searchModel':
                search_payload = data
                print("\n✅ FOUND 'searchModel' payload!")
                break
            # Fallback: look for keyword in variables
            if data and 'variables' in data and 'keyword' in data['variables']:
                if data['variables']['keyword'] == 'refrigerator':
                    search_payload = data
                    print(f"\n✅ FOUND payload with keyword match! Op: {data.get('operationName')}")
                    break
        
        if search_payload:
            print(json.dumps(search_payload, indent=2))
            with open("homedepot_payload.json", "w") as f:
                json.dump(search_payload, f, indent=2)
            print("\nSaved payload to homedepot_payload.json")
        else:
            print("❌ Could not find search payload.")
            # Dump all captured to debug
            with open("all_requests.json", "w") as f:
                json.dump(captured_requests, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_test())
