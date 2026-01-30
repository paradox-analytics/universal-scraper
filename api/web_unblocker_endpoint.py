"""
Add missing /api/v1/web-unblocker/test endpoint
"""

# Add after the proxy test endpoint (around line 1768)

@app.post("/api/v1/web-unblocker/test")
async def test_web_unblocker(
    request: dict,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Test Web Unblocker connection
    
    Request body:
    {
        "apiKey": "your-api-key",
        "zone": "web_unlocker1"
    }
    """
    try:
        api_key = request.get("apiKey")
        zone = request.get("zone", "web_unlocker1")
        
        if not api_key:
            return {"success": False, "message": "API key is required"}
        
        # Test Web Unblocker by making a simple request
        import requests
        
        # Use a simple test URL
        test_url = "https://lumtest.com/myip.json"
        
        # Web Unblocker proxy format
        proxy_url = f"http://brd-customer-{api_key}:{zone}@brd.superproxy.io:33335"
        
        proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        
        response = requests.get(
            test_url,
            proxies=proxies,
            timeout=10,
            verify=False
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "message": f"Web Unblocker connection successful! IP: {data.get('ip', 'unknown')}"
            }
        else:
            return {
                "success": False,
                "message": f"Web Unblocker returned status code {response.status_code}"
            }
            
    except Exception as e:
        logger.error(f"[{tenant_id}] Web Unblocker test failed: {e}")
        return {
            "success": False,
            "message": f"Web Unblocker test failed: {str(e)}"
        }
