# Bright Data Web Unblocker Setup Guide

## 🚀 Quick Start

### 1. Get Your API Key

1. Log in to Bright Data dashboard: https://brightdata.com/cp
2. Go to **Account** → **API**
3. Generate or copy your API key
4. Note your Web Unblocker zone name (usually `web_unlocker1`)

### 2. Set Environment Variables

```bash
export BRIGHT_DATA_API_KEY="your-api-key-here"
export BRIGHT_DATA_ZONE="web_unlocker1"  # Optional, defaults to web_unlocker1
```

### 3. Test API Connection

```bash
# Test the API directly
python3 test_web_unblocker_api.py

# Or test with curl (as shown in Bright Data docs)
curl https://api.brightdata.com/request \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"zone": "web_unlocker1","url": "https://geo.brdtest.com/welcome.txt?product=unlocker&method=api", "format": "raw"}'
```

### 4. Test Chewy.com Scraping

```bash
python3 test_chewy_web_unblocker.py
```

## 📋 API Test Endpoint

Bright Data provides a test endpoint to verify your API key:

```bash
curl https://api.brightdata.com/request \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "zone": "web_unlocker1",
    "url": "https://geo.brdtest.com/welcome.txt?product=unlocker&method=api",
    "format": "raw"
  }'
```

**Expected Response**: Text containing IP information and "unlocker" confirmation

## 🔧 Configuration

### In Code

```python
from universal_scraper import UniversalScraper

scraper = UniversalScraper(
    api_key="your-openai-key",
    proxy_config={
        'server': 'http://brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    },
    web_unblocker_api_key="your-bright-data-api-key",  # Required for fallback
    web_unblocker_zone="web_unlocker1",  # Optional
    use_camoufox=True
)
```

### Via Environment Variables

The scraper will automatically use `BRIGHT_DATA_API_KEY` if provided:

```python
scraper = UniversalScraper(
    api_key="your-openai-key",
    proxy_config={...},
    # web_unblocker_api_key will be read from BRIGHT_DATA_API_KEY env var
)
```

## ✅ Verification Steps

1. **Test API Key**:
   ```bash
   python3 test_web_unblocker_api.py
   ```
   Should show: `✅ SUCCESS!`

2. **Test Chewy.com**:
   ```bash
   python3 test_chewy_web_unblocker.py
   ```
   Should extract products successfully

3. **Check Logs**:
   Look for: `🌐 Falling back to Bright Data Web Unblocker...`
   Then: `✅ Web Unblocker fetch successful!`

## 🐛 Troubleshooting

### Authentication Failed (401)

- **Issue**: Invalid API key
- **Solution**: 
  1. Verify API key at https://brightdata.com/cp/account/api
  2. Ensure no extra spaces in environment variable
  3. Check if API key has Web Unblocker permissions

### Insufficient Credits (402)

- **Issue**: Account has no credits
- **Solution**: Add credits to Bright Data account

### Rate Limit (429)

- **Issue**: Too many requests
- **Solution**: Wait a few minutes and retry

### Still Getting Blocked

- **Issue**: Web Unblocker also returns Kasada challenge
- **Possible Causes**:
  1. Zone name incorrect
  2. Web Unblocker not enabled for your account
  3. Target site has advanced protection

- **Solution**: 
  1. Verify zone name in Bright Data dashboard
  2. Check Web Unblocker is enabled
  3. Contact Bright Data support

## 📊 Cost Considerations

- **Residential Proxies**: Lower cost, tried first
- **Web Unblocker**: Higher cost, used only when needed
- **Fallback Strategy**: Minimizes costs by using Web Unblocker only when residential proxies fail

## 🔗 Resources

- Bright Data Dashboard: https://brightdata.com/cp
- API Documentation: https://brightdata.com/products/web-unblocker
- Account API Keys: https://brightdata.com/cp/account/api

---

**Status**: ✅ Ready to use once API key is configured

