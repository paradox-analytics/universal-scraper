# Bright Data Web Unblocker Integration

## ✅ Implementation Complete

Web Unblocker has been integrated as an **automatic fallback** when residential proxies fail due to anti-bot protection (Kasada, Cloudflare, etc.).

## 🏗️ Architecture

```
HybridFetcher
    ├─ Try Static HTML
    ├─ Try Browser (Camoufox + Residential Proxy)
    │   └─ Detect Blocking (Kasada challenge)
    │       └─ Fallback to Web Unblocker API ✅ NEW
    └─ Return Results
```

## 📝 How It Works

1. **Normal Flow**: Residential proxy → Browser → Success
2. **Blocked Flow**: Residential proxy → Browser → Kasada detected → **Web Unblocker** → Success

### Blocking Detection

The system automatically detects blocking by checking for:
- **Kasada**: `kasada`, `kpsdk`, `ips.js`
- **Cloudflare**: `cf-browser-verification`, `checking your browser`
- **Generic**: `access denied`, `blocked`, `forbidden`, `403`

If HTML is small (<2000 bytes) and contains blocking indicators, fallback is triggered.

## 🔧 Configuration

### Environment Variables

```bash
export BRIGHT_DATA_API_KEY="your-api-key-here"
export BRIGHT_DATA_ZONE="web_unlocker1"  # Optional, defaults to web_unlocker1
```

### Code Configuration

```python
from universal_scraper import UniversalScraper

scraper = UniversalScraper(
    api_key="your-openai-key",
    proxy_config={
        'server': 'http://brd.superproxy.io:33335',
        'username': 'brd-customer-REDACTED_CUSTOMER_ID-zone-residential_proxy2',
        'password': 'REDACTED_PROXY_PASS'
    },
    web_unblocker_api_key="your-bright-data-api-key",  # NEW
    web_unblocker_zone="web_unlocker1",  # NEW (optional)
    use_camoufox=True,
    fetch_mode='browser'
)

result = await scraper.scrape(
    url="https://www.chewy.com/b/wet-food-389",
    fields=["name", "price", "rating"]
)
```

## 📦 Components

### 1. WebUnblockerFetcher (`core/web_unblocker_fetcher.py`)

- Bright Data Web Unblocker API client
- Handles authentication, retries, error handling
- Supports async operations

**Features**:
- Automatic retry on failure (configurable)
- Rate limit handling
- Credit checking
- Timeout management

### 2. HybridFetcher Integration (`core/hybrid_fetcher.py`)

- Detects blocking after browser fetch
- Automatically falls back to Web Unblocker
- Logs fallback events

**New Parameters**:
- `web_unblocker_api_key`: Bright Data API key
- `web_unblocker_zone`: Zone name (default: `web_unlocker1`)

### 3. UniversalScraper Integration (`core/scraper.py`)

- Passes Web Unblocker config to HybridFetcher
- Transparent to end user

## 🧪 Testing

### Test Script

```bash
# Set API key
export BRIGHT_DATA_API_KEY="your-api-key"
export BRIGHT_DATA_ZONE="web_unlocker1"  # Optional

# Run test
python3 test_chewy_web_unblocker.py
```

### Expected Behavior

1. **Without Web Unblocker**: 
   - Residential proxy → Kasada challenge → No data
   - Logs: "⚠️ Browser fetch appears blocked"

2. **With Web Unblocker**:
   - Residential proxy → Kasada challenge → **Web Unblocker** → Success
   - Logs: "🌐 Falling back to Bright Data Web Unblocker..."
   - Logs: "✅ Web Unblocker fetch successful!"

## 📊 API Usage

### Bright Data API Endpoint

```
POST https://api.brightdata.com/request
Headers:
  Content-Type: application/json
  Authorization: Bearer {api_key}
Body:
  {
    "zone": "web_unlocker1",
    "url": "https://example.com",
    "format": "raw"
  }
```

### Response Codes

- `200`: Success
- `401`: Authentication failed (check API key)
- `402`: Insufficient credits
- `429`: Rate limit exceeded

## 💡 Benefits

1. **Automatic Fallback**: No manual intervention needed
2. **Cost Efficient**: Only uses Web Unblocker when needed
3. **Transparent**: Works seamlessly with existing code
4. **Reliable**: Handles Kasada, Cloudflare, and other protections

## 🔍 Monitoring

The system logs Web Unblocker usage:

```
🌐 Falling back to Bright Data Web Unblocker...
✅ Web Unblocker fetch successful!
```

Check logs to see when fallback occurs.

## 📝 Notes

- Web Unblocker is **only used as fallback** - residential proxies are tried first
- Web Unblocker costs more than residential proxies, so fallback minimizes costs
- If Web Unblocker is not configured, system continues with blocked result
- Web Unblocker doesn't capture API requests (unlike browser mode)

## 🚀 Next Steps

1. **Get API Key**: 
   - Visit https://brightdata.com/cp/account/api
   - Generate API key
   - Set `BRIGHT_DATA_API_KEY` environment variable

2. **Test**:
   ```bash
   python3 test_chewy_web_unblocker.py
   ```

3. **Monitor**: Check logs for fallback frequency and adjust as needed

---

**Status**: ✅ **Ready for Testing**

