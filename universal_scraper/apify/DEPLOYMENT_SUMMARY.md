# Apify Deployment Summary - Web Unblocker Support

## ✅ Changes Made

### 1. Updated Main Actor (`main.py`)
- ✅ Now uses main `UniversalScraper` class (not custom V2)
- ✅ Includes Web Unblocker fallback support
- ✅ Supports external proxy configuration (Bright Data, etc.)
- ✅ Proper error handling and logging
- ✅ Pushes results to Apify dataset

### 2. Updated Input Schema (`INPUT_SCHEMA.json`)
- ✅ Added `webUnblockerApiKey` field
- ✅ Added `webUnblockerZone` field
- ✅ Added `useExternalProxy` toggle
- ✅ Added `externalProxyServer`, `externalProxyUsername`, `externalProxyPassword`
- ✅ Added `enableAutoPagination` toggle
- ✅ Added `headless` browser option

### 3. Updated Dockerfile
- ✅ Uses `main.py` as entry point
- ✅ Includes all dependencies (Camoufox, etc.)

## 🚀 Deployment Steps

### 1. Verify Files

```bash
cd universal_scraper/apify
ls -la main.py INPUT_SCHEMA.json Dockerfile
```

### 2. Test Locally (Optional)

```bash
# Set environment variables
export OPENAI_API_KEY="your-key"
export BRIGHT_DATA_API_KEY="your-key"  # Optional

# Test run
python main.py
```

### 3. Deploy to Apify

```bash
# Login to Apify
apify login

# Push actor
apify push

# Or use Apify CLI
apify actors push
```

### 4. Configure in Apify Dashboard

1. Go to your actor in Apify
2. Navigate to **Settings** → **Environment Variables**
3. Optionally set:
   - `BRIGHT_DATA_API_KEY`: Your Bright Data API key
   - `OPENAI_API_KEY`: Your OpenAI API key (if not in input)

### 5. Test Run

Create a test run with:
```json
{
  "startUrls": [
    {"url": "https://www.chewy.com/b/wet-food-389"}
  ],
  "fields": ["name", "price", "rating"],
  "webUnblockerApiKey": "your-api-key",
  "useExternalProxy": true,
  "externalProxyServer": "http://brd.superproxy.io:33335",
  "externalProxyUsername": "brd-customer-xxx-zone-web_unlocker1",
  "externalProxyPassword": "your-password"
}
```

## 📋 Input Schema Overview

### Required Fields
- `startUrls`: Array of URLs
- `fields`: Array of field names

### Optional Fields

#### Proxy Configuration
- `proxyConfiguration`: Apify proxy config (object)
- `useExternalProxy`: Use external proxy (boolean)
- `externalProxyServer`: External proxy URL (string)
- `externalProxyUsername`: External proxy username (string, secret)
- `externalProxyPassword`: External proxy password (string, secret)

#### Web Unblocker
- `webUnblockerApiKey`: Bright Data API key (string, secret)
- `webUnblockerZone`: Zone name (string, default: "web_unlocker1")

#### Other Options
- `enableAutoPagination`: Auto-scrape all pages (boolean, default: false)
- `headless`: Headless browser (boolean, default: true)

## 🔧 Configuration Examples

### Example 1: Chewy.com with Web Unblocker

```json
{
  "startUrls": [{"url": "https://www.chewy.com/b/wet-food-389"}],
  "fields": ["name", "price", "rating", "reviewCount"],
  "webUnblockerApiKey": "your-bright-data-api-key",
  "webUnblockerZone": "web_unlocker1",
  "useExternalProxy": true,
  "externalProxyServer": "http://brd.superproxy.io:33335",
  "externalProxyUsername": "brd-customer-xxx-zone-web_unlocker1",
  "externalProxyPassword": "your-password",
  "enableAutoPagination": false
}
```

### Example 2: Apify Proxy with Web Unblocker Fallback

```json
{
  "startUrls": [{"url": "https://example.com"}],
  "fields": ["title", "price"],
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"]
  },
  "webUnblockerApiKey": "your-api-key"
}
```

### Example 3: Environment Variables Only

Set in Apify Actor Settings:
- `BRIGHT_DATA_API_KEY`
- `OPENAI_API_KEY`

Input:
```json
{
  "startUrls": [{"url": "https://example.com"}],
  "fields": ["title", "description"]
}
```

## ✅ Verification Checklist

- [x] `main.py` uses `UniversalScraper` from `core.scraper`
- [x] Web Unblocker parameters passed to `UniversalScraper`
- [x] External proxy support added
- [x] Input schema updated with all new fields
- [x] Dockerfile updated to use `main.py`
- [x] Error handling and logging included
- [x] Results pushed to Apify dataset

## 📝 Notes

- **Proxy Priority**: External proxy > Apify proxy > No proxy
- **Web Unblocker**: Only used as fallback when standard proxy is blocked
- **Premium Domains**: Some sites require Premium domains enabled in Bright Data dashboard
- **Auto-Pagination**: Disabled by default for safety (can enable per-run)

## 🐛 Troubleshooting

### Actor Fails to Start
- Check `main.py` syntax
- Verify all imports are available
- Check Dockerfile CMD points to `main.py`

### Web Unblocker Not Working
- Verify API key is set (input or env var)
- Check logs for "Web Unblocker: Disabled" messages
- Verify zone name is correct

### Proxy Issues
- Check proxy credentials
- Verify proxy server URL format
- Check if external proxy takes priority over Apify proxy

## 📚 Related Documentation

- `WEB_UNBLOCKER_DEPLOYMENT.md`: Detailed Web Unblocker guide
- `DEPLOYMENT_GUIDE.md`: General deployment guide
- `README.md`: Actor overview

