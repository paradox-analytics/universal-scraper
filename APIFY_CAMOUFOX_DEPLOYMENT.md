# Apify Deployment with Camoufox Support

## 🦊 What's New

This deployment includes **Camoufox** integration - an advanced anti-detection browser that significantly improves scraping success on challenging websites like Reddit, eBay, and more.

### Key Features
- ✅ **Camoufox Browser** (default): Advanced fingerprinting + humanization
- ✅ **Fallback to Playwright**: If Camoufox not available
- ✅ **Apify Residential Proxies**: Native support within Apify platform
- ✅ **LLM-First Architecture**: Dynamic structure detection, not hardcoded heuristics
- ✅ **Auto-caching**: LLM analysis results cached for speed & cost savings

---

## 📋 Prerequisites

1. **Apify Account** with API token
2. **OpenAI API Key** (for AI-powered extraction)
3. **Apify CLI** installed:
   ```bash
   npm install -g apify-cli
   ```

---

## 🚀 Deployment Steps

### 1. Navigate to Apify Directory
```bash
cd /Users/jevon_williams/Dev/universal-scraper/universal_scraper/apify
```

### 2. Login to Apify CLI
```bash
apify login
```

### 3. Deploy to Apify
```bash
apify push
```

This will:
- Build the Docker image with Camoufox
- Upload to your Apify account
- Make it available as an actor

---

## 🧪 Testing with Proxies

### Option 1: Test via Apify Console

1. Go to https://console.apify.com/
2. Find your "universal-scraper" actor
3. Click "Run" and use this input:

```json
{
  "mode": "scrape_only",
  "urls": [
    {"url": "https://www.reddit.com/r/webscraping/"}
  ],
  "fields": ["title", "author", "upvotes", "comments"],
  "browserConfig": {
    "useCamoufox": true,
    "headless": true
  },
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"]
  },
  "apiKeys": {
    "openaiApiKey": "<YOUR_OPENAI_KEY>"
  },
  "crawlConfig": {
    "maxDepth": 0,
    "maxPages": 1,
    "handlePagination": false
  }
}
```

### Option 2: Test via API

```bash
curl -X POST https://api.apify.com/v2/acts/YOUR_USERNAME~universal-scraper/runs \
  -H "Authorization: Bearer YOUR_APIFY_TOKEN" \
  -H "Content-Type: application/json" \
  -d @test_apify_camoufox.json
```

---

## 🔧 Configuration Options

### Browser Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `useCamoufox` | boolean | `true` | Use Camoufox (recommended for anti-detection) |
| `headless` | boolean | `true` | Run browser in headless mode |
| `waitForNetworkIdle` | boolean | `true` | Wait for all network requests |
| `captureApiRequests` | boolean | `true` | Intercept API calls |

### Proxy Configuration (Apify Native)

```json
{
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"]
  }
}
```

**Available Proxy Groups:**
- `RESIDENTIAL` - Residential IPs (best for anti-blocking)
- `DATACENTER` - Faster but more detectable
- `GOOGLE_SERP` - For Google searches

---

## 📊 Expected Results

### Reddit (r/webscraping)
- **Without Proxy + Camoufox**: 62 items ✅
- **With Apify Proxy + Camoufox**: Expected similar or better

### eBay Search
- **Status**: Under investigation (0 items locally)
- **With Apify Proxy + Camoufox**: May improve

### Hacker News
- **Without Proxy + Camoufox**: 30 items (97% quality) ✅
- **With Apify Proxy + Camoufox**: Expected similar

---

## 🐛 Troubleshooting

### "Camoufox not available" Error

**Solution**: The Dockerfile should auto-install Camoufox. If not:
```dockerfile
RUN pip install camoufox>=0.4.0 camoufox[geoip]
```

### Proxy Connection Refused

**Solution**: Ensure `proxyConfiguration` is correctly formatted:
```json
{
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"]
  }
}
```

### High LLM Costs

**Expected on First Run**: The system analyzes structure using LLM
**Cached on Subsequent Runs**: Structure analysis is cached per domain

---

## 💰 Cost Estimate

### Per New Website (First Run)
- **Structure Analysis**: ~$0.01 (cached)
- **Code Generation**: ~$0.02 (cached)
- **Total LLM Cost**: ~$0.03 first time, then FREE (cached)

### Apify Proxy Costs
- **Residential Proxies**: ~$7.50 per 1GB
- **Typical Scrape**: ~1-10 MB per page

---

## 📝 Notes

1. **Camoufox is ENABLED by default** - This gives best anti-detection
2. **Proxies are OPTIONAL** - Camoufox alone works well for most sites
3. **Structure Analysis is CACHED** - Only pays LLM cost once per domain structure
4. **Use Residential Proxies** - For maximum success with challenging sites

---

## 🎯 Test Sites

After deployment, test with these to verify:

1. ✅ **Reddit** (custom elements): https://www.reddit.com/r/webscraping/
2. ✅ **Hacker News** (simple HTML): https://news.ycombinator.com/
3. ⚠️ **eBay** (obfuscated): https://www.ebay.com/sch/i.html?_nkw=laptop
4. 🔄 **Metacritic** (nested): https://www.metacritic.com/browse/game/

Expected: Reddit and Hacker News should work perfectly with Camoufox alone!







