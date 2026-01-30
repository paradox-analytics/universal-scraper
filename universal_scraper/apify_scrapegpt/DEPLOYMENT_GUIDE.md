# 🚀 Universal Web Scraper - Deployment Guide

Complete guide for deploying the Universal Web Scraper to Apify.

---

## 📋 Prerequisites

1. **Apify Account**
   - Sign up at [apify.com](https://apify.com)
   - Get your API token from Settings

2. **OpenAI API Key**
   - Get from [platform.openai.com](https://platform.openai.com)
   - Required for scraping modes

3. **Apify CLI** (recommended)
   ```bash
   npm install -g apify-cli
   apify login
   ```

---

## 🚀 Deployment Methods

### Method 1: Automated Deployment (Recommended)

```bash
cd /Users/jevon_williams/Dev/universal-scraper
./deploy_to_apify.sh
```

The script will:
1. Copy all necessary files
2. Build Docker image
3. Push to Apify
4. Set up actor configuration

---

### Method 2: Manual Deployment via CLI

```bash
# Navigate to apify directory
cd universal_scraper/apify

# Initialize (first time only)
apify init

# Push to Apify
apify push
```

---

### Method 3: Manual Deployment via Web

1. Go to Apify Console
2. Click "Actors" → "Create new"
3. Select "From template" → "Blank Actor"
4. Copy files from `universal_scraper/apify/`:
   - `actor.py`
   - `INPUT_SCHEMA.json`
   - `.actor/actor.json`
   - `Dockerfile`
   - `README.md`
5. Click "Build"

---

## ⚙️ Configuration

### 1. Set Up Apify Secrets (Recommended)

Store your OpenAI API key securely:

1. Go to **Apify Console** → **Settings** → **Secrets**
2. Click **"Add Secret"**
3. Name: `OPENAI_API_KEY_SECRET`
4. Value: `sk-...` (your OpenAI API key)
5. Click **Save**

Now you can run the actor without exposing your API key in the input!

---

### 2. Configure Proxy (Recommended)

For best results, use Apify residential proxies:

```json
{
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"],
    "apifyProxyCountry": "US"
  }
}
```

**Proxy Options:**
- `RESIDENTIAL` - Best for anti-bot sites (recommended)
- `DATACENTER` - Faster, cheaper
- `GOOGLE_SERP` - For Google Search

---

## 🧪 Testing

### Test 1: Scrape Single URL

```json
{
  "mode": "scrape_only",
  "startUrls": [
    {"url": "https://example.com"}
  ],
  "scrapeConfig": {
    "fields": ["title", "description"],
    "fetchMode": "hybrid"
  },
  "openaiApiKey": "sk-..."
}
```

**Expected:** Dataset with title and description extracted.

---

### Test 2: Crawl Website

```json
{
  "mode": "crawl_only",
  "startUrls": [
    {"url": "https://example.com"}
  ],
  "crawlConfig": {
    "maxDepth": 1,
    "maxPages": 10,
    "followPatterns": ["/product/"],
    "ignorePatterns": ["/cart"]
  }
}
```

**Expected:** Dataset with discovered URLs.

---

### Test 3: Full Pipeline

```json
{
  "mode": "full_pipeline",
  "startUrls": [
    {"url": "https://example.com/category"}
  ],
  "crawlConfig": {
    "maxDepth": 2,
    "maxPages": 20,
    "followPatterns": ["/product/", "/category/"]
  },
  "scrapeConfig": {
    "fields": ["name", "price"],
    "fetchMode": "hybrid"
  },
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"]
  },
  "openaiApiKey": "sk-..."
}
```

**Expected:** Dataset with scraped product data.

---

## 📊 Monitoring

### View Logs

1. Go to **Actor Run** page
2. Click **"Log"** tab
3. Monitor progress:
   ```
   🚀 Universal Scraper Actor started
   🕷️  Starting CRAWL ONLY mode
   📊 Crawl Configuration:
      Max Depth: 2
      Max Pages: 20
   🌐 Crawling from 1 start URL(s)
   ✅ Crawl complete: 15 URLs discovered
   ```

### View Dataset

1. Click **"Dataset"** tab
2. See extracted data in table format
3. Download as JSON, CSV, or Excel

### View Key-Value Store

1. Click **"Key-value store"** tab
2. See cached data and API responses

---

## 🔧 Optimization

### 1. Reduce Costs

**Use Static Mode:**
```json
{
  "scrapeConfig": {
    "fetchMode": "static"
  }
}
```
- Faster
- Cheaper
- No browser overhead

**Limit Crawl Depth:**
```json
{
  "crawlConfig": {
    "maxDepth": 1,
    "maxPages": 50
  }
}
```

**Use Specific Patterns:**
```json
{
  "crawlConfig": {
    "followPatterns": ["/product/"],
    "ignorePatterns": ["/cart", "/auth", ".pdf"]
  }
}
```

---

### 2. Improve Speed

**Enable Caching:**
```json
{
  "advancedConfig": {
    "enableCache": true
  }
}
```

**Use Datacenter Proxies:**
```json
{
  "proxyConfiguration": {
    "apifyProxyGroups": ["DATACENTER"]
  }
}
```

**Reduce Field Count:**
```json
{
  "scrapeConfig": {
    "fields": ["name", "price"]
  }
}
```

---

### 3. Improve Quality

**Use Residential Proxies:**
```json
{
  "proxyConfiguration": {
    "apifyProxyGroups": ["RESIDENTIAL"]
  }
}
```

**Use Browser Mode:**
```json
{
  "scrapeConfig": {
    "fetchMode": "browser"
  }
}
```

**Define Schema:**
```json
{
  "scrapeConfig": {
    "schema": {
      "name": "products",
      "fields": [
        {"name": "name", "type": "string", "required": true},
        {"name": "price", "type": "number", "required": true}
      ]
    }
  }
}
```

---

## 🎯 Example Configurations

### E-commerce Site

**File:** `examples/ecommerce_config.json`

```json
{
  "mode": "full_pipeline",
  "startUrls": [
    {"url": "https://shop.example.com/electronics"}
  ],
  "crawlConfig": {
    "maxDepth": 2,
    "maxPages": 500,
    "followPatterns": ["/product/", "/category/electronics"],
    "ignorePatterns": ["/cart", "/checkout", "/account", "/wishlist"],
    "handlePagination": true,
    "discoverApis": true
  },
  "scrapeConfig": {
    "fields": [
      "name",
      "brand",
      "price",
      "originalPrice",
      "discount",
      "rating",
      "reviewCount",
      "stock",
      "sku",
      "description",
      "specifications",
      "imageUrl"
    ],
    "fetchMode": "hybrid",
    "schema": {
      "name": "electronics_products",
      "version": "1.0",
      "fields": [
        {"name": "name", "type": "string", "required": true},
        {"name": "price", "type": "number", "required": true},
        {"name": "brand", "type": "string", "required": false},
        {"name": "rating", "type": "number", "required": false}
      ]
    }
  },
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"],
    "apifyProxyCountry": "US"
  }
}
```

---

### News Site

**File:** `examples/news_config.json`

```json
{
  "mode": "full_pipeline",
  "startUrls": [
    {"url": "https://news.example.com/2024"}
  ],
  "crawlConfig": {
    "maxDepth": 2,
    "maxPages": 1000,
    "followPatterns": ["/2024/", "/article/", "/news/"],
    "ignorePatterns": ["/author/", "/tag/", "/category/", "/video/"],
    "handlePagination": true
  },
  "scrapeConfig": {
    "fields": [
      "headline",
      "subheadline",
      "author",
      "publishDate",
      "updateDate",
      "content",
      "summary",
      "tags",
      "category",
      "imageUrl"
    ],
    "fetchMode": "static"
  },
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["DATACENTER"]
  }
}
```

---

### Leafly Dispensaries

**File:** `examples/leafly_config.json`

```json
{
  "mode": "full_pipeline",
  "startUrls": [
    {"url": "https://www.leafly.com/dispensaries/nevada"}
  ],
  "crawlConfig": {
    "maxDepth": 2,
    "maxPages": 100,
    "followPatterns": ["/dispensaries/", "/dispensary-info/"],
    "ignorePatterns": [
      "/products",
      "/strains",
      "/news",
      "/brands",
      "/doctors",
      "/learn"
    ],
    "handlePagination": true,
    "discoverApis": true
  },
  "scrapeConfig": {
    "fields": [
      "name",
      "address",
      "city",
      "state",
      "zip",
      "phone",
      "website",
      "hours",
      "rating",
      "reviewCount",
      "description",
      "amenities",
      "products"
    ],
    "fetchMode": "browser"
  },
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"],
    "apifyProxyCountry": "US"
  }
}
```

---

### Real Estate

**File:** `examples/realestate_config.json`

```json
{
  "mode": "full_pipeline",
  "startUrls": [
    {"url": "https://realestate.example.com/city/seattle"}
  ],
  "crawlConfig": {
    "maxDepth": 2,
    "maxPages": 300,
    "followPatterns": ["/property/", "/listing/", "/for-sale/"],
    "ignorePatterns": ["/agent/", "/mortgage-calculator/", "/blog/"],
    "handlePagination": true
  },
  "scrapeConfig": {
    "fields": [
      "address",
      "price",
      "bedrooms",
      "bathrooms",
      "sqft",
      "lotSize",
      "yearBuilt",
      "propertyType",
      "description",
      "features",
      "images",
      "listingAgent",
      "mls"
    ],
    "fetchMode": "hybrid"
  },
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"]
  }
}
```

---

## 🔐 Security Best Practices

### 1. Use Apify Secrets

**Never** put API keys directly in input:
```json
{
  "openaiApiKey": "sk-..."  // ❌ DON'T DO THIS
}
```

**Instead,** use Apify secrets:
1. Store key in Apify Secrets as `OPENAI_API_KEY_SECRET`
2. Leave `openaiApiKey` field empty in input
3. Actor will automatically use the secret

### 2. Restrict Access

- Set actor visibility to **Private**
- Use **Apify API tokens** with limited scope
- Rotate API keys regularly

### 3. Monitor Usage

- Check Apify usage dashboard
- Set up alerts for high usage
- Review actor logs regularly

---

## 📈 Scaling

### Run Multiple Instances

```bash
# Via API
curl -X POST https://api.apify.com/v2/acts/YOUR_ACTOR/runs \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d @config1.json

curl -X POST https://api.apify.com/v2/acts/YOUR_ACTOR/runs \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d @config2.json
```

### Use Apify Schedules

1. Go to **Actor** → **Schedules**
2. Click **"Create schedule"**
3. Set frequency (hourly, daily, weekly)
4. Configure input
5. Save

### Batch Processing

```json
{
  "mode": "scrape_only",
  "startUrls": [
    {"url": "https://example.com/product/1"},
    {"url": "https://example.com/product/2"},
    // ... up to 1000 URLs
  ]
}
```

---

## 🐛 Troubleshooting

### Issue: Actor Fails to Start

**Solution:**
- Check Dockerfile syntax
- Verify all files are present
- Check actor logs for errors

### Issue: No Data in Dataset

**Solution:**
- Check `followPatterns` and `ignorePatterns`
- Verify `fields` match page content
- Try `fetchMode: "browser"`

### Issue: Rate Limited / Blocked

**Solution:**
- Enable Apify proxies
- Use `RESIDENTIAL` proxy group
- Reduce `maxPages`

### Issue: High Costs

**Solution:**
- Use `static` fetch mode
- Reduce `maxDepth` and `maxPages`
- Use more specific `followPatterns`
- Enable caching

---

## 📞 Support

### Documentation
- **Actor README:** See full actor documentation
- **Project Docs:** Check `/Users/jevon_williams/Dev/universal-scraper/docs/`
- **Examples:** See `examples/` directory

### Apify Support
- **Community Forum:** [community.apify.com](https://community.apify.com)
- **Help Center:** [help.apify.com](https://help.apify.com)

---

## ✅ Deployment Checklist

- [ ] Apify account created
- [ ] API token obtained
- [ ] OpenAI API key stored as secret
- [ ] Actor deployed to Apify
- [ ] Test run completed successfully
- [ ] Proxy configuration verified
- [ ] Example configurations tested
- [ ] Monitoring set up
- [ ] Production input prepared
- [ ] Schedule configured (if needed)

---

**Status:** ✅ Ready for Production  
**Last Updated:** November 7, 2025








