# 🧪 Local Apify Actor Testing Guide

Test your Apify actor **locally** without consuming any Apify credits. This setup simulates the exact Apify environment.

---

## 🚀 Quick Start

### 1. Set Your OpenAI API Key
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

### 2. Navigate to Apify Directory
```bash
cd /Users/jevon_williams/Dev/universal-scraper/universal_scraper/apify
```

### 3. Run the Test
```bash
./test-local.sh
```

That's it! The script will:
- ✅ Set up local Apify storage
- ✅ Use your OpenAI API key from environment
- ✅ Run the actor locally (no credits used)
- ✅ Save results to `apify_storage_local/datasets/default/`
- ✅ Show you a summary of results

---

## 📋 Configuration

### Edit Test Input
Modify `test-input.json` to change:
- Target URLs
- Scraping mode (scrape_only, crawl_only, full_pipeline)
- Fields to extract
- Pagination settings
- Advanced options

### Example: Test Different URL
```json
{
  "mode": "scrape_only",
  "startUrls": [
    {
      "url": "https://example.com/products"
    }
  ],
  "scrapeConfig": {
    "fields": ["name", "price", "description"]
  }
}
```

---

## 🔍 View Results

### Using jq (Pretty Print)
```bash
cat apify_storage_local/datasets/default/*.json | jq
```

### Using Python
```bash
python3 -c "
import json
with open('apify_storage_local/datasets/default/000000001.json') as f:
    data = json.load(f)
    print(f'Total items: {len(data)}')
    print('First item:', json.dumps(data[0], indent=2))
"
```

### Check Logs
All actor logs are printed to the console during execution.

---

## 🎯 Testing Scenarios

### Test 1: Basic Scraping
```bash
# Edit test-input.json
{
  "mode": "scrape_only",
  "startUrls": [{"url": "https://example.com"}],
  "scrapeConfig": {"fetchMode": "browser", "fields": []}
}

# Run test
./test-local.sh
```

### Test 2: Auto-Pagination
```bash
# Edit test-input.json
{
  "mode": "scrape_only",
  "startUrls": [{"url": "https://example.com/products"}],
  "scrapeConfig": {"fetchMode": "browser", "fields": []},
  "advancedConfig": {"enableLlmPagination": true}
}

# Run test
./test-local.sh
```

### Test 3: Crawling + Scraping
```bash
# Edit test-input.json
{
  "mode": "full_pipeline",
  "startUrls": [{"url": "https://example.com"}],
  "crawlConfig": {
    "mode": "smart",
    "maxDepth": 2,
    "maxPages": 50
  },
  "scrapeConfig": {"fields": ["title", "content"]}
}

# Run test
./test-local.sh
```

---

## 🐛 Debugging

### View Full Logs
Logs are printed to console in real-time. Look for:
- `INFO` - Normal operation
- `WARNING` - Potential issues
- `ERROR` - Problems that need fixing

### Check Local Storage
```bash
# View all stored data
ls -lah apify_storage_local/

# Key-value stores (INPUT, OUTPUT, etc.)
ls apify_storage_local/key_value_stores/default/

# Dataset results
ls apify_storage_local/datasets/default/
```

### Clean and Restart
```bash
# Remove all local data
rm -rf apify_storage_local/

# Run again
./test-local.sh
```

---

## 🔧 Manual Testing (Without Script)

If you prefer manual control:

```bash
# Set up environment
export APIFY_LOCAL_STORAGE_DIR="./apify_storage_local"
export APIFY_TOKEN=""
export OPENAI_API_KEY="sk-your-key-here"

# Create storage
mkdir -p apify_storage_local/key_value_stores/default

# Copy input
cp test-input.json apify_storage_local/key_value_stores/default/INPUT.json

# Run actor
apify run
```

---

## 📊 What Gets Tested

When you run locally with `apify run`, it tests:

✅ **Actor Dependencies**
- All Python packages from `requirements.txt`
- Playwright browser installation
- Apify SDK integration

✅ **Actor Logic**
- Input validation and parsing
- Scraper initialization
- Data extraction
- Dataset storage

✅ **Apify Integration**
- Local storage simulation
- Dataset writing
- Key-value store access
- Actor lifecycle (init, run, teardown)

✅ **Environment Compatibility**
- Docker environment simulation
- Python version compatibility
- System dependencies (browsers, etc.)

---

## ⚠️ Limitations of Local Testing

| Feature | Local | Cloud |
|---------|-------|-------|
| Actor logic | ✅ Tested | ✅ Same |
| Dependencies | ✅ Tested | ✅ Same |
| Apify SDK | ✅ Tested | ✅ Same |
| Residential Proxies | ❌ Not available | ✅ Available |
| Scalability | ❌ Single machine | ✅ Auto-scaled |
| Monitoring | ❌ Console only | ✅ Full dashboard |

**Recommendation**: Test locally for development, then do 1-2 cloud runs for final validation.

---

## 💡 Tips

1. **Fast Iteration**: Edit `test-input.json` and rerun `./test-local.sh`
2. **Debug Mode**: Set `"logLevel": "DEBUG"` in `advancedConfig`
3. **Save Bandwidth**: Use `"headless": true` (already default)
4. **Test Small First**: Start with 1 URL before scaling up
5. **Check Results**: Always verify output before deploying to cloud

---

## 🆘 Troubleshooting

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### "apify: command not found"
```bash
npm install -g apify-cli
```

### Browser doesn't launch
```bash
# Install Playwright browsers
cd /Users/jevon_williams/Dev/universal-scraper/universal_scraper/apify
python -m playwright install chromium
```

### Results are empty
1. Check logs for errors
2. Try simpler URL first
3. Set `"logLevel": "DEBUG"` to see more details

---

## ✅ Success Checklist

Before deploying to Apify cloud, ensure:

- [ ] Local test runs without errors
- [ ] Results are extracted correctly
- [ ] Pagination works (if applicable)
- [ ] Input validation works
- [ ] No dependency errors
- [ ] Dataset structure is correct

**Once all checked, deploy to Apify with confidence!** 🚀








