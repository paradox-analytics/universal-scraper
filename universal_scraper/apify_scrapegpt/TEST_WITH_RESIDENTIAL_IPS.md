# 🌐 Testing with Residential IPs Locally

## Quick Start

```bash
# 1. Set your API keys
export OPENAI_API_KEY="sk-proj-qbN90vro..."
export APIFY_TOKEN="apify_api_zcB3PUc54SUFwNyLtfs6MXB8mbfegJ2UiFq4"

# 2. Navigate to the directory
cd /Users/jevon_williams/Dev/universal-scraper/universal_scraper/apify

# 3. Test Amazon with Residential IPs
./test-site.sh amazon

# 4. Test Leafly (verify nothing broke)
./test-site.sh leafly

# 5. Test Ticketmaster (verify nothing broke)
./test-site.sh ticketmaster
```

---

## 🔑 What You Need

### 1. OpenAI API Key
Already have: `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`

### 2. Apify Token
Already have: `apify_api_zcB3PUc54SUFwNyLtfs6MXB8mbfegJ2UiFq4`

---

## 🧪 Test Sequence

### Test 1: Amazon (with Residential IPs)
```bash
export OPENAI_API_KEY="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"
export APIFY_TOKEN="apify_api_zcB3PUc54SUFwNyLtfs6MXB8mbfegJ2UiFq4"
cd /Users/jevon_williams/Dev/universal-scraper/universal_scraper/apify
./test-site.sh amazon
```

**Expected Result:**
- ✅ Residential IPs enabled
- ✅ Anti-detection working
- ✅ Should extract actual products (not just configs)
- ⏱️ May take 30-60 seconds

### Test 2: Leafly (verify nothing broke)
```bash
./test-site.sh leafly
```

**Expected Result:**
- ✅ Should still extract 523+ items (all pages)
- ✅ Auto-pagination working
- ✅ No regressions

### Test 3: Ticketmaster (verify JS rendering)
```bash
./test-site.sh ticketmaster
```

**Expected Result:**
- ✅ JS-rendered content working
- ✅ Events extracted
- ✅ Anti-detection not breaking public sites

---

## 📊 What to Look For

### In the Logs:

**Proxy Status:**
```
Configuration:
  ...
  Apify Token: apify_api_zcB3...fegJ ✅
  Proxy: Residential IPs enabled 🌐
```

**Anti-Detection:**
```
📷 Images loaded
🎯 DOM stabilized
```

**Data Extraction:**
```
✅ JSON sources sufficient, extracting from JSON...
✅ Extraction complete: X items in Y.ZZs
```

### In the Results:

**Amazon:**
```bash
cat apify_storage_local/datasets/default/*.json | head -50
```
Look for:
- Product names (not config IDs)
- Prices ($X.XX)
- "Add to cart" related data
- Actual product info

**Leafly:**
```bash
cat apify_storage_local/datasets/default/*.json | jq 'length'
# Should show: 523+ items
```

**Ticketmaster:**
```bash
cat apify_storage_local/datasets/default/*.json | jq '.[0]'
# Should show: Event data (name, date, venue)
```

---

## 🐛 Troubleshooting

### "WARNING: APIFY_TOKEN not set"
```bash
# Make sure you exported it:
export APIFY_TOKEN="apify_api_zcB3PUc54SUFwNyLtfs6MXB8mbfegJ2UiFq4"

# Verify it's set:
echo $APIFY_TOKEN
```

### "Proxy: Local only (no residential IPs)"
- Token wasn't exported correctly
- Re-run the export command above

### Still Getting Config Data from Amazon
- Residential IPs may not be connected
- Check Apify dashboard for proxy usage
- May need to wait a bit for proxy warmup

### Leafly/Ticketmaster Broke
- Check logs for errors
- Make sure test was run with tokens set
- Clear cache: `rm -rf apify_storage_local/`

---

## 💰 Cost Estimate

**Apify Residential Proxies: $8/GB**

| Test | Data Transfer | Est. Cost |
|------|---------------|-----------|
| Amazon (1 page) | ~1-2 MB | $0.01-0.02 |
| Leafly (27 pages) | ~15-20 MB | $0.12-0.16 |
| Ticketmaster (1 page) | ~2-3 MB | $0.02-0.03 |
| **Total** | **~18-25 MB** | **$0.15-0.21** |

Very affordable for testing!

---

## ✅ Success Criteria

### Amazon
- [ ] Extracted >10 items
- [ ] Items are actual products (not configs)
- [ ] Has product names/prices
- [ ] No "sign in" errors

### Leafly
- [ ] Extracted 500+ items
- [ ] Auto-pagination worked
- [ ] Data quality good

### Ticketmaster
- [ ] Extracted events
- [ ] JS rendering worked
- [ ] Data complete

---

## 🚀 Ready to Test!

Run this complete test sequence:

```bash
#!/bin/bash
# Complete test sequence

# Set keys
export OPENAI_API_KEY="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"
export APIFY_TOKEN="apify_api_zcB3PUc54SUFwNyLtfs6MXB8mbfegJ2UiFq4"

cd /Users/jevon_williams/Dev/universal-scraper/universal_scraper/apify

echo "🧪 Test 1/3: Amazon (with Residential IPs)"
./test-site.sh amazon
echo ""
echo "Press Enter to continue to Leafly test..."
read

echo "🧪 Test 2/3: Leafly (verify no regressions)"
./test-site.sh leafly
echo ""
echo "Press Enter to continue to Ticketmaster test..."
read

echo "🧪 Test 3/3: Ticketmaster (verify JS rendering)"
./test-site.sh ticketmaster
echo ""
echo "✅ All tests complete!"
```

Save as `run-all-tests.sh`, make executable, and run!








