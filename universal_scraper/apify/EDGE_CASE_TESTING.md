# 🧪 Edge Case Testing Guide

This guide helps you test the universal scraper against challenging websites to identify edge cases and limitations.

---

## 🎯 Quick Start

### Test Amazon Same-Day Store
```bash
cd /Users/jevon_williams/Dev/universal-scraper/universal_scraper/apify
export OPENAI_API_KEY="sk-your-key-here"
./test-site.sh amazon
```

### Test Ticketmaster
```bash
./test-site.sh ticketmaster
```

### Test Leafly (Known Working)
```bash
./test-site.sh leafly
```

---

## 🔍 Test Case 1: Amazon Same-Day Store

**URL**: https://www.amazon.com/fmc/ssd-storefront

### Expected Challenges

#### 1. **Authentication Required** ⚠️
- Site shows: "Sign-in to shop from The Same-Day Store"
- **Expected Behavior**: Scraper extracts generic page structure (categories, links)
- **Limitation**: Cannot access product listings without login
- **Status**: **EXPECTED LIMITATION** (No auth support yet)

#### 2. **Location-Based Content** 🌍
- Page content changes based on user's location (delivery zip code)
- Default shows: "Delivering to Ashburn 20149"
- **Expected Behavior**: Scrapes whatever location Amazon defaults to
- **Limitation**: Cannot set custom delivery locations
- **Status**: **EXPECTED LIMITATION**

#### 3. **Dynamic Categories** 📦
- Categories load dynamically via JavaScript
- **Expected Behavior**: Browser mode should capture rendered content
- **Status**: **SHOULD WORK** ✅

### What to Look For

```bash
# After running test, check:

# 1. Was any data extracted?
cat apify_storage_local/datasets/default/*.json | jq 'length'

# 2. What fields were found?
cat apify_storage_local/datasets/default/*.json | jq '.[0] | keys'

# 3. Check for auth-related content
cat apify_storage_local/datasets/default/*.json | jq '.[] | select(.text? | contains("sign-in"))'
```

### Expected Results

**Scenario A: No Login** (Most Likely)
- ✅ Extracts: Page structure, navigation, category links
- ❌ Missing: Actual products (requires auth)
- **Conclusion**: Works as expected for public content

**Scenario B: Anti-Bot Block**
- ❌ No data or captcha page
- **Next Step**: Add stealth mode, rotate user agents

---

## 🔍 Test Case 2: Ticketmaster

**URL**: https://www.ticketmaster.com/

### Expected Challenges

#### 1. **Heavy Anti-Bot Protection** 🛡️
- Ticketmaster uses sophisticated bot detection
- May show captcha or block requests
- **Expected Behavior**: May get blocked on first try
- **Status**: **HIGH RISK EDGE CASE**

#### 2. **Complex JavaScript Rendering** ⚡
- Events load dynamically via multiple AJAX calls
- Content loads progressively as user scrolls
- **Expected Behavior**: Browser mode + scroll should capture most content
- **Status**: **SHOULD WORK** with `scrollToBottom: true` ✅

#### 3. **Regional Content** 🌎
- Different events shown based on user's location/IP
- **Expected Behavior**: Shows events for detected location
- **Status**: **EXPECTED BEHAVIOR**

#### 4. **Infinite Scroll** ♾️
- Events continue loading as user scrolls
- No clear pagination
- **Expected Behavior**: LLM pagination should detect and handle
- **Status**: **TEST THIS** 🧪

### What to Look For

```bash
# After running test, check:

# 1. Did it get blocked?
cat apify_storage_local/datasets/default/*.json | jq -r '.[] | select(.title? | contains("captcha", "blocked", "denied"))'

# 2. How many events extracted?
cat apify_storage_local/datasets/default/*.json | jq '[.[] | select(.eventName? or .title? or .name?)] | length'

# 3. What data structure?
cat apify_storage_local/datasets/default/*.json | jq '.[0]'

# 4. Check for Next.js data (Ticketmaster uses it)
grep -i "next_data" apify_storage_local/key_value_stores/default/*.html
```

### Expected Results

**Scenario A: Success** ✅
- Extracts 20-50+ events with names, dates, venues
- Captures `__NEXT_DATA__` from page
- **Conclusion**: Universal scraper handles it well

**Scenario B: Partial Success** ⚠️
- Extracts some data but incomplete
- Missing specific fields
- **Action**: Adjust field detection or add specific selectors

**Scenario C: Blocked** ❌
- Captcha page or empty results
- **Action**: Need to add:
  - Stealth mode
  - Better fingerprinting
  - Residential proxies (Apify has these)

---

## 📊 Comparison Matrix

| Feature | Leafly | Amazon SSD | Ticketmaster |
|---------|--------|------------|--------------|
| **Auth Required** | ❌ No | ✅ Yes | ❌ No |
| **Anti-Bot** | 🟢 Low | 🟡 Medium | 🔴 High |
| **JavaScript** | 🟢 Heavy | 🟢 Heavy | 🔴 Very Heavy |
| **Pagination** | ✅ URL-based | ⚠️ Auth wall | ⚠️ Infinite scroll |
| **JSON Data** | ✅ `__NEXT_DATA__` | ✅ Expected | ✅ `__NEXT_DATA__` |
| **Expected Success** | ✅ 95% | ⚠️ 60% | ⚠️ 50% |

---

## 🐛 Common Edge Cases to Identify

### 1. Authentication Walls
**Symptoms:**
- "Sign in" messages in extracted data
- Empty product lists
- Generic landing pages

**Detection:**
```bash
cat apify_storage_local/datasets/default/*.json | grep -i "sign.in\|login\|authenticate"
```

**Solution:**
- Document as limitation
- Consider adding auth support (future)

### 2. Bot Detection / Captcha
**Symptoms:**
- No data extracted
- "Access denied" messages
- Captcha challenges

**Detection:**
```bash
cat apify_storage_local/datasets/default/*.json | grep -i "captcha\|blocked\|denied\|suspicious"
```

**Solution:**
- Add stealth mode (playwright-stealth)
- Use residential proxies (Apify has these)
- Slow down requests

### 3. Location-Gated Content
**Symptoms:**
- Content varies by location
- "Not available in your area"
- Different results on different runs

**Detection:**
- Run test multiple times, compare results
- Check for location-related fields

**Solution:**
- Document as expected behavior
- Consider adding location spoofing (future)

### 4. Incomplete Data Extraction
**Symptoms:**
- Some items have data, others don't
- Missing key fields
- Inconsistent structure

**Detection:**
```bash
# Check for null/empty fields
cat apify_storage_local/datasets/default/*.json | jq '[.[] | to_entries | .[] | select(.value == null or .value == "")] | length'
```

**Solution:**
- Improve field detection
- Add fallback selectors
- Use LLM for complex extractions

### 5. Infinite Scroll Not Triggering
**Symptoms:**
- Only first screen of content extracted
- No pagination detected
- Low item count

**Detection:**
- Compare item count to website
- Check pagination metadata

**Solution:**
- Enable `scrollToBottom: true`
- Adjust scroll timing
- Use LLM pagination analyzer

---

## 📋 Testing Checklist

### For Each Site, Document:

- [ ] **Success Rate**: Did it extract data?
- [ ] **Item Count**: How many items extracted?
- [ ] **Field Quality**: Are fields complete and accurate?
- [ ] **Pagination**: Did it detect/handle pagination?
- [ ] **Speed**: How long did it take?
- [ ] **Errors**: Any errors in logs?
- [ ] **Edge Cases**: What failed or was unexpected?

### Create Issue Report Template:

```markdown
## Site: [URL]

### Results:
- Items Extracted: X
- Time Taken: Y seconds
- Success: ✅/⚠️/❌

### Edge Cases Found:
1. [Description]
   - Expected: [behavior]
   - Actual: [behavior]
   - Severity: High/Medium/Low

### Logs:
[Paste relevant error logs]

### Suggested Fix:
[Your recommendation]
```

---

## 🔧 Advanced Testing

### Test with Different Configurations

#### Minimal Config (Fast)
```json
{
  "scrapeConfig": {
    "fetchMode": "static",
    "fields": ["title", "price"]
  }
}
```

#### Full Browser (Slow but Comprehensive)
```json
{
  "scrapeConfig": {
    "fetchMode": "browser",
    "fields": [],
    "scrollToBottom": true,
    "waitForSelector": ".product-list"
  },
  "advancedConfig": {
    "enableLlmPagination": true,
    "browserTimeout": 120000
  }
}
```

### Compare Results
```bash
# Test 1: Static
./test-site.sh amazon
mv apify_storage_local/datasets/default results-amazon-static

# Test 2: Browser
# (Edit config to use browser mode)
./test-site.sh amazon
mv apify_storage_local/datasets/default results-amazon-browser

# Compare
diff -u results-amazon-static results-amazon-browser
```

---

## 📈 Success Criteria

### Tier 1: Basic Success ✅
- [ ] Extracts at least 1 data item
- [ ] No critical errors
- [ ] Completes in reasonable time (<5 min)

### Tier 2: Good Success 🌟
- [ ] Extracts 10+ items
- [ ] Fields are mostly complete
- [ ] Handles pagination if present
- [ ] No major data quality issues

### Tier 3: Excellent Success 🏆
- [ ] Extracts all available data
- [ ] Perfect field detection
- [ ] Automatic pagination works
- [ ] Fast and efficient
- [ ] Works consistently across runs

---

## 🆘 Troubleshooting

### "No data extracted"
1. Check if site requires auth
2. Look for bot detection
3. Try `headless: false` to see browser
4. Check logs for JavaScript errors

### "Only partial data"
1. Increase `browserTimeout`
2. Add `scrollToBottom: true`
3. Add `waitForSelector` for specific elements
4. Check if pagination was detected

### "Scraper hangs"
1. Reduce timeout values
2. Disable LLM pagination
3. Use simpler `fetchMode: "static"`
4. Check for infinite loops in pagination

---

## 📊 Results Tracking

Create a spreadsheet to track results:

| Site | URL | Items | Time | Pagination | Edge Cases | Status |
|------|-----|-------|------|------------|------------|--------|
| Leafly | [...] | 523 | 371s | ✅ URL-based | None | ✅ |
| Amazon SSD | [...] | 0 | 15s | ❌ Auth wall | Needs login | ⚠️ |
| Ticketmaster | [...] | ? | ?s | ? | ? | 🧪 |

---

## 🎯 Next Steps

After testing:

1. **Document Edge Cases**: Create issues for any limitations found
2. **Prioritize Fixes**: High-impact issues first
3. **Add Tests**: Create automated tests for edge cases
4. **Update Docs**: Add known limitations to README
5. **Iterate**: Test fixes and repeat

**Let's find those edge cases!** 🚀








