# 🔬 Metacritic Issue Analysis + CSV Sample Review

## Executive Summary

**All 4 test sites extracted the WRONG JSON data.**

The context-aware JSON ranking system is **NOT working as intended**. Instead of extracting the target data (posts, actors, games, products), the system is extracting:
- Config/settings JSON
- Analytics tracking data
- UI state/action objects
- Library/plugin metadata

---

## 🔍 What We Found in the CSVs

### 1. Reddit (reddit_sample.csv) ❌

**Expected:** Reddit posts with title, author, upvotes  
**Got:** Application config data (SSO settings, manifest URLs, CDN origins)

**Sample Row:**
```csv
ACCOUNT_MANAGER_ORIGIN,APPLE_SSO_CLIENT_ID,DEVVIT_GATEWAY_ORIGIN...
https://www.reddit.com,com.reddit.RedditAppleSSO,https://devvit-gateway.reddit.com...
```

**Problem:** The JSON ranking selected Reddit's app configuration instead of the GraphQL API response containing actual posts.

---

### 2. Apify (apify_sample.csv) ❌

**Expected:** Apify Actors with name, description, author  
**Got:** JavaScript library configs (Algolia plugins, analytics integrations)

**Sample Row:**
```csv
libraryName,name,settings,url
algolia-pluginsDestination,Algolia Insights (Actions)...
```

**Problem:** The JSON ranking selected Segment/Algolia analytics configs instead of the actual actor listing data.

---

### 3. Metacritic (metacritic_sample.csv) ❌

**Expected:** Video game listings with title, platform, score  
**Got:** Ad banner configs, GDPR settings, schema.org metadata

**Sample Row:**
```csv
@context,@type,BannerPushesDown,Conditions,Countries,Default...
,,,,['us'],,,True,,0195f047-3446-7608-a99a-b9fba5214b31
```

**Problem:** The JSON ranking selected advertising/tracking JSON instead of game listing data. This is the **critical flaw** in the context-aware ranking - it's not actually using the context effectively.

---

### 4. eBay (ebay_sample.csv) ⚠️

**Expected:** Laptop listings with title, price, condition  
**Got:** UI action tracking objects (click handlers, browse operations)

**Sample Row:**
```csv
_type,about,action
Group,,"{'_type': 'Action', 'type': 'OPERATION', 'name': 'REFRESH_BROWSE'..."
```

**Problem:** The JSON ranking selected UI state/tracking data instead of product listings.

---

## 🎯 Root Cause Analysis

### Why is Context-Aware JSON Ranking Failing?

Based on the CSV results, here are the likely issues:

### 1. **Pre-Filter is Too Aggressive** ❌
The `_pre_filter_sources` in `json_analyzer.py` is supposed to remove analytics/config JSON, but it's clearly not working. The fact that **all 4 sites** extracted non-data JSON suggests:
- Analytics patterns aren't being caught
- Config/settings JSON isn't being filtered
- The keyword matching isn't effective

### 2. **LLM Ranking is Not Running** ❌
If the LLM was properly analyzing the JSON with the user's context, it would never rank:
- "SSO settings" as relevant to "Reddit posts with title, author"
- "Algolia plugins" as relevant to "Actors with name, description"
- "Ad banner configs" as relevant to "games with title, platform, score"

**This suggests:**
- The LLM isn't being called at all
- OR the LLM is being called but its output is ignored
- OR there's a code path that bypasses the context system entirely

### 3. **Fallback to Simple JSON Detection** ⚠️
Looking at the data, it appears the system is falling back to the **old, non-context-aware JSON detection** which just picks the first/largest JSON source it finds (which is often config/analytics data).

---

## 🔍 Debugging the Code Flow

Let me trace the execution path:

### Expected Flow (Context-Aware):
```
1. Fetch HTML + captured JSON ✅
2. Detect all JSON sources ✅
3. Pre-filter (remove analytics/config) ❌ FAILED
4. Rank remaining sources with LLM + context ❓ UNKNOWN
5. Validate top source with context ❓ UNKNOWN
6. Extract data ✅
```

### Actual Flow (Based on Results):
```
1. Fetch HTML + captured JSON ✅
2. Detect all JSON sources ✅
3. Pre-filter does nothing (all sources pass) ❌
4. LLM ranking skipped OR ignored ❌
5. Default to first JSON source found ❌
6. Extract wrong data ✅
```

---

## 🐛 Specific Issues to Fix

### Issue #1: Pre-Filter Not Catching Analytics
**File:** `universal_scraper/core/json_analyzer.py`  
**Method:** `_pre_filter_sources`

**Current Logic:**
- Checks for analytics keywords in source names
- Checks for empty/small JSON
- **Does NOT check JSON structure/content**

**What's Missing:**
- Reddit config has no "analytics" in the name → passes filter
- Apify libraries have no "analytics" in the name → passes filter
- Metacritic ad configs have no "analytics" in the name → passes filter

**Fix Needed:**
- Check JSON structure (does it have arrays of items?)
- Check JSON content (does it match context keywords?)
- Be more aggressive with non-data patterns

---

### Issue #2: Context Keywords Not Working
**File:** `universal_scraper/core/json_analyzer.py`  
**Method:** `_could_contain_target_data`

**Current Logic:**
```python
# Extract keywords from context
context_keywords = set()
if context.data_type:
    context_keywords.add(context.data_type.lower())
if context.fields:
    context_keywords.update(f.lower() for f in context.fields)

# Check if keywords appear in JSON
if matches >= 2:
    return True
```

**What's Wrong:**
- Reddit config JSON has "ORIGIN", "CLIENT_ID" → no match with "posts", "title", "author"
- But it's still being extracted!

**This means:** The keyword matching isn't actually being enforced, OR the LLM ranking is overriding it.

---

### Issue #3: LLM Ranking Not Being Applied
**File:** `universal_scraper/core/scraper.py`  
**Method:** `scrape` (JSON detection flow)

**Expected:**
```python
if self.json_analyzer and extraction_context:
    rankings = await self.json_analyzer.rank_sources(...)
    for ranked_source in rankings:
        # Try top sources
```

**What Might Be Happening:**
- `self.json_analyzer` is None (not initialized)
- OR `extraction_context` is missing/empty
- OR rankings return empty/fail
- OR there's a fallback that bypasses ranking

**Fix Needed:**
- Add logging to confirm LLM ranking is called
- Add logging to show ranked results
- Ensure fallback doesn't skip context validation

---

## 📊 Test Results Summary

| Site | Context Provided | Expected Data | Actual Data | Status |
|------|-----------------|---------------|-------------|--------|
| **Reddit** | "posts with title, author" | Reddit posts (4) | App config | ❌ WRONG |
| **Apify** | "actors with name, description" | Apify Actors (10+) | JS libraries (2) | ❌ WRONG |
| **Metacritic** | "games with title, platform, score" | Game listings (25+) | Ad configs (5) | ❌ WRONG |
| **eBay** | "laptops with title, price" | Product listings (50+) | UI actions (4) | ❌ WRONG |

**Success Rate:** 0/4 (0%)

---

## 🔧 Recommended Fixes

### Priority 1: Verify Context System is Active

Add extensive logging to `scraper.py`:
```python
# In scrape() method, JSON detection flow
logger.info(f"🎯 Context-aware ranking: {bool(self.json_analyzer)}")
logger.info(f"🎯 Extraction context: {extraction_context[:100] if extraction_context else 'NONE'}")

if self.json_analyzer and extraction_context:
    logger.info(f"🎯 Calling LLM JSON ranking...")
    rankings = await self.json_analyzer.rank_sources(...)
    logger.info(f"🎯 LLM ranked {len(rankings)} sources")
    for i, rank in enumerate(rankings[:3]):
        logger.info(f"   {i+1}. {rank['source']}: confidence {rank['confidence']}")
else:
    logger.warning(f"⚠️ CONTEXT SYSTEM NOT ACTIVE - falling back to simple detection")
```

### Priority 2: Fix Pre-Filter

Make `_pre_filter_sources` more aggressive:
```python
def _pre_filter_sources(self, json_sources, context):
    """
    AGGRESSIVE filtering - reject anything that's obviously not data
    """
    filtered = {}
    
    for name, data in json_sources.items():
        # Reject if no arrays (most data is in arrays)
        if not self._contains_arrays(data):
            logger.debug(f"   ✗ {name}: No arrays found")
            continue
        
        # Reject if keywords don't match context AT ALL
        if not self._could_contain_target_data(data, context):
            logger.debug(f"   ✗ {name}: No context match")
            continue
        
        # Reject if looks like config/settings
        if self._is_config_or_settings(name, data):
            logger.debug(f"   ✗ {name}: Config/settings pattern")
            continue
        
        filtered[name] = data
    
    return filtered
```

### Priority 3: Add Validation

After extracting data, validate it matches the context:
```python
# After extraction
if len(extracted_data) > 0:
    # Check if data looks right
    sample_item = extracted_data[0]
    has_expected_fields = any(
        field.lower() in str(sample_item).lower()
        for field in context.fields or []
    )
    
    if not has_expected_fields:
        logger.warning(f"⚠️ Extracted data doesn't match context!")
        logger.warning(f"   Expected fields: {context.fields}")
        logger.warning(f"   Got keys: {list(sample_item.keys())[:10]}")
        # Fall back to HTML extraction
```

---

## 🎯 Next Steps

1. **Add Logging** - Confirm context system is being called
2. **Test Logging** - Re-run Reddit/Apify with debug logs
3. **Fix Pre-Filter** - Be more aggressive with non-data JSON
4. **Fix Keyword Matching** - Ensure context keywords are enforced
5. **Test Again** - Verify correct data is extracted

---

## 📝 Conclusion

**Phase 1 (HTML Cleaner):** ✅ Working  
**Phase 2 (Code Generation):** ✅ Working  
**Context-Aware JSON Ranking:** ❌ **NOT working**

The fundamental issue is that the **context system is not being applied to JSON extraction**. All 4 sites extracted the wrong JSON, which would be impossible if the LLM was properly ranking sources based on user context.

**This is a critical bug that must be fixed before the system can be considered truly universal.**

---

**Files Generated:**
- `reddit_sample.csv` - 4 rows of config data (wrong)
- `apify_sample.csv` - 2 rows of library data (wrong)
- `metacritic_sample.csv` - 5 rows of ad configs (wrong)
- `ebay_sample.csv` - 4 rows of UI actions (wrong)








