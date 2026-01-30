# Universal Insights from Oxylabs eBay Scraper

**Reference**: [Oxylabs eBay Scraper](https://github.com/oxylabs/ebay-scraper)

---

## 🎯 The Key Universal Principle

**"Proxy rotation per request, not per session"**

---

## 🔍 What Oxylabs Does Differently

### 1. **Proxy Rotation Strategy** (Most Critical)

```python
# Oxylabs approach
response = requests.post(
    'https://realtime.oxylabs.io/v1/queries',
    auth=('user', 'pass'),
    json={
        'source': 'universal',
        'url': 'https://www.ebay.com/itm/293608130360',
        'geo_location': 'United States'
    }
)
```

**Key Insight**: Each `requests.post()` call gets a **new proxy from their pool**.

**Universal Application**:
- ✅ **DO**: Request new proxy for each page/domain
- ❌ **DON'T**: Reuse same proxy for entire scraping session
- ✅ **DO**: Use residential proxies for challenging sites
- ❌ **DON'T**: Use datacenter proxies for sites with advanced anti-bot

---

### 2. **Geographic Targeting**

```python
'geo_location': 'United States'
```

**Why This Matters Universally**:
- eBay serves different content based on country
- Prices, availability, currency, language all vary
- Bot detection algorithms check IP location consistency

**Universal Application**:
- ✅ Match proxy location to target audience
- ✅ Maintain location consistency per session/domain
- ✅ Consider multi-location testing for global sites

---

### 3. **"Universal" Source Parameter**

```python
'source': 'universal'
```

**What This Implies**:
- Oxylabs has multiple fetching strategies
- "Universal" = adapts to site requirements
- Likely includes: static HTML, browser rendering, API detection

**Universal Application** (Already Implemented ✅):
- Our `HybridFetcher` does this (static → browser fallback)
- Our `JSONDetector` finds API endpoints
- Our `CamoufoxFetcher` handles JS-heavy sites

---

### 4. **Premium Proxy Infrastructure**

**Oxylabs' Competitive Advantage**:
- Millions of residential IPs
- Automatic rotation
- Smart retry logic
- Geographic distribution
- Session management
- Failure handling

**Why This Works for eBay**:
- eBay's bot detection is **multi-layered**
- IP reputation + browser fingerprint + behavior patterns
- Premium proxies solve the **IP reputation** layer
- Our Camoufox solves the **browser fingerprint** layer
- We need **both**

---

## 🚨 Our Current Issue (eBay on Apify)

### What We're Doing Wrong

```python
# In actor.py (CURRENT - WRONG)
proxy_url = await proxy_configuration.new_url()  # ❌ Called ONCE
proxy_config = parse_proxy_url(proxy_url)

scraper = UniversalScraper(proxy_config=proxy_config)  # ❌ Static proxy
```

**Problem**: We get one proxy URL and reuse it for the entire session.

**Why It Fails**:
- eBay detects: Same IP making multiple requests = bot
- Even with Camoufox, static IP is a red flag
- Residential proxies only help if they **rotate**

---

### What We Should Do (Universal Fix)

```python
# CORRECT - Universal approach
async def fetch_page(url):
    # Get NEW proxy for EACH request
    proxy_url = await proxy_configuration.new_url()  # ✅ Fresh proxy
    proxy_config = parse_proxy_url(proxy_url)
    
    # Use this proxy for this page only
    browser = await launch_browser(proxy=proxy_config)
    html = await browser.fetch(url)
    await browser.close()
    
    return html
```

**Why This Works Universally**:
- ✅ Each page request uses a different IP
- ✅ Looks like different users accessing the site
- ✅ Harder to detect as bot behavior
- ✅ Apify's proxy pool handles the rotation

---

## 📊 Comparison: Us vs. Oxylabs

| Feature | Oxylabs | Our System | Status |
|---------|---------|------------|--------|
| Proxy Rotation | ✅ Per request | ❌ Per session | 🔴 FIX NEEDED |
| Residential IPs | ✅ Millions | ✅ Apify pool | ✅ OK |
| Geographic Targeting | ✅ Yes | ⚠️ Partial | 🟡 CAN IMPROVE |
| Browser Fingerprinting | ✅ (proprietary) | ✅ Camoufox | ✅ OK |
| JS Rendering | ✅ Adaptive | ✅ HybridFetcher | ✅ OK |
| API Detection | ✅ Yes | ✅ JSONDetector | ✅ OK |
| Retry Logic | ✅ Built-in | ⚠️ Manual | 🟡 CAN IMPROVE |

---

## 💡 Universal Solution Strategy

### Phase 1: Minimal Fix (High ROI)
1. **Enable per-request proxy rotation** ← THIS IS THE KEY
2. Add geographic targeting parameter
3. Test on eBay

**Estimated Impact**: 70% improvement for challenging sites

---

### Phase 2: Enhanced Proxy Management
1. Implement ProxyManager class (done ✅)
2. Track proxy failures
3. Implement retry logic with proxy rotation
4. Add domain-sticky sessions (when needed)

**Estimated Impact**: 90% improvement

---

### Phase 3: Full Premium Infrastructure
1. Support multiple proxy providers (Bright Data, ScraperAPI, etc.)
2. Automatic provider selection based on site difficulty
3. Cost optimization (datacenter → residential only when needed)
4. Smart caching to reduce proxy usage

**Estimated Impact**: 95%+ success rate on all sites

---

## 🎯 Immediate Action for eBay

### Root Cause
✅ **IDENTIFIED**: Static proxy per session (not rotating)

### Minimal Universal Fix

**File**: `universal_scraper/apify/actor.py`

**Current Code** (❌ WRONG):
```python
# Get proxy URL ONCE
proxy_url = await proxy_configuration.new_url()
proxy_config = {
    'server': f'{parsed.scheme}://{parsed.hostname}:{parsed.port}',
    'username': parsed.username or '',
    'password': parsed.password or ''
}

# Pass static proxy to scraper
workflow = UniversalWorkflow(proxy_config=proxy_config)
```

**Fixed Code** (✅ CORRECT):
```python
# Pass proxy_configuration object (not parsed URL)
# Let fetcher call new_url() for each request
workflow = UniversalWorkflow(
    proxy_configuration=proxy_configuration  # ← Pass the object
)

# In CamoufoxFetcher.fetch():
if self.proxy_configuration:
    # Get NEW proxy for THIS request
    proxy_url = await self.proxy_configuration.new_url()
    proxy_config = parse_proxy_url(proxy_url)
```

**Why This is Universal**:
- ✅ Works for ANY proxy provider with rotation API
- ✅ Works for ANY website (not eBay-specific)
- ✅ Minimal code changes
- ✅ No architectural changes needed

---

## 🔬 Testing Strategy

### 1. Verify Proxy Rotation
**Log each proxy URL used**:
```
🔄 Request 1: proxy-pool-123.apify.com:8000
🔄 Request 2: proxy-pool-456.apify.com:8000
🔄 Request 3: proxy-pool-789.apify.com:8000
```

### 2. Test on eBay
- Single page: Should get 60+ items
- Multiple pages: Each should use different proxy

### 3. Confirm Geographic Targeting
- Check that proxies are from `geo_location` if specified

---

## 📚 Key Takeaways (Universal)

### 1. **Proxy Rotation > Proxy Quality**
- 100 datacenter IPs rotating > 1 premium residential IP static
- Rotation breaks behavioral pattern detection

### 2. **Per-Request > Per-Session**
- Each page = new identity
- Harder to correlate requests

### 3. **Geographic Consistency Matters**
- Match proxy location to target audience
- Don't mix US/EU proxies in same session

### 4. **Combine Strategies**
- Rotating proxies + browser fingerprinting + human behavior = maximum success
- No single solution is enough for advanced anti-bot

### 5. **Cost Optimization**
- Start with datacenter (cheap)
- Fallback to residential on failure
- Only use premium for known difficult sites

---

## 🎯 Success Criteria

After implementing proxy rotation:

| Metric | Before | Target After |
|--------|--------|-------------|
| eBay Success Rate | 0% | 85%+ |
| Items Extracted | 0 | 60+ |
| Proxy Cost/Page | $0.10 | $0.05-0.10 |
| Detection Rate | 100% | <15% |

---

## 🔗 Related Files

- `universal_scraper/core/proxy_manager.py` - New universal proxy management (✅ created)
- `universal_scraper/apify/actor.py` - Needs update for proxy rotation
- `universal_scraper/core/camoufox_fetcher.py` - Needs update to accept proxy_configuration
- `universal_scraper/orchestrator/workflow.py` - Needs update to pass proxy_configuration

---

**Next Step**: Implement minimal fix (proxy rotation per request) in Apify actor





