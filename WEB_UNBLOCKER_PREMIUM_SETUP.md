# Web Unblocker Premium Domain Setup

## ✅ Proxy Configuration Working

The Web Unblocker proxy is configured correctly and working:
- **Host**: `brd.superproxy.io:33335`
- **Username**: `brd-customer-hl_803e8195-zone-web_unlocker1`
- **Password**: `t8mhp1qev1i1`
- **Zone**: `web_unlocker1`

## ⚠️ Premium Domain Required

Chewy.com requires **Premium permissions** to be enabled in your Web Unlocker zone.

### Current Status

The proxy is working, but Chewy.com returns:
```
Targeting chewy.com requires Premium permissions. 
To enable 'Premium domains', go to your Web Unlocker zone configuration page
```

### Solution

1. **Go to Bright Data Dashboard**:
   - Visit: https://brightdata.com/cp/zones/web_unlocker1/edit?id=hl_803e8195
   - Or navigate to: Zones → web_unlocker1 → Edit

2. **Enable Premium Domains**:
   - Find "Premium Domains" or "Premium Permissions" section
   - Add `chewy.com` to the premium domains list
   - Save configuration

3. **Wait for Activation**:
   - Changes may take a few minutes to propagate
   - Test again after enabling

### Alternative: Use API Method

If you prefer using the API instead of proxy:

```bash
export BRIGHT_DATA_API_KEY="your-api-key"
export BRIGHT_DATA_ZONE="web_unlocker1"
```

Then use the API-based Web Unblocker fetcher (which may have different premium domain requirements).

## 🧪 Testing

Once Premium domains are enabled, test with:

```bash
python3 test_web_unblocker_fetch_only.py
```

Expected result:
- ✅ HTML size: > 50,000 bytes (full page)
- ✅ Contains product content
- ✅ No Kasada challenge

## 📋 Current Configuration

### Proxy Format

```python
web_unblocker_proxy = {
    'server': 'http://brd.superproxy.io:33335',
    'username': 'brd-customer-hl_803e8195-zone-web_unlocker1',
    'password': 't8mhp1qev1i1'
}
```

### Usage in Code

```python
from universal_scraper import UniversalScraper

scraper = UniversalScraper(
    proxy_config=web_unblocker_proxy,
    use_camoufox=True,
    fetch_mode='browser'
)
```

## 🔗 Resources

- **Zone Configuration**: https://brightdata.com/cp/zones/web_unlocker1/edit?id=hl_803e8195
- **Premium Domains Docs**: https://docs.brightdata.com/scraping-automation/web-unlocker/features#web-unlocker-api-premium-domains
- **Bright Data Dashboard**: https://brightdata.com/cp

---

**Status**: ⚠️ **Waiting for Premium Domain Activation**

Once Premium domains are enabled for Chewy.com, the Web Unblocker proxy will bypass Kasada automatically.

