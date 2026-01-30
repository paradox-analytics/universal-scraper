# 🌐 Using Web Unblocker with Chewy.com on Apify

## Overview

This guide shows you how to configure the Universal Scraper Actor on Apify to scrape Chewy.com using Bright Data's Web Unblocker to bypass Kasada anti-bot protection.

## Prerequisites

1. **Bright Data Account** with Web Unblocker enabled
2. **Web Unblocker Credentials** (proxy format or API key)
3. **OpenAI API Key** (for LLM extraction)

## Step 1: Get Your Web Unblocker Credentials

### Option A: Proxy Credentials Format (Recommended)

From your Bright Data dashboard, get your Web Unblocker credentials in one of these formats:

**Comma-separated format:**
```
brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS
```

**Colon-separated format:**
```
brd.superproxy.io:33335:brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1:REDACTED_PROXY_PASS
```

**Format breakdown:**
- `brd.superproxy.io` = Host
- `33335` = Port
- `brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1` = Username
- `REDACTED_PROXY_PASS` = Password

### Option B: API Key Format (Bearer Token)

If you have a Bright Data API key (Bearer token), you can use that instead:
```
your-bright-data-api-key-here
```

## Step 2: Configure Apify Actor Input

### Basic Configuration

1. **Start URLs:**
   ```json
   [
     {
       "url": "https://www.chewy.com/b/wet-food-389"
     }
   ]
   ```

2. **Fields to Extract:**
   ```json
   [
     "title",
     "rating",
     "review count",
     "price"
   ]
   ```

3. **OpenAI API Key:**
   - Enter your OpenAI API key
   - Or set `OPENAI_API_KEY` as an environment variable

### Web Unblocker Configuration

4. **Web Unblocker API Key:**
   - **For Proxy Credentials:** Enter your credentials in either format:
     - `brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS`
     - `brd.superproxy.io:33335:brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1:REDACTED_PROXY_PASS`
   - **For API Key:** Enter your Bright Data Bearer token
   - **Note:** You can also set `BRIGHT_DATA_API_KEY` as an environment variable

5. **Web Unblocker Zone:**
   - Default: `web_unlocker1`
   - Change if your zone has a different name

### Pagination Configuration

6. **Enable Auto Pagination:**
   - Set to `true` to scrape all pages automatically

7. **Max Pages (Optional):**
   - **Leave blank or set to 0** = Scrape all pages
   - **Set to a number** = Limit to that many pages (e.g., `3` for testing)
   - **Recommended for testing:** Set to `3` to limit scraping time

### Other Recommended Settings

8. **Use Direct LLM Extraction:**
   - Set to `true` (recommended for maximum accuracy)

9. **Direct LLM Quality Mode:**
   - `balanced` (default) - Good balance of quality and quantity
   - `conservative` - Highest quality, fewer items
   - `aggressive` - Maximum items, may include lower quality

10. **Proxy Configuration:**
    - **Leave empty** - Web Unblocker handles proxy routing
    - Or configure Apify residential proxies as backup

## Step 3: Complete Input Example

Here's a complete JSON input example for Chewy.com:

```json
{
  "startUrls": [
    {
      "url": "https://www.chewy.com/b/wet-food-389"
    }
  ],
  "fields": [
    "title",
    "rating",
    "review count",
    "price"
  ],
  "openaiApiKey": "sk-proj-YOUR-OPENAI-KEY-HERE",
  "webUnblockerApiKey": "brd.superproxy.io,33335,brd-customer-REDACTED_CUSTOMER_ID-zone-web_unlocker1,REDACTED_PROXY_PASS",
  "webUnblockerZone": "web_unlocker1",
  "enableAutoPagination": true,
  "maxPages": 3,
  "useDirectLLM": true,
  "directLLMQualityMode": "balanced"
}
```

## Step 4: How It Works

### Automatic Blocking Detection

1. **Initial Attempt:** The scraper tries to fetch Chewy.com using standard methods
2. **Blocking Detected:** If the response is suspiciously small (< 5KB) or contains Kasada indicators, blocking is detected
3. **Web Unblocker Fallback:** Automatically switches to Web Unblocker to bypass protection
4. **Success:** Web Unblocker returns full HTML content (6-7MB) instead of blocked content (840 bytes)

### What Gets Logged

Look for these log messages:
- `🔍 Blocked: E-commerce site with small HTML and no product indicators`
- `🌐 Falling back to Bright Data Web Unblocker...`
- `🌐 Using Native Proxy-Based Access (proxy credentials)` or `🌐 Using Direct API Access (Bearer token)`
- `✅ Web Unblocker fetch successful: 6,763,689 bytes`

## Step 5: Testing Tips

### For Quick Testing

1. **Set `maxPages` to 1-3** to limit scraping time
2. **Monitor logs** to see Web Unblocker fallback in action
3. **Check dataset** for extracted items

### For Production

1. **Set `maxPages` to 0** (or leave blank) to scrape all pages
2. **Monitor costs** - Web Unblocker has usage costs
3. **Check rate limits** - Bright Data may have rate limits

## Troubleshooting

### Issue: Web Unblocker Not Triggering

**Symptoms:**
- Small HTML responses (< 1KB)
- Kasada challenge pages detected
- No "Falling back to Web Unblocker" message

**Solutions:**
1. Verify Web Unblocker credentials are correct
2. Check that Web Unblocker zone is correct
3. Ensure Web Unblocker is enabled in Bright Data dashboard
4. Check that Chewy.com is allowed in your Web Unblocker zone

### Issue: Authentication Failed

**Symptoms:**
- `❌ Authentication failed - check API key`
- `401 Unauthorized` errors

**Solutions:**
1. **For Proxy Credentials:** Verify format is correct (comma or colon separated)
2. **For API Key:** Ensure you're using a Bright Data Bearer token, not proxy password
3. Check credentials in Bright Data dashboard

### Issue: Still Getting Blocked

**Symptoms:**
- Web Unblocker returns small HTML
- Kasada challenges still present

**Solutions:**
1. Verify Web Unblocker zone has "Premium domains" enabled for `chewy.com`
2. Check Bright Data account balance
3. Try increasing timeout values
4. Contact Bright Data support if issue persists

## Cost Considerations

### Web Unblocker Costs

- **Default:** ~$1.80/CPM (cost per 1000 requests)
- **Premium:** ~$2.80/CPM (for premium domains like Chewy.com)
- **Check your Bright Data dashboard** for exact pricing

### Optimization Tips

1. **Use caching:** Direct LLM results are cached, reducing repeated requests
2. **Limit pages:** Use `maxPages` to control scraping scope
3. **Monitor usage:** Check Bright Data dashboard for usage stats

## Example Output

After running, you'll get a dataset with extracted items like:

```json
{
  "title": "Purina Pro Plan Complete Essentials Adult Canned Wet Dog Food",
  "rating": "4.5",
  "review count": "1,234",
  "price": "$45.99"
}
```

## Support

- **Actor Build:** https://console.apify.com/organization/YwaXmjFoleWBUmZdU/actors/MSwDish8FXKQKiIyx#/builds/2.0.26
- **Actor Detail:** https://console.apify.com/organization/YwaXmjFoleWBUmZdU/actors/MSwDish8FXKQKiIyx
- **Bright Data Docs:** https://docs.brightdata.com/scraping-automation/web-unlocker

## Quick Reference

| Setting | Value for Chewy |
|---------|----------------|
| **URL** | `https://www.chewy.com/b/wet-food-389` |
| **Web Unblocker API Key** | Your proxy credentials or API key |
| **Web Unblocker Zone** | `web_unlocker1` (or your zone name) |
| **Max Pages** | `3` (testing) or `0` (all pages) |
| **Enable Auto Pagination** | `true` |
| **Use Direct LLM** | `true` |

---

**Last Updated:** Build 2.0.26
**Deployment Date:** 2025-11-28







