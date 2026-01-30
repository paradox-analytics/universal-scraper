# Web Unblocker Deployment Guide

## Overview

The Universal Scraper Apify Actor now includes **Bright Data Web Unblocker** support as a fallback mechanism for sites protected by Kasada, Cloudflare, and other advanced anti-bot solutions.

## Features

✅ **Automatic Fallback**: When standard proxies are blocked, automatically falls back to Web Unblocker  
✅ **Kasada Bypass**: Specifically handles Kasada challenges  
✅ **Cloudflare Bypass**: Works with Cloudflare protection  
✅ **Zero Configuration**: Works out-of-the-box with just API key  
✅ **Premium Domains**: Supports premium domain configuration  

## Input Schema

### Required Fields

- `startUrls`: Array of URLs to scrape
- `fields`: Array of field names to extract (e.g., `["name", "price", "rating"]`)

### Optional Fields

#### Web Unblocker Configuration

```json
{
  "webUnblockerApiKey": "your-bright-data-api-key",
  "webUnblockerZone": "web_unlocker1"
}
```

**Note**: You can also set `BRIGHT_DATA_API_KEY` as an environment variable in Apify.

#### External Proxy Configuration

```json
{
  "useExternalProxy": true,
  "externalProxyServer": "http://brd.superproxy.io:33335",
  "externalProxyUsername": "brd-customer-xxx-zone-xxx",
  "externalProxyPassword": "your-password"
}
```

#### Apify Proxy Configuration

```json
{
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"]
  }
}
```

**Priority**: External proxy > Apify proxy > No proxy

## Usage Examples

### Example 1: Chewy.com with Web Unblocker

```json
{
  "startUrls": [
    {"url": "https://www.chewy.com/b/wet-food-389"}
  ],
  "fields": ["name", "price", "rating", "reviewCount"],
  "webUnblockerApiKey": "your-api-key",
  "webUnblockerZone": "web_unlocker1",
  "useExternalProxy": true,
  "externalProxyServer": "http://brd.superproxy.io:33335",
  "externalProxyUsername": "brd-customer-xxx-zone-web_unlocker1",
  "externalProxyPassword": "your-password",
  "enableAutoPagination": false
}
```

### Example 2: Using Apify Proxy with Web Unblocker Fallback

```json
{
  "startUrls": [
    {"url": "https://example.com/products"}
  ],
  "fields": ["title", "price", "description"],
  "proxyConfiguration": {
    "useApifyProxy": true,
    "apifyProxyGroups": ["RESIDENTIAL"]
  },
  "webUnblockerApiKey": "your-api-key",
  "webUnblockerZone": "web_unlocker1"
}
```

### Example 3: Environment Variable Configuration

Set in Apify Actor environment variables:
- `BRIGHT_DATA_API_KEY`: Your Bright Data API key
- `OPENAI_API_KEY`: Your OpenAI API key

Then use minimal input:
```json
{
  "startUrls": [{"url": "https://example.com"}],
  "fields": ["title", "description"]
}
```

## How It Works

1. **Primary**: Attempts scraping with configured proxy (external or Apify)
2. **Detection**: Detects if blocked (Kasada challenge, small HTML, etc.)
3. **Fallback**: Automatically switches to Web Unblocker if blocked
4. **Extraction**: Extracts data using universal JSON-first approach

## Premium Domains

Some sites (like Chewy.com) require **Premium domains** to be enabled in your Bright Data dashboard:

1. Go to Bright Data dashboard
2. Navigate to your Web Unblocker zone
3. Enable "Premium domains" for the target domain
4. Wait 5-15 minutes for changes to propagate

## Deployment Steps

### 1. Update Actor Code

The actor code (`main.py`) is already updated with Web Unblocker support.

### 2. Update Input Schema

The `INPUT_SCHEMA.json` includes all new fields.

### 3. Deploy to Apify

```bash
cd universal_scraper/apify
apify push
```

### 4. Configure Environment Variables (Optional)

In Apify Actor settings:
- `BRIGHT_DATA_API_KEY`: Your Bright Data API key
- `OPENAI_API_KEY`: Your OpenAI API key

### 5. Test Run

Run a test with a protected site (e.g., Chewy.com) to verify Web Unblocker fallback works.

## Troubleshooting

### Web Unblocker Not Activating

- Check that `webUnblockerApiKey` is set (or `BRIGHT_DATA_API_KEY` env var)
- Verify API key is valid
- Check logs for "Web Unblocker: Disabled" messages

### Premium Permissions Error

- Enable "Premium domains" in Bright Data dashboard for the target domain
- Wait 5-15 minutes for propagation
- Verify domain is added correctly (try both `example.com` and `www.example.com`)

### Proxy Configuration Issues

- External proxy takes priority over Apify proxy
- If both are configured, external proxy is used
- Check proxy credentials are correct

## Cost Considerations

- **Web Unblocker**: Pay-per-use pricing (check Bright Data pricing)
- **Standard Proxy**: Standard proxy costs
- **Fallback Only**: Web Unblocker only used when standard proxy fails

## Best Practices

1. **Start with Standard Proxy**: Use Apify or external proxy first
2. **Enable Fallback**: Set Web Unblocker API key for automatic fallback
3. **Monitor Logs**: Check logs to see when Web Unblocker is used
4. **Premium Domains**: Enable for sites that require it
5. **Test First**: Test with a single URL before large-scale scraping

## Support

For issues:
1. Check actor logs in Apify
2. Verify API keys are correct
3. Check Bright Data dashboard for zone configuration
4. Review Web Unblocker documentation

