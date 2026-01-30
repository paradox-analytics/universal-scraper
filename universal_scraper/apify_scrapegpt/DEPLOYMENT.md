# Apify Actor Deployment Guide

This guide explains how to deploy and test the Universal Scraper as an Apify Actor.

## 📋 Prerequisites

1. **Apify Account**: Create a free account at [https://apify.com](https://apify.com)
2. **Apify CLI**: Will be automatically installed by deployment script if not present
3. **API Keys**: Have an OpenAI, Gemini, or Claude API key ready

## 🧪 Local Testing (Recommended)

Before deploying, test the actor locally to ensure everything works:

### Step 1: Set up environment

```bash
cd universal_scraper/apify
export OPENAI_API_KEY="your-api-key-here"
```

### Step 2: Run local test

```bash
python test_actor_local.py
```

This will:
- Create a sample `test_input.json` file
- Run the actor logic locally
- Save output to `test_output.json`

### Step 3: Customize test input

Edit `test_input.json` to test different scenarios:

```json
{
  "urls": [
    "https://example.com/page1",
    "https://example.com/page2"
  ],
  "fields": ["title", "price", "description"],
  "proxyType": "none",
  "aiModel": "gpt-4o-mini",
  "apiKeys": {
    "openai_api_key": "sk-..."
  }
}
```

### Step 4: Review output

Check `test_output.json` for the scraped data and verify it matches expectations.

## 🚀 Deployment to Apify

### Quick Deployment

From the project root directory, run:

```bash
./deploy_to_apify.sh
```

This script will:
1. Check if Apify CLI is installed (install if needed)
2. Verify you're logged in (prompt login if needed)
3. Copy necessary files to deployment structure
4. Push the actor to Apify platform
5. Clean up temporary files

### Manual Deployment

If you prefer manual deployment:

```bash
# 1. Install Apify CLI
npm install -g apify-cli

# 2. Login to Apify
apify login

# 3. Navigate to project root
cd /path/to/universal-scraper

# 4. Prepare deployment structure
mkdir -p .actor
cp universal_scraper/apify/.actor/actor.json .actor/
cp universal_scraper/apify/INPUT_SCHEMA.json .actor/
cp universal_scraper/apify/Dockerfile .
cp universal_scraper/apify/README.md .

# 5. Deploy
apify push

# 6. Cleanup (optional)
rm -rf .actor/
rm Dockerfile
```

## ⚙️ Configuration

### Setting Up API Keys on Apify

You have two options for providing API keys:

#### Option 1: Environment Variables (Recommended for Security)

1. Go to [Apify Console](https://console.apify.com)
2. Navigate to your actor
3. Click "Settings" → "Environment Variables"
4. Add your API key:
   - Key: `OPENAI_API_KEY` (or `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`)
   - Value: Your actual API key
   - Check "Secret" to hide the value

#### Option 2: Input Schema (Per-run Configuration)

Provide API keys in the input when running the actor:

```json
{
  "urls": ["https://example.com"],
  "fields": ["title", "price"],
  "apiKeys": {
    "openai_api_key": "sk-..."
  }
}
```

### Proxy Configuration

The actor supports three proxy types:

1. **Residential** (Recommended): Best for production, highest success rate
   - Requires Apify proxy subscription or free trial
2. **Datacenter**: Cheaper alternative, suitable for less protected sites
   - Included with Apify subscription
3. **None**: No proxy (for testing or sites without protection)

## 🧩 Actor Structure

```
universal_scraper/apify/
├── actor.py              # Main actor logic
├── __init__.py           # Package init
├── Dockerfile            # Docker image definition
├── INPUT_SCHEMA.json     # Apify input schema
├── README.md             # Actor marketplace description
├── DEPLOYMENT.md         # This file
├── test_actor_local.py   # Local testing script
└── .actor/
    └── actor.json        # Actor metadata
```

## 📊 Monitoring & Debugging

### Check Actor Logs

After deployment, you can monitor runs:

1. Go to [Apify Console](https://console.apify.com)
2. Navigate to your actor
3. Click "Runs" tab
4. Select a run to view detailed logs

### Common Issues

#### "Actor failed to start"
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt
- Review build logs in Apify console

#### "No data extracted"
- Check API key is valid and has credits
- Verify URL is accessible
- Review actor logs for specific errors
- Test locally with `test_actor_local.py`

#### "Proxy error"
- Ensure you have Apify proxy subscription/trial
- Try switching proxy type (residential → datacenter → none)
- Check proxy configuration in actor logs

#### "AI model error"
- Verify API key matches selected model
- Check AI provider account has sufficient credits
- Try a different model (e.g., gpt-4o-mini instead of gpt-4o)

## 💰 Cost Optimization Tips

1. **Use Caching**: The actor caches extraction code by page structure
   - First page: ~$0.001
   - Cached pages: ~$0.00001
   - 90%+ cost savings on similar pages!

2. **Choose Right Model**:
   - `gpt-4o-mini`: Best value ($0.15/1M tokens)
   - `gemini-2.0-flash-exp`: Free tier available
   - `claude-3-haiku`: Fast and cheap

3. **Batch Similar URLs**: Group URLs with same structure for maximum cache hits

4. **Monitor Usage**: Check "Statistics" tab in Apify console

## 🔄 Updating the Actor

To update after making code changes:

```bash
# Make your changes to the code
# Then redeploy
./deploy_to_apify.sh
```

The deployment script will:
- Build a new Docker image
- Push updated code to Apify
- Keep your existing runs and data

## 📈 Scaling

### Running at Scale

The actor supports:
- Multiple URLs in single run
- Concurrent processing
- Automatic retry on failures
- Comprehensive error logging

### Best Practices

1. **Group Similar Pages**: URLs with similar structure benefit from caching
2. **Use Metamorph**: Chain with other Apify actors for complex workflows
3. **Schedule Runs**: Set up scheduled runs for regular data updates
4. **Monitor Quotas**: Keep eye on Apify platform usage and AI provider quotas

## 🆘 Support

If you encounter issues:

1. **Check Logs**: Always start with the run logs in Apify console
2. **Test Locally**: Use `test_actor_local.py` to reproduce issues
3. **Review Documentation**: Check main README.md for general usage
4. **Open Issue**: Report bugs on GitHub with logs and reproduction steps

## 📚 Additional Resources

- [Apify Documentation](https://docs.apify.com)
- [Apify SDK for Python](https://docs.apify.com/sdk/python)
- [Main Project README](../../README.md)
- [Examples Directory](../../examples/)

## 🎉 Success Checklist

- [ ] Tested locally with `test_actor_local.py`
- [ ] API keys configured (environment variables or input)
- [ ] Deployment successful (`./deploy_to_apify.sh`)
- [ ] Test run completed successfully on Apify
- [ ] Reviewed output data quality
- [ ] Monitoring set up (if needed)
- [ ] Ready for production! 🚀


