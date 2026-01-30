# 🧪 Local Testing Guide

Test your Apify Actor locally before deploying to catch errors early.

---

## Method 1: Direct Python Testing (Fastest)

Test the core functionality without Docker:

```bash
# Test the crawler directly
python3 test_local_simple.py
```

**Pros:**
- ✅ Fast (no Docker build)
- ✅ Easy to debug
- ✅ See real-time output

**Cons:**
- ❌ Doesn't test the full Apify environment
- ❌ Requires local dependencies

---

## Method 2: Docker Testing (Most Accurate)

Test the actual Docker image that will run on Apify:

### Step 1: Build the Docker image locally

```bash
cd universal_scraper/apify
docker build -t universal-scraper-test .
```

### Step 2: Create a test input file

```bash
cat > /tmp/test-input.json << 'EOF'
{
  "mode": "crawl_only",
  "startUrls": [
    {"url": "https://www.leafly.com/dispensaries/nevada"}
  ],
  "crawlConfig": {
    "mode": "smart",
    "maxDepth": 1,
    "maxPages": 3,
    "followPatterns": ["/dispensary-info/"],
    "ignorePatterns": ["/products", "/strains", "/news", "?filter"],
    "handlePagination": false,
    "discoverApis": false,
    "respectRobotsTxt": false
  }
}
EOF
```

### Step 3: Run the container

```bash
docker run --rm \
  -v /tmp/test-input.json:/tmp/input.json \
  -e APIFY_INPUT_FILE=/tmp/input.json \
  -e APIFY_IS_AT_HOME=false \
  universal-scraper-test
```

**Pros:**
- ✅ Tests the exact environment that runs on Apify
- ✅ Catches Docker-specific issues
- ✅ Validates dependencies

**Cons:**
- ❌ Slower (Docker build time)
- ❌ Harder to debug

---

## Method 3: Apify CLI Testing (Most Realistic)

Test using the Apify CLI, which simulates the Apify platform:

```bash
# Navigate to the actor directory
cd universal_scraper/apify

# Run the actor locally
apify run --input '
{
  "mode": "crawl_only",
  "startUrls": [
    {"url": "https://www.leafly.com/dispensaries/nevada"}
  ],
  "crawlConfig": {
    "mode": "smart",
    "maxDepth": 1,
    "maxPages": 3,
    "followPatterns": ["/dispensary-info/"],
    "ignorePatterns": ["/products", "/strains", "/news"]
  }
}
'
```

**Pros:**
- ✅ Most realistic Apify simulation
- ✅ Tests storage, datasets, etc.
- ✅ Easy to iterate

**Cons:**
- ❌ Requires Apify CLI
- ❌ Still slower than direct Python

---

## Method 4: Syntax Check Only (Instant)

Quick syntax validation without running:

```bash
# Check for Python syntax errors
python3 -m py_compile universal_scraper/apify/actor.py
python3 -m py_compile universal_scraper/core/*.py
python3 -m py_compile universal_scraper/crawler/*.py
python3 -m py_compile universal_scraper/orchestrator/*.py

# Run linter
pylint universal_scraper/apify/actor.py
```

---

## Recommended Workflow

1. **Quick iteration:** Use Method 1 (Direct Python)
2. **Before deploying:** Use Method 2 (Docker)
3. **Final check:** Deploy to Apify and test there

---

## Common Issues

### ImportError: No module named 'X'

**Solution:** Install dependencies locally

```bash
pip install -r universal_scraper/apify/requirements.txt
```

### Browser launch failed

**Solution:** Install Playwright browsers

```bash
playwright install chromium
```

### Docker build fails

**Solution:** Check Dockerfile paths and build context

```bash
# Build from the correct directory
cd universal_scraper/apify
docker build -t test .
```

---

## Environment Variables for Testing

When testing locally, you can set these environment variables:

```bash
# Optional: Set OpenAI API key for AI scraping
export OPENAI_API_KEY="sk-..."

# Optional: Mock Apify environment
export APIFY_IS_AT_HOME=false
export APIFY_DEFAULT_DATASET_ID=test-dataset
```

---

## Next Steps

After successful local testing:

1. Commit your changes
2. Deploy to Apify: `cd universal_scraper/apify && apify push`
3. Run a test task on Apify
4. Monitor the logs

---

## Debugging Tips

### Enable verbose logging

Set environment variable:
```bash
export LOG_LEVEL=DEBUG
```

### Check async issues

Look for these warnings:
- `RuntimeWarning: coroutine 'X' was never awaited`
- `SyntaxError: 'await' outside async function`

### Test specific components

```python
# Test browser fetcher only
from universal_scraper.core import BrowserFetcher
import asyncio

async def test():
    async with BrowserFetcher(headless=True) as fetcher:
        result = await fetcher.fetch("https://example.com")
        print(result['html'][:200])

asyncio.run(test())
```

---

Happy testing! 🚀








