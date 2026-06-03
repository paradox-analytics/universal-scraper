# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Universal Scraper is an AI-powered web scraping platform with a JSON-first architecture. It's deployed as a FastAPI backend (GCP Cloud Run) with a React/TypeScript frontend (Firebase hosting). The system intelligently detects and extracts structured data from websites using LLM-driven analysis and pattern learning.

### Key Architecture Principles

1. **JSON-First**: Always detect JSON sources (API responses, embedded JSON-LD, GraphQL) before falling back to HTML
2. **Cost-Optimized**: Use LLMs for understanding structure only, not for executing extractions (cached patterns handle execution)
3. **Multi-Layered Caching**: Structure hash (Layer 1) → DOM digest (Layer 2) → Smart patterns (Layer 3) → Direct LLM (Layer 4)
4. **Adaptive Escalation**: Start with static HTML, escalate to browser rendering, then Web Unblocker for anti-bot sites
5. **Schema-Driven**: Optional output validation via Pydantic models or custom schemas

## Tech Stack

### Backend
- **Python 3.8+** | FastAPI 0.104+ | Uvicorn ASGI server
- **AI/LLM**: OpenAI (gpt-4o-mini default) | Gemini | Claude via LiteLLM
- **Fetching**: CloudScraper | Playwright | Camoufox (anti-detection) | Web Unblocker
- **Data Processing**: BeautifulSoup4 | lxml | JSONSchema | Pydantic v2
- **Caching**: Redis 5.0+ | DiskCache | Code-level caching
- **Proxies**: Bright Data | Oxylabs | ScraperAPI | Custom HTTP/SOCKS
- **Testing**: pytest | pytest-asyncio | pytest-cov
- **Cloud**: Google Cloud Run | Cloud Scheduler | Cloud Tasks

### Frontend
- **React 18.2** | TypeScript 5.2 | Vite 5.0
- **State**: Zustand | React Context
- **Data Fetching**: TanStack React Query | Axios
- **Auth**: Firebase Auth | JWT tokens
- **UI**: Tailwind CSS 3.4 | Headless UI | Heroicons
- **Code Display**: React Syntax Highlighter | Recharts (analytics)
- **File Processing**: Mammoth (DOCX) | pdfjs-dist (PDF) | Tesseract (OCR)

## Directory Structure

```
universal-scraper/
├── api/                           # FastAPI backend
│   ├── main.py                    # Main FastAPI app + endpoints (/scrape, /crawl, /preview, etc.)
│   ├── models/
│   │   └── adaptive_scrape_models.py
│   └── middleware/
│       ├── auth.py                # Firebase + JWT token verification, tenant ID extraction
│       ├── rate_limit.py          # Per-tenant rate limiting (Redis-backed)
│       └── usage_tracking.py      # Usage metrics tracking
├── universal_scraper/             # Core scraping engine
│   ├── __init__.py                # Exports UniversalScraper
│   ├── cli.py                     # CLI interface
│   └── core/
│       ├── scraper.py             # Main orchestration class (critical: read first)
│       ├── hybrid_fetcher.py      # Auto-detects best fetch method (static/browser)
│       ├── html_fetcher.py        # Static HTML fetching with CloudScraper
│       ├── browser_fetcher.py     # Playwright-based browser automation
│       ├── camoufox_fetcher.py    # Anti-detection browser (Playwright + Camoufox)
│       ├── web_unblocker_fetcher.py # Bright Data Web Unblocker for Cloudflare/Kasada
│       ├── json_detector.py       # JSON source detection + extraction (priority over HTML)
│       ├── direct_llm_extractor.py # Direct LLM extraction (fallback/supplement)
│       ├── code_cache.py          # Caches generated extraction code
│       ├── smart_pattern_cache.py # Caches reusable extraction patterns (Layer 3)
│       ├── dom_digest_cache.py    # Fast fingerprint matching (Layer 2)
│       ├── redis_cache.py         # Redis wrapper for multi-tenant caching
│       ├── proxy_manager.py       # Proxy rotation + per-request distribution
│       └── [40+ more modules]     # Various extraction, validation, and utility modules
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── App.tsx                # Main app shell + routing
│   │   ├── index.css              # Tailwind + global styles
│   │   ├── pages/                 # Route pages (WebScraping, DocumentProcessing, Dashboard, etc.)
│   │   ├── components/
│   │   │   ├── Auth/              # Login, auth guards
│   │   │   └── Layout/            # Sidebar, Header, shared layout
│   │   ├── contexts/
│   │   │   └── AuthContext.tsx    # Firebase auth state + JWT token management
│   │   └── config/
│   │       ├── api.ts             # API endpoint URLs + getApiKey()
│   │       └── firebase.ts        # Firebase config (Firestore, Auth, Storage)
│   └── public/
├── tests/
│   └── test_smart_json_first.py   # pytest-based tests
├── requirements.txt               # Python dependencies
├── setup.py                       # Package distribution config
├── .env.example                   # Environment variables template
├── .firebaserc                    # Firebase project config
├── .env                           # Local environment (GIT IGNORED)
└── [README, deployment docs, etc.]
```

## Core Architecture: The Extraction Pipeline

The `UniversalScraper.scrape()` method follows this flow:

```
1. FETCH (with adaptive escalation)
   └─ Strategy detector → cached strategy?
      ├─ Try static HTML first (fast)
      ├─ Escalate to browser if JSON needed or site is JS-heavy
      └─ Escalate to Web Unblocker if Cloudflare/Kasada detected

2. DETECT PAGINATION (fast patterns + LLM fallback)
   └─ FastPaginationDetector → find page URLs
      └─ Auto-scrape all pages if enable_auto_pagination=True

3. EXTRACT JSON (primary method)
   └─ JSONDetector.detect_and_extract() → find JSON sources
      ├─ Context-driven JSON ranking (if user provided context)
      └─ Quality validation (QualityValidator)
         ├─ If high quality + no missing fields → Early exit (Layer 1 optimization)
         ├─ If unhealthy → Fall through to Direct LLM
         └─ If missing fields → Supplement with Direct LLM (merge, don't replace)

4. DIRECT LLM EXTRACTION (fallback/supplement)
   └─ Smart pattern cache check
      ├─ Pattern cache hit (instant) → execute cached pattern
      ├─ Subset pattern hit (incremental) → use pattern + fill missing fields
      └─ Cache miss → run Direct LLM
         └─ Learn pattern from extraction results (seed pattern cache)

5. RETURN RESULTS
   └─ Merged JSON + supplemented Direct LLM fields
      └─ Metadata includes: source, quality score, cache status, strategy
```

## API Endpoints

### Core Scraping
- `POST /scrape` - Scrape single URL with field extraction
  - Auth: Bearer token or X-API-Key header
  - Returns: `{data: [], metadata: {...}}`

### Field Discovery & Preview
- `POST /api/v1/suggest-fields` - Analyze page and suggest extractable fields
- `POST /api/v1/preview` - Get interactive HTML preview with element highlighting
- `POST /api/v1/generate-fields-from-prompt` - Generate field names from natural language

### Utility
- `POST /api/v1/proxy/test` - Test proxy connectivity
- `POST /crawl` - Scrape multiple URLs
- `POST /document-processing/extract` - Extract text from PDF/DOCX/images (with OCR)
- `GET /` or `/health` - Health check

### Authentication
All scraping endpoints require either:
1. `Authorization: Bearer <firebase-token>` (production), OR
2. `X-API-Key: <openai-api-key>` header (direct API key), OR
3. `X-Tenant-ID: <tenant-id>` header (testing/admin)

Auth middleware (`api/middleware/auth.py`) verifies tokens and extracts tenant ID for multi-tenant rate limiting.

## Environment Variables

See `.env.example` for the full list. Key groups:
- **AI Provider Keys**: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`
- **Proxy**: `BRIGHTDATA_PROXY_HOST`, `BRIGHTDATA_PROXY_PORT`, `BRIGHTDATA_PROXY_USER`, `BRIGHTDATA_PROXY_PASS`
- **Web Unblocker**: `WEB_UNBLOCKER_API_KEY`, `WEB_UNBLOCKER_ZONE`, `WEB_UNBLOCKER_CUSTOMER_ID`
- **Firebase**: `FIREBASE_PROJECT_ID`
- **Cache**: `CACHE_DIR`, `CACHE_TTL`, `REDIS_URL`
- **Frontend (VITE_ prefix)**: `VITE_API_BASE_URL`, `VITE_FIREBASE_API_KEY`, `VITE_FIREBASE_AUTH_DOMAIN`, etc.

## Common Development Commands

### Backend Setup & Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run API locally (hot-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run pytest (all tests)
pytest

# Run specific test file
pytest tests/test_smart_json_first.py -v

# Run test with coverage
pytest --cov=universal_scraper --cov-report=html

# Run with logging output
pytest -s -v tests/test_smart_json_first.py::test_name

# Lint Python (ESLint-equivalent not configured, but can use flake8 if added)
# Currently no built-in linting, consider adding black/ruff to requirements
```

### Frontend Setup & Development
```bash
# Install dependencies
cd frontend
npm install

# Development server (Vite, hot module reload)
npm run dev
# Runs on http://localhost:5173

# Build for production
npm run build
# Outputs to dist/

# Preview production build locally
npm run preview

# Lint TypeScript/React
npm run lint
# Uses ESLint with TypeScript parser

# Check specific file
npx eslint src/pages/WebScraping.tsx --fix
```

### Deployment
```bash
# Deploy backend to Google Cloud Run
gcloud run deploy universal-scraper-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=sk-... \
  --allow-unauthenticated

# Deploy frontend to Firebase Hosting
firebase deploy --only hosting

# Both
firebase deploy
```

## Code Conventions

### Python
- **Type hints**: Full type annotations on function signatures (PEP 484)
- **Async**: Core methods are async (`async def scrape()`), use `await` for I/O
- **Logging**: `logger = logging.getLogger(__name__)` at module level, structured messages with context
- **Error handling**: Specific exceptions in try-except, log with `exc_info=True` for debugging
- **Naming**: snake_case for functions/variables, UPPER_CASE for constants, CamelCase for classes

### TypeScript/React
- **Components**: Functional components with hooks, TypeScript interfaces for props
- **State**: Zustand stores for complex state, React Context for auth
- **Async**: TanStack Query (React Query) for data fetching, avoid direct fetch()
- **Styling**: Tailwind CSS utilities, avoid inline styles
- **Router**: React Router v6, use `<Navigate>` for redirects

## Critical Files to Understand First

1. **`universal_scraper/core/scraper.py`** (1700+ lines)
   - The orchestrator: coordinates all components
   - Key method: `async scrape(url, fields, ...)` 
   - Implements the 5-step pipeline above
   - Contains the multi-layer caching strategy and fallback logic

2. **`api/main.py`** (3400+ lines)
   - FastAPI app definition + all endpoints
   - Request models (ScrapeRequest, CrawlRequest, etc.)
   - Proxy config parsing (handles comma-separated format, Bright Data specifics)
   - Tenant-aware scraper pooling via `get_scraper()`

3. **`universal_scraper/core/json_detector.py`**
   - Parses HTML for JSON sources: `<script>` tags, GraphQL endpoints, JSON-LD
   - Method: `detect_and_extract(html, url, captured_json)`
   - Returns: `{json_found, data: [], sources: []}`

4. **`universal_scraper/core/direct_llm_extractor.py`**
   - Uses LLM to extract data directly from HTML
   - Implements result caching by structure hash + fields
   - Method: `async extract(html, fields, context, url)`

5. **`universal_scraper/core/html_cleaner.py`**
   - Reduces HTML by ~98% while preserving extractability
   - Removes: ads, scripts, stylesheets, comments, tracking
   - Method: `clean(html)` returns `{html, reduction_percent}`

6. **`api/middleware/auth.py`**
   - Tenant identification from Bearer tokens or API keys
   - Verifies Firebase tokens (project ID checking)
   - Returns tenant_id for rate limiting + multi-tenant context

7. **`frontend/src/App.tsx`**
   - Route structure: protected routes vs public (Login/Signup)
   - Layout: Sidebar + Header + main content
   - All pages require AuthProvider wrapper

8. **`frontend/src/config/api.ts`**
   - API_BASE_URL (GCP Cloud Run endpoint)
   - getApiKey() / setApiKey() for localStorage management
   - Shared with all API calls via interceptors

## Multi-Tenant Architecture

The system is built for SaaS with per-tenant isolation:

- **Tenant ID**: Derived from Firebase UID (via Bearer token) or API key hash
- **Rate Limiting**: Per-minute and per-day limits per tenant (Redis-backed)
- **Usage Tracking**: Requests, items extracted, cache hits tracked per tenant
- **Caching**: Redis keys include tenant_id for data isolation
- **Strategy Detector**: Caches scraping strategies per tenant to enable learning

Key flows:
```
Frontend request → Firebase Auth → Bearer token → get_tenant_id() → check rate limits → scrape → track usage
Direct API call → X-API-Key header → derive tenant_id → check rate limits → scrape → track usage
```

## Caching Strategy (4 Layers)

Understanding caching is critical for performance:

1. **Layer 1 - Execution Cache** (TenantCache)
   - Full result caching: `cache_key = (URL, fields_hash)`
   - TTL: Per-tenant (default 3600s)
   - Use case: Exact same scrape request repeated

2. **Layer 2 - DOM Digest Cache** (DOMDigestCache)
   - Fast fingerprint matching (<10ms) without LLM
   - Detects "same layout" across similar pages
   - Maps digest → template_id for template reuse

3. **Layer 3 - Smart Pattern Cache** (SmartPatternCache + Redis)
   - Caches extraction PATTERNS (not results)
   - Pattern types: CSS selectors, XPath, regex, JSON paths
   - Supports exact match + subset match (incremental extraction)
   - Per-domain + per-field-set caching

4. **Layer 4 - Direct LLM Cache** (DirectLLMExtractor)
   - Caches LLM extraction results by structure hash
   - Fallback when patterns miss
   - Enables learning: LLM output → pattern learning → Layer 3 cache hit next time

## Learning System

After successful extraction:
1. Extract learns pattern from LLM results
2. Pattern stored in SmartPatternCache with success rate
3. DOM digest cached to enable fast template lookup
4. Selectors learned into SelectorLibrary for bootstrapping

Next similar request → pattern cache hit → instant execution (no LLM).

## Fetching Escalation Strategy

The system adaptively escalates to bypass blocks:

```
1. Try Static HTML (fast, 95% of sites)
   ↓ (if JSON needed or site JS-heavy)
2. Try Browser Rendering (Playwright/Camoufox, most JS sites)
   ↓ (if Cloudflare/Kasada detected)
3. Escalate to Web Unblocker (Bright Data, anti-bot sites)
   ↓ (if still blocked)
4. Custom strategy (detect page structure, adjust timeout, etc.)
```

Strategy detector caches which method worked for each domain.

## Key Dependencies & Their Roles

| Library | Purpose | Notes |
|---------|---------|-------|
| `litellm` | Multi-provider LLM routing | Unified API for OpenAI/Gemini/Claude/etc. |
| `beautifulsoup4` | HTML parsing | Primary for CSS selector extraction |
| `playwright` | Browser automation | Camoufox integration for anti-detection |
| `cloudscraper` | CloudFlare bypass | Static requests with JS challenge handling |
| `pydantic` | Data validation | Request/response schemas, strict type checking |
| `redis` | Distributed caching | Multi-tenant data isolation, rate limiting |
| `diskcache` | Local file caching | Fallback when Redis unavailable |
| `httpx` | Async HTTP | Used in some fetchers for speed |
| `firebase-admin` | Firebase integration | Token verification (if using Admin SDK) |

## Common Debugging Patterns

**Problem: HTML extraction returns empty or wrong data**
- Check: Did JSON detection find usable JSON? (JSONDetector logs)
- If JSON found but validation failed (quality < 0.3), Direct LLM kicks in
- If both fail, check extraction_source in metadata (should show 'json', 'html', or 'json+direct_llm')

**Problem: Site returns 403/403 Cloudflare**
- Logs show this in fetch_result.fetch_method = 'browser' or 'static'
- Solution: Enable Web Unblocker or Camoufox (use_camoufox=True, web_unblocker_api_key=...)
- Check metadata.unblocker_log for details

**Problem: Pagination not detecting**
- FastPaginationDetector runs first (instant, pattern-based)
- If no pattern found, LLMPaginationAnalyzer runs (requires api_key)
- Set enable_auto_pagination=False to disable, or set scroll_to_bottom=True for infinite scroll

**Problem: Rate limiting on repeated calls**
- Check: Is Redis available? (logs will show "Redis: ✅" or "❌")
- Verify tenant_id is consistent (should be Firebase UID or API key hash)
- Check Redis key format: `rate:{tenant_id}:minute:{timestamp}`

**Problem: LLM extraction is slow**
- Check: Is pattern cache enabled? (logs show "⚡ PATTERN CACHE HIT")
- If not, could be DOM digest cache miss + fresh LLM call
- Consider increasing browser_timeout for slow sites, not decreasing it

## Testing Strategy

- `pytest` is configured with asyncio support
- Tests use `test_` prefix and reside in `tests/` directory
- Run with `-s -v` for debug output
- Current test: `test_smart_json_first.py` validates JSON detection pipeline

Add new tests following existing pattern:
```python
import pytest
from universal_scraper import UniversalScraper

@pytest.mark.asyncio
async def test_json_first_extraction():
    scraper = UniversalScraper(api_key="test-key")
    result = await scraper.scrape(url="...", fields=[...])
    assert result['success']
    assert len(result['data']) > 0
```

## Deployment Notes

- **Backend**: GCP Cloud Run (containerized)
  - HTTP/2, auto-scaling, regional
  - Cold starts: Playwright may take 5-10s first load
  - Env vars set via Cloud Run console or `gcloud run deploy --set-env-vars`

- **Frontend**: Firebase Hosting
  - Static files, CDN-backed
  - Deployment: `firebase deploy` syncs dist/ to CDN
  - Auth: Firebase Auth via browser SDK

- **Redis**: Optional, required for multi-tenant rate limiting
  - Can use Cloud Memorystore or local Redis
  - Set REDIS_URL env var, or system uses DiskCache fallback

- **Database**: Currently no persistent DB (Firestore can be added)
  - Tenant config hardcoded in auth.py (TODO: lookup from Firestore)
  - Usage tracking stored in Redis (ephemeral)

## Known Limitations & TODOs

1. **Database Integration**: Tenant lookup, usage persistence, and SaaS billing require Firestore or PostgreSQL
2. **Admin SDK**: Token verification uses lightweight JWT decoding (production should use Firebase Admin SDK)
3. **Scraping Patterns**: Only learns selectors for CSS; could extend to XPath, regex
4. **Pagination**: Only supports URL-param and load-more; infinite scroll via scroll_to_bottom
5. **OCR**: Requires Tesseract system package; not all deployments have it
6. **Error Recovery**: Some edge cases fall back to fresh extraction instead of retry-with-timeout

