# Universal Scraper - Codebase Index

Updated for current Firebase UI + GCP backend + Redis cache layout.

## Project Overview

Universal scraping platform with two primary flows:

- Universal web scraper agent (probabilistic → deterministic, JSON/CSS/HTML extraction, Camoufox anti-detection, device fingerprinting, proxy + web unlocker support).
- Document processor (similar extraction pipeline for PDFs and other documents).

Primary stack:

- Frontend: React + Vite + Tailwind (Firebase Auth)
- Backend: FastAPI on GCP Cloud Run (Redis cache, Playwright/Camoufox)
- Core engine: `universal_scraper/` Python modules

---

## Top-Level Directories

- `api/` FastAPI service (Cloud Run entrypoint)
- `frontend/` React app (Firebase Auth UI, web + document workflows)
- `universal_scraper/` core scraping engine
- `infrastructure/` GCP + Redis setup scripts
- `auth/` local Firebase service account JSON
- `migrations/` Firestore or data migrations (if used)
- `services/` placeholder for service split (currently empty)
- `cache/`, `local_cache/`, `apify_storage_local/`, `output/` run artifacts
- Root `test_*.py` scripts for local/regression testing

---

## Backend (FastAPI)

Entry: `api/main.py`

Key endpoints (selected):

- `POST /api/v1/preview` browser/static render + HTML for preview
- `POST /api/v1/suggest-fields` field discovery (LLM + heuristics)
- `POST /scrape` main universal scrape endpoint
- `POST /document-processing/extract` document processor flow
- `POST /api/v1/generate-fields-from-prompt` LLM field generation
- `POST /api/v1/proxy/test` proxy connectivity checks
- `GET/POST /api/v1/patterns/*` pattern cache CRUD
- `POST /api/v1/agents/*` agent create/execute/schedule
- `POST /api/v1/browser/session/*` browser session API (navigate/click/scroll/screenshot/html)

Middleware:

- `api/middleware/auth.py` Firebase Auth + API key support (tenant ID)
- `api/middleware/rate_limit.py` per-tenant throttling
- `api/middleware/usage_tracking.py` usage + quota tracking

Browser automation:

- `api/browser_session.py` session lifecycle + Playwright operations

Caching:

- `universal_scraper/core/redis_cache.py` Redis-backed cache for Cloud Run
- `universal_scraper/core/tenant_cache.py` and `tenant_pattern_cache.py` per-tenant pattern cache

---

## Frontend (React + Vite)

Entry: `frontend/src/App.tsx`

Routing:

- `/web-scraping` universal agent UI
- `/document-processing` document processor UI
- `/history`, `/cache`, `/settings`
- `/agents/:id`, `/scrapers/:id`, `/processors/:id` agent builder

Key pages:

- `frontend/src/pages/WebScraping.tsx` main web scraping workflow
- `frontend/src/pages/DocumentProcessing.tsx` document workflow
- `frontend/src/pages/AgentBuilder.tsx` agent design/runner

Key components:

- `frontend/src/components/BrowserWorkspace/BrowserWorkspace.tsx` browser preview + element selection + agent log
- `frontend/src/components/DocumentViewer/DocumentViewer.tsx` document preview + extraction
- `frontend/src/components/Layout/Sidebar.tsx`, `Header.tsx` shell layout
- `frontend/src/components/Common/GlobalStatusIndicators.tsx` status badges

Services & config:

- `frontend/src/services/api.ts` API client + auth token injection
- `frontend/src/services/auth.ts` Firebase auth + profile
- `frontend/src/config/firebase.ts` Firebase config
- `frontend/src/config/api.ts` backend base URL

---

## Core Scraper Engine (`universal_scraper/`)

Primary modules:

- `core/scraper.py` orchestrates fetch → detect → extract → validate
- `core/hybrid_fetcher.py` static + browser fallback (Camoufox supported)
- `core/camoufox_fetcher.py` anti-detection browser fetcher
- `core/json_detector.py` JSON/embedded/Next.js detection
- `core/direct_llm_extractor.py` direct LLM extraction fallback
- `core/html_cleaner.py` HTML reduction + structure preservation
- `core/pattern_*` pattern detection + caching
- `core/pagination_*` pagination detection + execution
- `core/pdf_extractor.py` document extraction

Supporting modules:

- `crawler/` URL discovery crawler
- `orchestrator/` workflow coordination
- `apify/` Apify actor packaging + duplicated core for Apify builds

---

## Infrastructure

- `infrastructure/cloudbuild/cloudbuild.yaml` Cloud Build pipeline
- `infrastructure/setup_redis.sh`, `infrastructure/redis_setup.sh` Redis setup
- `deploy_to_gcp.sh`, `deploy_firebase.sh`, `deploy_frontend.sh` deployment helpers

---

## Local Testing / Debug

Web scraping:

- `test_producthunt_*`, `debug_product_hunt.py`, `extract_product_hunt_*.py`
- `test_browser_*`, `test_camoufox_*`, `test_web_unblocker_*`

Document processing:

- `test_local_pdf.py`, `test_fusa_pdf.py`, `test_pdf_extraction.py`

End-to-end:

- `test_full_pipeline_v2.py`, `test_end_to_end_*`

---

## Key Config + Credentials

- Firebase: `frontend/src/config/firebase.ts`, `firebase.json`
- Auth service account: `auth/soma-data-467016-d0118961514a.json`
- Redis: `REDIS_URL` env var (defaults to `redis://localhost:6379`)
- Proxies/web unlocker: configured in UI settings or API payloads
