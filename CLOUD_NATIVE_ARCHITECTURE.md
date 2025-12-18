# Cloud-Native SaaS Architecture Recommendations
## Universal Scraper → AI Copilot for Web Scraping & Document Processing

---

## Executive Summary

This document provides a comprehensive architecture plan to transform the Universal Scraper into a **cloud-native, multi-tenant SaaS platform** with subsecond processing capabilities, separating web scraping and document processing into distinct modules, and building a React/Tailwind frontend similar to OxyLabs/Sequentum.

---

## 1. Cloud-Native Architecture (Google Cloud)

### 1.1 Core Infrastructure Components

#### **Compute Layer**
```
┌─────────────────────────────────────────────────────────────┐
│                    Google Cloud Run                          │
│  (Serverless Containers - Auto-scaling, Subsecond Cold Start)│
│                                                              │
│  • Web Scraping Service (Python/FastAPI)                    │
│  • Document Processing Service (Python/FastAPI)             │
│  • API Gateway (Cloud Endpoints)                            │
│  • Frontend (Cloud Run or Cloud Storage + CDN)              │
└─────────────────────────────────────────────────────────────┘
```

**Why Cloud Run:**
- **Subsecond cold starts** with min instances = 1
- **Auto-scaling** from 0 to 1000+ instances
- **Pay-per-use** pricing model
- **Built-in load balancing** and HTTPS
- **Multi-region deployment** support

#### **Caching & State Management**
```
┌─────────────────────────────────────────────────────────────┐
│              Google Cloud Memorystore (Redis)                │
│                                                              │
│  • Request deduplication                                    │
│  • Structural hash cache (page structure fingerprints)      │
│  • Extraction code cache                                    │
│  • Session state (multi-tenant isolation)                   │
│  • Rate limiting per tenant                                 │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
- **Tier**: Standard (HA) or Basic (dev)
- **Memory**: Start with 1GB, scale to 10GB+
- **Multi-region replication** for global latency

#### **Object Storage**
```
┌─────────────────────────────────────────────────────────────┐
│              Google Cloud Storage (GCS)                      │
│                                                              │
│  • Raw HTML/PDF storage (temporary)                         │
│  • Extracted data archives                                  │
│  • Browser session storage (screenshots, HAR files)         │
│  • User uploads (PDFs, documents)                           │
└─────────────────────────────────────────────────────────────┘
```

**Lifecycle Policies:**
- **Hot tier** (0-7 days): Frequently accessed data
- **Cold tier** (7-30 days): Archive data
- **Delete** (>30 days): Auto-cleanup

#### **Database Layer**
```
┌─────────────────────────────────────────────────────────────┐
│         Cloud SQL (PostgreSQL) + Cloud Spanner              │
│                                                              │
│  Cloud SQL (PostgreSQL):                                    │
│  • User accounts & authentication                           │
│  • Job metadata & status                                    │
│  • API keys & credentials                                   │
│  • Billing & usage tracking                                 │
│                                                              │
│  Cloud Spanner (Global Scale):                              │
│  • Multi-region job queue                                   │
│  • Real-time analytics                                      │
│  • Cross-region consistency                                 │
└─────────────────────────────────────────────────────────────┘
```

#### **Message Queue & Event Streaming**
```
┌─────────────────────────────────────────────────────────────┐
│         Cloud Pub/Sub + Cloud Tasks                          │
│                                                              │
│  Cloud Pub/Sub:                                             │
│  • Async job processing                                     │
│  • Event-driven architecture                                │
│  • Cross-service communication                              │
│                                                              │
│  Cloud Tasks:                                               │
│  • Scheduled scraping jobs                                  │
│  • Retry logic for failed jobs                             │
│  • Rate limiting enforcement                                │
└─────────────────────────────────────────────────────────────┘
```

#### **AI/ML Services**
```
┌─────────────────────────────────────────────────────────────┐
│         Vertex AI + External LLM APIs                       │
│                                                              │
│  Vertex AI:                                                 │
│  • Custom extraction models (fine-tuned)                    │
│  • Embedding models (for structural similarity)            │
│  • Batch prediction for cost optimization                  │
│                                                              │
│  External APIs (via LiteLLM):                              │
│  • OpenAI (GPT-4o, GPT-4o-mini)                           │
│  • Anthropic (Claude 3.5 Sonnet)                           │
│  • Google (Gemini 2.0 Flash)                               │
│  • Fallback routing & load balancing                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Subsecond Processing Architecture

#### **Request Flow (Target: <500ms P95)**
```
User Request
    ↓
Cloud Load Balancer (Global Anycast) [~10ms]
    ↓
API Gateway (Cloud Endpoints) [~5ms]
    ↓
┌─────────────────────────────────────────┐
│  Cache Check (Redis) [~2ms]            │
│  • Structural hash lookup              │
│  • Extraction code cache               │
│  • Recent results cache                │
└─────────────────────────────────────────┘
    ↓ (Cache Hit: ~50ms total)
    OR
    ↓ (Cache Miss: Continue)
┌─────────────────────────────────────────┐
│  Fast Path Detection [~10ms]            │
│  • JSON API detection                  │
│  • Static HTML check                   │
│  • PDF content type check              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Processing (Parallel)                  │
│  • Fetch content (Cloud Run)            │
│  • Extract data (Cloud Run)            │
│  • Validate & transform                │
└─────────────────────────────────────────┘
    ↓
Response [~200-400ms for cache miss]
```

#### **Optimization Strategies**

1. **Pre-warming**
   - Keep min instances = 1 per service
   - Cloud Scheduler triggers periodic health checks
   - Pre-load common extraction patterns

2. **Edge Caching**
   - Cloud CDN for static assets
   - Cloud Armor for DDoS protection
   - Regional caching for popular URLs

3. **Parallel Processing**
   - Async/await for I/O operations
   - Concurrent LLM calls (when needed)
   - Batch processing for multiple URLs

4. **Smart Caching**
   - Structural hash-based cache (same page structure = reuse)
   - Time-based invalidation (TTL per domain)
   - User-specific cache isolation

---

## 2. Multi-Tenant Architecture

### 2.1 Tenant Isolation Strategy

#### **Database-Level Isolation**
```python
# Tenant ID in every table
CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    url TEXT NOT NULL,
    status TEXT,
    created_at TIMESTAMP,
    INDEX idx_tenant_status (tenant_id, status)
);

# Row-level security
CREATE POLICY tenant_isolation ON jobs
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
```

#### **Redis Namespace Isolation**
```python
# Key pattern: tenant:{tenant_id}:{resource_type}:{resource_id}
redis.set(f"tenant:{tenant_id}:cache:{hash}", data)
redis.set(f"tenant:{tenant_id}:rate_limit:{user_id}", count)
```

#### **Cloud Storage Bucket Strategy**
```
Option 1: Separate buckets per tenant (enterprise)
  gs://universal-scraper-tenant-{tenant_id}/

Option 2: Prefix-based (cost-effective)
  gs://universal-scraper-data/tenant/{tenant_id}/jobs/{job_id}/
```

### 2.2 Resource Quotas & Rate Limiting

```python
# Per-tenant limits (stored in Redis)
TENANT_LIMITS = {
    "free": {
        "requests_per_minute": 10,
        "concurrent_jobs": 2,
        "storage_gb": 1,
        "llm_calls_per_day": 100
    },
    "pro": {
        "requests_per_minute": 100,
        "concurrent_jobs": 10,
        "storage_gb": 10,
        "llm_calls_per_day": 10000
    },
    "enterprise": {
        "requests_per_minute": 1000,
        "concurrent_jobs": 100,
        "storage_gb": 1000,
        "llm_calls_per_day": 1000000
    }
}
```

### 2.3 Threading & Concurrency Model

#### **Architecture Pattern: Event-Driven + Worker Pools**

```python
# FastAPI with async/await (not threads)
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()

# Worker pool for CPU-bound tasks (browser automation)
browser_pool = ThreadPoolExecutor(max_workers=10)

# Async for I/O-bound tasks (HTTP requests, DB queries)
@app.post("/api/v1/scrape")
async def scrape(request: ScrapeRequest):
    tenant_id = get_tenant_id(request.api_key)
    
    # Check rate limits (Redis)
    if not await check_rate_limit(tenant_id):
        raise HTTPException(429, "Rate limit exceeded")
    
    # Check cache first
    cache_key = f"tenant:{tenant_id}:cache:{hash_url(request.url)}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Process async
    result = await process_scrape(request, tenant_id)
    
    # Cache result
    await redis.setex(cache_key, 3600, json.dumps(result))
    
    return result
```

#### **Scaling Strategy**
- **Horizontal**: Cloud Run auto-scales based on request rate
- **Vertical**: Use Cloud Run CPU/memory allocation (up to 8 vCPU, 32GB RAM)
- **Regional**: Deploy to multiple regions (us-central1, europe-west1, asia-east1)

---

## 3. Module Separation: Web Scraping vs Document Processing

### 3.1 Proposed Architecture

```
universal-scraper/
├── services/
│   ├── web-scraping-service/          # Module 1: Web Scraping
│   │   ├── src/
│   │   │   ├── api/
│   │   │   │   └── routes.py         # FastAPI routes
│   │   │   ├── core/
│   │   │   │   ├── scraper.py        # Main scraper logic
│   │   │   │   ├── hybrid_fetcher.py
│   │   │   │   ├── json_detector.py
│   │   │   │   └── html_cleaner.py
│   │   │   ├── models/
│   │   │   │   └── schemas.py        # Pydantic models
│   │   │   └── utils/
│   │   │       └── cache.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py
│   │
│   ├── document-processing-service/    # Module 2: Document Processing
│   │   ├── src/
│   │   │   ├── api/
│   │   │   │   └── routes.py
│   │   │   ├── core/
│   │   │   │   ├── pdf_extractor.py
│   │   │   │   ├── docx_extractor.py
│   │   │   │   ├── xlsx_extractor.py
│   │   │   │   └── ocr_processor.py
│   │   │   ├── models/
│   │   │   │   └── schemas.py
│   │   │   └── utils/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── main.py
│   │
│   └── shared/                        # Shared libraries
│       ├── auth/
│       │   └── tenant_manager.py
│       ├── cache/
│       │   └── redis_client.py
│       ├── database/
│       │   └── models.py
│       └── llm/
│           └── llm_client.py
│
├── frontend/                          # React + Tailwind
│   ├── src/
│   │   ├── components/
│   │   │   ├── ScrapePreview/
│   │   │   ├── DocumentViewer/
│   │   │   ├── DataTable/
│   │   │   └── CacheIndicator/
│   │   ├── pages/
│   │   │   ├── WebScraping.tsx
│   │   │   ├── DocumentProcessing.tsx
│   │   │   └── Dashboard.tsx
│   │   └── services/
│   │       └── api.ts
│   ├── package.json
│   └── tailwind.config.js
│
└── infrastructure/
    ├── terraform/                     # Infrastructure as Code
    ├── cloudbuild/                    # CI/CD
    └── kubernetes/                    # Optional: GKE for advanced use cases
```

### 3.2 Service APIs

#### **Web Scraping Service**
```python
# POST /api/v1/web-scraping/scrape
{
    "url": "https://example.com/products",
    "fields": ["title", "price", "rating"],
    "options": {
        "wait_for_selector": ".product-list",
        "scroll_to_bottom": true,
        "use_browser": false
    }
}

# Response
{
    "job_id": "uuid",
    "status": "completed",
    "data": [...],
    "metadata": {
        "cache_hit": true,
        "processing_time_ms": 150,
        "source": "json_api"
    }
}
```

#### **Document Processing Service**
```python
# POST /api/v1/document-processing/extract
{
    "file_url": "gs://bucket/document.pdf",
    "fields": ["title", "date", "amount"],
    "options": {
        "use_ocr": false,
        "max_pages": 10
    }
}

# Response
{
    "job_id": "uuid",
    "status": "completed",
    "data": [...],
    "metadata": {
        "pages_processed": 5,
        "processing_time_ms": 800
    }
}
```

### 3.3 Shared Components

**Common Libraries:**
- `shared/auth/`: Tenant authentication & authorization
- `shared/cache/`: Redis client wrapper
- `shared/database/`: SQLAlchemy models
- `shared/llm/`: LiteLLM client with fallback routing
- `shared/monitoring/`: OpenTelemetry instrumentation

---

## 4. Frontend Architecture (React + Tailwind)

### 4.1 Design Reference: OxyLabs/Sequentum

**Key Features to Implement:**
1. **Browser Preview**: Show page structure before scraping
2. **Cache Indicator**: Visual indicator if page is cached
3. **Raw Data View**: Show extracted JSON/HTML
4. **Field Mapping**: Visual field selector
5. **Real-time Status**: WebSocket for job progress

### 4.2 Component Structure

```
frontend/src/
├── components/
│   ├── Layout/
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── Footer.tsx
│   │
│   ├── WebScraping/
│   │   ├── UrlInput.tsx
│   │   ├── BrowserPreview.tsx      # Embedded browser view
│   │   ├── FieldSelector.tsx        # Drag-drop field mapping
│   │   ├── CacheIndicator.tsx       # Shows if cached
│   │   ├── ResultsTable.tsx
│   │   └── RawDataViewer.tsx        # JSON/HTML viewer
│   │
│   ├── DocumentProcessing/
│   │   ├── FileUpload.tsx
│   │   ├── DocumentViewer.tsx       # PDF/DOCX preview
│   │   ├── FieldSelector.tsx
│   │   └── ResultsTable.tsx
│   │
│   └── Common/
│       ├── DataTable.tsx
│       ├── CodeViewer.tsx
│       └── StatusBadge.tsx
│
├── pages/
│   ├── Dashboard.tsx
│   ├── WebScraping.tsx
│   ├── DocumentProcessing.tsx
│   ├── Jobs.tsx
│   └── Settings.tsx
│
├── services/
│   ├── api.ts                       # API client
│   ├── websocket.ts                 # Real-time updates
│   └── cache.ts                     # Frontend cache
│
└── hooks/
    ├── useScrape.ts
    ├── useDocument.ts
    └── useWebSocket.ts
```

### 4.3 Key Features Implementation

#### **Browser Preview Component**
```typescript
// components/WebScraping/BrowserPreview.tsx
import { useEffect, useRef } from 'react';

export function BrowserPreview({ url }: { url: string }) {
    const iframeRef = useRef<HTMLIFrameElement>(null);
    
    useEffect(() => {
        // Load page in iframe (or use headless browser screenshot API)
        iframeRef.current.src = `/api/v1/preview?url=${encodeURIComponent(url)}`;
    }, [url]);
    
    return (
        <div className="border rounded-lg overflow-hidden">
            <div className="bg-gray-100 px-4 py-2 flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="ml-4 text-sm text-gray-600">{url}</span>
            </div>
            <iframe
                ref={iframeRef}
                className="w-full h-96"
                sandbox="allow-same-origin allow-scripts"
            />
        </div>
    );
}
```

#### **Cache Indicator**
```typescript
// components/WebScraping/CacheIndicator.tsx
export function CacheIndicator({ url }: { url: string }) {
    const { data: cacheStatus } = useQuery(
        ['cache-status', url],
        () => api.get(`/api/v1/cache/status?url=${url}`)
    );
    
    if (cacheStatus?.is_cached) {
        return (
            <div className="flex items-center gap-2 text-green-600">
                <CheckCircleIcon className="w-5 h-5" />
                <span>Cached - Instant results</span>
            </div>
        );
    }
    
    return (
        <div className="flex items-center gap-2 text-yellow-600">
            <ClockIcon className="w-5 h-5" />
            <span>Not cached - Processing...</span>
        </div>
    );
}
```

#### **Raw Data Viewer**
```typescript
// components/Common/CodeViewer.tsx
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export function RawDataViewer({ data, language = 'json' }) {
    return (
        <div className="border rounded-lg overflow-hidden">
            <div className="bg-gray-800 px-4 py-2 flex justify-between items-center">
                <span className="text-white text-sm">Raw Data</span>
                <button className="text-white text-sm">Copy</button>
            </div>
            <SyntaxHighlighter
                language={language}
                style={vscDarkPlus}
                customStyle={{ margin: 0 }}
            >
                {JSON.stringify(data, null, 2)}
            </SyntaxHighlighter>
        </div>
    );
}
```

### 4.4 Tech Stack

- **Framework**: React 18+ with TypeScript
- **Styling**: Tailwind CSS + Headless UI
- **State Management**: React Query (TanStack Query) + Zustand
- **API Client**: Axios + React Query
- **WebSocket**: Socket.io-client
- **Code Highlighting**: react-syntax-highlighter
- **Charts**: Recharts or Chart.js
- **Forms**: React Hook Form + Zod validation

---

## 5. Data Warehouse Connectors

### 5.1 Architecture Pattern

```
Extracted Data
    ↓
┌─────────────────────────────────────────┐
│  Data Transformation Layer              │
│  • Schema normalization                 │
│  • Data type conversion                 │
│  • Validation & cleaning                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Warehouse Connector (Abstract)         │
│  • Snowflake                            │
│  • PostgreSQL                           │
│  • BigQuery                             │
│  • Redshift                             │
│  • Databricks                           │
└─────────────────────────────────────────┘
    ↓
Target Warehouse
```

### 5.2 Connector Implementation

```python
# shared/warehouse/connector.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class WarehouseConnector(ABC):
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> None:
        """Establish connection to warehouse"""
        pass
    
    @abstractmethod
    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """Create table if not exists"""
        pass
    
    @abstractmethod
    def insert_batch(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        """Insert batch of records"""
        pass
    
    @abstractmethod
    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute query"""
        pass

# shared/warehouse/snowflake.py
import snowflake.connector

class SnowflakeConnector(WarehouseConnector):
    def __init__(self, config: Dict[str, Any]):
        self.conn = snowflake.connector.connect(
            user=config['user'],
            password=config['password'],
            account=config['account'],
            warehouse=config['warehouse'],
            database=config['database'],
            schema=config['schema']
        )
    
    def insert_batch(self, table_name: str, data: List[Dict[str, Any]]):
        cursor = self.conn.cursor()
        # Use COPY INTO for bulk insert
        cursor.execute(f"COPY INTO {table_name} FROM ...")
        cursor.close()

# shared/warehouse/postgres.py
import psycopg2
from psycopg2.extras import execute_batch

class PostgresConnector(WarehouseConnector):
    def __init__(self, config: Dict[str, Any]):
        self.conn = psycopg2.connect(
            host=config['host'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
    
    def insert_batch(self, table_name: str, data: List[Dict[str, Any]]):
        cursor = self.conn.cursor()
        columns = ', '.join(data[0].keys())
        placeholders = ', '.join(['%s'] * len(data[0]))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        execute_batch(cursor, query, [tuple(row.values()) for row in data])
        self.conn.commit()
        cursor.close()

# shared/warehouse/factory.py
def create_connector(warehouse_type: str, config: Dict[str, Any]) -> WarehouseConnector:
    connectors = {
        'snowflake': SnowflakeConnector,
        'postgres': PostgresConnector,
        'bigquery': BigQueryConnector,
        'redshift': RedshiftConnector,
        'databricks': DatabricksConnector
    }
    return connectors[warehouse_type](config)
```

### 5.3 Supported Warehouses

1. **Snowflake**
   - Use `snowflake-connector-python`
   - COPY INTO for bulk loads
   - Support for stages and file formats

2. **PostgreSQL / Cloud SQL**
   - Use `psycopg2` or `asyncpg`
   - Batch inserts with `execute_batch`
   - Connection pooling

3. **BigQuery**
   - Use `google-cloud-bigquery`
   - Load jobs for large datasets
   - Streaming inserts for real-time

4. **Redshift**
   - Use `psycopg2` (PostgreSQL-compatible)
   - COPY FROM S3 for bulk loads
   - Spectrum for external tables

5. **Databricks**
   - Use `databricks-sql-connector`
   - Delta Lake support
   - Spark integration

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up Google Cloud project & billing
- [ ] Deploy Cloud Run services (web scraping + document processing)
- [ ] Set up Cloud SQL (PostgreSQL) for metadata
- [ ] Implement Redis caching layer
- [ ] Basic FastAPI endpoints for both services
- [ ] Multi-tenant authentication (API keys)

### Phase 2: Core Features (Weeks 5-8)
- [ ] Migrate existing scraper logic to web-scraping-service
- [ ] Migrate PDF extractor to document-processing-service
- [ ] Implement cache layer (structural hash, extraction code)
- [ ] Add rate limiting per tenant
- [ ] Set up Cloud Pub/Sub for async processing
- [ ] Basic monitoring (Cloud Monitoring)

### Phase 3: Frontend (Weeks 9-12)
- [ ] Set up React + Tailwind project (Replit or local)
- [ ] Implement browser preview component
- [ ] Build cache indicator
- [ ] Create data table component
- [ ] Add raw data viewer
- [ ] WebSocket integration for real-time updates

### Phase 4: Warehouse Connectors (Weeks 13-14)
- [ ] Implement connector abstraction
- [ ] Build Snowflake connector
- [ ] Build PostgreSQL connector
- [ ] Build BigQuery connector
- [ ] Add connector configuration UI

### Phase 5: Optimization (Weeks 15-16)
- [ ] Optimize for subsecond processing
- [ ] Implement pre-warming strategies
- [ ] Add edge caching (Cloud CDN)
- [ ] Performance testing & tuning
- [ ] Cost optimization

---

## 7. Cost Estimates (Google Cloud)

### Monthly Costs (Estimated)

**Cloud Run** (2 services, ~1000 requests/day):
- Compute: ~$50-100/month
- Requests: ~$10/month

**Cloud SQL** (PostgreSQL, db-f1-micro):
- Instance: ~$7/month
- Storage: ~$0.17/GB/month

**Memorystore Redis** (1GB):
- ~$50/month

**Cloud Storage** (100GB):
- Storage: ~$2/month
- Operations: ~$1/month

**Cloud Pub/Sub**:
- ~$10/month

**Vertex AI** (if using):
- Pay-per-use (varies)

**Total**: ~$130-180/month (base) + usage-based costs

---

## 8. Development Tools

### Replit Setup
1. Create new Replit project (React + TypeScript)
2. Install dependencies: `npm install`
3. Connect to GitHub repo
4. Use Replit's built-in preview

### Cursor Setup
1. Open project in Cursor
2. Use Cursor's AI features for:
   - Code generation
   - Architecture decisions
   - Bug fixes
   - Documentation

### Local Development
```bash
# Backend services
cd services/web-scraping-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## 9. Next Steps

1. **Review & Approve Architecture**
2. **Set up Google Cloud Project**
3. **Create GitHub Repository Structure**
4. **Begin Phase 1 Implementation**
5. **Set up CI/CD Pipeline (Cloud Build)**

---

## References

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [OxyLabs Platform](https://oxylabs.io/)
- [Sequentum Platform](https://sequentum.com/)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Tailwind CSS Documentation](https://tailwindcss.com/)

