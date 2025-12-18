# Quick Start Implementation Guide

## Overview

This guide provides step-by-step instructions to begin implementing the cloud-native SaaS architecture.

---

## Phase 1: Set Up Google Cloud Infrastructure

### 1.1 Create Google Cloud Project

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Create project
gcloud projects create universal-scraper-saas --name="Universal Scraper SaaS"

# Set as active project
gcloud config set project universal-scraper-saas

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    sql-component.googleapis.com \
    sqladmin.googleapis.com \
    redis.googleapis.com \
    storage-component.googleapis.com \
    pubsub.googleapis.com \
    cloudbuild.googleapis.com
```

### 1.2 Set Up Cloud SQL (PostgreSQL)

```bash
# Create Cloud SQL instance
gcloud sql instances create universal-scraper-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=us-central1 \
    --root-password=YOUR_SECURE_PASSWORD

# Create database
gcloud sql databases create universal_scraper \
    --instance=universal-scraper-db

# Get connection name
gcloud sql instances describe universal-scraper-db \
    --format="value(connectionName)"
```

### 1.3 Set Up Memorystore Redis

```bash
# Create Redis instance
gcloud redis instances create universal-scraper-cache \
    --size=1 \
    --region=us-central1 \
    --redis-version=redis_7_0
```

### 1.4 Create Cloud Storage Bucket

```bash
# Create bucket
gsutil mb -p universal-scraper-saas -l us-central1 gs://universal-scraper-data

# Set lifecycle policy (optional)
gsutil lifecycle set lifecycle.json gs://universal-scraper-data
```

---

## Phase 2: Create Service Structure

### 2.1 Initialize Project Structure

```bash
mkdir -p services/web-scraping-service/src/{api,core,models,utils}
mkdir -p services/document-processing-service/src/{api,core,models,utils}
mkdir -p services/shared/{auth,cache,database,llm,warehouse}
mkdir -p frontend/src/{components,pages,services,hooks}
mkdir -p infrastructure/{terraform,cloudbuild}
```

### 2.2 Web Scraping Service - FastAPI Setup

```bash
cd services/web-scraping-service

# Create requirements.txt
cat > requirements.txt << EOF
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
redis>=5.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
google-cloud-storage>=2.10.0
litellm>=1.30.0
cloudscraper>=1.2.71
beautifulsoup4>=4.12.0
EOF

# Create main.py
cat > main.py << 'EOF'
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="Web Scraping Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScrapeRequest(BaseModel):
    url: str
    fields: List[str]
    options: Optional[dict] = {}

class ScrapeResponse(BaseModel):
    job_id: str
    status: str
    data: List[dict]
    metadata: dict

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/v1/scrape", response_model=ScrapeResponse)
async def scrape(request: ScrapeRequest):
    # TODO: Implement scraping logic
    return ScrapeResponse(
        job_id="test-123",
        status="completed",
        data=[],
        metadata={}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
EOF
```

### 2.3 Deploy to Cloud Run

```bash
# Build and deploy
gcloud run deploy web-scraping-service \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --min-instances 1 \
    --max-instances 10 \
    --memory 2Gi \
    --cpu 2

# Get service URL
gcloud run services describe web-scraping-service \
    --region us-central1 \
    --format="value(status.url)"
```

---

## Phase 3: Frontend Setup (Replit)

### 3.1 Create React + TypeScript Project

1. Go to [Replit](https://replit.com)
2. Create new Repl: "React + TypeScript"
3. Install dependencies:

```bash
npm install \
    react react-dom \
    @tanstack/react-query \
    axios \
    tailwindcss \
    @headlessui/react \
    react-syntax-highlighter \
    @types/react-syntax-highlighter
```

### 3.2 Initialize Tailwind

```bash
npx tailwindcss init -p
```

Update `tailwind.config.js`:

```javascript
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### 3.3 Create Basic Components

```typescript
// src/components/WebScraping/UrlInput.tsx
import { useState } from 'react';

export function UrlInput({ onScrape }: { onScrape: (url: string) => void }) {
    const [url, setUrl] = useState('');
    
    return (
        <div className="flex gap-2">
            <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Enter URL to scrape..."
                className="flex-1 px-4 py-2 border rounded-lg"
            />
            <button
                onClick={() => onScrape(url)}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
                Scrape
            </button>
        </div>
    );
}
```

---

## Phase 4: Database Schema

### 4.1 Create Migration

```sql
-- migrations/001_initial.sql

-- Tenants table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    plan VARCHAR(50) NOT NULL DEFAULT 'free',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    email VARCHAR(255) NOT NULL UNIQUE,
    api_key VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    user_id UUID NOT NULL REFERENCES users(id),
    job_type VARCHAR(50) NOT NULL, -- 'web_scraping' or 'document_processing'
    url TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    result JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX idx_jobs_tenant_status ON jobs(tenant_id, status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at);
```

---

## Phase 5: Warehouse Connector Example

### 5.1 Snowflake Connector

```python
# services/shared/warehouse/snowflake.py
import snowflake.connector
from typing import List, Dict, Any

class SnowflakeConnector:
    def __init__(self, config: Dict[str, Any]):
        self.conn = snowflake.connector.connect(
            user=config['user'],
            password=config['password'],
            account=config['account'],
            warehouse=config['warehouse'],
            database=config['database'],
            schema=config['schema']
        )
    
    def create_table(self, table_name: str, schema: Dict[str, str]):
        """Create table if not exists"""
        columns = ', '.join([
            f"{col} {dtype}" for col, dtype in schema.items()
        ])
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {columns},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.conn.cursor().execute(query)
    
    def insert_batch(self, table_name: str, data: List[Dict[str, Any]]):
        """Insert batch of records"""
        if not data:
            return
        
        cursor = self.conn.cursor()
        columns = ', '.join(data[0].keys())
        placeholders = ', '.join(['%s'] * len(data[0]))
        
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        values = [tuple(row.values()) for row in data]
        
        cursor.executemany(query, values)
        cursor.close()
        self.conn.commit()
```

---

## Next Steps

1. **Complete Web Scraping Service**: Migrate existing scraper logic
2. **Complete Document Processing Service**: Migrate PDF extractor
3. **Build Frontend**: Implement all components
4. **Add Authentication**: Implement API key validation
5. **Add Monitoring**: Set up Cloud Monitoring dashboards
6. **Load Testing**: Test with concurrent users

---

## Useful Commands

```bash
# View Cloud Run logs
gcloud run services logs read web-scraping-service --region us-central1

# Update service
gcloud run services update web-scraping-service --region us-central1

# Test locally with Cloud SQL proxy
cloud-sql-proxy universal-scraper-saas:us-central1:universal-scraper-db

# Connect to Redis
gcloud redis instances describe universal-scraper-cache --region us-central1
```

