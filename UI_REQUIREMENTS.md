# Frontend UI Requirements & Configuration Inputs
## Complete Analysis of Required UI Components

---

## Overview

This document catalogs all configuration inputs, settings, and UI functionality required for the cloud-native SaaS frontend based on the Universal Scraper codebase analysis.

---

## 1. Core Scraping Configuration

### 1.1 URL Input
**Component**: `UrlInput.tsx`
**Type**: Text input / URL list
**Required**: Yes

```typescript
interface UrlInput {
  // Single URL mode
  url?: string;
  
  // Multiple URLs mode
  urls?: Array<{
    url: string;
    label?: string; // Optional label for organization
  }>;
  
  // URL source options
  sourceType: 'single' | 'multiple' | 'file_upload' | 'crawl';
}
```

**UI Features Needed:**
- Single URL input field
- Multiple URL input (textarea or list)
- URL validation (format checking)
- URL history/autocomplete
- Import from file (CSV, TXT)
- URL preview/validation before scraping

---

### 1.2 Fields to Extract
**Component**: `FieldSelector.tsx`
**Type**: Multi-select / Text input
**Required**: Yes

```typescript
interface FieldInput {
  // Option 1: Natural language (recommended for beginners)
  fieldsNaturalLanguage?: string;
  // Example: "Extract product names, prices, and descriptions"
  
  // Option 2: Structured field list (advanced)
  fields?: string[];
  // Example: ["title", "price", "description", "rating"]
  
  // Field type hints (optional)
  fieldTypes?: {
    [fieldName: string]: 'text' | 'number' | 'date' | 'url' | 'image';
  };
}
```

**UI Features Needed:**
- Natural language input (textarea) with examples
- Structured field list builder
- Field type selector (text, number, date, URL, image)
- Field validation (required/optional)
- Field preview from sample page
- Field suggestions based on page content

---

## 2. Proxy Configuration

### 2.1 Proxy Provider Selection
**Component**: `ProxyConfig.tsx`
**Type**: Tabbed interface / Dropdown
**Required**: No (optional)

```typescript
interface ProxyConfiguration {
  // Provider selection
  provider: 'none' | 'apify' | 'brightdata' | 'oxylabs' | 'scraperapi' | 'custom';
  
  // Apify Proxy (if provider === 'apify')
  apifyProxy?: {
    useApifyProxy: boolean;
    apifyProxyGroups: string[]; // ['RESIDENTIAL', 'DATACENTER']
    apifyProxyCountry?: string; // ISO2 country code
  };
  
  // Bright Data / External Proxy (if provider === 'brightdata' or 'custom')
  externalProxy?: {
    server: string; // e.g., "http://brd.superproxy.io:33335"
    username: string;
    password: string; // Secret field
    country?: string; // Geographic targeting
  };
  
  // Proxy rotation strategy
  rotationStrategy?: 'per_request' | 'per_domain' | 'on_failure' | 'session';
  
  // Geographic targeting
  geoLocation?: string; // ISO2 country code: 'US', 'GB', 'DE', etc.
}
```

**UI Features Needed:**
- Provider selector (dropdown/tabs)
- Provider-specific configuration forms
- Test proxy connection button
- Proxy status indicator (connected/disconnected)
- Geographic location selector (country dropdown)
- Rotation strategy selector
- Proxy credentials management (secure storage)
- Multiple proxy pool management (for custom provider)

**UI Layout:**
```
┌─────────────────────────────────────┐
│ Proxy Configuration                  │
├─────────────────────────────────────┤
│ [ ] No Proxy                         │
│ [✓] Apify Proxy                      │
│ [ ] Bright Data                      │
│ [ ] OxyLabs                          │
│ [ ] Custom                           │
├─────────────────────────────────────┤
│ Apify Proxy Settings:                │
│ ☑ Use Residential Proxies           │
│ ☐ Use Datacenter Proxies            │
│ Country: [Select Country ▼]          │
│ [Test Connection]                    │
└─────────────────────────────────────┘
```

---

## 3. Web Unblocker Configuration

### 3.1 Bright Data Web Unblocker
**Component**: `WebUnblockerConfig.tsx`
**Type**: Form with API key input
**Required**: No (optional, fallback for blocked sites)

```typescript
interface WebUnblockerConfig {
  // Enable Web Unblocker
  enabled: boolean;
  
  // API Key (can be Bearer token or proxy credentials)
  apiKey: string; // Secret field
  
  // Zone name
  zone: string; // Default: "web_unlocker1"
  
  // Method selection (auto-detected from API key format)
  method?: 'api' | 'proxy'; // Auto-detected
  
  // Timeout settings
  timeout?: number; // Default: 120 seconds
  
  // Retry settings
  retryOnFailure?: boolean; // Default: true
  maxRetries?: number; // Default: 3
}
```

**UI Features Needed:**
- Enable/disable toggle
- API key input (masked/secret field)
- Zone name input (with default)
- Method indicator (shows detected method: API vs Proxy)
- Test connection button
- Status indicator (active/inactive)
- Usage statistics (requests made, success rate)
- Cost estimator (if applicable)

**UI Layout:**
```
┌─────────────────────────────────────┐
│ Web Unblocker (Bright Data)          │
├─────────────────────────────────────┤
│ ☑ Enable Web Unblocker              │
│                                     │
│ API Key: [••••••••••••••••]        │
│ Zone: [web_unlocker1        ]       │
│                                     │
│ Method: Direct API Access ✓         │
│ [Test Connection]                   │
│                                     │
│ Status: ● Active                    │
│ Usage: 1,234 requests (98% success) │
└─────────────────────────────────────┘
```

---

## 4. Browser Configuration

### 4.1 Browser Settings
**Component**: `BrowserConfig.tsx`
**Type**: Checkboxes and dropdowns
**Required**: No (has defaults)

```typescript
interface BrowserConfiguration {
  // Browser selection
  useCamoufox: boolean; // Default: true (recommended)
  // Alternative: Playwright (if useCamoufox === false)
  
  // Headless mode
  headless: boolean; // Default: true
  
  // Browser timeout
  browserTimeout: number; // Milliseconds, default: 60000 (60s)
  
  // Network settings
  waitForNetworkIdle: boolean; // Default: true
  captureApiRequests: boolean; // Default: true
  
  // Advanced browser options
  userAgent?: string; // Custom user agent
  viewport?: {
    width: number;
    height: number;
  };
  
  // JavaScript execution
  enableJavaScript: boolean; // Default: true
}
```

**UI Features Needed:**
- Browser type selector (Camoufox vs Playwright)
- Headless mode toggle
- Timeout slider/input
- Network idle wait toggle
- API capture toggle
- Custom user agent input (optional)
- Viewport size inputs (optional)
- Browser preview toggle (for debugging)

**UI Layout:**
```
┌─────────────────────────────────────┐
│ Browser Configuration               │
├─────────────────────────────────────┤
│ Browser: [Camoufox ▼] ✓ Recommended│
│                                     │
│ ☑ Headless Mode                    │
│ ☑ Wait for Network Idle             │
│ ☑ Capture API Requests             │
│                                     │
│ Timeout: [60] seconds               │
│                                     │
│ [Advanced Options ▼]               │
│   User Agent: [Custom...]           │
│   Viewport: 1920 x 1080            │
└─────────────────────────────────────┘
```

---

## 5. Pagination Configuration

### 5.1 Pagination Settings
**Component**: `PaginationConfig.tsx`
**Type**: Toggles and inputs
**Required**: No (has defaults)

```typescript
interface PaginationConfiguration {
  // Auto-pagination
  enableAutoPagination: boolean; // Default: false
  
  // Max pages limit
  maxPages?: number; // 0 = unlimited, default: 0
  
  // Pagination hints
  scrollToBottom?: boolean; // For infinite scroll
  clickLoadMore?: string; // CSS selector for "Load More" button
  waitForSelector?: string; // CSS selector to wait for
  
  // Pagination detection
  enableLLMPagination: boolean; // Default: true
  paginationStrategy?: 'auto' | 'url_param' | 'infinite_scroll' | 'load_more';
}
```

**UI Features Needed:**
- Enable auto-pagination toggle
- Max pages input (with "unlimited" option)
- Pagination type selector (auto-detect vs manual)
- Infinite scroll toggle
- "Load More" button selector input
- Wait for selector input
- Pagination preview (shows detected pagination pattern)

**UI Layout:**
```
┌─────────────────────────────────────┐
│ Pagination Settings                  │
├─────────────────────────────────────┤
│ ☑ Enable Auto-Pagination            │
│                                     │
│ Max Pages: [0] (0 = unlimited)      │
│                                     │
│ Pagination Type: [Auto-detect ▼]   │
│                                     │
│ ☐ Scroll to Bottom (infinite scroll)│
│ ☐ Click "Load More" Button          │
│   Selector: [.load-more-btn]        │
│ ☐ Wait for Selector                 │
│   Selector: [.product-list]          │
│                                     │
│ Detected Pattern: URL-based (page=) │
└─────────────────────────────────────┘
```

---

## 6. AI/LLM Configuration

### 6.1 AI Provider Settings
**Component**: `AIConfig.tsx`
**Type**: API key inputs and dropdowns
**Required**: Yes (for AI extraction)

```typescript
interface AIConfiguration {
  // Provider selection
  provider: 'openai' | 'anthropic' | 'google' | 'custom';
  
  // API Keys (stored securely)
  apiKeys: {
    openai?: string; // Secret
    anthropic?: string; // Secret
    google?: string; // Secret
    custom?: {
      endpoint: string;
      apiKey: string; // Secret
    };
  };
  
  // Model selection
  modelName: string; // Default: "gpt-4o-mini"
  // Options: "gpt-4o-mini", "gpt-4o", "claude-3-haiku", "gemini-2.0-flash"
  
  // Extraction mode
  useDirectLLM: boolean; // Default: true
  directLLMQualityMode: 'conservative' | 'balanced' | 'aggressive'; // Default: 'balanced'
  
  // Pattern generation (if useDirectLLM === false)
  enableLLMPatternGeneration: boolean; // Default: true
  similarityThreshold: number; // 0.0-1.0, default: 0.75
  cachePatterns: boolean; // Default: true
}
```

**UI Features Needed:**
- Provider selector (OpenAI, Anthropic, Google, Custom)
- API key inputs (masked/secret fields)
- Model selector dropdown
- Quality mode selector (conservative/balanced/aggressive)
- Extraction mode toggle (Direct LLM vs Pattern-based)
- Pattern cache toggle
- Similarity threshold slider
- Cost estimator (shows estimated cost per request)
- API key validation/test

**UI Layout:**
```
┌─────────────────────────────────────┐
│ AI Configuration                    │
├─────────────────────────────────────┤
│ Provider: [OpenAI ▼]                │
│                                     │
│ API Key: [••••••••••••••••]        │
│ [Test API Key]                      │
│                                     │
│ Model: [gpt-4o-mini ▼]             │
│                                     │
│ Extraction Mode:                    │
│ ○ Direct LLM (Recommended)         │
│ ○ Pattern-Based                    │
│                                     │
│ Quality Mode: [Balanced ▼]          │
│   Conservative | Balanced | Aggressive│
│                                     │
│ ☑ Cache Patterns                    │
│ Similarity Threshold: [0.75]        │
│                                     │
│ Estimated Cost: ~$0.02 per domain  │
└─────────────────────────────────────┘
```

---

## 7. Document Processing Configuration

### 7.1 PDF/Document Settings
**Component**: `DocumentConfig.tsx`
**Type**: File upload and settings
**Required**: Yes (for document processing module)

```typescript
interface DocumentProcessingConfig {
  // File input
  file?: File; // Uploaded file
  fileUrl?: string; // Or URL to file
  
  // Processing options
  maxPages?: number; // Limit pages to process (null = all)
  useOCR: boolean; // Default: false (for scanned PDFs)
  
  // Fields to extract (same as web scraping)
  fields: string[] | string; // Natural language or structured
  
  // Context (optional)
  context?: string; // Additional context about document type
}
```

**UI Features Needed:**
- File upload component (drag-drop)
- File URL input (alternative to upload)
- File preview (PDF viewer, document preview)
- Max pages input
- OCR toggle (for scanned documents)
- Field selector (same as web scraping)
- Context input (optional)
- Document type detection indicator

**UI Layout:**
```
┌─────────────────────────────────────┐
│ Document Processing                 │
├─────────────────────────────────────┤
│ Upload File:                        │
│ [Drag & Drop or Click to Upload]   │
│                                     │
│ Or Enter URL: [https://...]         │
│                                     │
│ File: document.pdf (2.5 MB)        │
│ Pages: 10                           │
│                                     │
│ Max Pages: [All] or [10]           │
│ ☐ Use OCR (for scanned documents)  │
│                                     │
│ Fields to Extract:                  │
│ [title, date, amount, ...]          │
│                                     │
│ Context (optional):                 │
│ [Invoice, Receipt, Contract...]     │
└─────────────────────────────────────┘
```

---

## 8. Advanced Configuration

### 8.1 Crawl Configuration
**Component**: `CrawlConfig.tsx`
**Type**: Advanced settings panel
**Required**: No (for crawl mode)

```typescript
interface CrawlConfiguration {
  // Crawl limits
  maxDepth: number; // Default: 3
  maxPages: number; // Default: 1000
  maxItems: number; // Default: 10000
  
  // URL patterns
  followPatterns?: string[]; // e.g., ['/product/', '/item/']
  ignorePatterns?: string[]; // e.g., ['/login', '/cart']
  
  // Features
  handlePagination: boolean; // Default: true
  discoverApis: boolean; // Default: true
  enableSearchDiscovery: boolean; // Default: true
  
  // Search discovery (if enabled)
  searchConfig?: {
    strategy: 'auto' | 'alphabetic' | 'numeric' | 'date';
    maxDepth: number; // Default: 4
    resultLimit: number; // 0 = auto-detect
  };
}
```

**UI Features Needed:**
- Max depth slider/input
- Max pages input
- Max items input
- URL pattern builder (include/exclude patterns)
- Feature toggles (pagination, API discovery, search)
- Search discovery configuration (if enabled)

---

### 8.2 Schema Configuration
**Component**: `SchemaConfig.tsx`
**Type**: Schema builder
**Required**: No (optional)

```typescript
interface SchemaConfiguration {
  // Schema usage
  useSchema: boolean; // Default: false
  
  // Schema type
  schemaType: 'auto' | 'ecommerce' | 'custom';
  
  // Custom schema (if schemaType === 'custom')
  customSchema?: {
    [fieldName: string]: {
      type: 'string' | 'number' | 'date' | 'boolean' | 'array';
      required: boolean;
      format?: string; // e.g., 'email', 'url', 'date-time'
    };
  };
  
  // Strict mode
  strictSchema: boolean; // Default: false (fail on missing required fields)
}
```

**UI Features Needed:**
- Schema type selector
- Schema builder (visual field editor)
- Field type selector
- Required/optional toggle per field
- Format selector (email, URL, date-time, etc.)
- Schema preview (JSON view)
- Schema validation

---

### 8.3 Output Configuration
**Component**: `OutputConfig.tsx`
**Type**: Format selector
**Required**: No (has defaults)

```typescript
interface OutputConfiguration {
  // Format
  outputFormat: 'json' | 'jsonl' | 'csv' | 'xlsx';
  
  // Destination
  destination?: 'download' | 'warehouse' | 'webhook' | 'storage';
  
  // Warehouse config (if destination === 'warehouse')
  warehouse?: {
    type: 'snowflake' | 'postgres' | 'bigquery' | 'redshift' | 'databricks';
    config: {
      // Provider-specific config
      host?: string;
      database?: string;
      schema?: string;
      table?: string;
      credentials: {
        username?: string;
        password?: string; // Secret
        apiKey?: string; // Secret
      };
    };
  };
  
  // Webhook (if destination === 'webhook')
  webhookUrl?: string;
  
  // Concurrency
  maxConcurrency?: number; // Default: 10
}
```

**UI Features Needed:**
- Output format selector
- Destination selector (download, warehouse, webhook)
- Warehouse connector configuration (if selected)
- Webhook URL input (if selected)
- Concurrency slider
- Test connection button (for warehouse)

---

## 9. Cache & Performance Settings

### 9.1 Cache Configuration
**Component**: `CacheConfig.tsx`
**Type**: Toggles and settings
**Required**: No (has defaults)

```typescript
interface CacheConfiguration {
  // Enable caching
  enableCache: boolean; // Default: true
  
  // Cache TTL
  cacheTTL: number; // Seconds, default: 3600 (1 hour)
  
  // Cache types
  cacheStructuralHash: boolean; // Default: true
  cacheExtractionCode: boolean; // Default: true
  cacheResults: boolean; // Default: true
  
  // Cache invalidation
  invalidateOnChange: boolean; // Default: true
}
```

**UI Features Needed:**
- Enable cache toggle
- Cache TTL input
- Cache type checkboxes
- Cache status indicator (shows cache hit rate)
- Clear cache button
- Cache statistics (size, hit rate, etc.)

---

## 10. UI Components Required

### 10.1 Main Dashboard Components

1. **Browser Preview Component**
   - Embedded iframe or screenshot API
   - Shows page structure before scraping
   - Highlights detected elements
   - Interactive element selector

2. **Cache Indicator**
   - Visual badge showing cache status
   - Cache hit rate display
   - Cache age indicator

3. **Raw Data Viewer**
   - JSON/HTML syntax highlighting
   - Expandable/collapsible sections
   - Copy to clipboard
   - Export options

4. **Field Mapping Interface**
   - Drag-drop field mapper
   - Visual field selector from page
   - Field preview with sample data
   - Field validation

5. **Job Status Monitor**
   - Real-time job progress (WebSocket)
   - Job history table
   - Success/failure indicators
   - Retry options

6. **Results Table**
   - Sortable columns
   - Filterable rows
   - Export buttons
   - Pagination

7. **Settings Panel**
   - Collapsible sections
   - Form validation
   - Save/cancel buttons
   - Reset to defaults

---

## 11. User Account & Authentication

### 11.1 Account Settings
**Component**: `AccountSettings.tsx`

```typescript
interface AccountSettings {
  // User info
  email: string;
  name: string;
  
  // API Keys (for external services)
  apiKeys: {
    openai?: string; // Stored securely
    anthropic?: string;
    brightdata?: string;
    // ... other providers
  };
  
  // Subscription
  plan: 'free' | 'pro' | 'enterprise';
  usage: {
    requestsThisMonth: number;
    requestsLimit: number;
    llmCallsThisMonth: number;
    llmCallsLimit: number;
  };
  
  // Billing
  billingEmail?: string;
  paymentMethod?: {
    type: 'card' | 'invoice';
    last4?: string;
  };
}
```

**UI Features Needed:**
- Profile editor
- API key management (add/edit/delete)
- Usage dashboard (charts, limits)
- Subscription management
- Billing information
- Invoice history

---

## 12. Data Warehouse Connectors UI

### 12.1 Warehouse Configuration
**Component**: `WarehouseConnector.tsx`

```typescript
interface WarehouseConnectorConfig {
  // Warehouse type
  type: 'snowflake' | 'postgres' | 'bigquery' | 'redshift' | 'databricks';
  
  // Connection details (varies by type)
  config: {
    // Common
    host?: string;
    port?: number;
    database: string;
    schema?: string;
    table: string;
    
    // Credentials
    username?: string;
    password?: string; // Secret
    apiKey?: string; // Secret (for BigQuery, Databricks)
    
    // Snowflake specific
    account?: string;
    warehouse?: string;
    
    // BigQuery specific
    projectId?: string;
    dataset?: string;
    
    // Redshift specific
    clusterId?: string;
  };
  
  // Table creation
  createTableIfNotExists: boolean; // Default: true
  tableSchema?: {
    [fieldName: string]: string; // SQL type
  };
}
```

**UI Features Needed:**
- Warehouse type selector
- Connection form (dynamic based on type)
- Test connection button
- Table selector/builder
- Schema mapper (map extracted fields to table columns)
- Preview data before insert
- Connection status indicator

**UI Layout:**
```
┌─────────────────────────────────────┐
│ Data Warehouse Connector            │
├─────────────────────────────────────┤
│ Warehouse: [Snowflake ▼]            │
│                                     │
│ Account: [your-account]             │
│ Username: [username]                │
│ Password: [••••••••]                │
│ Warehouse: [COMPUTE_WH]              │
│ Database: [PRODUCTION]              │
│ Schema: [PUBLIC]                    │
│ Table: [scraped_data]               │
│                                     │
│ ☑ Create table if not exists       │
│                                     │
│ [Test Connection]                   │
│ Status: ● Connected                │
│                                     │
│ Field Mapping:                      │
│ title → TITLE (VARCHAR)             │
│ price → PRICE (DECIMAL)            │
│ [Map Fields...]                     │
└─────────────────────────────────────┘
```

---

## 13. Form Validation & Error Handling

### 13.1 Validation Rules

**Required Fields:**
- URLs (at least one)
- Fields to extract (at least one)
- API key (if using AI extraction)

**Format Validation:**
- URL format (http/https)
- Email format (for account)
- API key format (provider-specific)
- SQL table names (for warehouse)

**Business Logic Validation:**
- Max pages > 0 if auto-pagination enabled
- Proxy credentials complete if proxy enabled
- Warehouse config complete if warehouse destination selected

---

## 14. UI/UX Patterns

### 14.1 Design System

**Color Scheme:**
- Primary: Blue (for actions)
- Success: Green (for cache hits, success)
- Warning: Yellow (for cache miss, processing)
- Error: Red (for failures)
- Info: Gray (for neutral information)

**Component Library:**
- Use Headless UI for accessible components
- Tailwind CSS for styling
- React Hook Form for form management
- Zod for validation schemas

### 14.2 User Flows

**Flow 1: Quick Scrape (Beginner)**
1. Enter URL
2. Enter fields (natural language)
3. Click "Scrape"
4. View results

**Flow 2: Advanced Scrape (Power User)**
1. Enter URL
2. Configure proxy
3. Configure browser settings
4. Set pagination options
5. Configure AI settings
6. Set output destination
7. Click "Scrape"
8. Monitor job progress
9. View/download results

**Flow 3: Document Processing**
1. Upload PDF/document
2. Enter fields to extract
3. Configure OCR (if needed)
4. Click "Process"
5. View extracted data
6. Export or send to warehouse

---

## 15. Real-time Features

### 15.1 WebSocket Integration

**Events to Subscribe:**
- `job.created` - Job started
- `job.progress` - Job progress update
- `job.completed` - Job finished
- `job.failed` - Job failed
- `cache.hit` - Cache hit detected
- `cache.miss` - Cache miss detected

**UI Updates:**
- Real-time job status
- Progress bars
- Live cache indicators
- Notification toasts

---

## 16. Summary: Required UI Components

### Core Components
1. ✅ `UrlInput` - URL entry
2. ✅ `FieldSelector` - Field extraction configuration
3. ✅ `ProxyConfig` - Proxy provider settings
4. ✅ `WebUnblockerConfig` - Bright Data Web Unblocker
5. ✅ `BrowserConfig` - Browser automation settings
6. ✅ `PaginationConfig` - Pagination options
7. ✅ `AIConfig` - AI provider and model settings
8. ✅ `DocumentConfig` - Document processing settings
9. ✅ `OutputConfig` - Output format and destination
10. ✅ `CacheConfig` - Cache settings

### Display Components
11. ✅ `BrowserPreview` - Page preview
12. ✅ `CacheIndicator` - Cache status
13. ✅ `RawDataViewer` - JSON/HTML viewer
14. ✅ `ResultsTable` - Extracted data table
15. ✅ `JobStatus` - Job monitoring
16. ✅ `FieldMapper` - Visual field mapping

### Advanced Components
17. ✅ `CrawlConfig` - Crawl settings
18. ✅ `SchemaConfig` - Schema builder
19. ✅ `WarehouseConnector` - Data warehouse setup
20. ✅ `AccountSettings` - User account management

### Layout Components
21. ✅ `Dashboard` - Main dashboard
22. ✅ `Sidebar` - Navigation
23. ✅ `Header` - Top bar with user menu
24. ✅ `SettingsPanel` - Collapsible settings

---

## 17. Implementation Priority

### Phase 1 (MVP)
- URL input
- Field selector (natural language)
- Basic AI config (API key)
- Results table
- Job status

### Phase 2 (Core Features)
- Proxy configuration
- Browser settings
- Pagination config
- Cache indicator
- Browser preview

### Phase 3 (Advanced)
- Web Unblocker
- Document processing
- Warehouse connectors
- Schema builder
- Advanced crawl config

---

## 18. API Endpoints Required

### Scraping Endpoints
- `POST /api/v1/web-scraping/scrape` - Start scraping job
- `GET /api/v1/web-scraping/jobs/:id` - Get job status
- `GET /api/v1/web-scraping/jobs/:id/results` - Get results
- `GET /api/v1/web-scraping/preview?url=...` - Get page preview

### Document Endpoints
- `POST /api/v1/document-processing/extract` - Process document
- `GET /api/v1/document-processing/jobs/:id` - Get job status

### Configuration Endpoints
- `GET /api/v1/cache/status?url=...` - Check cache status
- `POST /api/v1/proxy/test` - Test proxy connection
- `POST /api/v1/warehouse/test` - Test warehouse connection

### Account Endpoints
- `GET /api/v1/account/settings` - Get account settings
- `PUT /api/v1/account/settings` - Update settings
- `GET /api/v1/account/usage` - Get usage statistics

---

This comprehensive analysis covers all UI inputs and functionality needed based on the codebase. The frontend should implement these components to provide a complete, user-friendly interface for the Universal Scraper SaaS platform.

