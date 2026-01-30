// Core types for the application

export interface ScrapeRequest {
  url: string;
  fields: string[];
  options?: {
    mode?: 'hybrid' | 'html' | 'browser' | 'json';
    wait_for_selector?: string;
    scroll_to_bottom?: boolean;
    click_load_more?: string;
    use_browser?: boolean;
    forceHtml?: boolean;
    forceGenerate?: boolean;
  };
  proxyConfig?: {
    server: string;
    username: string;
    password: string;
  } | null;
}

export interface ScrapeResponse {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  data?: any[];
  metadata?: {
    cache_hit?: boolean;
    cache_stored?: boolean;
    cache_type?: string;
    processing_time_ms?: number;
    source?: string;
    items_extracted?: number;
    strategy?: {
      method?: string;
      proxy?: string;
      confidence?: number;
      source?: string;
    };
  };
}

export interface ProxyConfiguration {
  provider: 'none' | 'apify' | 'brightdata' | 'oxylabs' | 'scraperapi' | 'custom' | 'web_unlocker' | 'bright_data';
  apifyProxy?: {
    useApifyProxy: boolean;
    apifyProxyGroups: string[];
    apifyProxyCountry?: string;
  };
  externalProxy?: {
    server: string;
    username: string;
    password: string;
    country?: string;
  };
  webUnblocker?: {
    enabled: boolean;
    apiKey?: string; // API key (Bearer token) OR proxy credentials
    zone?: string; // Zone name (default: web_unlocker1)
    useProxyMethod?: boolean; // If true, use proxy credentials instead of API key
  };
  rotationStrategy?: 'per_request' | 'per_domain' | 'on_failure' | 'session';
  geoLocation?: string;
}

export interface WebUnblockerConfig {
  enabled: boolean;
  apiKey: string;
  zone: string;
  timeout?: number;
  retryOnFailure?: boolean;
  maxRetries?: number;
}

export interface BrowserConfiguration {
  useCamoufox: boolean;
  headless: boolean;
  browserTimeout: number;
  waitForNetworkIdle: boolean;
  captureApiRequests: boolean;
  userAgent?: string;
  viewport?: {
    width: number;
    height: number;
  };
}

export interface PaginationConfiguration {
  enableAutoPagination: boolean;
  maxPages?: number;
  scrollToBottom?: boolean;
  clickLoadMore?: string;
  waitForSelector?: string;
  enableLLMPagination: boolean;
  paginationStrategy?: 'auto' | 'url_param' | 'infinite_scroll' | 'load_more';
}

export interface AIConfiguration {
  provider: 'openai' | 'anthropic' | 'google' | 'custom';
  apiKeys: {
    openai?: string;
    anthropic?: string;
    google?: string;
    custom?: {
      endpoint: string;
      apiKey: string;
    };
  };
  modelName: string;
  useDirectLLM: boolean;
  directLLMQualityMode: 'conservative' | 'balanced' | 'aggressive';
  enableLLMPatternGeneration: boolean;
  similarityThreshold: number;
  cachePatterns: boolean;
}

export interface DocumentProcessingConfig {
  file?: File;
  fileUrl?: string;
  maxPages?: number;
  useOCR: boolean;
  fields: string[] | string;
  context?: string;
}

export interface WarehouseConnectorConfig {
  type: 'snowflake' | 'postgres' | 'bigquery' | 'redshift' | 'databricks';
  config: {
    host?: string;
    port?: number;
    database: string;
    schema?: string;
    table: string;
    username?: string;
    password?: string;
    apiKey?: string;
    account?: string;
    warehouse?: string;
    projectId?: string;
    dataset?: string;
    clusterId?: string;
  };
  createTableIfNotExists: boolean;
  tableSchema?: {
    [fieldName: string]: string;
  };
}

export interface Job {
  id: string;
  tenant_id: string;
  user_id: string;
  job_type: 'web_scraping' | 'document_processing';
  url?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: any;
  metadata?: any;
  created_at: string;
  completed_at?: string;
}

export interface CacheStatus {
  domain?: string | null;
  is_cached: boolean;
  cache_age?: number;
  cache_key?: string;
  structural_hash?: string;
}

// ==================== NEW: Agent Type System ====================

// Agent type discriminator - two main categories
export type AgentType = 'SCRAPER' | 'DOC_PROCESSOR';
export type AgentStatus = 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

// Legacy sub-types for backward compatibility
export type ScraperSubType = 'web_scraping' | 'batch_scraping';
export type ProcessorSubType = 'document_processing';
export type LegacyAgentType = ScraperSubType | ProcessorSubType;

// Discriminated union for agents
export interface BaseAgent {
  id: string;
  name: string;
  description?: string;
  type: AgentType;
  status: AgentStatus;
  spaceId?: string;
  created_at: string;
  updated_at?: string;
  last_run_id?: string;
  tenant_id?: string;
  metadata?: {
    total_runs?: number;
    success_count?: number;
    failure_count?: number;
    avg_duration?: number;
    last_success_at?: string;
    last_failure_at?: string;
  };
}

export interface ScraperAgent extends BaseAgent {
  type: 'SCRAPER';
  definition: {
    subType: ScraperSubType;
    url?: string;
    urls?: string[];
    fields?: string[];
    schedule?: string;
    mode?: 'hybrid' | 'browser' | 'static';
    proxy_config?: ProxyConfiguration;
    pagination_config?: PaginationConfiguration;
    browser_timeout?: number;
    extraction_context?: string;
    cache_enabled?: boolean;
    export?: {
      format: 'csv' | 'json' | 'parquet';
      destination?: string;
    };
  };
}

export interface ProcessorAgent extends BaseAgent {
  type: 'DOC_PROCESSOR';
  definition: {
    subType: ProcessorSubType;
    sources?: Array<{
      id: string;
      name: string;
      type: 'upload' | 'url' | 's3' | 'drive';
      uri?: string;
    }>;
    fields?: string[];
    schedule?: string;
    use_ocr?: boolean;
    max_pages?: number;
    context?: string;
    chunking?: {
      enabled: boolean;
      strategy: 'fixed' | 'semantic' | 'paragraph';
      chunk_size?: number;
      overlap?: number;
    };
    enrichment?: {
      classify?: boolean;
      embed?: boolean;
      summarize?: boolean;
    };
    output?: {
      format: 'json' | 'csv' | 'parquet';
      include_chunks?: boolean;
      include_embeddings?: boolean;
    };
  };
}

// Union type for all agents
export type Agent = ScraperAgent | ProcessorAgent;

// Run outputs - type-specific
export interface ScraperRunOutput {
  rows: Array<Record<string, any>>;
  schema?: {
    columns: Array<{
      name: string;
      type: string;
      required?: boolean;
    }>;
  };
  selection?: {
    selectors: Record<string, string>;
    detectedFields: string[];
  };
  htmlSnapshots?: string[];
  screenshots?: string[];
  pagination?: {
    detected: boolean;
    totalPages?: number;
  };
}

export interface ProcessorRunOutput {
  documents: Array<{
    id: string;
    name: string;
    type: string;
    sourceUri?: string;
    pages?: number;
  }>;
  chunks?: Array<{
    docId: string;
    chunkId: string;
    text: string;
    page?: number;
    tokens?: number;
    embeddingId?: string;
  }>;
  fields: Array<{
    name: string;
    value: any;
    confidence?: number;
    sourceChunkId?: string;
  }>;
  artifacts?: Array<{
    type: 'index' | 'export' | 'embedding' | 'summary';
    uri: string;
    metadata?: Record<string, any>;
  }>;
}

export interface BaseAgentRun {
  id: string;
  agent_id: string;
  agentType: AgentType;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at: string;
  finished_at?: string;
  duration?: number;
  logs?: string[];
  metadata?: any;
}

export interface ScraperRun extends BaseAgentRun {
  agentType: 'SCRAPER';
  outputs: ScraperRunOutput;
  metrics?: {
    items_extracted: number;
    pages_crawled: number;
    cache_hits: number;
    extraction_quality: number;
  };
}

export interface ProcessorRun extends BaseAgentRun {
  agentType: 'DOC_PROCESSOR';
  outputs: ProcessorRunOutput;
  metrics?: {
    documents_processed: number;
    chunks_created: number;
    fields_extracted: number;
    avg_confidence: number;
  };
}

export type AgentRun = ScraperRun | ProcessorRun;

// ==================== Legacy Types (Backward Compatibility) ====================

// Legacy agent interface - maps to new discriminated union
export interface LegacyAgent {
  id: string;
  tenant_id?: string;
  type: LegacyAgentType;
  status: AgentStatus;
  config: {
    url?: string;
    urls?: string[];
    fields?: string[];
    mode?: string;
    scroll_to_bottom?: boolean;
    wait_for_selector?: string;
  };
  result?: any;
  error?: string;
  created_at: number;
  started_at?: number;
  completed_at?: number;
  progress: number;
  progress_message: string;
  // Scheduling
  schedule?: string;
  schedule_id?: string;
  next_run?: number;
  last_run?: number;
  run_count: number;
  // Cache reference
  from_cache: boolean;
  cache_domain?: string;
  cache_visibility?: 'public' | 'private';
}

// Legacy run interface
export interface LegacyAgentRun {
  id: string;
  agent_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at: string;
  finished_at?: string;
  duration?: number;
  result?: {
    items_extracted?: number;
    data?: any[];
    error?: string;
  };
  metadata?: any;
  logs?: string[];
}

// ==================== Adapter Functions ====================

// Convert legacy agent to new discriminated union
export function adaptLegacyAgent(legacy: LegacyAgent): Agent {
  const isProcessor = legacy.type === 'document_processing';

  if (isProcessor) {
    return {
      id: legacy.id,
      name: `Agent ${legacy.id.slice(0, 8)}`,
      type: 'DOC_PROCESSOR',
      status: legacy.status,
      tenant_id: legacy.tenant_id,
      created_at: new Date(legacy.created_at).toISOString(),
      updated_at: legacy.completed_at ? new Date(legacy.completed_at).toISOString() : undefined,
      definition: {
        subType: 'document_processing',
        fields: legacy.config.fields,
        schedule: legacy.schedule,
      },
    } as ProcessorAgent;
  }

  return {
    id: legacy.id,
    name: `Agent ${legacy.id.slice(0, 8)}`,
    type: 'SCRAPER',
    status: legacy.status,
    tenant_id: legacy.tenant_id,
    created_at: new Date(legacy.created_at).toISOString(),
    updated_at: legacy.completed_at ? new Date(legacy.completed_at).toISOString() : undefined,
    definition: {
      subType: legacy.type === 'batch_scraping' ? 'batch_scraping' : 'web_scraping',
      url: legacy.config.url,
      urls: legacy.config.urls,
      fields: legacy.config.fields,
      schedule: legacy.schedule,
      mode: legacy.config.mode as any || 'hybrid',
    },
  } as ScraperAgent;
}

// Pattern cache types (unchanged)
export type PatternVisibility = 'public' | 'private';

export interface CachedPattern {
  tenant_id?: string;
  domain: string;
  fields: string[];
  fields_hash: string;
  pattern_data: any;
  visibility: PatternVisibility;
  url?: string;
  created_at: number;
  updated_at: number;
  usage_count: number;
  shared_from?: string;
}
