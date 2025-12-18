// Core types for the application

export interface ScrapeRequest {
  url: string;
  fields: string[];
  options?: {
    wait_for_selector?: string;
    scroll_to_bottom?: boolean;
    click_load_more?: string;
    use_browser?: boolean;
  };
}

export interface ScrapeResponse {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  data?: any[];
  metadata?: {
    cache_hit?: boolean;
    processing_time_ms?: number;
    source?: string;
    items_extracted?: number;
  };
}

export interface ProxyConfiguration {
  provider: 'none' | 'apify' | 'brightdata' | 'oxylabs' | 'scraperapi' | 'custom';
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
  is_cached: boolean;
  cache_age?: number;
  cache_key?: string;
  structural_hash?: string;
}

