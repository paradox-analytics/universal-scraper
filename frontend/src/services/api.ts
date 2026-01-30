import axios from 'axios';
import type { ScrapeRequest, ScrapeResponse, ProxyConfiguration, WarehouseConnectorConfig, CacheStatus, Agent, CachedPattern, PatternVisibility } from '../types';
import { API_BASE_URL, getApiKey } from '../config/api';
import { getAuthToken } from './auth';

console.log('API_BASE_URL:', API_BASE_URL);

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Token getter function (can be updated from auth context)
let tokenGetter: (() => Promise<string | null>) | null = null;

export const setTokenGetter = (getter: () => Promise<string | null>) => {
  tokenGetter = getter;
};

// Add auth token interceptor
api.interceptors.request.use(async (config) => {
  console.log('API Request:', {
    method: config.method,
    url: config.url,
    fullUrl: `${config.baseURL}${config.url}`,
    headers: config.headers,
  });
  // Get Firebase auth token
  const authToken = tokenGetter ? await tokenGetter() : await getAuthToken();
  if (authToken) {
    config.headers.Authorization = `Bearer ${authToken}`;
  }

  // Also include API key if available (for backward compatibility)
  const apiKey = getApiKey();
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }

  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', {
      status: response.status,
      url: response.config.url,
      data: response.data,
    });
    return response;
  },
  (error) => {
    console.error('API Error Response:', {
      status: error.response?.status,
      url: error.config?.url,
      data: error.response?.data,
      message: error.message,
    });

    if (error.response?.status === 401) {
      // Handle unauthorized - could redirect to login in future
      console.error('Unauthorized - check API key');
    }
    return Promise.reject(error);
  }
);

// Scraping endpoints - adapted for Cloud Run backend
export const scrapingApi = {
  scrape: async (data: ScrapeRequest): Promise<{ data: ScrapeResponse }> => {
    try {
      // Transform to Cloud Run API format
      const requestPayload: any = {
        url: data.url,
        fields: data.fields || [],
        mode: data.options?.mode || 'hybrid',
        force_html: data.options?.forceHtml || false,
        force_generate: data.options?.forceGenerate || false,
        scroll_to_bottom: data.options?.scroll_to_bottom || false,
      };

      // Only include optional fields if they're provided
      if (data.options?.click_load_more) {
        requestPayload.click_load_more = data.options.click_load_more;
      }
      if (data.options?.wait_for_selector) {
        requestPayload.wait_for_selector = data.options.wait_for_selector;
      }

      // Include proxy config if provided
      if (data.proxyConfig) {
        requestPayload.proxy_config = data.proxyConfig;
      }

      const response = await api.post('/scrape', requestPayload);

      // Transform response to expected format
      return {
        data: {
          job_id: `scrape-${Date.now()}`,
          status: 'completed',
          data: Array.isArray(response.data.data) ? response.data.data : [],
          metadata: {
            ...response.data.metadata,
            source: response.data.source || 'unknown',
          },
        } as ScrapeResponse,
      };
    } catch (error: any) {
      // Better error handling with detailed logging
      console.error('API Error:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
      });

      const errorMessage =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        'Failed to scrape URL';
      throw new Error(errorMessage);
    }
  },

  suggestFields: async (url: string, useLlm: boolean = false, proxyConfig?: ProxyConfiguration): Promise<{ data: { fields: string[], confidence: number, source: string, reasoning: string } }> => {
    try {
      const requestPayload: any = {
        url,
        use_llm: useLlm,
      };

      if (proxyConfig) {
        requestPayload.proxy_config = proxyConfig;
      }

      const response = await api.post('/api/v1/suggest-fields', requestPayload);
      return {
        data: {
          fields: response.data.fields || [],
          confidence: response.data.confidence || 0,
          source: response.data.source || 'unknown',
          reasoning: response.data.reasoning || '',
        },
      };
    } catch (error: any) {
      console.error('Field suggestion error:', error);
      throw new Error(error.response?.data?.detail || error.message || 'Failed to suggest fields');
    }
  },

  getJob: (_jobId: string): Promise<{ data: ScrapeResponse }> => {
    // TODO: Implement job tracking endpoint
    return Promise.resolve({
      data: {
        job_id: _jobId,
        status: 'completed',
        data: [],
      } as ScrapeResponse,
    });
  },

  getResults: (_jobId: string): Promise<{ data: any }> => {
    // TODO: Implement results endpoint
    return Promise.resolve({ data: [] });
  },

  preview: (_url: string): Promise<{ data: { html: string; screenshot?: string } }> => {
    // TODO: Implement preview endpoint
    return Promise.resolve({ data: { html: '' } });
  },

  // Batch scraping - creates agents for multiple URLs
  batchScrape: async (urls: string[], fields: string[], options?: any, proxyConfig?: any): Promise<{ data: { agents: Agent[] } }> => {
    const agents: Agent[] = [];
    for (const url of urls) {
      try {
        const agentResponse = await agentApi.createAgent({
          type: 'web_scraping',
          config: {
            url,
            fields,
            mode: options?.mode || 'hybrid',
            scroll_to_bottom: options?.scroll_to_bottom || false,
            click_load_more: options?.click_load_more,
            wait_for_selector: options?.wait_for_selector,
            proxy_config: proxyConfig,
          },
          queue_immediately: true,
        });
        if (agentResponse.data.agent) {
          agents.push(agentResponse.data.agent);
        }
      } catch (err) {
        console.error(`Failed to create agent for ${url}:`, err);
      }
    }
    return { data: { agents } };
  },

  checkCache: async (url: string, fields?: string[]): Promise<{ data: CacheStatus }> => {
    try {
      const fieldsParam = fields && fields.length > 0 ? fields.join(',') : undefined;
      const params = new URLSearchParams({ url });
      if (fieldsParam) {
        params.append('fields', fieldsParam);
      }

      const response = await api.get(`/api/v1/cache/check?${params.toString()}`);
      return {
        data: {
          is_cached: response.data.is_cached || false,
          cache_age: response.data.cache_age || undefined,
          domain: response.data.domain || null,
        } as CacheStatus,
      };
    } catch (error: any) {
      console.error('Cache check failed:', error);
      return {
        data: {
          is_cached: false,
          cache_age: undefined,
          domain: null,
        } as CacheStatus,
      };
    }
  },
};

// Document processing endpoints
export const documentApi = {
  extract: async (data: FormData): Promise<{ data: ScrapeResponse }> => {
    try {
      // Create a new axios instance for multipart requests to avoid Content-Type override
      const multipartApi = axios.create({
        baseURL: API_BASE_URL,
      });

      // Add API key to headers
      const apiKey = getApiKey();
      if (apiKey) {
        multipartApi.defaults.headers.common['X-API-Key'] = apiKey;
      }

      // Don't set Content-Type - let axios set it automatically with boundary
      const response = await multipartApi.post('/document-processing/extract', data);

      console.log('Document processing response:', response.data);

      // Transform response to expected format
      return {
        data: {
          job_id: `doc-${Date.now()}`,
          status: 'completed',
          data: Array.isArray(response.data.data) ? response.data.data : [],
          metadata: {
            ...response.data.metadata,
          },
        } as ScrapeResponse,
      };
    } catch (error: any) {
      console.error('Document processing error:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        config: error.config,
      });

      const errorMessage =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message ||
        'Failed to process document';
      throw new Error(errorMessage);
    }
  },

  // Batch document processing - creates agents for multiple files
  batchExtract: async (files: File[], fields: string[], config?: any): Promise<{ data: { agents: Agent[] } }> => {
    const agents: Agent[] = [];
    for (const file of files) {
      try {
        // For now, we'll process files sequentially and create agents
        // In the future, we could upload files to storage and reference them
        const formData = new FormData();
        formData.append('file', file);
        formData.append('fields', JSON.stringify(fields.length > 0 ? fields : ['text', 'title', 'metadata']));
        if (config?.useOCR) {
          formData.append('use_ocr', String(config.useOCR));
        }
        if (config?.maxPages) {
          formData.append('max_pages', String(config.maxPages));
        }
        if (config?.context) {
          formData.append('context', config.context);
        }

        // Process file and create agent with result
        const agentResponse = await agentApi.createAgent({
          type: 'document_processing',
          config: {
            filename: file.name,
            fields,
            useOCR: config?.useOCR || false,
            maxPages: config?.maxPages,
            context: config?.context,
          },
          queue_immediately: true,
        });
        if (agentResponse.data.agent) {
          agents.push(agentResponse.data.agent);
        }
      } catch (err) {
        console.error(`Failed to create agent for ${file.name}:`, err);
      }
    }
    return { data: { agents } };
  },

  getJob: (jobId: string): Promise<{ data: ScrapeResponse }> =>
    Promise.resolve({
      data: {
        job_id: jobId,
        status: 'completed',
        data: [],
      } as ScrapeResponse,
    }),
};

// Configuration endpoints
export const configApi = {
  testProxy: (config: ProxyConfiguration | { provider: string; webUnblocker?: any }): Promise<{ data: { success: boolean; message?: string; ip?: string } }> => {
    // Transform ProxyConfiguration to backend format
    const requestBody: any = {
      provider: config.provider,
    };

    // Handle Web Unblocker
    if (config.provider === 'web_unlocker' || config.provider === 'webunblocker') {
      const webUnblocker = (config as any).webUnblocker;
      if (webUnblocker) {
        requestBody.webUnblocker = {
          apiKey: webUnblocker.apiKey || webUnblocker.api_key,
          zone: webUnblocker.zone || 'web_unlocker1',
        };
      }
      return api.post('/api/v1/proxy/test', requestBody);
    }

    // Type guard for ProxyConfiguration
    const proxyConfig = config as ProxyConfiguration;

    if (proxyConfig.provider === 'brightdata' || proxyConfig.provider === 'oxylabs' || proxyConfig.provider === 'scraperapi' || proxyConfig.provider === 'custom') {
      if (proxyConfig.externalProxy) {
        requestBody.server = proxyConfig.externalProxy.server;
        requestBody.username = proxyConfig.externalProxy.username;
        requestBody.password = proxyConfig.externalProxy.password;
        requestBody.country = proxyConfig.externalProxy.country;
        requestBody.externalProxy = proxyConfig.externalProxy;
      }
    } else if (proxyConfig.provider === 'apify' && proxyConfig.apifyProxy) {
      requestBody.apifyProxy = proxyConfig.apifyProxy;
    }

    return api.post('/api/v1/proxy/test', requestBody);
  },

  testWarehouse: (config: WarehouseConnectorConfig): Promise<{ data: { success: boolean; message?: string } }> =>
    api.post('/api/v1/warehouse/test', config),

  testWebUnblocker: (apiKey: string, zone: string): Promise<{ data: { success: boolean; message?: string } }> =>
    api.post('/api/v1/web-unblocker/test', { apiKey, zone }),

  getCachedPatterns: (domain?: string): Promise<{ data: { success: boolean; patterns: any[]; domains: string[]; total_patterns: number } }> =>
    api.get('/api/v1/cache/patterns', { params: domain ? { domain } : {} }),
};

// Account endpoints
export const accountApi = {
  getSettings: (): Promise<{ data: any }> =>
    api.get('/api/v1/account/settings'),

  updateSettings: (settings: any): Promise<{ data: any }> =>
    api.put('/api/v1/account/settings', settings),

  getUsage: (): Promise<{ data: any }> =>
    api.get('/api/v1/account/usage'),
};

// Pattern cache endpoints (multi-tenant with public/private visibility)
export const patternApi = {
  // List current user's patterns
  listMyPatterns: (domain?: string, visibility?: PatternVisibility): Promise<{ data: { success: boolean; patterns: CachedPattern[]; total: number } }> =>
    api.get('/api/v1/patterns/mine', { params: { domain, visibility } }),

  // List all public patterns
  listPublicPatterns: (domain?: string, limit?: number): Promise<{ data: { success: boolean; patterns: CachedPattern[]; total: number } }> =>
    api.get('/api/v1/patterns/public', { params: { domain, limit } }),

  // Store a new pattern
  storePattern: (data: { domain: string; fields: string[]; pattern_data: any; visibility: PatternVisibility; url?: string }): Promise<{ data: { success: boolean; message: string } }> =>
    api.post('/api/v1/patterns/store', data),

  // Update pattern visibility
  updateVisibility: (domain: string, fields: string[], visibility: PatternVisibility): Promise<{ data: { success: boolean; message: string } }> =>
    api.put('/api/v1/patterns/visibility', { visibility }, { params: { domain, fields: fields.join(',') } }),

  // Copy a public pattern to your cache
  copyPublicPattern: (domain: string, fields: string[]): Promise<{ data: { success: boolean; message: string } }> =>
    api.post('/api/v1/patterns/copy-public', null, { params: { domain, fields: fields.join(',') } }),

  // Delete a pattern
  deletePattern: (domain: string, fields: string[]): Promise<{ data: { success: boolean; message: string } }> =>
    api.delete('/api/v1/patterns', {
      params: {
        domain,
        fields: fields.join(',')
      }
    }),

  // Get pattern stats
  getStats: (): Promise<{ data: { success: boolean; total_patterns: number; private_patterns: number; public_patterns: number; domains: string[] } }> =>
    api.get('/api/v1/patterns/stats'),

  // Generate Python code for a pattern
  generatePythonCode: (data: { url: string; fields: string[]; selectors: any; target?: string }): Promise<{ data: { success: boolean; code: string; filename: string } }> =>
    api.post('/api/v1/generate-python-code', data),
};

// Agent (Job) endpoints
export const agentApi = {
  // Create a new agent
  createAgent: (request: { type: string; config: any; queue_immediately?: boolean }): Promise<{ data: { success: boolean; agent: Agent } }> =>
    api.post('/api/v1/agents', request),

  create: (type: string, config: any, queueImmediately: boolean = true): Promise<{ data: { success: boolean; agent: Agent } }> =>
    api.post('/api/v1/agents', { type, config, queue_immediately: queueImmediately }),

  // List agents
  list: (status?: string, type?: string, limit?: number): Promise<{ data: { success: boolean; agents: Agent[]; total: number } }> =>
    api.get('/api/v1/agents', { params: { status, type, limit } }),

  // Get agent details
  get: (agentId: string): Promise<{ data: { success: boolean; agent: Agent } }> =>
    api.get(`/api/v1/agents/${agentId}`),

  // Cancel an agent
  cancel: (agentId: string): Promise<{ data: { success: boolean; message: string } }> =>
    api.post(`/api/v1/agents/${agentId}/cancel`),

  // Get agent stats
  getStats: (): Promise<{ data: { success: boolean; total_agents: number; by_status: Record<string, number>; by_type: Record<string, number>; scheduled_count: number } }> =>
    api.get('/api/v1/agents/stats'),

  // Schedule an agent
  scheduleAgent: (agentId: string, request: { schedule: string; timezone?: string }): Promise<{ data: { success: boolean; message: string } }> =>
    api.post(`/api/v1/agents/${agentId}/schedule`, request),

  schedule: (agentId: string, schedule: string, timezone: string = 'UTC'): Promise<{ data: { success: boolean; message: string } }> =>
    api.post(`/api/v1/agents/${agentId}/schedule`, { schedule, timezone }),

  // Unschedule an agent
  unschedule: (agentId: string): Promise<{ data: { success: boolean; message: string } }> =>
    api.delete(`/api/v1/agents/${agentId}/schedule`),

  // Create agent from cache
  createFromCache: (domain: string, fields: string[], url: string, visibility: string = 'private', schedule?: string): Promise<{ data: { success: boolean; agent: Agent; message: string } }> =>
    api.post('/api/v1/agents/from-cache', { domain, fields, url, visibility, schedule }),
};
