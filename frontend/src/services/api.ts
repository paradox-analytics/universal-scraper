import axios from 'axios';
import type { ScrapeRequest, ScrapeResponse, ProxyConfiguration, WarehouseConnectorConfig, CacheStatus } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token interceptor (when auth is implemented)
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('api_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized - redirect to login
      localStorage.removeItem('api_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Scraping endpoints
export const scrapingApi = {
  scrape: (data: ScrapeRequest): Promise<{ data: ScrapeResponse }> =>
    api.post('/api/v1/web-scraping/scrape', data),
  
  getJob: (jobId: string): Promise<{ data: ScrapeResponse }> =>
    api.get(`/api/v1/web-scraping/jobs/${jobId}`),
  
  getResults: (jobId: string): Promise<{ data: any }> =>
    api.get(`/api/v1/web-scraping/jobs/${jobId}/results`),
  
  preview: (url: string): Promise<{ data: { html: string; screenshot?: string } }> =>
    api.get(`/api/v1/web-scraping/preview`, { params: { url } }),
  
  checkCache: (url: string): Promise<{ data: CacheStatus }> =>
    api.get(`/api/v1/cache/status`, { params: { url } }),
};

// Document processing endpoints
export const documentApi = {
  extract: (data: FormData): Promise<{ data: ScrapeResponse }> =>
    api.post('/api/v1/document-processing/extract', data, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  
  getJob: (jobId: string): Promise<{ data: ScrapeResponse }> =>
    api.get(`/api/v1/document-processing/jobs/${jobId}`),
};

// Configuration endpoints
export const configApi = {
  testProxy: (config: ProxyConfiguration): Promise<{ data: { success: boolean; message?: string } }> =>
    api.post('/api/v1/proxy/test', config),
  
  testWarehouse: (config: WarehouseConnectorConfig): Promise<{ data: { success: boolean; message?: string } }> =>
    api.post('/api/v1/warehouse/test', config),
  
  testWebUnblocker: (apiKey: string, zone: string): Promise<{ data: { success: boolean; message?: string } }> =>
    api.post('/api/v1/web-unblocker/test', { apiKey, zone }),
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

