// API Configuration
// Backend API URL - Cloud Run service
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 
  'https://universal-scraper-api-r3crozpq7q-uc.a.run.app';

export const API_ENDPOINTS = {
  health: `${API_BASE_URL}/health`,
  scrape: `${API_BASE_URL}/scrape`,
  crawl: `${API_BASE_URL}/crawl`,
} as const;

// API Key - should be stored securely (consider using Firebase Auth or environment variables)
export const getApiKey = (): string => {
  // Check localStorage first, then environment variable
  if (typeof window !== 'undefined') {
    const storedKey = localStorage.getItem('api_key');
    if (storedKey) {
      return storedKey;
    }
  }
  // Fallback to environment variable
  return import.meta.env.VITE_API_KEY || '';
};

// Function to set API key (for runtime updates)
export const setApiKey = (key: string): void => {
  if (typeof window !== 'undefined') {
    localStorage.setItem('api_key', key);
  }
};

