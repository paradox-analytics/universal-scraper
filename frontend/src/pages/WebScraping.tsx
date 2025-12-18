import { useState } from 'react';
import { UrlInput } from '../components/WebScraping/UrlInput';
import { FieldSelector } from '../components/WebScraping/FieldSelector';
import { CacheIndicator } from '../components/WebScraping/CacheIndicator';
import { DataTable } from '../components/Common/DataTable';
import { CodeViewer } from '../components/Common/CodeViewer';
import { scrapingApi } from '../services/api';

export default function WebScraping() {
  const [url, setUrl] = useState('');
  const [fields, setFields] = useState<string[]>([]);
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const handleScrape = async (scrapeUrl: string) => {
    setUrl(scrapeUrl);
    setLoading(true);
    setError('');
    setResults([]);

    try {
      const response = await scrapingApi.scrape({
        url: scrapeUrl,
        fields: fields.length > 0 ? fields : ['title', 'description', 'url'],
        options: {},
      });

      if (response.data.status === 'completed' && response.data.data) {
        setResults(response.data.data);
      } else {
        setError('Scraping in progress... Check Jobs page for status.');
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to scrape URL');
      console.error('Scraping error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Web Scraping</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Configuration */}
        <div className="lg:col-span-1 space-y-6">
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Configuration</h2>
            
            <div className="space-y-4">
              <UrlInput onScrape={handleScrape} initialUrl={url} />
              
              <FieldSelector fields={fields} onChange={setFields} />
              
              {url && <CacheIndicator url={url} />}
            </div>
          </div>
          
          {/* Settings Panel - Collapsible */}
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Advanced Settings</h2>
            <p className="text-sm text-gray-500">
              Proxy, browser, pagination, and AI settings will go here.
            </p>
          </div>
        </div>
        
        {/* Right Column - Results */}
        <div className="lg:col-span-2 space-y-6">
          {loading && (
            <div className="card text-center py-12">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
              <p className="mt-4 text-gray-600">Scraping in progress...</p>
            </div>
          )}
          
          {error && (
            <div className="card bg-red-50 border border-red-200">
              <p className="text-red-800">{error}</p>
            </div>
          )}
          
          {results.length > 0 && (
            <>
              <div className="card">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">
                    Results ({results.length} items)
                  </h2>
                  <button className="btn-secondary text-sm">
                    Export CSV
                  </button>
                </div>
                <DataTable data={results} />
              </div>
              
              <div className="card">
                <h2 className="text-lg font-semibold mb-4">Raw Data</h2>
                <CodeViewer data={results} language="json" />
              </div>
            </>
          )}
          
          {!loading && !error && results.length === 0 && (
            <div className="card text-center py-12 text-gray-500">
              <p>Enter a URL and fields to start scraping</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

