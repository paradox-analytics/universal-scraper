import { useState } from 'react';
import { PlusIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface BatchUrlInputProps {
  urls: string[];
  onUrlsChange: (urls: string[]) => void;
  onScrape: (urls: string[]) => void;
  loading?: boolean;
}

export function BatchUrlInput({ urls, onUrlsChange, onScrape, loading = false }: BatchUrlInputProps) {
  const [newUrl, setNewUrl] = useState('');

  const handleAddUrl = () => {
    const trimmedUrl = newUrl.trim();
    if (trimmedUrl && !urls.includes(trimmedUrl)) {
      // Basic URL validation
      try {
        new URL(trimmedUrl);
        onUrlsChange([...urls, trimmedUrl]);
        setNewUrl('');
      } catch {
        alert('Please enter a valid URL');
      }
    }
  };

  const handleRemoveUrl = (index: number) => {
    onUrlsChange(urls.filter((_, i) => i !== index));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddUrl();
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Enter URL to scrape (e.g., https://example.com)"
          value={newUrl}
          onChange={(e) => setNewUrl(e.target.value)}
          onKeyPress={handleKeyPress}
          className="input-field flex-1"
          disabled={loading}
        />
        <button
          onClick={handleAddUrl}
          disabled={loading || !newUrl.trim()}
          className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <PlusIcon className="h-5 w-5" />
          Add
        </button>
      </div>

      {urls.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-300">
              URLs to scrape ({urls.length})
            </label>
            <button
              onClick={() => onUrlsChange([])}
              className="text-xs text-gray-400 hover:text-gray-300"
              disabled={loading}
            >
              Clear all
            </button>
          </div>
          <div className="max-h-48 overflow-y-auto space-y-2 border border-gray-700 rounded-lg p-3 bg-gray-900">
            {urls.map((url, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors"
              >
                <span className="text-sm text-gray-300 truncate flex-1 mr-2">{url}</span>
                <button
                  onClick={() => handleRemoveUrl(index)}
                  className="text-gray-400 hover:text-gray-200 flex-shrink-0"
                  disabled={loading}
                >
                  <XMarkIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
          <button
            onClick={() => onScrape(urls)}
            disabled={loading || urls.length === 0}
            className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Processing...' : `Scrape ${urls.length} URL${urls.length > 1 ? 's' : ''}`}
          </button>
        </div>
      )}
    </div>
  );
}




