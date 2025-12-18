import { useState } from 'react';

interface UrlInputProps {
  onScrape: (url: string) => void;
  initialUrl?: string;
}

export function UrlInput({ onScrape, initialUrl = '' }: UrlInputProps) {
  const [url, setUrl] = useState(initialUrl);
  const [error, setError] = useState<string>('');

  const validateUrl = (urlString: string): boolean => {
    try {
      const url = new URL(urlString);
      return url.protocol === 'http:' || url.protocol === 'https:';
    } catch {
      return false;
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    if (!validateUrl(url)) {
      setError('Please enter a valid URL (must start with http:// or https://)');
      return;
    }

    onScrape(url);
  };

  return (
    <div>
      <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-2">
        URL to Scrape
      </label>
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          id="url"
          type="text"
          value={url}
          onChange={(e) => {
            setUrl(e.target.value);
            setError('');
          }}
          placeholder="https://example.com/products"
          className={`input-field flex-1 ${error ? 'border-red-500' : ''}`}
        />
        <button type="submit" className="btn-primary whitespace-nowrap">
          Scrape
        </button>
      </form>
      {error && (
        <p className="mt-1 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
}

