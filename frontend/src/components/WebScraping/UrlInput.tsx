import { useState } from 'react';

interface UrlInputProps {
  onScrape?: (url: string) => void;
  onChange?: (url: string) => void;
  initialUrl?: string;
}

export function UrlInput({ onScrape, onChange, initialUrl = '' }: UrlInputProps) {
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

    // Call onChange if provided, otherwise call onScrape
    if (onChange) {
      onChange(url);
    } else if (onScrape) {
      onScrape(url);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newUrl = e.target.value;
    setUrl(newUrl);
    setError('');
    if (onChange) {
      onChange(newUrl);
    }
  };

  return (
    <div>
      <label htmlFor="url" className="block text-sm font-medium text-gray-300 mb-2">
        URL to Scrape
      </label>
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          id="url"
          type="text"
          value={url}
          onChange={handleChange}
          placeholder="https://example.com/products"
          className={`input-field flex-1 ${error ? 'border-red-500' : ''}`}
        />
        {onScrape && (
          <button type="submit" className="btn-primary whitespace-nowrap">
            Scrape
          </button>
        )}
      </form>
      {error && (
        <p className="mt-1 text-sm text-red-400">{error}</p>
      )}
    </div>
  );
}

