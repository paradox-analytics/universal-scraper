import { useQuery } from '@tanstack/react-query';
import { scrapingApi } from '../../services/api';
import { CheckCircleIcon, ClockIcon } from '@heroicons/react/24/outline';

interface CacheIndicatorProps {
  url: string;
  fields?: string[];
}

export function CacheIndicator({ url, fields }: CacheIndicatorProps) {
  const { data: cacheStatus, isLoading, error } = useQuery({
    queryKey: ['cache-status', url, fields?.join(',')],
    queryFn: () => scrapingApi.checkCache(url, fields),
    enabled: !!url,
    refetchInterval: 30000, // Refetch every 30 seconds
    retry: 1, // Only retry once
    staleTime: 10000, // Consider data stale after 10 seconds
  });

  if (!url) {
    return null;
  }

  // Show loading state only briefly, then show error or cached status
  if (isLoading && !cacheStatus) {
    return (
      <div className="flex items-center gap-2 text-yellow-400 bg-yellow-900/20 border border-yellow-700 px-3 py-2 rounded-lg">
        <ClockIcon className="h-5 w-5 animate-spin" />
        <span className="text-sm font-medium">Checking cache...</span>
      </div>
    );
  }

  // If error, don't show anything (let main component handle it)
  if (error) {
    return null;
  }

  const isCached = cacheStatus?.data?.is_cached || false;
  const cacheAge = cacheStatus?.data?.cache_age;

  if (isCached) {
    return (
      <div className="flex items-center gap-2 text-green-400 bg-green-900/20 border border-green-700 px-3 py-2 rounded-lg">
        <CheckCircleIcon className="h-5 w-5" />
        <span className="text-sm font-medium">
          Cached - Instant results
          {cacheAge && (
            <span className="text-green-300 ml-1">
              ({Math.round(cacheAge / 60)}m ago)
            </span>
          )}
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-yellow-400 bg-yellow-900/20 border border-yellow-700 px-3 py-2 rounded-lg">
      <ClockIcon className="h-5 w-5" />
      <span className="text-sm font-medium">Not cached - Processing...</span>
    </div>
  );
}

