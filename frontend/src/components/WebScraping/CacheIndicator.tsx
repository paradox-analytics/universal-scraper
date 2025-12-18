import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { scrapingApi } from '../../services/api';
import { CheckCircleIcon, ClockIcon } from '@heroicons/react/24/outline';

interface CacheIndicatorProps {
  url: string;
}

export function CacheIndicator({ url }: CacheIndicatorProps) {
  const { data: cacheStatus, isLoading } = useQuery({
    queryKey: ['cache-status', url],
    queryFn: () => scrapingApi.checkCache(url),
    enabled: !!url,
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  if (!url || isLoading) {
    return null;
  }

  const isCached = cacheStatus?.data?.is_cached || false;
  const cacheAge = cacheStatus?.data?.cache_age;

  if (isCached) {
    return (
      <div className="flex items-center gap-2 text-green-600 bg-green-50 px-3 py-2 rounded-lg">
        <CheckCircleIcon className="h-5 w-5" />
        <span className="text-sm font-medium">
          Cached - Instant results
          {cacheAge && (
            <span className="text-green-500 ml-1">
              ({Math.round(cacheAge / 60)}m ago)
            </span>
          )}
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-yellow-600 bg-yellow-50 px-3 py-2 rounded-lg">
      <ClockIcon className="h-5 w-5" />
      <span className="text-sm font-medium">Not cached - Processing...</span>
    </div>
  );
}

