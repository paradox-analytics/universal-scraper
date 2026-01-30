import { useState, useEffect } from 'react';
import { ArrowPathIcon } from '@heroicons/react/24/outline';
import { agentApi } from '../../services/api';

export function ActiveJobIndicator() {
  const [activeCount, setActiveCount] = useState(0);
  const [loading, setLoading] = useState(true);

  const loadActiveJobs = async () => {
    try {
      const response = await agentApi.list('running', undefined, 100);
      if (response.data.success) {
        const active = response.data.agents.filter(
          (job: any) => job.status === 'pending' || job.status === 'queued' || job.status === 'running'
        );
        setActiveCount(active.length);
      }
    } catch (err) {
      console.error('Failed to load active jobs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadActiveJobs();
    const interval = setInterval(loadActiveJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading || activeCount === 0) return null;

  return (
    <a
      href="/agents?filter=active"
      className="inline-flex items-center gap-2 px-3 py-1.5 bg-blue-900/30 border border-blue-700 rounded-lg text-blue-300 hover:bg-blue-900/50 transition-colors text-sm font-medium"
    >
      <ArrowPathIcon className="h-4 w-4 animate-spin" />
      <span>{activeCount} active job{activeCount > 1 ? 's' : ''}</span>
    </a>
  );
}




