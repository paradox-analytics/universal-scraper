import { useState, useEffect } from 'react';
import { ClockIcon, CheckCircleIcon, XCircleIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import { agentApi } from '../../services/api';

// Generic Job interface for JobQueue - accepts any agent type
interface Job {
  id: string;
  type?: string;
  status: string;
  config?: any;
  definition?: any; // For new Agent type
  progress?: number;
  progress_message?: string;
  created_at?: string | number;
  completed_at?: string | number;
  error?: string;
}

interface JobQueueProps {
  onJobClick?: (job: Job) => void;
}

export function JobQueue({ onJobClick }: JobQueueProps) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'active' | 'recent'>('active');

  const loadJobs = async () => {
    try {
      setLoading(true);
      const response = await agentApi.list();
      if (response.data.success) {
        setJobs(response.data.agents || []);
      }
    } catch (err) {
      console.error('Failed to load jobs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadJobs();
    // Poll for updates every 5 seconds
    const interval = setInterval(loadJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const activeJobs = jobs.filter(
    (job) => job.status === 'pending' || job.status === 'queued' || job.status === 'running'
  );
  const recentJobs = jobs
    .filter((job) => job.status === 'completed' || job.status === 'failed')
    .sort((a, b) => {
      const aTime = typeof a.completed_at === 'number' ? a.completed_at : (a.completed_at ? new Date(a.completed_at).getTime() : 0);
      const bTime = typeof b.completed_at === 'number' ? b.completed_at : (b.completed_at ? new Date(b.completed_at).getTime() : 0);
      return bTime - aTime;
    })
    .slice(0, 20);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-400" />;
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-400" />;
      case 'running':
        return <ArrowPathIcon className="h-5 w-5 text-blue-400 animate-spin" />;
      default:
        return <ClockIcon className="h-5 w-5 text-yellow-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-400 bg-green-900/20 border-green-700';
      case 'failed':
        return 'text-red-400 bg-red-900/20 border-red-700';
      case 'running':
        return 'text-blue-400 bg-blue-900/20 border-blue-700';
      default:
        return 'text-yellow-400 bg-yellow-900/20 border-yellow-700';
    }
  };

  const formatTime = (timestamp?: string | number) => {
    if (!timestamp) return 'N/A';
    const date = typeof timestamp === 'number' ? new Date(timestamp * 1000) : new Date(timestamp);
    return date.toLocaleTimeString();
  };

  return (
    <div className="card-elevated">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-50">Job Queue</h2>
        <button
          onClick={loadJobs}
          disabled={loading}
          className="text-sm text-gray-400 hover:text-gray-300 disabled:opacity-50"
        >
          Refresh
        </button>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-700 mb-4">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('active')}
            className={`${
              activeTab === 'active'
                ? 'border-purple-500 text-purple-400'
                : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors flex items-center gap-2`}
          >
            Active ({activeJobs.length})
          </button>
          <button
            onClick={() => setActiveTab('recent')}
            className={`${
              activeTab === 'recent'
                ? 'border-purple-500 text-purple-400'
                : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors`}
          >
            Recent ({recentJobs.length})
          </button>
        </nav>
      </div>

      {/* Job List */}
      {loading ? (
        <div className="text-center py-8">
          <ArrowPathIcon className="h-8 w-8 text-gray-400 animate-spin mx-auto mb-2" />
          <p className="text-sm text-gray-400">Loading jobs...</p>
        </div>
      ) : activeTab === 'active' ? (
        activeJobs.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <ClockIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>No active jobs</p>
          </div>
        ) : (
          <div className="space-y-3">
            {activeJobs.map((job) => (
              <div
                key={job.id}
                onClick={() => onJobClick?.(job)}
                className={`p-4 rounded-lg border cursor-pointer hover:border-purple-500 transition-colors ${getStatusColor(
                  job.status
                )}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3 flex-1 min-w-0">
                    {getStatusIcon(job.status)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm capitalize">{(job.type || 'agent').replace('_', ' ')}</span>
                        <span className={`text-xs px-2 py-0.5 rounded border ${getStatusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </div>
                      {job.config?.url && (
                        <p className="text-xs text-gray-300 truncate">{job.config.url}</p>
                      )}
                      {job.progress_message && (
                        <p className="text-xs text-gray-400 mt-1">{job.progress_message}</p>
                      )}
                      {job.progress !== undefined && (
                        <div className="mt-2">
                          <div className="w-full bg-gray-700 rounded-full h-1.5">
                            <div
                              className="bg-purple-600 h-1.5 rounded-full transition-all"
                              style={{ width: `${job.progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-xs text-gray-400 ml-2">
                    {formatTime(job.created_at)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )
      ) : (
        recentJobs.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <CheckCircleIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p>No recent jobs</p>
          </div>
        ) : (
          <div className="space-y-3">
            {recentJobs.map((job) => (
              <div
                key={job.id}
                onClick={() => onJobClick?.(job)}
                className={`p-4 rounded-lg border cursor-pointer hover:border-purple-500 transition-colors ${getStatusColor(
                  job.status
                )}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3 flex-1 min-w-0">
                    {getStatusIcon(job.status)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm capitalize">{(job.type || 'agent').replace('_', ' ')}</span>
                        <span className={`text-xs px-2 py-0.5 rounded border ${getStatusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </div>
                      {job.config?.url && (
                        <p className="text-xs text-gray-300 truncate">{job.config.url}</p>
                      )}
                      {job.error && (
                        <p className="text-xs text-red-400 mt-1">{job.error}</p>
                      )}
                    </div>
                  </div>
                  <div className="text-xs text-gray-400 ml-2">
                    {formatTime(job.completed_at)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )
      )}
    </div>
  );
}


