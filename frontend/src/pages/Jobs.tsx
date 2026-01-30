import { useState } from 'react';
import { CodeViewer } from '../components/Common/CodeViewer';
import {
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

interface Job {
  id: string;
  type: 'web_scraping' | 'document_processing';
  url?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  result?: any;
  metadata?: any;
}

export default function Jobs() {
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [filter, setFilter] = useState<'all' | 'pending' | 'completed' | 'failed'>('all');

  // Mock jobs data - in production, fetch from API
  const mockJobs: Job[] = [
    {
      id: '1',
      type: 'web_scraping',
      url: 'https://example.com/products',
      status: 'completed',
      created_at: new Date(Date.now() - 3600000).toISOString(),
      completed_at: new Date(Date.now() - 3500000).toISOString(),
      result: { items: 10 },
    },
    {
      id: '2',
      type: 'document_processing',
      status: 'processing',
      created_at: new Date(Date.now() - 1800000).toISOString(),
    },
    {
      id: '3',
      type: 'web_scraping',
      url: 'https://example.com/page',
      status: 'failed',
      created_at: new Date(Date.now() - 7200000).toISOString(),
      metadata: { error: 'Connection timeout' },
    },
  ];

  const filteredJobs = mockJobs.filter((job) => {
    if (filter === 'all') return true;
    return job.status === filter;
  });

  const getStatusIcon = (status: Job['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'processing':
        return <ArrowPathIcon className="h-5 w-5 text-blue-500 animate-spin" />;
      default:
        return <ClockIcon className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getStatusBadge = (status: Job['status']) => {
    const baseClasses = 'px-2 py-1 text-xs font-medium rounded-full';
    switch (status) {
      case 'completed':
        return `${baseClasses} bg-green-100 text-green-800`;
      case 'failed':
        return `${baseClasses} bg-red-100 text-red-800`;
      case 'processing':
        return `${baseClasses} bg-blue-100 text-blue-800`;
      default:
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Jobs</h1>
        <div className="flex gap-2">
          {(['all', 'pending', 'completed', 'failed'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                filter === f
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Jobs List */}
        <div className="lg:col-span-2 space-y-4">
          {filteredJobs.length === 0 ? (
            <div className="card text-center py-12 text-gray-500">
              <p>No jobs found</p>
            </div>
          ) : (
            filteredJobs.map((job) => (
              <div
                key={job.id}
                className="card cursor-pointer hover:shadow-md transition-shadow"
                onClick={() => setSelectedJob(job)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    {getStatusIcon(job.status)}
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-gray-900">
                          {job.type === 'web_scraping' ? 'Web Scraping' : 'Document Processing'}
                        </span>
                        <span className={getStatusBadge(job.status)}>{job.status}</span>
                      </div>
                      {job.url && (
                        <p className="text-sm text-gray-500 mt-1 truncate max-w-md">{job.url}</p>
                      )}
                      <p className="text-xs text-gray-400 mt-1">
                        Created: {formatDate(job.created_at)}
                      </p>
                    </div>
                  </div>
                  {job.completed_at && (
                    <p className="text-xs text-gray-400">
                      {Math.round(
                        (new Date(job.completed_at).getTime() -
                          new Date(job.created_at).getTime()) /
                          1000
                      )}{' '}
                      seconds
                    </p>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Job Details */}
        <div className="lg:col-span-1">
          {selectedJob ? (
            <div className="card space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Job Details</h2>
                <button
                  onClick={() => setSelectedJob(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>

              <div className="space-y-2 text-sm">
                <div>
                  <span className="font-medium text-gray-700">ID:</span>{' '}
                  <span className="text-gray-900">{selectedJob.id}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Type:</span>{' '}
                  <span className="text-gray-900 capitalize">
                    {selectedJob.type.replace('_', ' ')}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Status:</span>{' '}
                  <span className={getStatusBadge(selectedJob.status)}>
                    {selectedJob.status}
                  </span>
                </div>
                {selectedJob.url && (
                  <div>
                    <span className="font-medium text-gray-700">URL:</span>{' '}
                    <span className="text-gray-900 break-all">{selectedJob.url}</span>
                  </div>
                )}
                <div>
                  <span className="font-medium text-gray-700">Created:</span>{' '}
                  <span className="text-gray-900">{formatDate(selectedJob.created_at)}</span>
                </div>
                {selectedJob.completed_at && (
                  <div>
                    <span className="font-medium text-gray-700">Completed:</span>{' '}
                    <span className="text-gray-900">{formatDate(selectedJob.completed_at)}</span>
                  </div>
                )}
              </div>

              {selectedJob.result && (
                <div>
                  <h3 className="font-medium text-gray-700 mb-2">Results</h3>
                  <CodeViewer data={selectedJob.result} language="json" />
                </div>
              )}

              {selectedJob.metadata && (
                <div>
                  <h3 className="font-medium text-gray-700 mb-2">Metadata</h3>
                  <CodeViewer data={selectedJob.metadata} language="json" />
                </div>
              )}
            </div>
          ) : (
            <div className="card text-center py-12 text-gray-500">
              <p>Select a job to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
