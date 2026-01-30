import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { agentApi } from '../services/api';
import {
  ChartBarIcon,
  CheckCircleIcon,
  XCircleIcon,
  SparklesIcon,
  DocumentTextIcon,
  ArrowRightIcon,
  GlobeAltIcon,
  BoltIcon,
} from '@heroicons/react/24/outline';

export default function Dashboard() {
  const [stats, setStats] = useState({
    total_tasks: 0,
    active_tasks: 0,
    success_rate: 100,
    most_used_template: 'None',
    by_status: {} as Record<string, number>,
    by_type: {} as Record<string, number>,
  });
  const [loading, setLoading] = useState(true);
  const [dateRange, setDateRange] = useState('Last 7 days');

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const response = await agentApi.getStats();
      if (response.data && response.data.success) {
        const totalTasks = response.data.total_agents || 0;
        const completed = response.data.by_status?.completed || 0;
        const failed = response.data.by_status?.failed || 0;
        const successRate = totalTasks > 0 ? Math.round((completed / (completed + failed)) * 100) : 100;
        
        // Determine most used template
        const byType = response.data.by_type || {};
        const mostUsed = Object.entries(byType).sort((a, b) => (b[1] as number) - (a[1] as number))[0];
        const mostUsedTemplate = mostUsed ? mostUsed[0].replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'None';
        
        setStats({
          total_tasks: totalTasks,
          active_tasks: (response.data.by_status?.running || 0) + (response.data.by_status?.queued || 0),
          success_rate: successRate,
          most_used_template: mostUsedTemplate,
          by_status: response.data.by_status || {},
          by_type: response.data.by_type || {},
        });
      }
    } catch (err) {
      console.error('Failed to load stats:', err);
    } finally {
      setLoading(false);
    }
  };

  const completed = stats.by_status.completed || 0;
  const failed = stats.by_status.failed || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Welcome to ParaDocs</h1>
        <p className="text-gray-400">Monitor your scraping and document processing activity</p>
      </div>

      {/* This Week's Statistics */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-900/20 rounded-lg border border-purple-700">
              <ChartBarIcon className="h-5 w-5 text-purple-400" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">This Week's Statistics</h2>
              <p className="text-sm text-gray-400">
                {new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toLocaleDateString()} - {new Date().toLocaleDateString()}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
              className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:border-purple-500"
            >
              <option value="Last 7 days">Last 7 days</option>
              <option value="Last 30 days">Last 30 days</option>
              <option value="Last 90 days">Last 90 days</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-400">Total Agents</span>
              <ChartBarIcon className="h-4 w-4 text-gray-500" />
            </div>
            <p className="text-3xl font-bold text-white">{loading ? '...' : stats.total_tasks.toLocaleString()}</p>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-400">Active Agents</span>
              <BoltIcon className="h-4 w-4 text-yellow-500" />
            </div>
            <p className="text-3xl font-bold text-white">{loading ? '...' : stats.active_tasks.toLocaleString()}</p>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-400">Success Rate</span>
              <CheckCircleIcon className="h-4 w-4 text-green-500" />
            </div>
            <p className={`text-3xl font-bold ${stats.success_rate >= 90 ? 'text-green-400' : stats.success_rate >= 70 ? 'text-yellow-400' : 'text-red-400'}`}>
              {loading ? '...' : `${stats.success_rate}%`}
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-400">Most Used Pattern</span>
              <SparklesIcon className="h-4 w-4 text-purple-500" />
            </div>
            <p className="text-lg font-semibold text-white truncate">{loading ? '...' : stats.most_used_template}</p>
          </div>
        </div>
      </div>

      {/* Task Activity */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-900/20 rounded-lg border border-blue-700">
              <ChartBarIcon className="h-5 w-5 text-blue-400" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">Agent Activity</h2>
              <p className="text-sm text-gray-400">Performance overview for the last 7 days</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1.5 bg-purple-600 text-white rounded-lg text-sm font-medium">
              Line
            </button>
            <button className="px-3 py-1.5 bg-gray-700 text-gray-300 rounded-lg text-sm font-medium hover:bg-gray-600">
              Bar
            </button>
            <button className="px-3 py-1.5 bg-gray-700 text-gray-300 rounded-lg text-sm font-medium hover:bg-gray-600">
              Area
            </button>
          </div>
        </div>

        {/* Chart Placeholder */}
        <div className="bg-gray-900 rounded-lg border border-gray-700 p-8 mb-6" style={{ minHeight: '300px' }}>
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <ChartBarIcon className="h-12 w-12 mx-auto mb-3 text-gray-600" />
              <p className="text-sm">Chart visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center gap-2 mb-2">
              <ChartBarIcon className="h-4 w-4 text-gray-400" />
              <span className="text-sm font-medium text-gray-400">Total Agents</span>
            </div>
            <p className="text-2xl font-bold text-white">{loading ? '...' : stats.total_tasks.toLocaleString()}</p>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircleIcon className="h-4 w-4 text-green-400" />
              <span className="text-sm font-medium text-gray-400">Completed</span>
            </div>
            <p className="text-2xl font-bold text-green-400">{loading ? '...' : completed.toLocaleString()}</p>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center gap-2 mb-2">
              <XCircleIcon className="h-4 w-4 text-red-400" />
              <span className="text-sm font-medium text-gray-400">Failed</span>
            </div>
            <p className="text-2xl font-bold text-red-400">{loading ? '...' : failed.toLocaleString()}</p>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <div className="flex items-center gap-2 mb-2">
              <ChartBarIcon className="h-4 w-4 text-purple-400" />
              <span className="text-sm font-medium text-gray-400">Success Rate</span>
            </div>
            <p className={`text-2xl font-bold ${stats.success_rate >= 90 ? 'text-green-400' : stats.success_rate >= 70 ? 'text-yellow-400' : 'text-red-400'}`}>
              {loading ? '...' : `${stats.success_rate}%`}
            </p>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link
          to="/web-scraping"
          className="bg-gray-800 rounded-lg border border-gray-700 p-6 hover:border-purple-500 transition-colors group"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-purple-900/20 rounded-lg border border-purple-700">
              <GlobeAltIcon className="h-6 w-6 text-purple-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-white mb-1 group-hover:text-purple-400 transition-colors">
                Scraper
              </h3>
              <p className="text-sm text-gray-400">Extract data from any webpage</p>
            </div>
            <ArrowRightIcon className="h-5 w-5 text-gray-400 group-hover:text-purple-400 group-hover:translate-x-1 transition-all" />
          </div>
        </Link>

        <Link
          to="/document-processing"
          className="bg-gray-800 rounded-lg border border-gray-700 p-6 hover:border-purple-500 transition-colors group"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 bg-purple-900/20 rounded-lg border border-purple-700">
              <DocumentTextIcon className="h-6 w-6 text-purple-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-white mb-1 group-hover:text-purple-400 transition-colors">
                Document Processor
              </h3>
              <p className="text-sm text-gray-400">Process PDFs, Word docs, and more</p>
            </div>
            <ArrowRightIcon className="h-5 w-5 text-gray-400 group-hover:text-purple-400 group-hover:translate-x-1 transition-all" />
          </div>
        </Link>
      </div>
    </div>
  );
}
