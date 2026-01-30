import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CircleStackIcon, ShareIcon, TrashIcon, MagnifyingGlassIcon, ArrowDownTrayIcon, GlobeAltIcon, LockClosedIcon, EyeIcon, DocumentDuplicateIcon, BoltIcon, PlayIcon } from '@heroicons/react/24/outline';
import { patternApi, configApi, agentApi } from '../services/api';
import type { CachedPattern, PatternVisibility } from '../types';

type TabType = 'my-patterns' | 'public-patterns';

export default function Cache() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<TabType>('my-patterns');
  const [myPatterns, setMyPatterns] = useState<CachedPattern[]>([]);
  const [publicPatterns, setPublicPatterns] = useState<CachedPattern[]>([]);
  const [domains, setDomains] = useState<string[]>([]);
  const [selectedDomain, setSelectedDomain] = useState<string>('');
  const [visibilityFilter, setVisibilityFilter] = useState<PatternVisibility | ''>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [stats, setStats] = useState<{ total_patterns: number; private_patterns: number; public_patterns: number }>({ total_patterns: 0, private_patterns: 0, public_patterns: 0 });
  
  // Deploy as Agent modal state
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [deployPattern, setDeployPattern] = useState<CachedPattern | null>(null);
  const [deployUrl, setDeployUrl] = useState('');
  const [deploySchedule, setDeploySchedule] = useState('');
  const [deploying, setDeploying] = useState(false);

  useEffect(() => {
    loadData();
  }, [selectedDomain, visibilityFilter, activeTab]);

  const loadData = async () => {
    setLoading(true);
    setError('');
    try {
      if (activeTab === 'my-patterns') {
        const response = await patternApi.listMyPatterns(selectedDomain || undefined, visibilityFilter || undefined);
        if (response.data.success) {
          setMyPatterns(response.data.patterns || []);
        }
        // Also get stats
        const statsResponse = await patternApi.getStats();
        if (statsResponse.data.success) {
          setStats({
            total_patterns: statsResponse.data.total_patterns || 0,
            private_patterns: statsResponse.data.private_patterns || 0,
            public_patterns: statsResponse.data.public_patterns || 0,
          });
          setDomains(statsResponse.data.domains || []);
        }
      } else {
        const response = await patternApi.listPublicPatterns(selectedDomain || undefined, 100);
        if (response.data.success) {
          setPublicPatterns(response.data.patterns || []);
        }
      }
      
      // Also load legacy cache patterns for backwards compatibility
      try {
        const legacyResponse = await configApi.getCachedPatterns(selectedDomain || undefined);
        if (legacyResponse.data.success && legacyResponse.data.patterns?.length > 0) {
          // Merge legacy patterns (they don't have visibility, so treat as private)
          const legacyPatterns = legacyResponse.data.patterns.map((p: any) => ({
            ...p,
            visibility: 'private' as PatternVisibility,
            tenant_id: 'legacy',
            fields_hash: '',
            pattern_data: {},
            updated_at: p.created_at || 0,
            usage_count: 0,
          }));
          
          if (activeTab === 'my-patterns') {
            setMyPatterns(prev => [...prev, ...legacyPatterns.filter((lp: CachedPattern) => !prev.some(p => p.domain === lp.domain))]);
          }
          
          // Update domains list
          const allDomains = new Set([...domains, ...legacyResponse.data.domains || []]);
          setDomains(Array.from(allDomains));
        }
      } catch (legacyErr) {
        console.warn('Legacy cache load failed:', legacyErr);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load cached patterns');
      console.error('Cache load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleShare = async (pattern: CachedPattern) => {
    const shareData = {
      domain: pattern.domain,
      fields: pattern.fields,
      visibility: pattern.visibility,
      url: pattern.url,
    };
    navigator.clipboard.writeText(JSON.stringify(shareData, null, 2));
    alert('Cache pattern copied to clipboard! Share this with others to reduce parsing costs.');
  };

  const handleMakePublic = async (pattern: CachedPattern) => {
    if (!confirm('Make this pattern public? Other users will be able to use it.')) {
      return;
    }
    try {
      const response = await patternApi.updateVisibility(pattern.domain, pattern.fields, 'public');
      if (response.data.success) {
        alert('Pattern is now public!');
        loadData();
      } else {
        alert(response.data.message || 'Failed to update visibility');
      }
    } catch (err: any) {
      alert(`Failed to update visibility: ${err.message}`);
    }
  };

  const handleMakePrivate = async (pattern: CachedPattern) => {
    if (!confirm('Make this pattern private? It will no longer be visible to other users.')) {
      return;
    }
    try {
      const response = await patternApi.updateVisibility(pattern.domain, pattern.fields, 'private');
      if (response.data.success) {
        alert('Pattern is now private.');
        loadData();
      } else {
        alert(response.data.message || 'Failed to update visibility');
      }
    } catch (err: any) {
      alert(`Failed to update visibility: ${err.message}`);
    }
  };

  const handleCopyPublic = async (pattern: CachedPattern) => {
    try {
      const response = await patternApi.copyPublicPattern(pattern.domain, pattern.fields);
      if (response.data.success) {
        alert('Pattern copied to your cache!');
        setActiveTab('my-patterns');
        loadData();
      } else {
        alert(response.data.message || 'Failed to copy pattern');
      }
    } catch (err: any) {
      alert(`Failed to copy pattern: ${err.message}`);
    }
  };

  const handleExportAll = async () => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL || 'https://universal-scraper-api-968720932091.us-central1.run.app'}/api/v1/cache/export${selectedDomain ? `?domain=${encodeURIComponent(selectedDomain)}` : ''}`,
        {
          headers: {
            'X-API-Key': localStorage.getItem('api_key') || '',
          },
        }
      );
      
      if (!response.ok) {
        throw new Error('Export failed');
      }
      
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `paradocs-cache-export-${selectedDomain || 'all'}-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      
      alert('Cache exported successfully!');
    } catch (err: any) {
      alert(`Export failed: ${err.message}`);
      console.error('Export error:', err);
    }
  };

  const handleDelete = async (pattern: CachedPattern) => {
    if (!confirm('Are you sure you want to delete this cached pattern?')) {
      return;
    }
    try {
      console.log('Deleting pattern:', {
        domain: pattern.domain,
        fields: pattern.fields
      });
      
      if (!pattern.domain || !pattern.fields || pattern.fields.length === 0) {
        alert('Invalid pattern data: missing domain or fields');
        return;
      }
      
      const response = await patternApi.deletePattern(pattern.domain, pattern.fields);
      console.log('Delete response:', response);
      
      if (response.data.success) {
        alert('Pattern deleted successfully.');
        loadData();
      } else {
        alert(response.data.message || 'Failed to delete pattern');
      }
    } catch (err: any) {
      console.error('Delete error:', err);
      console.error('Error response:', err.response?.data);
      const errorMsg = err.response?.data?.detail || err.response?.data?.message || err.message || 'Unknown error';
      alert(`Failed to delete pattern: ${errorMsg}`);
    }
  };

  const handleDeployAsAgent = (pattern: CachedPattern) => {
    setDeployPattern(pattern);
    setDeployUrl(pattern.url || `https://${pattern.domain}/`);
    setDeploySchedule('');
    setShowDeployModal(true);
  };

  const handleDeploySubmit = async () => {
    if (!deployPattern || !deployUrl) {
      alert('Please enter a URL');
      return;
    }
    
    setDeploying(true);
    try {
      const response = await agentApi.createFromCache(
        deployPattern.domain,
        deployPattern.fields,
        deployUrl,
        deployPattern.visibility,
        deploySchedule || undefined
      );
      
      if (response.data.success) {
        setShowDeployModal(false);
        setDeployPattern(null);
        alert(response.data.message || 'Agent created successfully!');
        navigate('/agents');
      } else {
        alert('Failed to create agent');
      }
    } catch (err: any) {
      alert(`Failed to create agent: ${err.message}`);
    } finally {
      setDeploying(false);
    }
  };

  const patterns = activeTab === 'my-patterns' ? myPatterns : publicPatterns;

  const filteredPatterns = patterns.filter((pattern) => {
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      return (
        pattern.domain?.toLowerCase().includes(searchLower) ||
        pattern.fields?.some((f) => f.toLowerCase().includes(searchLower)) ||
        pattern.url?.toLowerCase().includes(searchLower)
      );
    }
    return true;
  });

  const formatDate = (timestamp: number) => {
    if (!timestamp) return 'Unknown';
    return new Date(timestamp * 1000).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 bg-purple-900/20 rounded-2xl border border-purple-700">
            <CircleStackIcon className="h-7 w-7 text-purple-400" />
          </div>
          <div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-50 mb-2">Cache Management</h1>
            <p className="text-lg md:text-xl text-gray-300">
              View and manage cached extraction patterns. Share patterns to reduce parsing costs.
            </p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActiveTab('my-patterns')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'my-patterns'
              ? 'bg-purple-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          <LockClosedIcon className="h-5 w-5" />
          My Patterns
        </button>
        <button
          onClick={() => setActiveTab('public-patterns')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'public-patterns'
              ? 'bg-purple-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          <GlobeAltIcon className="h-5 w-5" />
          Public Patterns
        </button>
      </div>

      {/* Filters */}
      <div className="card mb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="label">Search</label>
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-500" />
              <input
                type="text"
                className="input-field pl-10"
                placeholder="Search by domain, fields, or URL..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>
          <div>
            <label className="label">Filter by Domain</label>
            <select
              className="select-field"
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
            >
              <option value="">All Domains</option>
              {domains.map((domain) => (
                <option key={domain} value={domain}>
                  {domain}
                </option>
              ))}
            </select>
          </div>
          {activeTab === 'my-patterns' && (
            <div>
              <label className="label">Visibility</label>
              <select
                className="select-field"
                value={visibilityFilter}
                onChange={(e) => setVisibilityFilter(e.target.value as PatternVisibility | '')}
              >
                <option value="">All</option>
                <option value="private">Private</option>
                <option value="public">Public</option>
              </select>
            </div>
          )}
          <div className="flex items-end gap-2">
            <button onClick={loadData} className="btn-secondary flex-1">
              Refresh
            </button>
            {activeTab === 'my-patterns' && (
              <button onClick={handleExportAll} className="btn-secondary flex items-center justify-center gap-2">
                <ArrowDownTrayIcon className="h-5 w-5" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <div className="card">
          <div className="text-sm font-medium text-gray-400 mb-1">Total Patterns</div>
          <div className="text-3xl font-bold text-gray-50">{stats.total_patterns}</div>
        </div>
        <div className="card">
          <div className="text-sm font-medium text-gray-400 mb-1">Private</div>
          <div className="text-3xl font-bold text-blue-400 flex items-center gap-2">
            <LockClosedIcon className="h-6 w-6" />
            {stats.private_patterns}
          </div>
        </div>
        <div className="card">
          <div className="text-sm font-medium text-gray-400 mb-1">Public (Shared)</div>
          <div className="text-3xl font-bold text-green-400 flex items-center gap-2">
            <GlobeAltIcon className="h-6 w-6" />
            {stats.public_patterns}
          </div>
        </div>
        <div className="card">
          <div className="text-sm font-medium text-gray-400 mb-1">Unique Domains</div>
          <div className="text-3xl font-bold text-purple-400">{domains.length}</div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="card bg-red-900/20 border-red-700 mb-6">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="card text-center py-16">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-4 border-purple-600"></div>
          <p className="mt-6 text-lg text-gray-400">Loading cached patterns...</p>
        </div>
      )}

      {/* Patterns List */}
      {!loading && filteredPatterns.length === 0 && (
        <div className="card text-center py-16">
          <CircleStackIcon className="mx-auto h-12 w-12 text-gray-600 mb-4" />
          <p className="text-lg text-gray-400">
            {activeTab === 'my-patterns' ? 'No cached patterns found' : 'No public patterns available'}
          </p>
          <p className="text-sm text-gray-500 mt-2">
            {activeTab === 'my-patterns'
              ? 'Start scraping pages to build your cache library.'
              : 'Be the first to share a pattern with the community!'}
          </p>
        </div>
      )}

      {!loading && filteredPatterns.length > 0 && (
        <div className="space-y-4">
          {filteredPatterns.map((pattern, index) => (
            <div key={`${pattern.domain}-${pattern.fields_hash || index}`} className="card hover:border-purple-600 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-3">
                    <h3 className="text-xl font-semibold text-gray-50">{pattern.domain || 'Unknown Domain'}</h3>
                    <span
                      className={`px-2 py-1 rounded-lg text-xs font-medium flex items-center gap-1 ${
                        pattern.visibility === 'public'
                          ? 'bg-green-900/20 text-green-300 border border-green-700'
                          : 'bg-blue-900/20 text-blue-300 border border-blue-700'
                      }`}
                    >
                      {pattern.visibility === 'public' ? (
                        <>
                          <GlobeAltIcon className="h-3 w-3" /> Public
                        </>
                      ) : (
                        <>
                          <LockClosedIcon className="h-3 w-3" /> Private
                        </>
                      )}
                    </span>
                    {pattern.usage_count > 0 && (
                      <span className="px-2 py-1 rounded-lg text-xs font-medium bg-gray-700 text-gray-300">
                        Used {pattern.usage_count}x
                      </span>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div>
                      <div className="text-xs font-medium text-gray-500 mb-1">Fields</div>
                      <div className="flex flex-wrap gap-2">
                        {pattern.fields?.map((field) => (
                          <span
                            key={field}
                            className="px-2 py-1 bg-gray-800 text-gray-300 rounded-md text-xs border border-gray-700"
                          >
                            {field}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-medium text-gray-500 mb-1">Metadata</div>
                      <div className="text-sm text-gray-400 space-y-1">
                        <div>Created: {formatDate(pattern.created_at)}</div>
                        <div className="truncate">URL: {pattern.url || 'N/A'}</div>
                        {pattern.shared_from && (
                          <div className="text-xs text-purple-400">
                            Copied from community
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="flex gap-2 ml-4">
                  {/* Deploy as Agent button - available for all patterns */}
                  <button
                    onClick={() => handleDeployAsAgent(pattern)}
                    className="p-2 text-gray-400 hover:text-yellow-400 hover:bg-yellow-900/30 rounded-lg transition-colors"
                    title="Deploy as Agent"
                  >
                    <BoltIcon className="h-5 w-5" />
                  </button>
                  
                  {activeTab === 'my-patterns' ? (
                    <>
                      {pattern.visibility === 'private' ? (
                        <button
                          onClick={() => handleMakePublic(pattern)}
                          className="p-2 text-gray-400 hover:text-green-400 hover:bg-green-900/30 rounded-lg transition-colors"
                          title="Make public"
                        >
                          <GlobeAltIcon className="h-5 w-5" />
                        </button>
                      ) : (
                        <button
                          onClick={() => handleMakePrivate(pattern)}
                          className="p-2 text-gray-400 hover:text-blue-400 hover:bg-blue-900/30 rounded-lg transition-colors"
                          title="Make private"
                        >
                          <LockClosedIcon className="h-5 w-5" />
                        </button>
                      )}
                      <button
                        onClick={() => handleShare(pattern)}
                        className="p-2 text-gray-400 hover:text-purple-400 hover:bg-purple-900/30 rounded-lg transition-colors"
                        title="Copy pattern data"
                      >
                        <ShareIcon className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => handleDelete(pattern)}
                        className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-900/30 rounded-lg transition-colors"
                        title="Delete pattern"
                      >
                        <TrashIcon className="h-5 w-5" />
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => handleCopyPublic(pattern)}
                        className="p-2 text-gray-400 hover:text-purple-400 hover:bg-purple-900/30 rounded-lg transition-colors"
                        title="Copy to my cache"
                      >
                        <DocumentDuplicateIcon className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => handleShare(pattern)}
                        className="p-2 text-gray-400 hover:text-purple-400 hover:bg-purple-900/30 rounded-lg transition-colors"
                        title="View pattern data"
                      >
                        <EyeIcon className="h-5 w-5" />
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Deploy as Agent Modal */}
      {showDeployModal && deployPattern && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900 rounded-xl border border-gray-700 p-6 max-w-lg w-full mx-4">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-yellow-900/20 rounded-lg border border-yellow-700">
                <BoltIcon className="h-6 w-6 text-yellow-400" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-50">Deploy as Agent</h3>
                <p className="text-sm text-gray-400">Create an agent from this cached pattern</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="label">Domain</label>
                <input
                  type="text"
                  className="input-field bg-gray-800"
                  value={deployPattern.domain}
                  disabled
                />
              </div>
              
              <div>
                <label className="label">Fields</label>
                <div className="flex flex-wrap gap-2 p-3 bg-gray-800 rounded-lg border border-gray-700">
                  {deployPattern.fields?.map((field) => (
                    <span
                      key={field}
                      className="px-2 py-1 bg-gray-700 text-gray-300 rounded-md text-xs"
                    >
                      {field}
                    </span>
                  ))}
                </div>
              </div>
              
              <div>
                <label className="label">URL to Scrape</label>
                <input
                  type="url"
                  className="input-field"
                  placeholder={`https://${deployPattern.domain}/...`}
                  value={deployUrl}
                  onChange={(e) => setDeployUrl(e.target.value)}
                />
              </div>
              
              <div>
                <label className="label">Schedule (Optional)</label>
                <select
                  className="select-field"
                  value={deploySchedule}
                  onChange={(e) => setDeploySchedule(e.target.value)}
                >
                  <option value="">Run Once (No Schedule)</option>
                  <option value="0 * * * *">Every Hour</option>
                  <option value="0 */6 * * *">Every 6 Hours</option>
                  <option value="0 0 * * *">Daily at Midnight</option>
                  <option value="0 0 * * 0">Weekly on Sunday</option>
                  <option value="0 0 1 * *">Monthly on the 1st</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  Scheduled agents use Google Cloud Scheduler
                </p>
              </div>
            </div>
            
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => {
                  setShowDeployModal(false);
                  setDeployPattern(null);
                }}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleDeploySubmit}
                disabled={deploying || !deployUrl}
                className="btn-primary flex items-center gap-2"
              >
                {deploying ? (
                  <>
                    <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                    Deploying...
                  </>
                ) : (
                  <>
                    <PlayIcon className="h-5 w-5" />
                    Deploy Agent
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
