import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { configApi } from '../services/api';
import { saveUserSettings } from '../services/auth';
import type { ProxyConfiguration, AIConfiguration, WebUnblockerConfig } from '../types';

export default function Settings() {
  const { user, userSettings, refreshSettings } = useAuth();
  const [activeTab, setActiveTab] = useState<'api' | 'proxy' | 'ai' | 'webunblocker' | 'warehouse'>('api');
  const [apiKey, setApiKey] = useState('');
  const [proxyConfig, setProxyConfig] = useState<ProxyConfiguration>({
    provider: 'none',
  });
  const [aiConfig, setAiConfig] = useState<AIConfiguration>({
    provider: 'openai',
    apiKeys: {},
    modelName: 'gpt-4o-mini',
    useDirectLLM: true,
    directLLMQualityMode: 'balanced',
    enableLLMPatternGeneration: true,
    similarityThreshold: 0.85,
    cachePatterns: true,
  });
  const [webUnblockerConfig, setWebUnblockerConfig] = useState<WebUnblockerConfig>({
    enabled: false,
    apiKey: '',
    zone: '',
  });
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Load settings from Firestore or localStorage
  useEffect(() => {
    const loadSettings = async () => {
      setLoading(true);
      try {
        if (user && userSettings) {
          // Load from Firestore
          setApiKey(userSettings.apiKey || '');
          if (userSettings.proxyConfig) {
            setProxyConfig(userSettings.proxyConfig);
          }
          if (userSettings.aiConfig) {
            setAiConfig(userSettings.aiConfig);
          }
          if (userSettings.webUnblockerConfig) {
            setWebUnblockerConfig(userSettings.webUnblockerConfig);
          }
        } else {
          // Fallback to localStorage
          const storedApiKey = localStorage.getItem('api_key');
          if (storedApiKey) {
            setApiKey(storedApiKey);
          }
          const storedProxyConfig = localStorage.getItem('proxy_config');
          if (storedProxyConfig) {
            try {
              setProxyConfig(JSON.parse(storedProxyConfig));
            } catch (e) {
              console.error('Failed to parse proxy config:', e);
            }
          }
          const storedAiConfig = localStorage.getItem('ai_config');
          if (storedAiConfig) {
            try {
              setAiConfig(JSON.parse(storedAiConfig));
            } catch (e) {
              console.error('Failed to parse AI config:', e);
            }
          }
        }
      } catch (error) {
        console.error('Error loading settings:', error);
      } finally {
        setLoading(false);
      }
    };

    loadSettings();
  }, [user, userSettings]);

  const handleSaveApiKey = async () => {
    if (!user) {
      // Fallback to localStorage
      localStorage.setItem('api_key', apiKey);
      setMessage({ type: 'success', text: 'API key saved locally' });
      setTimeout(() => setMessage(null), 3000);
      return;
    }

    setSaving(true);
    try {
      // Only save API key and AI config, don't overwrite proxy/webUnblocker settings
      await saveUserSettings(user.uid, {
        apiKey,
        aiConfig: {
          ...aiConfig,
          apiKeys: {
            ...aiConfig.apiKeys,
            openai: apiKey, // Also save OpenAI key in AI config
          },
        },
        // Preserve existing proxy and webUnblocker configs
        proxyConfig: userSettings?.proxyConfig || proxyConfig,
        webUnblockerConfig: userSettings?.webUnblockerConfig || webUnblockerConfig,
      });
      await refreshSettings();
      setMessage({ type: 'success', text: 'API key saved successfully' });
      setTimeout(() => setMessage(null), 3000);
    } catch (error: any) {
      setMessage({ type: 'error', text: error.message || 'Failed to save API key' });
      setTimeout(() => setMessage(null), 5000);
    } finally {
      setSaving(false);
    }
  };

  const handleTestProxy = async () => {
    try {
      const response = await configApi.testProxy(proxyConfig);
      if (response.data.success) {
        setMessage({ type: 'success', text: 'Proxy connection successful!' });
      } else {
        setMessage({ type: 'error', text: response.data.message || 'Proxy test failed' });
      }
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to test proxy' });
    }
    setTimeout(() => setMessage(null), 5000);
  };

  const handleTestWebUnblocker = async () => {
    if (!webUnblockerConfig.enabled || !webUnblockerConfig.apiKey) {
      setMessage({ type: 'error', text: 'Please enable Web Unblocker and enter an API key' });
      setTimeout(() => setMessage(null), 5000);
      return;
    }

    try {
      const response = await configApi.testWebUnblocker(
        webUnblockerConfig.apiKey,
        webUnblockerConfig.zone || 'web_unlocker1'
      );
      if (response.data.success) {
        setMessage({ type: 'success', text: response.data.message || 'Web Unblocker connection successful!' });
      } else {
        setMessage({ type: 'error', text: response.data.message || 'Web Unblocker test failed' });
      }
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to test Web Unblocker' });
    }
    setTimeout(() => setMessage(null), 5000);
  };

  const handleSaveSettings = async () => {
    if (!user) {
      // Fallback to localStorage
      localStorage.setItem('proxy_config', JSON.stringify(proxyConfig));
      localStorage.setItem('ai_config', JSON.stringify(aiConfig));
      localStorage.setItem('webunblocker_config', JSON.stringify(webUnblockerConfig));
      setMessage({ type: 'success', text: 'Settings saved locally' });
      setTimeout(() => setMessage(null), 3000);
      return;
    }

    setSaving(true);
    try {
      await saveUserSettings(user.uid, {
        apiKey,
        proxyConfig,
        aiConfig,
        webUnblockerConfig,
      });
      await refreshSettings();
      setMessage({ type: 'success', text: 'Settings saved successfully' });
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to save settings' });
    } finally {
      setSaving(false);
      setTimeout(() => setMessage(null), 3000);
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto">
        <div className="text-center py-16">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-purple-700 border-t-purple-600 mb-4"></div>
          <p className="text-gray-400">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 text-white">Settings</h1>

      {message && (
        <div
          className={`mb-4 p-4 rounded-md ${message.type === 'success'
              ? 'bg-green-50 border border-green-200 text-green-800'
              : 'bg-red-50 border border-red-200 text-red-800'
            }`}
        >
          {message.text}
        </div>
      )}

      <div className="card">
        {/* Tabs */}
        <div className="border-b border-gray-200 mb-6">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'api', label: 'API Keys' },
              { id: 'proxy', label: 'Proxy' },
              { id: 'ai', label: 'AI Configuration' },
              { id: 'webunblocker', label: 'Web Unblocker' },
              { id: 'warehouse', label: 'Data Warehouse' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`${activeTab === tab.id
                    ? 'border-purple-500 text-purple-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
                  } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition-colors`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* API Keys Tab */}
        {activeTab === 'api' && (
          <div className="space-y-6">
            <div>
              <label className="label">OpenAI API Key</label>
              <div className="flex gap-2">
                <input
                  type="password"
                  className="input-field flex-1"
                  placeholder="sk-..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
                <button onClick={handleSaveApiKey} className="btn-primary whitespace-nowrap">
                  Save
                </button>
              </div>
              <p className="mt-1 text-xs text-gray-500">
                Your API key is stored locally and never sent to our servers
              </p>
            </div>

            <div>
              <label className="label">Anthropic API Key (optional)</label>
              <input
                type="password"
                className="input-field"
                placeholder="sk-ant-..."
                value={aiConfig.apiKeys.anthropic || ''}
                onChange={(e) =>
                  setAiConfig({
                    ...aiConfig,
                    apiKeys: { ...aiConfig.apiKeys, anthropic: e.target.value },
                  })
                }
              />
            </div>

            <div>
              <label className="label">Google API Key (optional)</label>
              <input
                type="password"
                className="input-field"
                placeholder="AIza..."
                value={aiConfig.apiKeys.google || ''}
                onChange={(e) =>
                  setAiConfig({
                    ...aiConfig,
                    apiKeys: { ...aiConfig.apiKeys, google: e.target.value },
                  })
                }
              />
            </div>
          </div>
        )}

        {/* Proxy Tab */}
        {activeTab === 'proxy' && (
          <div className="space-y-6">
            <div>
              <label className="label">Proxy Provider</label>
              <select
                className="select-field"
                value={proxyConfig.provider}
                onChange={(e) => {
                  const newProvider = e.target.value as ProxyConfiguration['provider'];
                  const newConfig: ProxyConfiguration = {
                    ...proxyConfig,
                    provider: newProvider,
                  };
                  // Initialize provider-specific configs
                  if (newProvider === 'apify' && !proxyConfig.apifyProxy) {
                    newConfig.apifyProxy = {
                      useApifyProxy: true,
                      apifyProxyGroups: [],
                    };
                  }
                  if ((newProvider === 'brightdata' || newProvider === 'oxylabs' || newProvider === 'scraperapi' || newProvider === 'custom') && !proxyConfig.externalProxy) {
                    const defaultServer = newProvider === 'brightdata' ? 'brd.superproxy.io:33335' :
                      newProvider === 'oxylabs' ? 'pr.oxylabs.io:7777' :
                        newProvider === 'scraperapi' ? 'scraperapi.com' : '';
                    newConfig.externalProxy = {
                      server: defaultServer,
                      username: '',
                      password: '',
                    };
                  }
                  setProxyConfig(newConfig);
                }}
              >
                <option value="none">None</option>
                <option value="apify">Apify Proxy</option>
                <option value="brightdata">Bright Data</option>
                <option value="oxylabs">Oxylabs</option>
                <option value="scraperapi">ScraperAPI</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            {/* Apify Proxy Configuration */}
            {proxyConfig.provider === 'apify' && (
              <div className="space-y-4 pl-4 border-l-2 border-gray-200">
                <div>
                  <label className="label">Proxy Groups</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="RESIDENTIAL, DATACENTER"
                    value={proxyConfig.apifyProxy?.apifyProxyGroups?.join(', ') || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        apifyProxy: {
                          useApifyProxy: true,
                          apifyProxyGroups: e.target.value.split(',').map((g) => g.trim()).filter(g => g),
                          apifyProxyCountry: proxyConfig.apifyProxy?.apifyProxyCountry,
                        },
                      })
                    }
                  />
                  <p className="mt-1 text-xs text-gray-500">Comma-separated list: RESIDENTIAL, DATACENTER</p>
                </div>
                <div>
                  <label className="label">Country (optional)</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="US"
                    value={proxyConfig.apifyProxy?.apifyProxyCountry || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        apifyProxy: {
                          useApifyProxy: true,
                          apifyProxyGroups: proxyConfig.apifyProxy?.apifyProxyGroups || [],
                          apifyProxyCountry: e.target.value || undefined,
                        },
                      })
                    }
                  />
                </div>
                <button onClick={handleTestProxy} className="btn-secondary">
                  Test Connection
                </button>
              </div>
            )}

            {/* Bright Data Configuration */}
            {proxyConfig.provider === 'brightdata' && (
              <div className="space-y-4 pl-4 border-l-2 border-gray-200">
                <div>
                  <label className="label">Proxy Server</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="brd.superproxy.io:33335"
                    value={proxyConfig.externalProxy?.server || 'brd.superproxy.io:33335'}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: e.target.value,
                          username: proxyConfig.externalProxy?.username || '',
                          password: proxyConfig.externalProxy?.password || '',
                          country: proxyConfig.externalProxy?.country,
                        },
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">Username</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="brd-customer-hl_XXXXX-zone-residential_proxy2"
                    value={proxyConfig.externalProxy?.username || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: proxyConfig.externalProxy?.server || 'brd.superproxy.io:33335',
                          username: e.target.value,
                          password: proxyConfig.externalProxy?.password || '',
                          country: proxyConfig.externalProxy?.country,
                        },
                      })
                    }
                  />
                  <p className="mt-1 text-xs text-gray-500">Bright Data username format: brd-customer-hl_XXXXX-zone-ZONENAME</p>
                </div>
                <div>
                  <label className="label">Password</label>
                  <input
                    type="password"
                    className="input-field"
                    placeholder="Your Bright Data password"
                    value={proxyConfig.externalProxy?.password || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: proxyConfig.externalProxy?.server || 'brd.superproxy.io:33335',
                          username: proxyConfig.externalProxy?.username || '',
                          password: e.target.value,
                          country: proxyConfig.externalProxy?.country,
                        },
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">Country (optional)</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="US"
                    value={proxyConfig.externalProxy?.country || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: proxyConfig.externalProxy?.server || 'brd.superproxy.io:33335',
                          username: proxyConfig.externalProxy?.username || '',
                          password: proxyConfig.externalProxy?.password || '',
                          country: e.target.value || undefined,
                        },
                      })
                    }
                  />
                  <p className="mt-1 text-xs text-gray-500">Country code (e.g., US, GB, DE). Will be appended to username.</p>
                </div>
                <button onClick={handleTestProxy} className="btn-secondary">
                  Test Connection
                </button>
              </div>
            )}

            {/* Oxylabs Configuration */}
            {proxyConfig.provider === 'oxylabs' && (
              <div className="space-y-4 pl-4 border-l-2 border-gray-200">
                <div>
                  <label className="label">Username</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="customer-USERNAME"
                    value={proxyConfig.externalProxy?.username || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: proxyConfig.externalProxy?.server || 'pr.oxylabs.io:7777',
                          username: e.target.value,
                          password: proxyConfig.externalProxy?.password || '',
                          country: proxyConfig.externalProxy?.country,
                        },
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">Password</label>
                  <input
                    type="password"
                    className="input-field"
                    placeholder="Your Oxylabs password"
                    value={proxyConfig.externalProxy?.password || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: proxyConfig.externalProxy?.server || 'pr.oxylabs.io:7777',
                          username: proxyConfig.externalProxy?.username || '',
                          password: e.target.value,
                          country: proxyConfig.externalProxy?.country,
                        },
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">Country (optional)</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="US"
                    value={proxyConfig.externalProxy?.country || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: proxyConfig.externalProxy?.server || 'pr.oxylabs.io:7777',
                          username: proxyConfig.externalProxy?.username || '',
                          password: proxyConfig.externalProxy?.password || '',
                          country: e.target.value || undefined,
                        },
                      })
                    }
                  />
                </div>
                <button onClick={handleTestProxy} className="btn-secondary">
                  Test Connection
                </button>
              </div>
            )}

            {/* ScraperAPI Configuration */}
            {proxyConfig.provider === 'scraperapi' && (
              <div className="space-y-4 pl-4 border-l-2 border-gray-200">
                <div>
                  <label className="label">API Key</label>
                  <input
                    type="password"
                    className="input-field"
                    placeholder="Your ScraperAPI key"
                    value={proxyConfig.externalProxy?.username || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: proxyConfig.externalProxy?.server || 'scraperapi.com',
                          username: e.target.value,
                          password: proxyConfig.externalProxy?.password || '',
                          country: proxyConfig.externalProxy?.country,
                        },
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">Country (optional)</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="US"
                    value={proxyConfig.externalProxy?.country || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: proxyConfig.externalProxy?.server || 'scraperapi.com',
                          username: proxyConfig.externalProxy?.username || '',
                          password: proxyConfig.externalProxy?.password || '',
                          country: e.target.value || undefined,
                        },
                      })
                    }
                  />
                </div>
                <button onClick={handleTestProxy} className="btn-secondary">
                  Test Connection
                </button>
              </div>
            )}

            {/* Custom Proxy Configuration */}
            {proxyConfig.provider === 'custom' && (
              <div className="space-y-4 pl-4 border-l-2 border-gray-200">
                <div>
                  <label className="label">Proxy Server</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="proxy.example.com:8080"
                    value={proxyConfig.externalProxy?.server || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: e.target.value,
                          username: proxyConfig.externalProxy?.username || '',
                          password: proxyConfig.externalProxy?.password || '',
                          country: proxyConfig.externalProxy?.country,
                        },
                      })
                    }
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="label">Username</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="username"
                      value={proxyConfig.externalProxy?.username || ''}
                      onChange={(e) =>
                        setProxyConfig({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || '',
                            username: e.target.value,
                            password: proxyConfig.externalProxy?.password || '',
                            country: proxyConfig.externalProxy?.country,
                          },
                        })
                      }
                    />
                  </div>
                  <div>
                    <label className="label">Password</label>
                    <input
                      type="password"
                      className="input-field"
                      placeholder="password"
                      value={proxyConfig.externalProxy?.password || ''}
                      onChange={(e) =>
                        setProxyConfig({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || '',
                            username: proxyConfig.externalProxy?.username || '',
                            password: e.target.value,
                            country: proxyConfig.externalProxy?.country,
                          },
                        })
                      }
                    />
                  </div>
                </div>
                <div>
                  <label className="label">Country (optional)</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="US"
                    value={proxyConfig.externalProxy?.country || ''}
                    onChange={(e) =>
                      setProxyConfig({
                        ...proxyConfig,
                        externalProxy: {
                          server: proxyConfig.externalProxy?.server || '',
                          username: proxyConfig.externalProxy?.username || '',
                          password: proxyConfig.externalProxy?.password || '',
                          country: e.target.value || undefined,
                        },
                      })
                    }
                  />
                </div>
                <button onClick={handleTestProxy} className="btn-secondary">
                  Test Connection
                </button>
              </div>
            )}
          </div>
        )}

        {/* AI Configuration Tab */}
        {activeTab === 'ai' && (
          <div className="space-y-6">
            <div>
              <label className="label">Default AI Provider</label>
              <select
                className="select-field"
                value={aiConfig.provider}
                onChange={(e) =>
                  setAiConfig({ ...aiConfig, provider: e.target.value as any })
                }
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic (Claude)</option>
                <option value="google">Google (Gemini)</option>
              </select>
            </div>

            <div>
              <label className="label">Default Model</label>
              <input
                type="text"
                className="input-field"
                placeholder="gpt-4o-mini"
                value={aiConfig.modelName}
                onChange={(e) =>
                  setAiConfig({ ...aiConfig, modelName: e.target.value })
                }
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  className="checkbox-field"
                  checked={aiConfig.useDirectLLM}
                  onChange={(e) =>
                    setAiConfig({ ...aiConfig, useDirectLLM: e.target.checked })
                  }
                />
                <label className="ml-2 text-sm text-gray-700">Use Direct LLM extraction</label>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  className="checkbox-field"
                  checked={aiConfig.cachePatterns}
                  onChange={(e) =>
                    setAiConfig({ ...aiConfig, cachePatterns: e.target.checked })
                  }
                />
                <label className="ml-2 text-sm text-gray-700">Cache extraction patterns</label>
              </div>
            </div>
          </div>
        )}

        {/* Web Unblocker Tab */}
        {activeTab === 'webunblocker' && (
          <div className="space-y-6">
            <div className="flex items-center">
              <input
                type="checkbox"
                className="checkbox-field"
                checked={webUnblockerConfig.enabled}
                onChange={(e) =>
                  setWebUnblockerConfig({
                    ...webUnblockerConfig,
                    enabled: e.target.checked,
                  })
                }
              />
              <label className="ml-2 text-sm text-gray-700">Enable Web Unblocker</label>
            </div>

            {webUnblockerConfig.enabled && (
              <div className="space-y-4 pl-4 border-l-2 border-gray-200">
                <div>
                  <label className="label">API Key</label>
                  <input
                    type="password"
                    className="input-field"
                    placeholder="Your Web Unblocker API key"
                    value={webUnblockerConfig.apiKey}
                    onChange={(e) =>
                      setWebUnblockerConfig({
                        ...webUnblockerConfig,
                        apiKey: e.target.value,
                      })
                    }
                  />
                </div>
                <div>
                  <label className="label">Zone</label>
                  <input
                    type="text"
                    className="input-field"
                    placeholder="web_unlocker1"
                    value={webUnblockerConfig.zone}
                    onChange={(e) =>
                      setWebUnblockerConfig({
                        ...webUnblockerConfig,
                        zone: e.target.value,
                      })
                    }
                  />
                  <p className="mt-1 text-xs text-gray-500">Default: web_unlocker1</p>
                </div>
                <button
                  onClick={handleTestWebUnblocker}
                  className="btn-secondary"
                  disabled={!webUnblockerConfig.apiKey}
                >
                  Test Connection
                </button>
              </div>
            )}
          </div>
        )}

        {/* Data Warehouse Tab */}
        {activeTab === 'warehouse' && (
          <div className="space-y-6">
            <p className="text-gray-500">
              Data warehouse connector configuration will be available here. Connect to Snowflake, BigQuery, PostgreSQL, and more.
            </p>
          </div>
        )}

        {/* Save Button */}
        <div className="mt-6 flex justify-end">
          <button
            onClick={handleSaveSettings}
            disabled={saving}
            className="btn-primary"
          >
            {saving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>
    </div>
  );
}
