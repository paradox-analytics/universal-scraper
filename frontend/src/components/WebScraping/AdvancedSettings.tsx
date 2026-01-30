import { useState } from 'react';
import type { ProxyConfiguration, PaginationConfiguration } from '../../types';

interface AdvancedSettingsProps {
  proxyConfig?: ProxyConfiguration;
  paginationConfig?: PaginationConfiguration;
  onProxyChange?: (config: ProxyConfiguration) => void;
  onPaginationChange?: (config: PaginationConfiguration) => void;
  defaultTab?: 'proxy' | 'pagination';
  highlightProxy?: boolean;
}

export function AdvancedSettings({
  proxyConfig,
  paginationConfig,
  onProxyChange,
  onPaginationChange,
  defaultTab = 'proxy',
  highlightProxy = false,
}: AdvancedSettingsProps) {
  const [activeTab, setActiveTab] = useState<'proxy' | 'pagination'>(defaultTab);

  return (
    <div className={`card-elevated ${highlightProxy && activeTab === 'proxy' ? 'ring-2 ring-orange-500 ring-opacity-50' : ''}`}>
      <h2 className="text-xl font-bold mb-6 text-gray-50">Advanced Settings</h2>

      <div className="space-y-6">
        {/* Tabs */}
        <div className="border-b border-gray-700">
          <nav className="-mb-px flex space-x-8">
            {(['proxy', 'pagination'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`${
                  activeTab === tab
                    ? 'border-purple-500 text-purple-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm capitalize transition-colors`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>

          {/* Proxy Settings */}
          {activeTab === 'proxy' && (
            <div className="space-y-4">
              <div>
                <label className="label">Proxy Provider</label>
                <select
                  className="select-field"
                  value={proxyConfig?.provider || 'none'}
                  onChange={(e) => {
                    const newProvider = e.target.value as ProxyConfiguration['provider'];
                    const newConfig: ProxyConfiguration = {
                      ...proxyConfig,
                      provider: newProvider,
                    };
                    // Initialize provider-specific configs
                    if (newProvider === 'apify' && !proxyConfig?.apifyProxy) {
                      newConfig.apifyProxy = {
                        useApifyProxy: true,
                        apifyProxyGroups: [],
                      };
                    }
                    if ((newProvider === 'brightdata' || newProvider === 'oxylabs' || newProvider === 'scraperapi' || newProvider === 'custom') && !proxyConfig?.externalProxy) {
                      newConfig.externalProxy = {
                        server: newProvider === 'brightdata' ? 'brd.superproxy.io:22225' : newProvider === 'oxylabs' ? 'pr.oxylabs.io:7777' : '',
                        username: '',
                        password: '',
                      };
                    }
                    if (newProvider === 'web_unlocker' && !proxyConfig?.webUnblocker) {
                      newConfig.webUnblocker = {
                        enabled: true,
                        apiKey: '',
                        zone: 'web_unlocker1',
                        useProxyMethod: false,
                      };
                    }
                    onProxyChange?.(newConfig);
                  }}
                >
                  <option value="none">None</option>
                  <option value="apify">Apify Proxy</option>
                  <option value="brightdata">Bright Data</option>
                  <option value="web_unlocker">Bright Data Web Unlocker</option>
                  <option value="oxylabs">Oxylabs</option>
                  <option value="scraperapi">ScraperAPI</option>
                  <option value="custom">Custom</option>
                </select>
              </div>

              {proxyConfig?.provider === 'apify' && (
                <div className="space-y-4 pl-4 border-l-2 border-gray-700">
                  <div>
                    <label className="label">Proxy Groups</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="RESIDENTIAL, DATACENTER"
                      value={proxyConfig.apifyProxy?.apifyProxyGroups?.join(', ') || ''}
                      onChange={(e) =>
                        onProxyChange?.({
                          ...proxyConfig,
                          apifyProxy: {
                            useApifyProxy: true,
                            apifyProxyGroups: e.target.value.split(',').map((g) => g.trim()).filter(g => g),
                            apifyProxyCountry: proxyConfig.apifyProxy?.apifyProxyCountry,
                          },
                        } as ProxyConfiguration)
                      }
                    />
                    <p className="mt-1 text-xs text-gray-400">Comma-separated list: RESIDENTIAL, DATACENTER</p>
                  </div>
                  <div>
                    <label className="label">Country (optional)</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="US"
                      value={proxyConfig.apifyProxy?.apifyProxyCountry || ''}
                      onChange={(e) =>
                        onProxyChange?.({
                          ...proxyConfig,
                          apifyProxy: {
                            ...proxyConfig.apifyProxy,
                            apifyProxyCountry: e.target.value,
                          },
                        } as ProxyConfiguration)
                      }
                    />
                  </div>
                </div>
              )}

              {/* Bright Data Configuration */}
              {proxyConfig?.provider === 'brightdata' && (
                <div className="space-y-4 pl-4 border-l-2 border-gray-700">
                  <div>
                    <label className="label">Proxy Server</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="brd.superproxy.io:22225"
                      value={proxyConfig.externalProxy?.server || 'brd.superproxy.io:22225'}
                      onChange={(e) =>
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: e.target.value,
                            username: proxyConfig.externalProxy?.username || '',
                            password: proxyConfig.externalProxy?.password || '',
                            country: proxyConfig.externalProxy?.country,
                          },
                        } as ProxyConfiguration)
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
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || 'brd.superproxy.io:22225',
                            username: e.target.value,
                            password: proxyConfig.externalProxy?.password || '',
                            country: proxyConfig.externalProxy?.country,
                          },
                        } as ProxyConfiguration)
                      }
                    />
                    <p className="mt-1 text-xs text-gray-400">Format: brd-customer-hl_XXXXX-zone-ZONENAME</p>
                  </div>
                  <div>
                    <label className="label">Password</label>
                    <input
                      type="password"
                      className="input-field"
                      placeholder="Your Bright Data password"
                      value={proxyConfig.externalProxy?.password || ''}
                      onChange={(e) =>
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || 'brd.superproxy.io:22225',
                            username: proxyConfig.externalProxy?.username || '',
                            password: e.target.value,
                            country: proxyConfig.externalProxy?.country,
                          },
                        } as ProxyConfiguration)
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
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || 'brd.superproxy.io:22225',
                            username: proxyConfig.externalProxy?.username || '',
                            password: proxyConfig.externalProxy?.password || '',
                            country: e.target.value || undefined,
                          },
                        } as ProxyConfiguration)
                      }
                    />
                  </div>
                </div>
              )}

              {/* Web Unlocker Configuration */}
              {proxyConfig?.provider === 'web_unlocker' && (
                <div className="space-y-4 pl-4 border-l-2 border-gray-700">
                  <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-3 mb-4">
                    <p className="text-xs text-blue-300">
                      <strong>Web Unlocker</strong> bypasses advanced anti-bot protection (Kasada, Cloudflare, etc.) 
                      using Bright Data's Web Unlocker service. You can use either API key (Bearer token) or proxy credentials.
                    </p>
                  </div>
                  
                  <div>
                    <label className="label flex items-center gap-2">
                      <input
                        type="radio"
                        name="web_unlocker_method"
                        checked={!proxyConfig.webUnblocker?.useProxyMethod}
                        onChange={() => onProxyChange?.({
                          ...proxyConfig,
                          webUnblocker: {
                            ...proxyConfig.webUnblocker,
                            enabled: true,
                            useProxyMethod: false,
                            apiKey: proxyConfig.webUnblocker?.apiKey || '',
                            zone: proxyConfig.webUnblocker?.zone || 'web_unlocker1',
                          },
                        } as ProxyConfiguration)}
                        className="form-radio h-4 w-4 text-purple-600"
                      />
                      <span>API Key Method (Bearer Token)</span>
                    </label>
                  </div>
                  
                  <div>
                    <label className="label flex items-center gap-2">
                      <input
                        type="radio"
                        name="web_unlocker_method"
                        checked={proxyConfig.webUnblocker?.useProxyMethod === true}
                        onChange={() => onProxyChange?.({
                          ...proxyConfig,
                          webUnblocker: {
                            ...proxyConfig.webUnblocker,
                            enabled: true,
                            useProxyMethod: true,
                            apiKey: proxyConfig.webUnblocker?.apiKey || '',
                            zone: proxyConfig.webUnblocker?.zone || 'web_unlocker1',
                          },
                        } as ProxyConfiguration)}
                        className="form-radio h-4 w-4 text-purple-600"
                      />
                      <span>Proxy Method (Proxy Credentials)</span>
                    </label>
                  </div>

                  {!proxyConfig.webUnblocker?.useProxyMethod ? (
                    <>
                      <div>
                        <label className="label">API Key (Bearer Token) *</label>
                        <input
                          type="password"
                          className="input-field"
                          placeholder="Your Bright Data API key (Bearer token)"
                          value={proxyConfig.webUnblocker?.apiKey || ''}
                          onChange={(e) =>
                            onProxyChange?.({
                              ...proxyConfig,
                              webUnblocker: {
                                ...proxyConfig.webUnblocker,
                                enabled: true,
                                apiKey: e.target.value,
                                zone: proxyConfig.webUnblocker?.zone || 'web_unlocker1',
                                useProxyMethod: false,
                              },
                            } as ProxyConfiguration)
                          }
                        />
                        <p className="mt-1 text-xs text-gray-400">
                          Get your API key from{' '}
                          <a href="https://brightdata.com/cp/account/api" target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:text-purple-300 underline">
                            Bright Data Dashboard
                          </a>
                        </p>
                      </div>
                      <div>
                        <label className="label">Zone</label>
                        <input
                          type="text"
                          className="input-field"
                          placeholder="web_unlocker1"
                          value={proxyConfig.webUnblocker?.zone || 'web_unlocker1'}
                          onChange={(e) =>
                            onProxyChange?.({
                              ...proxyConfig,
                              webUnblocker: {
                                ...proxyConfig.webUnblocker,
                                enabled: true,
                                apiKey: proxyConfig.webUnblocker?.apiKey || '',
                                zone: e.target.value || 'web_unlocker1',
                                useProxyMethod: false,
                              },
                            } as ProxyConfiguration)
                          }
                        />
                        <p className="mt-1 text-xs text-gray-400">Default: web_unlocker1</p>
                      </div>
                    </>
                  ) : (
                    <>
                      <div>
                        <label className="label">Proxy Server</label>
                        <input
                          type="text"
                          className="input-field"
                          placeholder="brd.superproxy.io:22225"
                          value={proxyConfig.externalProxy?.server || 'brd.superproxy.io:22225'}
                          onChange={(e) =>
                            onProxyChange?.({
                              ...proxyConfig,
                              externalProxy: {
                                server: e.target.value,
                                username: proxyConfig.externalProxy?.username || '',
                                password: proxyConfig.externalProxy?.password || '',
                              },
                              webUnblocker: {
                                ...proxyConfig.webUnblocker,
                                enabled: true,
                                useProxyMethod: true,
                                zone: proxyConfig.webUnblocker?.zone || 'web_unlocker1',
                              },
                            } as ProxyConfiguration)
                          }
                        />
                      </div>
                      <div>
                        <label className="label">Username *</label>
                        <input
                          type="text"
                          className="input-field"
                          placeholder="brd-customer-hl_XXXXX-zone-web_unlocker1"
                          value={proxyConfig.externalProxy?.username || ''}
                          onChange={(e) =>
                            onProxyChange?.({
                              ...proxyConfig,
                              externalProxy: {
                                server: proxyConfig.externalProxy?.server || 'brd.superproxy.io:22225',
                                username: e.target.value,
                                password: proxyConfig.externalProxy?.password || '',
                              },
                              webUnblocker: {
                                ...proxyConfig.webUnblocker,
                                enabled: true,
                                useProxyMethod: true,
                                zone: proxyConfig.webUnblocker?.zone || 'web_unlocker1',
                              },
                            } as ProxyConfiguration)
                          }
                        />
                        <p className="mt-1 text-xs text-gray-400">Format: brd-customer-hl_XXXXX-zone-web_unlocker1</p>
                      </div>
                      <div>
                        <label className="label">Password *</label>
                        <input
                          type="password"
                          className="input-field"
                          placeholder="Your Bright Data password"
                          value={proxyConfig.externalProxy?.password || ''}
                          onChange={(e) =>
                            onProxyChange?.({
                              ...proxyConfig,
                              externalProxy: {
                                server: proxyConfig.externalProxy?.server || 'brd.superproxy.io:22225',
                                username: proxyConfig.externalProxy?.username || '',
                                password: e.target.value,
                              },
                              webUnblocker: {
                                ...proxyConfig.webUnblocker,
                                enabled: true,
                                useProxyMethod: true,
                                zone: proxyConfig.webUnblocker?.zone || 'web_unlocker1',
                              },
                            } as ProxyConfiguration)
                          }
                        />
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* Oxylabs Configuration */}
              {proxyConfig?.provider === 'oxylabs' && (
                <div className="space-y-4 pl-4 border-l-2 border-gray-700">
                  <div>
                    <label className="label">Proxy Server</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="pr.oxylabs.io:7777"
                      value={proxyConfig.externalProxy?.server || 'pr.oxylabs.io:7777'}
                      onChange={(e) =>
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: e.target.value,
                            username: proxyConfig.externalProxy?.username || '',
                            password: proxyConfig.externalProxy?.password || '',
                            country: proxyConfig.externalProxy?.country,
                          },
                        } as ProxyConfiguration)
                      }
                    />
                  </div>
                  <div>
                    <label className="label">Username</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="Your Oxylabs username"
                      value={proxyConfig.externalProxy?.username || ''}
                      onChange={(e) =>
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || 'pr.oxylabs.io:7777',
                            username: e.target.value,
                            password: proxyConfig.externalProxy?.password || '',
                            country: proxyConfig.externalProxy?.country,
                          },
                        } as ProxyConfiguration)
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
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || 'pr.oxylabs.io:7777',
                            username: proxyConfig.externalProxy?.username || '',
                            password: e.target.value,
                            country: proxyConfig.externalProxy?.country,
                          },
                        } as ProxyConfiguration)
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
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || 'pr.oxylabs.io:7777',
                            username: proxyConfig.externalProxy?.username || '',
                            password: proxyConfig.externalProxy?.password || '',
                            country: e.target.value || undefined,
                          },
                        } as ProxyConfiguration)
                      }
                    />
                  </div>
                </div>
              )}

              {/* ScraperAPI Configuration */}
              {proxyConfig?.provider === 'scraperapi' && (
                <div className="space-y-4 pl-4 border-l-2 border-gray-700">
                  <div>
                    <label className="label">API Key</label>
                    <input
                      type="password"
                      className="input-field"
                      placeholder="Your ScraperAPI key"
                      value={proxyConfig.externalProxy?.password || ''}
                      onChange={(e) =>
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || '',
                            username: proxyConfig.externalProxy?.username || '',
                            password: e.target.value,
                            country: proxyConfig.externalProxy?.country,
                          },
                        } as ProxyConfiguration)
                      }
                    />
                    <p className="mt-1 text-xs text-gray-400">ScraperAPI uses an API key instead of username/password</p>
                  </div>
                  <div>
                    <label className="label">Country (optional)</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="US"
                      value={proxyConfig.externalProxy?.country || ''}
                      onChange={(e) =>
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || '',
                            username: proxyConfig.externalProxy?.username || '',
                            password: proxyConfig.externalProxy?.password || '',
                            country: e.target.value || undefined,
                          },
                        } as ProxyConfiguration)
                      }
                    />
                  </div>
                </div>
              )}

              {/* Custom Proxy Configuration */}
              {proxyConfig?.provider === 'custom' && (
                <div className="space-y-4 pl-4 border-l-2 border-gray-700">
                  <div>
                    <label className="label">Proxy Server</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="proxy.example.com:8080"
                      value={proxyConfig.externalProxy?.server || ''}
                      onChange={(e) =>
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: e.target.value,
                            username: proxyConfig.externalProxy?.username || '',
                            password: proxyConfig.externalProxy?.password || '',
                            country: proxyConfig.externalProxy?.country,
                          },
                        } as ProxyConfiguration)
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
                          onProxyChange?.({
                            ...proxyConfig,
                            externalProxy: {
                              server: proxyConfig.externalProxy?.server || '',
                              username: e.target.value,
                              password: proxyConfig.externalProxy?.password || '',
                              country: proxyConfig.externalProxy?.country,
                            },
                          } as ProxyConfiguration)
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
                          onProxyChange?.({
                            ...proxyConfig,
                            externalProxy: {
                              server: proxyConfig.externalProxy?.server || '',
                              username: proxyConfig.externalProxy?.username || '',
                              password: e.target.value,
                              country: proxyConfig.externalProxy?.country,
                            },
                          } as ProxyConfiguration)
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
                        onProxyChange?.({
                          ...proxyConfig,
                          externalProxy: {
                            server: proxyConfig.externalProxy?.server || '',
                            username: proxyConfig.externalProxy?.username || '',
                            password: proxyConfig.externalProxy?.password || '',
                            country: e.target.value || undefined,
                          },
                        } as ProxyConfiguration)
                      }
                    />
                  </div>
                </div>
              )}
            </div>
          )}


          {/* Pagination Settings */}
          {activeTab === 'pagination' && (
            <div className="space-y-4">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  className="checkbox-field"
                  checked={paginationConfig?.scrollToBottom || false}
                  onChange={(e) =>
                    onPaginationChange?.({
                      ...paginationConfig,
                      scrollToBottom: e.target.checked,
                    } as PaginationConfiguration)
                  }
                />
                <label className="ml-2 text-sm text-gray-300">
                  <span className="font-semibold">Scroll to bottom (infinite scroll)</span>
                  <span className="text-xs text-gray-400 ml-2">Auto-enabled for feed/listings</span>
                </label>
              </div>
              <p className="text-xs text-gray-400 ml-6">
                Automatically scrolls down to load more content. Required for sites that use infinite scroll pagination.
              </p>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  className="checkbox-field"
                  checked={paginationConfig?.enableAutoPagination || false}
                  onChange={(e) =>
                    onPaginationChange?.({
                      ...paginationConfig,
                      enableAutoPagination: e.target.checked,
                    } as PaginationConfiguration)
                  }
                />
                <label className="ml-2 text-sm text-gray-300">Enable automatic pagination (multi-page)</label>
              </div>

              {paginationConfig?.enableAutoPagination && (
                <div className="space-y-4 pl-4 border-l-2 border-gray-700">
                  <div>
                    <label className="label">Max Pages</label>
                    <input
                      type="number"
                      className="input-field"
                      value={paginationConfig.maxPages || 10}
                      onChange={(e) =>
                        onPaginationChange?.({
                          ...paginationConfig,
                          maxPages: parseInt(e.target.value) || 10,
                        } as PaginationConfiguration)
                      }
                    />
                  </div>

                  <div>
                    <label className="label">Click Load More Button (CSS selector)</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="button.load-more"
                      value={paginationConfig?.clickLoadMore || ''}
                      onChange={(e) =>
                        onPaginationChange?.({
                          ...paginationConfig,
                          clickLoadMore: e.target.value,
                        } as PaginationConfiguration)
                      }
                    />
                  </div>

                  <div>
                    <label className="label">Wait for Selector (CSS selector)</label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder=".product-list"
                      value={paginationConfig?.waitForSelector || ''}
                      onChange={(e) =>
                        onPaginationChange?.({
                          ...paginationConfig,
                          waitForSelector: e.target.value,
                        } as PaginationConfiguration)
                      }
                    />
                  </div>
                </div>
              )}
            </div>
          )}
      </div>
    </div>
  );
}

