import { ShieldCheckIcon, SparklesIcon, Cog6ToothIcon } from '@heroicons/react/24/outline';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function GlobalStatusIndicators() {
  const { userSettings } = useAuth();
  const navigate = useNavigate();

  const hasAIKey = userSettings?.apiKey || 
                   userSettings?.aiConfig?.apiKeys?.openai || 
                   userSettings?.aiConfig?.apiKeys?.anthropic || 
                   userSettings?.aiConfig?.apiKeys?.google;
  
  // Check if proxy is actually configured and enabled
  const hasProxy = (() => {
    // FIRST: Check Web Unblocker (stored separately)
    const webUnblocker = userSettings?.webUnblockerConfig;
    if (webUnblocker?.enabled === true && webUnblocker?.apiKey) {
      return true; // Web Unblocker is enabled
    }
    
    // SECOND: Check proxy config
    const provider = userSettings?.proxyConfig?.provider;
    if (!provider || provider === 'none') return false;
    
    // For other providers, check if credentials exist
    if (provider === 'brightdata' || provider === 'oxylabs' || provider === 'scraperapi' || provider === 'custom') {
      const externalProxy = userSettings?.proxyConfig?.externalProxy;
      return externalProxy?.server && externalProxy?.username && externalProxy?.password;
    }
    
    if (provider === 'apify') {
      const apifyProxy = userSettings?.proxyConfig?.apifyProxy;
      return apifyProxy?.useApifyProxy && apifyProxy?.apifyProxyGroups?.length > 0;
    }
    
    return false;
  })();
  
  const proxyLabel = (() => {
    if (!hasProxy) return '';
    
    // Check Web Unblocker first (it takes priority)
    const webUnblocker = userSettings?.webUnblockerConfig;
    if (webUnblocker?.enabled === true && webUnblocker?.apiKey) {
      return 'Web Unblocker';
    }
    
    // Then check proxy provider
    const provider = userSettings?.proxyConfig?.provider;
    if (provider === 'brightdata') return 'Bright Data';
    if (provider === 'oxylabs') return 'Oxylabs';
    if (provider === 'scraperapi') return 'ScraperAPI';
    if (provider === 'apify') return 'Apify Proxy';
    return 'Proxy';
  })();

  return (
    <div className="flex items-center gap-2">
      {/* AI Status Indicator */}
      {hasAIKey && (
        <button
          onClick={() => navigate('/settings')}
          className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-900/30 border border-purple-600 hover:bg-purple-900/50 transition-colors"
          title="AI API Key configured - Click to manage"
        >
          <SparklesIcon className="w-4 h-4 text-purple-400" />
          <span className="text-xs font-medium text-purple-400">AI Enabled</span>
        </button>
      )}

      {/* Proxy/Web Unblocker Status Indicator */}
      {hasProxy && (
        <button
          onClick={() => navigate('/settings')}
          className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-900/30 border border-green-600 hover:bg-green-900/50 transition-colors"
          title={`${proxyLabel} configured - Click to manage`}
        >
          <ShieldCheckIcon className="w-4 h-4 text-green-400" />
          <span className="text-xs font-medium text-green-400">{proxyLabel} Active</span>
        </button>
      )}

      {/* Settings Quick Access (if nothing configured) */}
      {!hasAIKey && !hasProxy && (
        <button
          onClick={() => navigate('/settings')}
          className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-700/50 border border-gray-600 hover:bg-gray-700 transition-colors"
          title="Configure AI and Proxy settings"
        >
          <Cog6ToothIcon className="w-4 h-4 text-gray-400" />
          <span className="text-xs font-medium text-gray-400">Setup</span>
        </button>
      )}
    </div>
  );
}

