import { useState, useEffect, useRef, useCallback } from 'react';
import {
  ArrowPathIcon,
  ArrowLeftIcon,
  ArrowRightIcon,
  Cog6ToothIcon,
  CircleStackIcon,
  PlayIcon,
  EyeIcon,
  CodeBracketIcon,
  DocumentTextIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  SparklesIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';
import type { ProxyConfiguration, PaginationConfiguration, CachedPattern } from '../../types';
import { API_BASE_URL } from '../../config/api';
import { getAuthToken } from '../../services/auth';
import { useAuth } from '../../contexts/AuthContext';

interface BrowserWorkspaceProps {
  onExtract?: (results: any[], fields: string[], metadata?: any) => void;
  onPatternSave?: (pattern: any) => void;
  cachedPatterns?: CachedPattern[];
  triggerExtraction?: number; // Increment to trigger extraction from parent
  onLog?: (message: string, level: 'info' | 'success' | 'warning' | 'error' | 'progress', details?: string) => void;
  className?: string;
}

interface DetectedElement {
  type: 'container' | 'field';
  field?: string;
  selector: string;
  count: number;
  sample: string;
}

interface SelectedField {
  name: string;
  selector: string;
  count: number;
  sample: string;
}

type ViewMode = 'browser' | 'json' | 'html' | 'network';

// Use the imported API_BASE_URL

export default function BrowserWorkspace({
  onExtract,
  onPatternSave,
  onLog,
  cachedPatterns = [],
  triggerExtraction = 0,
  className = ''
}: BrowserWorkspaceProps) {
  // Browser state
  const [, setSessionId] = useState<string | null>(null);
  const [url, setUrl] = useState('');
  const [currentUrl, setCurrentUrl] = useState('');
  const [screenshot, setScreenshot] = useState<string | null>(null);
  const [previewHtml, setPreviewHtml] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [browserRenderingFailed, setBrowserRenderingFailed] = useState(false);
  const [fallbackReason, setFallbackReason] = useState<string | null>(null);
  const [, setFetchMethod] = useState<string>('unknown');

  // View state
  const [viewMode, setViewMode] = useState<ViewMode>('browser');
  const [showOptionsMenu, setShowOptionsMenu] = useState(false);
  const [showCacheMenu, setShowCacheMenu] = useState(false);
  const [showClickTip, setShowClickTip] = useState(true);

  // Extraction state
  const [extractionPrompt, setExtractionPrompt] = useState<string>(''); // User's extraction goal/prompt
  const [extractionTarget, setExtractionTarget] = useState<string>(''); // NEW: Target hint (e.g., 'products')
  const [isSuggestingFields, setIsSuggestingFields] = useState(false);
  const [detectedElements, setDetectedElements] = useState<DetectedElement[]>([]);
  const [selectedFields, setSelectedFields] = useState<SelectedField[]>([]);
  const [extractionResults, setExtractionResults] = useState<any[]>([]);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractionStatus, setExtractionStatus] = useState<'idle' | 'cached' | 'direct_llm' | 'extracting'>('idle');
  const [extractionTime, setExtractionTime] = useState<number | null>(null);

  // Agent log state
  interface AgentLogEntry {
    id: string;
    timestamp: Date;
    level: 'info' | 'success' | 'warning' | 'error' | 'progress';
    message: string;
    details?: string;
    stage?: 'probabilistic' | 'deterministic' | 'cached';
  }
  const [agentLogs, setAgentLogs] = useState<AgentLogEntry[]>([]);


  const addAgentLog = (type: AgentLogEntry['level'], message: string, detail?: string, stage?: AgentLogEntry['stage']) => {
    const entry: AgentLogEntry = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
      level: type,
      message,
      details: detail,
      stage
    };
    setAgentLogs(prev => [entry, ...prev].slice(0, 50)); // Prepend and keep last 50 entries
    if (onLog) {
      onLog(message, type, detail);
    }
  };

  // JSON detection state
  const [jsonSources, setJsonSources] = useState<any[]>([]);
  const [jsonRecommended, setJsonRecommended] = useState(false);
  const [, setJsonConfidence] = useState(0);
  const [extractionMode, setExtractionMode] = useState<'json' | 'browser'>('browser');
  const [showJsonRecommendation, setShowJsonRecommendation] = useState(false);

  // Get global settings from AuthContext
  const { userSettings } = useAuth();

  // Configuration state - use global settings as default
  const [proxyConfig, setProxyConfig] = useState<ProxyConfiguration>(
    userSettings?.proxyConfig || { provider: 'none' }
  );
  const [paginationConfig, setPaginationConfig] = useState<PaginationConfiguration>({
    scrollToBottom: false,
    enableAutoPagination: false,
    enableLLMPagination: false,
    maxPages: 10,
  });
  const [browserTimeout, setBrowserTimeout] = useState(60000);

  // Pattern state
  const [patternName, setPatternName] = useState('');
  const [patternVisibility, setPatternVisibility] = useState<'private' | 'public'>('private');
  const [showSavePatternModal, setShowSavePatternModal] = useState(false);
  const [showProxySettingsModal, setShowProxySettingsModal] = useState(false);
  const [localProxyConfig, setLocalProxyConfig] = useState<ProxyConfiguration>({ provider: 'none' });

  // Refs
  const urlInputRef = useRef<HTMLInputElement>(null);

  // Load global settings when they change
  useEffect(() => {
    if (userSettings?.proxyConfig) {
      setProxyConfig(userSettings.proxyConfig);
      setLocalProxyConfig(userSettings.proxyConfig);
    }
  }, [userSettings?.proxyConfig]);

  // Load pagination config from localStorage (page-specific)
  useEffect(() => {
    const savedPagination = localStorage.getItem('pagination_config');
    if (savedPagination) {
      try {
        setPaginationConfig(JSON.parse(savedPagination));
      } catch (e) {
        console.error('Failed to load pagination config:', e);
      }
    }
  }, []);

  // Navigate to URL - loads page and auto-suggests fields
  const navigate = useCallback(async (targetUrl: string, forceUnblocker: boolean = false) => {
    if (!targetUrl) return;

    setIsLoading(true);
    setError(null);
    setExtractionStatus('idle');
    setExtractionResults([]);
    setSelectedFields([]); // Clear previous fields
    setSessionId(targetUrl);
    setScreenshot(null);
    setShowClickTip(true); // Show tip for new page
    setAgentLogs([]); // Clear agent logs
    addAgentLog('info', '🚀 Starting scan...', `URL: ${targetUrl}`);

    try {
      const token = await getAuthToken();
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };

      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
      const apiKey = localStorage.getItem('api_key');
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }

      // Build proxy config for request - include Web Unblocker if enabled
      let requestProxyConfig: any = null;

      // Check if Web Unblocker is enabled in global settings
      const webUnblocker = userSettings?.webUnblockerConfig;
      if (forceUnblocker || (webUnblocker?.enabled && webUnblocker?.apiKey)) {
        // Send Web Unblocker config
        requestProxyConfig = {
          provider: 'web_unlocker',
          webUnblocker: {
            apiKey: webUnblocker?.apiKey,
            zone: webUnblocker?.zone || 'web_unlocker1',
          },
        };
        if (forceUnblocker) {
          addAgentLog('info', '🛡️ Forcing Web Unblocker...', 'Bypassing anti-bot protection');
        }
      } else if (proxyConfig.provider !== 'none') {
        // Send regular proxy config
        requestProxyConfig = proxyConfig;
      }

      // Step 1: Load preview (rendered page with JS)
      addAgentLog('info', '🌐 Fetching page...', 'Loading page content');
      const previewResponse = await fetch(`${API_BASE_URL}/api/v1/preview`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          url: targetUrl,
          proxy_config: requestProxyConfig,
          browser_timeout: browserTimeout
        })
      });

      if (!previewResponse.ok) {
        let errorMessage = 'Navigation failed';
        try {
          const errorData = await previewResponse.json();
          errorMessage = errorData.detail || errorData.message || JSON.stringify(errorData);
        } catch (e) {
          errorMessage = `HTTP ${previewResponse.status}: ${previewResponse.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const previewData = await previewResponse.json();

      if (previewData.success !== false && previewData.html) {
        // Store HTML for iframe rendering
        setPreviewHtml(previewData.html);
        setCurrentUrl(targetUrl);
        setDetectedElements(previewData.detected_elements || []);

        // Check browser rendering status
        setBrowserRenderingFailed(previewData.browser_rendering_failed || false);
        setFallbackReason(previewData.fallback_reason || null);
        const fetchMethod = previewData.fetch_method || 'unknown';
        setFetchMethod(fetchMethod);

        addAgentLog('success', `✅ Page loaded (${fetchMethod})`, `HTML size: ${previewData.html.length.toLocaleString()} bytes`, 'probabilistic');

        if (previewData.browser_rendering_failed) {
          addAgentLog('warning', '⚠️ Browser rendering failed', previewData.fallback_reason || 'Using static HTML fallback', 'probabilistic');
        }

        // JSON-FIRST: Check if JSON extraction is recommended
        if (previewData.json_recommended) {
          setJsonRecommended(true);
          setJsonConfidence(previewData.json_confidence || 0);
          setJsonSources(previewData.json_sources || []);
          setExtractionMode('json'); // Default to JSON if recommended
          setShowJsonRecommendation(true);

          // Auto-populate fields from JSON if available
          const usableJsonSource = previewData.json_sources?.find((s: any) => s.usable);
          if (usableJsonSource && usableJsonSource.sample_fields) {
            const jsonFields: SelectedField[] = usableJsonSource.sample_fields.map((fieldName: string) => ({
              name: fieldName,
              selector: '', // JSON fields don't need selectors
              count: usableJsonSource.count || 0,
              sample: ''
            }));
            setSelectedFields(jsonFields);
          }
        } else {
          setJsonRecommended(false);
          setExtractionMode('browser');
          setShowJsonRecommendation(false);
        }

        // Step 2: Auto-suggest fields (only if no fields selected)
        if (selectedFields.length === 0) {
          addAgentLog('info', '🔍 Starting field discovery...', 'Field discovery initiated');
          try {
            const suggestResponse = await fetch(`${API_BASE_URL}/api/v1/suggest-fields`, {
              method: 'POST',
              headers,
              body: JSON.stringify({
                url: targetUrl,
                use_llm: true, // Use LLM for better suggestions
                proxy_config: requestProxyConfig, // Use same proxy config as preview
                browser_timeout: browserTimeout
              })
            });

            addAgentLog('info', '📊 Analyzing page structure...', 'Field discovery in progress');

            if (suggestResponse.ok) {
              const suggestData = await suggestResponse.json();
              if (suggestData.fields && suggestData.fields.length > 0) {
                addAgentLog('success', `✅ Discovered ${suggestData.fields.length} fields`, `Fields: ${suggestData.fields.join(', ')}`);

                // Automatically add suggested fields
                const suggestedFields: SelectedField[] = suggestData.fields.map((fieldName: string) => {
                  // Try to find matching detected element
                  const matchingElement = previewData.detected_elements?.find(
                    (el: DetectedElement) => el.field?.toLowerCase() === fieldName.toLowerCase()
                  );

                  return {
                    name: fieldName,
                    selector: matchingElement?.selector || '', // Will be determined on click if not found
                    count: matchingElement?.count || 0,
                    sample: matchingElement?.sample_value || ''
                  };
                });

                setSelectedFields(suggestedFields);
                addAgentLog('info', '📝 Fields auto-populated - ready for extraction', 'Using LLM-based field detection');
              } else {
                addAgentLog('warning', '⚠️ No fields detected automatically', 'Try clicking elements in the browser to add fields manually');
              }
            } else {
              addAgentLog('error', '❌ Field discovery failed', 'Will use manual field selection');
            }
          } catch (suggestErr: any) {
            console.warn('Field suggestion failed, continuing without suggestions:', suggestErr);
            addAgentLog('error', '❌ Field discovery error', suggestErr?.message || 'Unknown error');
            // Don't fail navigation if suggestion fails
          }
        }
      } else {
        throw new Error(previewData.error || previewData.detail || 'Failed to load preview');
      }

    } catch (err: any) {
      console.error('Navigation failed:', err);
      const errorMsg = err.message || err.toString() || 'Failed to load page';
      setError(errorMsg);
    } finally {
      setIsLoading(false);
    }
  }, [userSettings?.webUnblockerConfig, userSettings?.proxyConfig, browserTimeout]);

  const normalizeUrl = (value: string): string | null => {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const withProtocol = /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed)
      ? trimmed
      : `https://${trimmed}`;
    try {
      return new URL(withProtocol).toString();
    } catch (error) {
      return null;
    }
  };

  // Handle URL submit
  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const normalizedUrl = normalizeUrl(url);
    if (!normalizedUrl) {
      setError('Please enter a valid URL');
      return;
    }
    if (normalizedUrl !== url) {
      setUrl(normalizedUrl);
    }
    navigate(normalizedUrl);
  };

  // Refresh page
  const handleRefresh = () => {
    if (currentUrl) {
      navigate(currentUrl);
    }
  };

  // Handle element click from iframe - automatically detect selector and add field
  useEffect(() => {
    const handleMessage = async (event: MessageEvent) => {
      // 1. Handle Navigation
      if (event.data && event.data.type === 'paradocs-navigate') {
        const newUrl = event.data.url;
        if (newUrl && newUrl !== currentUrl) {
          addAgentLog('info', '🧭 Navigating to linked page...', newUrl);
          setUrl(newUrl);
          navigate(newUrl);
        }
        return;
      }

      // 2. Handle element selection
      if (!event.data || event.data.type !== 'paradocs-element-selected') {
        return;
      }

      const { element } = event.data;
      if (!element || !element.selector) return;

      // Ask user for field name
      const fieldName = window.prompt(`Name this field (e.g. "product_name")`,
        element.text?.substring(0, 20).trim() || element.tagName?.toLowerCase() || 'field'
      );

      if (fieldName === null) return; // Cancelled

      const cleanFieldName = fieldName
        .replace(/[^a-zA-Z0-9\s]/g, '')
        .replace(/\s+/g, '_')
        .toLowerCase()
        .substring(0, 30) || 'field';

      // Check if field already exists (by selector)
      const existingField = selectedFields.find(f => f.selector === element.selector);
      if (existingField) {
        // Toggle off if already selected
        setSelectedFields(prev => prev.filter(f => f.selector !== element.selector));
        return;
      }

      // Count how many elements match this selector
      let count = 1;
      try {
        const matchingDetected = detectedElements.find(el => el.selector === element.selector);
        count = matchingDetected?.count || 1;
      } catch (e) {
        // Ignore
      }

      // Add the field
      setSelectedFields(prev => [...prev, {
        name: cleanFieldName,
        selector: element.selector,
        count: count,
        sample: element.text?.substring(0, 100) || ''
      }]);

      addAgentLog('success', `📌 Field added: ${cleanFieldName}`, `Selector: ${element.selector}`);
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [selectedFields, detectedElements]);

  // Remove field
  // Suggest fields from prompt (before navigation)
  const suggestFieldsFromPrompt = async () => {
    if (!url) {
      addAgentLog('warning', 'Please enter a URL first');
      return;
    }

    setIsSuggestingFields(true);
    addAgentLog('info', extractionTarget ? `💡 Analyzing page for ${extractionTarget}...` : '💡 Analyzing page structure for field suggestions...');

    try {
      const apiKey = localStorage.getItem('api_key');
      const token = await getAuthToken();

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };
      if (token) headers['Authorization'] = `Bearer ${token}`;
      if (apiKey) headers['X-API-Key'] = apiKey;

      const response = await fetch(`${API_BASE_URL}/api/v1/suggest-fields`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          url,
          target: extractionTarget || undefined,
          use_llm: true,
          proxy_config: proxyConfig.provider !== 'none' ? proxyConfig : undefined
        })
      });

      if (!response.ok) {
        throw new Error('Failed to suggest fields');
      }

      const data = await response.json();

      if (data.fields && data.fields.length > 0) {
        const suggestedFields = data.fields;
        addAgentLog('success', `✅ Found ${suggestedFields.length} suggested fields`, data.reasoning);

        // Populate extraction prompt based on target and fields
        const targetText = extractionTarget || 'items';
        const fieldsText = suggestedFields.join(', ');
        setExtractionPrompt(`Extract ALL ${targetText} with these fields: ${fieldsText}`);

        // Update selected fields
        const newFields: SelectedField[] = suggestedFields.map((name: string) => ({
          name,
          selector: '',
          count: 0,
          sample: ''
        }));
        setSelectedFields(newFields);
      }
    } catch (err: any) {
      console.error('Field suggestion failed:', err);
      addAgentLog('error', '❌ Failed to suggest fields', err.message);
    } finally {
      setIsSuggestingFields(false);
    }
  };

  // Run extraction
  const runExtraction = async () => {
    if (!currentUrl || selectedFields.length === 0) {
      setError('Please navigate to a URL and select fields to extract');
      addAgentLog('error', '❌ Extraction failed', 'No fields selected or URL missing');
      return;
    }

    setIsExtracting(true);
    setExtractionStatus('extracting');
    setError(null);
    const startTime = Date.now();
    addAgentLog('progress', '🚀 Starting extraction...', `Extracting ${selectedFields.length} fields`);

    try {
      const token = await getAuthToken();
      const apiKey = localStorage.getItem('api_key');

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };

      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }

      // Build proxy config for request - include Web Unblocker if enabled
      let requestProxyConfig: any = null;

      // Check if Web Unblocker is enabled in global settings
      const webUnblocker = userSettings?.webUnblockerConfig;
      if (webUnblocker?.enabled && webUnblocker?.apiKey) {
        // Send Web Unblocker config
        requestProxyConfig = {
          provider: 'web_unlocker',
          webUnblocker: {
            apiKey: webUnblocker.apiKey,
            zone: webUnblocker.zone || 'web_unlocker1',
          },
        };
      } else if (proxyConfig.provider !== 'none') {
        // Send regular proxy config
        requestProxyConfig = proxyConfig;
      }

      const response = await fetch(`${API_BASE_URL}/scrape`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          url: currentUrl,
          fields: selectedFields.map(f => f.name),
          target: extractionTarget || undefined,
          mode: 'hybrid',
          scroll_to_bottom: paginationConfig.scrollToBottom,
          click_load_more: paginationConfig.clickLoadMore,
          wait_for_selector: paginationConfig.waitForSelector,
          proxy_config: requestProxyConfig,
          browser_timeout: browserTimeout
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail?.error || errorData.detail || 'Extraction failed');
      }

      const data = await response.json();
      const endTime = Date.now();
      const elapsed = (endTime - startTime) / 1000;
      setExtractionTime(elapsed);

      if (!data.success || !data.data) {
        throw new Error(data.detail?.error || data.detail || 'No data returned');
      }

      const items = data.data || [];

      // Validate and clean extraction results (fix encoding issues)
      const cleanedItems = items.map((item: any) => {
        const cleaned: any = {};
        for (const [key, value] of Object.entries(item)) {
          if (value === null || value === undefined) {
            cleaned[key] = null;
          } else if (typeof value === 'string') {
            // More aggressive corruption detection
            const nonPrintable = (value.match(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]/g) || []).length;
            const totalChars = value.length;
            const corruptionRatio = totalChars > 0 ? nonPrintable / totalChars : 0;

            // Lower threshold - catch more corruption (5% instead of 10%)
            if (corruptionRatio > 0.05 && totalChars > 20) {
              // Definitely corrupted - mark as invalid
              cleaned[key] = null; // Set to null instead of showing corrupted data
            } else if (corruptionRatio > 0.01 && totalChars > 100) {
              // Possibly corrupted - try to clean it
              try {
                // Remove non-printable characters
                const cleanedValue = value.replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]/g, '');
                // If cleaned value is much shorter, it was mostly garbage
                if (cleanedValue.length < value.length * 0.5) {
                  cleaned[key] = null;
                } else {
                  cleaned[key] = cleanedValue.trim() || null;
                }
              } catch {
                cleaned[key] = null;
              }
            } else {
              // Clean normal strings - remove any stray non-printable chars
              cleaned[key] = value.replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]/g, '').trim() || null;
            }
          } else if (typeof value === 'object' && value.constructor === Object) {
            // Recursively clean nested objects
            cleaned[key] = Object.fromEntries(
              Object.entries(value).map(([k, v]) => {
                if (typeof v === 'string') {
                  const nonPrintable = (v.match(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]/g) || []).length;
                  const totalChars = v.length;
                  const corruptionRatio = totalChars > 0 ? nonPrintable / totalChars : 0;
                  if (corruptionRatio > 0.05 && totalChars > 20) {
                    return [k, null];
                  }
                  return [k, v.replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]/g, '').trim() || null];
                }
                return [k, v];
              })
            );
          } else {
            cleaned[key] = value;
          }
        }
        return cleaned;
      }).filter((item: any) => {
        // Filter out items that are completely corrupted (all fields are null or corrupted)
        const hasValidData = Object.values(item).some((v: any) =>
          v !== null && v !== undefined && v !== '' && v !== '[Invalid encoding - binary data detected]'
        );
        return hasValidData;
      });

      setExtractionResults(cleanedItems);

      // Determine extraction mode from response
      const source = data.source || data.metadata?.extraction_source || 'unknown';
      const metadata = data.metadata || {};
      const isCached = metadata.cache_hit || metadata.pattern_cache_hit || source.includes('cache');
      const isDirectLLM = source === 'direct_llm' || source === 'json+direct_llm' || metadata.cache_type === 'direct_llm';

      // Phase 2/3 metadata logging
      if (metadata.template_spec_used) {
        addAgentLog('success', '⚡ Template Spec Executed', `Deterministic extraction (<50ms) - Template ID: ${metadata.template_spec_id?.substring(0, 16) || 'unknown'}`, 'deterministic');
      } else if (metadata.template_spec_generated) {
        addAgentLog('info', '📋 Template Spec Generated', `New template created and cached for future use`, 'deterministic');
      }

      if (metadata.dom_digest_cache_hit) {
        addAgentLog('success', '🔍 DOM Digest Cache Hit', `Fast template matching (<10ms) - Page type: ${metadata.dom_digest_page_type || 'unknown'}`, 'cached');
      }

      if (metadata.model_tier_used) {
        const tierLabels: Record<string, string> = {
          'router': 'Router Tier (Classification)',
          'template': 'Template Tier (Generation)',
          'recovery': 'Recovery Tier (Fallback)'
        };
        addAgentLog('info', `🤖 Model Tier: ${tierLabels[metadata.model_tier_used] || metadata.model_tier_used}`, 'Optimized model selection based on task');
      }

      if (metadata.pattern_learned) {
        addAgentLog('success', '📚 Pattern Learned', `Pattern type: ${metadata.pattern_type || 'unknown'} - Future scrapes will be instant!`, 'cached');
      }

      if (metadata.selector_library_updated) {
        addAgentLog('info', '📖 Selector Library Updated', 'Site-specific selectors saved for faster future extraction');
      }

      if (metadata.early_exit) {
        addAgentLog('success', '⚡ Early Exit', 'High-quality extraction detected - skipped optimization steps');
      }

      if (isCached) {
        setExtractionStatus('cached');
        addAgentLog('success', '✅ Using cached pattern', `Extraction completed in ${elapsed.toFixed(1)}s`, 'cached');
      } else if (isDirectLLM) {
        setExtractionStatus('direct_llm');
        addAgentLog('success', '✅ Deterministic extraction complete', `Extracted ${items.length} items in ${elapsed.toFixed(1)}s`, 'deterministic');
        if (!metadata.pattern_learned) {
          addAgentLog('info', '💾 Pattern ready for caching', 'This pattern can be saved and reused');
        }
      } else {
        setExtractionStatus('idle');
        addAgentLog('success', '✅ Extraction complete', `Extracted ${items.length} items in ${elapsed.toFixed(1)}s`);
      }

      onExtract?.(items, selectedFields.map(f => f.name), data.metadata || {});

    } catch (err: any) {
      console.error('Extraction failed:', err);
      setError(err.message);
      setExtractionStatus('idle');
    } finally {
      setIsExtracting(false);
    }
  };

  useEffect(() => {
    if (triggerExtraction && triggerExtraction > 0) {
      runExtraction();
    }
  }, [triggerExtraction, runExtraction]);

  // Save pattern
  const savePattern = async () => {
    if (!patternName || selectedFields.length === 0) {
      setError('Please enter a pattern name and select fields');
      return;
    }

    try {
      const token = await getAuthToken();
      const domain = new URL(currentUrl).hostname;

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };

      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(`${API_BASE_URL}/api/v1/patterns/store`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          domain,
          fields: selectedFields.map(f => f.name),
          pattern_data: {
            name: patternName,
            selectors: selectedFields.reduce((acc, f) => ({ ...acc, [f.name]: f.selector }), {}),
            url: currentUrl,
            created_at: Date.now()
          },
          visibility: patternVisibility,
          url: currentUrl
        })
      });

      if (!response.ok) {
        throw new Error('Failed to save pattern');
      }

      setShowSavePatternModal(false);
      setPatternName('');
      onPatternSave?.({ name: patternName, fields: selectedFields, url: currentUrl });

    } catch (err: any) {
      console.error('Pattern save failed:', err);
      setError(err.message);
    }
  };

  // Load cached pattern
  const loadCachedPattern = (pattern: CachedPattern) => {
    if (pattern.url) {
      setUrl(pattern.url);
      navigate(pattern.url);
    }
    if (pattern.fields) {
      setSelectedFields(pattern.fields.map(f => ({
        name: f,
        selector: '',
        count: 0,
        sample: ''
      })));
    }
    setShowCacheMenu(false);
  };

  return (
    <div className={`flex flex-col h-full bg-gray-900 ${className}`}>
      {/* URL Bar */}
      <div className="flex items-center gap-2 p-3 bg-gray-800 border-b border-gray-700">
        {/* Navigation buttons */}
        <button
          onClick={() => window.history.back()}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
          title="Back"
        >
          <ArrowLeftIcon className="w-4 h-4" />
        </button>
        <button
          onClick={() => window.history.forward()}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
          title="Forward"
        >
          <ArrowRightIcon className="w-4 h-4" />
        </button>
        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
          title="Refresh"
        >
          <ArrowPathIcon className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>

        {/* URL and Prompt Input */}
        <form onSubmit={handleUrlSubmit} className="flex-1 flex items-center gap-2">
          <div className="relative flex-1">
            <input
              ref={urlInputRef}
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Enter URL to preview..."
              className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
            {isLoading && (
              <div className="absolute right-3 top-1/2 -translate-y-1/2">
                <div className="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              </div>
            )}
          </div>
          <div className="relative flex-[0.3] flex items-center gap-2">
            <input
              type="text"
              value={extractionTarget}
              onChange={(e) => setExtractionTarget(e.target.value)}
              placeholder="Target (e.g. products)"
              className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div className="relative flex-1 flex items-center gap-2">
            <input
              type="text"
              value={extractionPrompt}
              onChange={(e) => setExtractionPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && extractionPrompt.trim()) {
                  e.preventDefault();
                  runExtraction();
                }
              }}
              placeholder="What data to extract? (e.g., 'product name', 'price')"
              className="flex-1 px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
            <button
              type="button"
              onClick={suggestFieldsFromPrompt}
              disabled={isSuggestingFields}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium flex items-center gap-2 whitespace-nowrap"
              title="Generate field suggestions"
            >
              {isSuggestingFields ? (
                <>
                  <ArrowPathIcon className="w-4 h-4 animate-spin" />
                  Suggesting...
                </>
              ) : (
                <>
                  <SparklesIcon className="w-4 h-4" />
                  Suggest Fields
                </>
              )}
            </button>
          </div>
          <button
            type="submit"
            disabled={!url || isLoading}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
          >
            Navigate
          </button>
          <button
            type="button"
            onClick={runExtraction}
            disabled={!url || isExtracting || selectedFields.length === 0}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium flex items-center gap-2"
          >
            {isExtracting ? (
              <>
                <ArrowPathIcon className="w-4 h-4 animate-spin" />
                Extracting...
              </>
            ) : (
              <>
                <PlayIcon className="w-4 h-4" />
                Extract
              </>
            )}
          </button>
        </form>

        {/* Toolbar buttons */}
        <div className="flex items-center gap-1">
          {/* Options dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowOptionsMenu(!showOptionsMenu)}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
              title="Options"
            >
              <Cog6ToothIcon className="w-5 h-5" />
            </button>
            {showOptionsMenu && (
              <div className="absolute right-0 top-full mt-1 w-64 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50">
                <div className="p-3 space-y-3">
                  <h3 className="text-sm font-medium text-white">Browser Options</h3>
                  <label className="flex items-center gap-2 text-sm text-gray-300">
                    <input
                      type="checkbox"
                      checked={paginationConfig.scrollToBottom}
                      onChange={(e) => setPaginationConfig(prev => ({ ...prev, scrollToBottom: e.target.checked }))}
                      className="rounded bg-gray-700 border-gray-600"
                    />
                    Infinite scroll
                  </label>
                  <div>
                    <label className="text-xs text-gray-400">Timeout (ms)</label>
                    <input
                      type="number"
                      value={browserTimeout}
                      onChange={(e) => setBrowserTimeout(parseInt(e.target.value) || 60000)}
                      className="w-full mt-1 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Cache dropdown */}
          <div className="relative">
            <button
              onClick={() => setShowCacheMenu(!showCacheMenu)}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
              title="Cached Patterns"
            >
              <CircleStackIcon className="w-5 h-5" />
            </button>
            {showCacheMenu && (
              <div className="absolute right-0 top-full mt-1 w-80 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 max-h-96 overflow-auto">
                <div className="p-3">
                  <h3 className="text-sm font-medium text-white mb-2">Cached Patterns</h3>
                  {cachedPatterns.length === 0 ? (
                    <p className="text-sm text-gray-400">No cached patterns</p>
                  ) : (
                    <div className="space-y-2">
                      {cachedPatterns.map((pattern, idx) => (
                        <button
                          key={idx}
                          onClick={() => loadCachedPattern(pattern)}
                          className="w-full p-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-left transition-colors"
                        >
                          <div className="text-sm text-white font-medium truncate">{pattern.domain}</div>
                          <div className="text-xs text-gray-400 truncate">{pattern.fields?.join(', ')}</div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Proxy Settings Button - Opens inline modal */}
          <button
            onClick={() => setShowProxySettingsModal(true)}
            className={`p-2 rounded-lg transition-colors ${(userSettings?.webUnblockerConfig?.enabled && userSettings?.webUnblockerConfig?.apiKey) ||
              (proxyConfig.provider !== 'none')
              ? 'text-green-400 hover:text-green-300 hover:bg-gray-700'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            title="Proxy/Web Unblocker Settings"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </button>
        </div>
      </div>

      {/* Main Content Area - Sequentum-style: Browser top, Tabbed results bottom */}
      <div className="flex-1 grid grid-cols-[1fr_320px] overflow-hidden">
        {/* Browser View - Takes most of the space */}
        <div className="min-h-0 flex flex-col overflow-hidden bg-gray-950 min-w-0">
          {/* View Mode Tabs */}
          <div className="flex items-center gap-1 px-3 py-2 bg-gray-800/50 border-b border-gray-700">
            <button
              onClick={() => setViewMode('browser')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${viewMode === 'browser' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
            >
              <EyeIcon className="w-4 h-4 inline mr-1" />
              Browser
            </button>

            {/* Trigger Unblocker Button */}
            {userSettings?.webUnblockerConfig?.apiKey && (
              <button
                onClick={() => navigate(currentUrl, true)}
                disabled={isLoading}
                className="px-3 py-1.5 rounded-md text-sm font-medium text-amber-400 hover:text-amber-300 hover:bg-gray-700 transition-colors flex items-center gap-1 border border-amber-900/30 bg-amber-900/10"
                title="Force re-fetch using Web Unblocker to bypass blocks"
              >
                <ShieldCheckIcon className="w-4 h-4" />
                Trigger Unblocker
              </button>
            )}
            <button
              onClick={() => setViewMode('json')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors relative ${viewMode === 'json' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
            >
              <CodeBracketIcon className="w-4 h-4 inline mr-1" />
              JSON
              {jsonSources.length > 0 && (
                <span className="ml-1.5 px-1.5 py-0.5 bg-green-600 text-white text-xs rounded-full">
                  {jsonSources.filter((s: any) => s.usable).length}
                </span>
              )}
            </button>
            <button
              onClick={() => setViewMode('html')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${viewMode === 'html' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
            >
              <DocumentTextIcon className="w-4 h-4 inline mr-1" />
              HTML
            </button>

            <div className="flex-1" />

            {/* Status indicators */}
            {extractionStatus !== 'idle' && (
              <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${extractionStatus === 'cached' ? 'bg-green-900/50 text-green-400' :
                extractionStatus === 'direct_llm' ? 'bg-amber-900/50 text-amber-400' :
                  'bg-indigo-900/50 text-indigo-400'
                }`}>
                {extractionStatus === 'cached' && <CheckCircleIcon className="w-4 h-4" />}
                {extractionStatus === 'direct_llm' && <SparklesIcon className="w-4 h-4" />}
                {extractionStatus === 'extracting' && <ArrowPathIcon className="w-4 h-4 animate-spin" />}
                {extractionStatus === 'cached' && 'Cached Pattern'}
                {extractionStatus === 'direct_llm' && extractionMode === 'json' ? 'JSON Extraction' : 'Direct LLM'}
                {extractionStatus === 'extracting' && 'Extracting...'}
                {extractionTime && ` (${extractionTime.toFixed(1)}s)`}
              </div>
            )}
          </div>

          {/* Browser Content */}
          <div className="min-h-0 flex-1 relative overflow-hidden">
            {!currentUrl ? (
              <div className="flex items-center justify-center h-full text-gray-400">
                <div className="text-center">
                  <EyeIcon className="w-16 h-16 mx-auto mb-4 opacity-30" />
                  <h3 className="text-lg font-medium mb-2">Enter a URL to get started</h3>
                  <p className="text-sm text-gray-500">Navigate to any webpage to preview and extract data</p>
                </div>
              </div>
            ) : viewMode === 'browser' && (previewHtml || screenshot) ? (
              <div className="relative w-full h-full overflow-hidden">
                {previewHtml ? (
                  <>
                    <iframe
                      srcDoc={previewHtml}
                      title="Page Preview"
                      className="w-full h-full border-none bg-white"
                      sandbox="allow-scripts allow-same-origin"
                      onLoad={() => {
                        // Hide tip after iframe loads
                        setTimeout(() => setShowClickTip(false), 5000);
                      }}
                    />
                    {/* JSON Recommendation Banner */}
                    {showJsonRecommendation && jsonRecommended && (
                      <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-green-600 text-white px-6 py-3 rounded-lg shadow-xl z-20 max-w-2xl">
                        <div className="flex items-start gap-3">
                          <div className="flex-shrink-0 mt-0.5">
                            <SparklesIcon className="w-6 h-6" />
                          </div>
                          <div className="flex-1">
                            <h4 className="font-semibold mb-1">✨ JSON Data Detected!</h4>
                            <p className="text-sm mb-2">
                              Found {jsonSources.find(s => s.usable)?.count || 0} items in JSON format.
                              JSON extraction is faster and more reliable than browser scraping.
                            </p>
                            {jsonSources.find(s => s.usable)?.sample_fields && (
                              <div className="mb-2">
                                <p className="text-xs opacity-90 mb-1">Available fields:</p>
                                <div className="flex flex-wrap gap-1">
                                  {jsonSources.find(s => s.usable)?.sample_fields.slice(0, 8).map((field: string, idx: number) => (
                                    <span key={idx} className="px-2 py-0.5 bg-green-700 rounded text-xs">
                                      {field}
                                    </span>
                                  ))}
                                  {(jsonSources.find(s => s.usable)?.sample_fields.length || 0) > 8 && (
                                    <span className="px-2 py-0.5 bg-green-700 rounded text-xs">
                                      +{(jsonSources.find(s => s.usable)?.sample_fields.length || 0) - 8} more
                                    </span>
                                  )}
                                </div>
                              </div>
                            )}
                            <div className="flex items-center gap-2 mt-2">
                              <button
                                onClick={() => {
                                  setExtractionMode('json');
                                  setShowJsonRecommendation(false);
                                }}
                                className="px-3 py-1 bg-white text-green-600 rounded text-sm font-medium hover:bg-gray-100"
                              >
                                Use JSON Extraction
                              </button>
                              <button
                                onClick={() => {
                                  setExtractionMode('browser');
                                  setShowJsonRecommendation(false);
                                }}
                                className="px-3 py-1 bg-green-700 text-white rounded text-sm font-medium hover:bg-green-800"
                              >
                                Use Browser Instead
                              </button>
                              <button
                                onClick={() => setShowJsonRecommendation(false)}
                                className="ml-auto text-green-200 hover:text-white"
                              >
                                <XMarkIcon className="w-5 h-5" />
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Click Tip (only show if not using JSON) */}
                    {showClickTip && !jsonRecommended && (
                      <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-indigo-600 text-white px-4 py-2 rounded-lg shadow-xl z-10 flex items-center gap-2 animate-pulse">
                        <EyeIcon className="w-5 h-5" />
                        <span className="text-sm font-medium">Click any element to add it as a field</span>
                        <button
                          onClick={() => setShowClickTip(false)}
                          className="ml-2 hover:bg-indigo-700 rounded p-1"
                        >
                          <XMarkIcon className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                  </>
                ) : screenshot ? (
                  <img
                    src={`data:image/png;base64,${screenshot}`}
                    alt="Page Preview"
                    className="max-w-none"
                  />
                ) : null}
              </div>
            ) : viewMode === 'json' ? (
              <div className="h-full overflow-auto p-4">
                {jsonSources.length > 0 ? (
                  <div className="space-y-4">
                    {jsonSources.map((source, idx) => (
                      <div key={idx} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${source.usable
                              ? 'bg-green-600/20 text-green-400 border border-green-600/50'
                              : 'bg-gray-700 text-gray-400'
                              }`}>
                              {source.type}
                            </span>
                            {source.usable && (
                              <>
                                <span className="text-xs text-green-400">
                                  ✓ {source.count} items
                                </span>
                                {jsonRecommended && (
                                  <span className="text-xs text-amber-400">
                                    Recommended
                                  </span>
                                )}
                              </>
                            )}
                          </div>
                          {source.usable && (
                            <button
                              onClick={() => {
                                setExtractionMode('json');
                                if (source.sample_fields) {
                                  setSelectedFields(source.sample_fields.map((f: string) => ({
                                    name: f,
                                    selector: '',
                                    count: source.count || 0,
                                    sample: ''
                                  })));
                                }
                              }}
                              className="px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700"
                            >
                              Use This JSON
                            </button>
                          )}
                        </div>
                        {source.sample_fields && source.sample_fields.length > 0 && (
                          <div className="mb-2">
                            <p className="text-xs text-gray-400 mb-1">Fields:</p>
                            <div className="flex flex-wrap gap-1">
                              {source.sample_fields.map((field: string, fIdx: number) => (
                                <span key={fIdx} className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300">
                                  {field}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        <pre className="text-xs text-gray-400 font-mono bg-gray-900 p-3 rounded overflow-auto max-h-64">
                          {source.preview}
                        </pre>
                      </div>
                    ))}
                    {extractionResults.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-700">
                        <h4 className="text-sm font-medium text-white mb-2">Extraction Results:</h4>
                        <pre className="text-sm text-gray-300 font-mono">
                          {JSON.stringify(extractionResults, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center text-gray-400">
                      <CodeBracketIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
                      <p>No JSON sources detected</p>
                      <p className="text-sm mt-1">This page uses browser-rendered content</p>
                    </div>
                  </div>
                )}
              </div>
            ) : viewMode === 'html' ? (
              <div className="h-full overflow-auto p-4">
                {previewHtml ? (
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-medium text-white">Raw HTML ({previewHtml.length.toLocaleString()} bytes)</h3>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(previewHtml);
                        }}
                        className="px-3 py-1 bg-gray-700 text-white text-xs rounded hover:bg-gray-600"
                      >
                        Copy HTML
                      </button>
                    </div>
                    <pre className="text-xs text-gray-300 font-mono bg-gray-900 p-4 rounded overflow-auto border border-gray-700 max-h-full whitespace-pre-wrap break-words">
                      {previewHtml}
                    </pre>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-gray-400 text-sm">No HTML available. Navigate to a page first.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full">
                <ArrowPathIcon className="w-8 h-8 text-indigo-400 animate-spin" />
              </div>
            )}

            {/* Browser rendering failed warning */}
            {browserRenderingFailed && (
              <div className="absolute inset-x-0 top-0 p-4 bg-amber-900/90 text-amber-200 z-30">
                <div className="flex items-start gap-2">
                  <ExclamationTriangleIcon className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-medium">⚠️ Browser Rendering Unavailable</p>
                    <p className="text-sm">
                      This page requires JavaScript to render fully, but browser rendering failed.
                      You're seeing the static HTML version, which may be incomplete.
                      {fallbackReason && (
                        <span className="block mt-1 text-xs opacity-75">Reason: {fallbackReason}</span>
                      )}
                    </p>
                  </div>
                  <button onClick={() => setBrowserRenderingFailed(false)} className="text-amber-300 hover:text-white">
                    <XMarkIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
            )}

            {/* Error overlay */}
            {error && (
              <div className={`absolute inset-x-0 ${browserRenderingFailed ? 'top-24' : 'top-0'} p-4 bg-red-900/90 text-red-200 z-20`}>
                <div className="flex items-start gap-2">
                  <ExclamationTriangleIcon className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-medium">Error</p>
                    <p className="text-sm">{error}</p>
                  </div>
                  <button onClick={() => setError(null)} className="text-red-300 hover:text-white">
                    <XMarkIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
            )}

          </div>
        </div>

        {/* Right Sidebar: Agent Log Only */}
        <div className="min-h-0 bg-gray-900 border-l border-gray-700 flex flex-col overflow-hidden" data-agent-log>
          {/* Agent Log Header */}
          <div className="p-3 border-b border-gray-700 bg-gray-800">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-white">Agent Log</h3>
                <p className="text-xs text-gray-400 mt-0.5">Live scan progression</p>
              </div>
              <div className="flex gap-1">
                {(extractionStatus === 'idle' || extractionStatus === 'extracting') && (
                  <span className="px-1.5 py-0.5 bg-purple-500/10 text-purple-400 text-[10px] rounded border border-purple-500/20">
                    Probabilistic
                  </span>
                )}
                {extractionStatus === 'cached' && (
                  <span className="px-1.5 py-0.5 bg-green-500/10 text-green-400 text-[10px] rounded border border-green-500/20">
                    Deterministic
                  </span>
                )}
                {extractionStatus === 'direct_llm' && (
                  <span className="px-1.5 py-0.5 bg-blue-500/10 text-blue-400 text-[10px] rounded border border-blue-500/20">
                    Probabilistic
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Log Content */}
          <div className="flex-1 overflow-y-auto p-3 space-y-2 bg-gray-900/50">
            {agentLogs.length === 0 ? (
              <div className="text-center py-12 px-4">
                <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-gray-800 flex items-center justify-center">
                  <SparklesIcon className="w-6 h-6 text-gray-600" />
                </div>
                <p className="text-sm text-gray-400 font-medium">No activity yet</p>
                <p className="text-xs text-gray-500 mt-1">Navigate to a URL to start the agent</p>
              </div>
            ) : (
              agentLogs.map((log) => {
                const levelStyles = {
                  info: 'border-blue-500/30 bg-blue-500/5 text-blue-200',
                  success: 'border-green-500/30 bg-green-500/5 text-green-200',
                  warning: 'border-amber-500/30 bg-amber-500/5 text-amber-200',
                  error: 'border-red-500/30 bg-red-500/5 text-red-200',
                  progress: 'border-purple-500/30 bg-purple-500/5 text-purple-200'
                };

                return (
                  <div
                    key={log.id}
                    className={`p-2.5 rounded border-l-2 text-xs ${levelStyles[log.level]} transition-all animate-in fade-in slide-in-from-right-4 duration-300`}
                  >
                    <div className="flex justify-between items-start gap-2">
                      <span className="font-medium leading-relaxed">
                        {log.message}
                      </span>
                      <span className="text-[10px] opacity-50 whitespace-nowrap mt-0.5">
                        {log.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                      </span>
                    </div>
                    {log.details && (
                      <p className="mt-1.5 opacity-70 text-[11px] font-mono bg-black/20 p-1.5 rounded border border-white/5 overflow-x-auto">
                        {log.details}
                      </p>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* Proxy Settings Modal */}
      {
        showProxySettingsModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowProxySettingsModal(false)}>
            <div className="bg-gray-800 rounded-xl p-6 w-96 max-w-full mx-4" onClick={(e) => e.stopPropagation()}>
              <h3 className="text-lg font-semibold text-white mb-4">Proxy & Web Unblocker Settings</h3>
              <p className="text-xs text-gray-400 mb-4">
                These settings are loaded from your global settings. Override them here for this session only.
              </p>

              <div className="space-y-4">
                {/* Web Unblocker */}
                <div>
                  <label className="text-sm text-gray-400 flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={localProxyConfig.provider === 'web_unlocker' || (userSettings?.webUnblockerConfig?.enabled && userSettings?.webUnblockerConfig?.apiKey)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setLocalProxyConfig({ provider: 'web_unlocker', webUnblocker: userSettings?.webUnblockerConfig });
                        } else {
                          setLocalProxyConfig({ provider: 'none' });
                        }
                      }}
                      className="rounded"
                    />
                    <span>Web Unblocker</span>
                    {userSettings?.webUnblockerConfig?.enabled && userSettings?.webUnblockerConfig?.apiKey && (
                      <span className="text-xs text-green-400">(Active)</span>
                    )}
                  </label>
                  {((localProxyConfig.provider === 'web_unlocker') || (userSettings?.webUnblockerConfig?.enabled && userSettings?.webUnblockerConfig?.apiKey)) && (
                    <div className="mt-2 ml-6 text-xs text-gray-400">
                      Zone: {userSettings?.webUnblockerConfig?.zone || 'web_unlocker1'}
                    </div>
                  )}
                </div>

                {/* Proxy Provider */}
                <div>
                  <label className="text-sm text-gray-400">Proxy Provider</label>
                  <select
                    value={localProxyConfig.provider === 'web_unlocker' ? 'none' : (localProxyConfig.provider || 'none')}
                    onChange={(e) => {
                      const provider = e.target.value as any;
                      if (provider === 'none') {
                        setLocalProxyConfig({ provider: 'none' });
                      } else {
                        setLocalProxyConfig({
                          provider,
                          externalProxy: userSettings?.proxyConfig?.externalProxy || localProxyConfig.externalProxy
                        });
                      }
                    }}
                    className="w-full mt-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="none">No Proxy</option>
                    <option value="brightdata">Bright Data</option>
                    <option value="oxylabs">Oxylabs</option>
                    <option value="scraperapi">ScraperAPI</option>
                    <option value="custom">Custom Proxy</option>
                  </select>
                </div>

                {localProxyConfig.provider !== 'none' && localProxyConfig.provider !== 'web_unlocker' && (
                  <div className="text-xs text-gray-400 bg-gray-700/50 p-3 rounded">
                    <p>Using global proxy settings:</p>
                    {userSettings?.proxyConfig?.externalProxy?.server && (
                      <p>Server: {userSettings.proxyConfig.externalProxy.server}</p>
                    )}
                  </div>
                )}
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => {
                    setLocalProxyConfig({ provider: 'none' });
                    setShowProxySettingsModal(false);
                  }}
                  className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    setProxyConfig(localProxyConfig);
                    setShowProxySettingsModal(false);
                  }}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  Apply
                </button>
              </div>
            </div>
          </div>
        )
      }

      {/* Save Pattern Modal */}
      {
        showSavePatternModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-xl p-6 w-96 max-w-full mx-4">
              <h3 className="text-lg font-semibold text-white mb-4">Save Extraction Pattern</h3>

              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-400">Pattern Name</label>
                  <input
                    type="text"
                    value={patternName}
                    onChange={(e) => setPatternName(e.target.value)}
                    placeholder="e.g., Product Listings"
                    className="w-full mt-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-400">Visibility</label>
                  <select
                    value={patternVisibility}
                    onChange={(e) => setPatternVisibility(e.target.value as 'private' | 'public')}
                    className="w-full mt-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="private">Private (only you)</option>
                    <option value="public">Public (share with community)</option>
                  </select>
                </div>

                <div className="bg-gray-700 rounded-lg p-3">
                  <p className="text-xs text-gray-400 mb-2">Fields to save:</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedFields.map((f, idx) => (
                      <span key={idx} className="px-2 py-1 bg-indigo-600/20 text-indigo-400 text-xs rounded">
                        {f.name}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => setShowSavePatternModal(false)}
                  className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={savePattern}
                  disabled={!patternName}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                >
                  Save Pattern
                </button>
              </div>
            </div>
          </div>
        )
      }
    </div >
  );
}
