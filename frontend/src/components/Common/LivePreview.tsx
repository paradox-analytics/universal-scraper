import { useState, useEffect, useRef, useCallback } from 'react';
import {
  EyeIcon,
  CodeBracketIcon,
  DocumentTextIcon,
  ArrowPathIcon,
  XMarkIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CheckCircleIcon,
  PlusCircleIcon
} from '@heroicons/react/24/outline';
import { API_BASE_URL } from '../../config/api';

interface DetectedElement {
  field: string;
  selector: string;
  sample_value?: string;
  count: number;
}

interface JsonSource {
  type: string;
  preview: string;
}

interface PageInfo {
  title: string;
  url: string;
  html_size: number;
  element_count: number;
  has_json: boolean;
}

interface SelectedElement {
  selector: string;
  text: string;
  tagName: string;
}

interface LivePreviewProps {
  url: string;
  proxyConfig?: any;
  browserTimeout?: number;
  onFieldSelect?: (field: string, selector: string) => void;
  onElementsSelected?: (elements: SelectedElement[]) => void;
  suggestedFields?: string[];
  className?: string;
}

type ViewMode = 'browser' | 'json' | 'elements';

export default function LivePreview({
  url,
  proxyConfig,
  browserTimeout = 60000,
  onFieldSelect,
  onElementsSelected,
  suggestedFields = [],
  className = ''
}: LivePreviewProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('browser');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [html, setHtml] = useState<string | null>(null);
  const [detectedElements, setDetectedElements] = useState<DetectedElement[]>([]);
  const [jsonSources, setJsonSources] = useState<JsonSource[]>([]);
  const [pageInfo, setPageInfo] = useState<PageInfo | null>(null);
  const [selectedElements, setSelectedElements] = useState<SelectedElement[]>([]);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['containers', 'fields']));
  const iframeRef = useRef<HTMLIFrameElement>(null);

  // Fetch preview when URL changes
  const fetchPreview = useCallback(async () => {
    if (!url) return;

    setLoading(true);
    setError(null);

    try {
      // Use the imported API_BASE_URL

      const response = await fetch(`${API_BASE_URL}/api/v1/preview`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url,
          fields: suggestedFields,
          proxy_config: proxyConfig,
          browser_timeout: browserTimeout
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch preview');
      }

      const data = await response.json();
      setHtml(data.html);
      setDetectedElements(data.detected_elements || []);
      setJsonSources(data.json_sources || []);
      setPageInfo(data.page_info);

    } catch (err: any) {
      setError(err.message || 'Failed to load preview');
    } finally {
      setLoading(false);
    }
  }, [url, proxyConfig, browserTimeout, suggestedFields]);

  // Listen for messages from iframe
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data.type === 'paradocs-element-selected') {
        setSelectedElements(event.data.allSelected || []);
        onElementsSelected?.(event.data.allSelected || []);
      } else if (event.data.type === 'paradocs-navigate') {
        // Handle navigation from within iframe
        console.log('Paradocs: Navigation requested to:', event.data.url);
        // We need to update the parent URL state. 
        // Note: the parent component should handle the URL state update to trigger re-fetch
        // For now, we can try to trigger a re-fetch if we have the update function
        if (event.data.url) {
          // We'll update the local URL if possible, but usually this is controlled by props
          // For a smoother experience, we'll suggest the parent update the URL
          window.dispatchEvent(new CustomEvent('paradocs-url-change', { detail: { url: event.data.url } }));
        }
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [onElementsSelected]);

  // Toggle section expansion
  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  };

  // Add element as field
  const handleAddAsField = (element: DetectedElement | SelectedElement, fieldName?: string) => {
    const selector = 'selector' in element ? element.selector : (element as any).selector;
    const name = fieldName || ('field' in element ? element.field : '');
    onFieldSelect?.(name, selector);
  };

  // Highlight selector in iframe
  const highlightSelector = (selector: string) => {
    if (iframeRef.current?.contentWindow) {
      iframeRef.current.contentWindow.postMessage({
        type: 'paradocs-highlight-selector',
        selector
      }, '*');
    }
  };

  // Group detected elements
  const containerElements = detectedElements.filter(e => e.field.startsWith('Container:'));
  const fieldElements = detectedElements.filter(e => !e.field.startsWith('Container:'));

  return (
    <div className={`bg-gray-900 rounded-xl border border-gray-700 overflow-hidden flex flex-col ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-800/50 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <EyeIcon className="w-5 h-5 text-indigo-400" />
          <span className="font-medium text-white">Live Preview</span>
          {pageInfo && (
            <span className="text-xs text-gray-400 ml-2">
              {pageInfo.title?.substring(0, 30)}...
            </span>
          )}
        </div>

        {/* View Mode Tabs */}
        <div className="flex items-center gap-1 bg-gray-900 rounded-lg p-1">
          <button
            onClick={() => setViewMode('browser')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${viewMode === 'browser'
              ? 'bg-indigo-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
          >
            <EyeIcon className="w-4 h-4 inline mr-1" />
            Browser
          </button>
          <button
            onClick={() => setViewMode('json')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${viewMode === 'json'
              ? 'bg-indigo-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
          >
            <CodeBracketIcon className="w-4 h-4 inline mr-1" />
            JSON
          </button>
          <button
            onClick={() => setViewMode('elements')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${viewMode === 'elements'
              ? 'bg-indigo-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
          >
            <DocumentTextIcon className="w-4 h-4 inline mr-1" />
            Elements
          </button>
        </div>

        {/* Refresh Button */}
        <button
          onClick={fetchPreview}
          disabled={loading || !url}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
        >
          <ArrowPathIcon className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {!url ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <EyeIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>Enter a URL to preview</p>
            </div>
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <ArrowPathIcon className="w-8 h-8 mx-auto mb-3 text-indigo-400 animate-spin" />
              <p className="text-gray-400">Loading preview...</p>
            </div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full text-red-400">
            <div className="text-center">
              <XMarkIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>{error}</p>
              <button
                onClick={fetchPreview}
                className="mt-3 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
              >
                Retry
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Browser View */}
            {viewMode === 'browser' && html && (
              <div className="h-full">
                <iframe
                  ref={iframeRef}
                  srcDoc={html}
                  className="w-full h-full bg-white"
                  sandbox="allow-scripts allow-same-origin"
                  title="Page Preview"
                />
              </div>
            )}

            {/* JSON View */}
            {viewMode === 'json' && (
              <div className="h-full overflow-auto p-4">
                {jsonSources.length === 0 ? (
                  <div className="text-center text-gray-400 py-8">
                    <CodeBracketIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No JSON sources detected</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {jsonSources.map((source, idx) => (
                      <div key={idx} className="bg-gray-800 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="px-2 py-1 bg-indigo-600/20 text-indigo-400 text-xs rounded">
                            {source.type}
                          </span>
                        </div>
                        <pre className="text-xs text-gray-300 overflow-auto max-h-64 bg-gray-900 p-3 rounded">
                          {source.preview}
                        </pre>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Elements View */}
            {viewMode === 'elements' && (
              <div className="h-full overflow-auto p-4 space-y-4">
                {/* Selected Elements */}
                {selectedElements.length > 0 && (
                  <div className="bg-green-900/20 border border-green-700 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-green-400 mb-2 flex items-center gap-2">
                      <CheckCircleIcon className="w-4 h-4" />
                      Selected Elements ({selectedElements.length})
                    </h3>
                    <div className="space-y-2">
                      {selectedElements.map((el, idx) => (
                        <div key={idx} className="flex items-center justify-between bg-gray-800 rounded p-2">
                          <div>
                            <code className="text-xs text-green-400">{el.selector}</code>
                            <p className="text-xs text-gray-400 truncate max-w-xs">{el.text}</p>
                          </div>
                          <button
                            onClick={() => onFieldSelect?.(el.tagName.toLowerCase(), el.selector)}
                            className="px-2 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700"
                          >
                            Add Field
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Container Elements */}
                <div className="bg-gray-800 rounded-lg overflow-hidden">
                  <button
                    onClick={() => toggleSection('containers')}
                    className="w-full flex items-center justify-between p-3 hover:bg-gray-700"
                  >
                    <span className="font-medium text-amber-400 flex items-center gap-2">
                      {expandedSections.has('containers') ? (
                        <ChevronDownIcon className="w-4 h-4" />
                      ) : (
                        <ChevronRightIcon className="w-4 h-4" />
                      )}
                      Item Containers ({containerElements.length})
                    </span>
                  </button>
                  {expandedSections.has('containers') && (
                    <div className="p-3 pt-0 space-y-2">
                      {containerElements.map((el, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between bg-gray-900 rounded p-2 hover:bg-gray-700 cursor-pointer"
                          onClick={() => highlightSelector(el.selector)}
                        >
                          <div>
                            <span className="text-sm text-white">{el.field.replace('Container: ', '')}</span>
                            <code className="block text-xs text-gray-400">{el.selector}</code>
                            <span className="text-xs text-amber-400">{el.count} items found</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Field Elements */}
                <div className="bg-gray-800 rounded-lg overflow-hidden">
                  <button
                    onClick={() => toggleSection('fields')}
                    className="w-full flex items-center justify-between p-3 hover:bg-gray-700"
                  >
                    <span className="font-medium text-indigo-400 flex items-center gap-2">
                      {expandedSections.has('fields') ? (
                        <ChevronDownIcon className="w-4 h-4" />
                      ) : (
                        <ChevronRightIcon className="w-4 h-4" />
                      )}
                      Detected Fields ({fieldElements.length})
                    </span>
                  </button>
                  {expandedSections.has('fields') && (
                    <div className="p-3 pt-0 space-y-2">
                      {fieldElements.map((el, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between bg-gray-900 rounded p-2 hover:bg-gray-700"
                        >
                          <div className="flex-1 min-w-0">
                            <span className="text-sm text-white capitalize">{el.field}</span>
                            <code className="block text-xs text-gray-400 truncate">{el.selector}</code>
                            {el.sample_value && (
                              <p className="text-xs text-gray-500 truncate">{el.sample_value}</p>
                            )}
                          </div>
                          <button
                            onClick={() => handleAddAsField(el)}
                            className="ml-2 p-1 text-indigo-400 hover:text-indigo-300 hover:bg-indigo-900/30 rounded"
                            title="Add as field"
                          >
                            <PlusCircleIcon className="w-5 h-5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Page Info */}
                {pageInfo && (
                  <div className="bg-gray-800 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-300 mb-2">Page Info</h3>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-gray-500">Size:</span>
                        <span className="text-gray-300 ml-2">{(pageInfo.html_size / 1024).toFixed(1)} KB</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Elements:</span>
                        <span className="text-gray-300 ml-2">{pageInfo.element_count}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Has JSON:</span>
                        <span className={`ml-2 ${pageInfo.has_json ? 'text-green-400' : 'text-gray-400'}`}>
                          {pageInfo.has_json ? 'Yes' : 'No'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Footer with instructions */}
      {viewMode === 'browser' && html && (
        <div className="px-4 py-2 bg-gray-800/50 border-t border-gray-700">
          <p className="text-xs text-gray-400">
            💡 Click on highlighted elements to select them as fields. Selected elements will appear in the Elements tab.
          </p>
        </div>
      )}
    </div>
  );
}

