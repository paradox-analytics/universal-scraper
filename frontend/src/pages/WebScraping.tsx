import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AgentBuilderLayout } from '../components/agent/layouts/AgentBuilderLayout';
import { AgentToolbar } from '../components/agent/shared/AgentToolbar';
import { BottomDock } from '../components/agent/shared/BottomDock';
import { BrowserWorkspace } from '../components/BrowserWorkspace';
import { UnsavedChangesModal } from '../components/agent/shared/UnsavedChangesModal';
import { ExtractionFlowIndicator } from '../components/agent/shared/ExtractionFlowIndicator';
import { useAgentDraft, useBeforeUnload } from '../hooks/useAgentDraft';
import { patternApi } from '../services/api';
import type { CachedPattern, ScraperAgent } from '../types';
import {
  TableCellsIcon,
  CursorArrowRaysIcon,
  CodeBracketSquareIcon,
  EyeIcon,
  ClockIcon,
  DocumentTextIcon,
} from '@heroicons/react/24/outline';

export default function WebScraping() {
  const navigate = useNavigate();
  const [cachedPatterns, setCachedPatterns] = useState<CachedPattern[]>([]);
  const [extractionResults, setExtractionResults] = useState<any[]>([]);
  const [bottomDockCollapsed, setBottomDockCollapsed] = useState(false);
  const [selectedFields, setSelectedFields] = useState<string[]>([]);
  const [extractionMetadata, setExtractionMetadata] = useState<any>(null);
  const [triggerExtraction, setTriggerExtraction] = useState(0);
  const [showUnsavedModal, setShowUnsavedModal] = useState(false);
  const [pendingNavigation, setPendingNavigation] = useState<string | null>(null);
  const [agentLogs, setAgentLogs] = useState<any[]>([]);

  // Draft management
  const draftId = `scraper-draft-${Date.now()}`;
  const {
    hasUnsavedChanges,
    lastSaved,
    autoSaveEnabled,
    saveDraft,
    markDirty,
    clearDirty,
  } = useAgentDraft(draftId);

  // Prompt before closing/refreshing with unsaved changes
  useBeforeUnload(hasUnsavedChanges);

  useEffect(() => {
    loadCachedPatterns();
  }, []);

  const loadCachedPatterns = async () => {
    try {
      let allPatterns: CachedPattern[] = [];

      // Load user's private patterns
      try {
        const response = await patternApi.listMyPatterns();
        if (response.data && response.data.success) {
          const scrapingPatterns = (response.data.patterns || []).filter((p: any) => p.url);
          allPatterns = [...scrapingPatterns];
        }
      } catch (e) {
        console.error('Failed to load user patterns:', e);
      }

      // Also load public patterns
      try {
        const publicResponse = await patternApi.listPublicPatterns();
        if (publicResponse.data && publicResponse.data.success) {
          const publicScrapingPatterns = (publicResponse.data.patterns || []).filter((p: any) => p.url);
          const existing = new Set(allPatterns.map(p => `${p.domain}-${p.fields_hash}`));
          const newPatterns = publicScrapingPatterns.filter((p: any) =>
            !existing.has(`${p.domain}-${p.fields_hash}`)
          );
          allPatterns = [...allPatterns, ...newPatterns];
        }
      } catch (e) {
        console.error('Failed to load public patterns:', e);
      }

      setCachedPatterns(allPatterns);
    } catch (err) {
      console.error('Failed to load cached patterns:', err);
      setCachedPatterns([]);
    }
  };

  const handleExtract = (results: any[], fields: string[], metadata?: any) => {
    setExtractionResults(results);
    setSelectedFields(fields);
    setExtractionMetadata(metadata);
    markDirty(); // Mark as having unsaved changes
  };

  const handleLog = (message: string, level: 'info' | 'success' | 'warning' | 'error' | 'progress' = 'info', details?: string) => {
    const newLog = {
      id: Math.random().toString(36).substring(7),
      timestamp: new Date(),
      message,
      level,
      details
    };
    setAgentLogs(prev => [...prev, newLog]);
  };

  const downloadPythonCode = async () => {
    if (!extractionMetadata?.template_spec) {
      handleLog('No extraction pattern available to generate code', 'warning');
      return;
    }

    try {
      handleLog('Generating Python code...', 'info');
      const response = await patternApi.generatePythonCode({
        url: extractionMetadata.url || '',
        fields: selectedFields,
        selectors: extractionMetadata.template_spec.selectors,
        target: extractionMetadata.target
      });

      if (response.data && response.data.success && response.data.code) {
        const blob = new Blob([response.data.code], { type: 'text/x-python' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = response.data.filename || 'scraper.py';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        handleLog('Python code downloaded successfully', 'success');
      } else {
        throw new Error('Failed to generate code');
      }
    } catch (err: any) {
      console.error('Code generation failed:', err);
      handleLog(`Failed to generate code: ${err.message}`, 'error');
    }
  };

  const handleSave = () => {
    const agent: Partial<ScraperAgent> = {
      id: draftId,
      name: 'New Web Scraper',
      type: 'SCRAPER',
      definition: {
        subType: 'web_scraping',
        fields: selectedFields,
        mode: 'hybrid',
      },
    };
    saveDraft(agent);
  };

  const handleNavigation = (to: string) => {
    if (hasUnsavedChanges) {
      setPendingNavigation(to);
      setShowUnsavedModal(true);
    } else {
      navigate(to);
    }
  };

  const handleSaveAndNavigate = () => {
    handleSave();
    if (pendingNavigation) {
      navigate(pendingNavigation);
    }
    setShowUnsavedModal(false);
  };

  const handleDiscardAndNavigate = () => {
    clearDirty();
    if (pendingNavigation) {
      navigate(pendingNavigation);
    }
    setShowUnsavedModal(false);
  };

  const handleCancelNavigation = () => {
    setPendingNavigation(null);
    setShowUnsavedModal(false);
  };

  // Will be used for Router-level navigation guards later
  console.log('Navigation handler ready:', handleNavigation);

  // Create a mock ScraperAgent for the toolbar
  const mockAgent: ScraperAgent = {
    id: 'new',
    name: 'New Web Scraper',
    type: 'SCRAPER',
    status: 'pending',
    tenant_id: 'local',
    created_at: new Date().toISOString(),
    definition: {
      subType: 'web_scraping',
      fields: selectedFields,
      mode: 'hybrid',
    },
  };

  // Bottom dock tabs
  const bottomTabs = [
    {
      id: 'extracted-content',
      label: 'Extracted Content',
      icon: <TableCellsIcon className="w-4 h-4" />,
      badge: extractionResults.length > 0 ? extractionResults.length : undefined,
      content: (
        <div className="p-4 overflow-auto h-full">
          {extractionResults.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-gray-850 sticky top-0">
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-2 px-3 text-gray-400 font-medium w-16">#</th>
                    {Object.keys(extractionResults[0] || {}).map((key) => (
                      <th key={key} className="text-left py-2 px-3 text-gray-400 font-medium">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {extractionResults.map((item, idx) => (
                    <tr key={idx} className="border-b border-gray-800 hover:bg-gray-800/30 transition-colors">
                      <td className="py-2 px-3 text-gray-500 font-mono">{idx + 1}</td>
                      {Object.values(item).map((value: any, vIdx) => (
                        <td key={vIdx} className="py-2 px-3 text-gray-300 max-w-md truncate">
                          {value !== null && value !== undefined
                            ? String(value)
                            : <span className="text-gray-600 italic">null</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <TableCellsIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p>No extracted data yet</p>
                <p className="text-sm mt-1">Enter a URL and fields to start scraping</p>
              </div>
            </div>
          )}
        </div>
      ),
    },
    {
      id: 'selection',
      label: 'Selection',
      icon: <CursorArrowRaysIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full text-gray-400">
          {extractionMetadata?.template_spec?.selectors ? (
            <div className="space-y-4">
              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-2">Deterministic Selectors</h3>
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(extractionMetadata.template_spec.selectors).map(([field, selector]: [string, any]) => (
                    <div key={field} className="flex flex-col gap-1">
                      <span className="text-xs text-gray-500 uppercase tracking-wider">{field}</span>
                      <code className="text-xs text-indigo-400 bg-indigo-900/20 px-2 py-1 rounded border border-indigo-500/20 truncate">
                        {typeof selector === 'string' ? selector : selector.selector}
                      </code>
                    </div>
                  ))}
                </div>
              </div>
              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-2">Extraction Rules</h3>
                <pre className="text-xs text-gray-400 font-mono">
                  {JSON.stringify(extractionMetadata.template_spec.rules || {}, null, 2)}
                </pre>
              </div>
            </div>
          ) : (
            <p className="text-sm italic">Run an extraction to see deterministic selectors and rules</p>
          )}
        </div>
      ),
    },
    {
      id: 'schema',
      label: 'Schema',
      icon: <CodeBracketSquareIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full text-gray-400">
          {extractionMetadata?.template_spec ? (
            <div className="bg-gray-900 rounded-lg p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-sm font-medium text-gray-300">Template Schema</h3>
                <button
                  onClick={downloadPythonCode}
                  className="px-3 py-1.5 bg-indigo-600 text-white text-xs rounded hover:bg-indigo-700 transition-colors flex items-center gap-2"
                >
                  <CodeBracketSquareIcon className="w-4 h-4" />
                  Download Python Code
                </button>
              </div>
              <pre className="text-xs text-gray-400 font-mono">
                {JSON.stringify({
                  fields: Object.keys(extractionMetadata.template_spec.selectors || {}),
                  version: extractionMetadata.template_spec.version || '1.0',
                  id: extractionMetadata.template_spec_id
                }, null, 2)}
              </pre>
            </div>
          ) : (
            <p className="text-sm italic">Schema will be generated after extraction</p>
          )}
        </div>
      ),
    },
    {
      id: 'data-preview',
      label: 'Data Preview',
      icon: <EyeIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full">
          {extractionResults.length > 0 ? (
            <pre className="text-xs text-gray-300 font-mono bg-gray-900 p-3 rounded overflow-auto">
              {JSON.stringify(extractionResults, null, 2)}
            </pre>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <p className="text-sm">No data to preview</p>
            </div>
          )}
        </div>
      ),
    },
    {
      id: 'activity',
      label: 'Activity',
      icon: <ClockIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full">
          {extractionMetadata ? (
            <ExtractionFlowIndicator
              currentStep={
                extractionMetadata.dom_digest_cache_hit || extractionMetadata.template_spec_used
                  ? 'deterministic'
                  : extractionMetadata.template_spec_generated
                    ? 'template'
                    : 'llm'
              }
              cacheHit={extractionMetadata.dom_digest_cache_hit || extractionMetadata.template_spec_used}
              templateId={extractionMetadata.template_spec_id}
              llmTokensUsed={extractionMetadata.llm_tokens_used || 0}
            />
          ) : (
            <div className="p-4 text-gray-400 text-sm">
              <p>Run an extraction to see the flow</p>
            </div>
          )}
        </div>
      ),
    },
    {
      id: 'logs',
      label: 'Logs',
      icon: <DocumentTextIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full">
          <div className="space-y-2">
            {agentLogs.length > 0 ? (
              agentLogs.map((log) => (
                <div key={log.id} className="text-xs font-mono border-l-2 border-gray-700 pl-3 py-1">
                  <div className="flex justify-between items-start">
                    <span className={`
                      ${log.level === 'error' ? 'text-red-400' :
                        log.level === 'success' ? 'text-green-400' :
                          log.level === 'warning' ? 'text-yellow-400' :
                            log.level === 'progress' ? 'text-purple-400' : 'text-blue-400'}
                    `}>
                      [{log.timestamp.toLocaleTimeString()}] {log.message}
                    </span>
                  </div>
                  {log.details && (
                    <div className="mt-1 text-gray-500 text-[10px] break-all">
                      {log.details}
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="text-gray-500 text-sm italic">No logs yet</div>
            )}
          </div>
        </div>
      ),
    },
    {
      id: 'strategy',
      label: 'Strategy',
      icon: <span className="text-xs font-bold">🎯</span>,
      content: (
        <div className="p-4 overflow-auto h-full">
          {extractionMetadata?.strategy ? (
            <div className="space-y-4">
              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
                  <span>Detected Strategy</span>
                  <span className="px-2 py-0.5 rounded-full bg-green-900/30 text-green-400 text-xs border border-green-500/20">
                    Active
                  </span>
                </h3>

                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Extraction Method</div>
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${extractionMetadata.strategy.method === 'json_ld' ? 'bg-green-400' :
                          extractionMetadata.strategy.method === 'graphql' ? 'bg-purple-400' :
                            extractionMetadata.strategy.method === 'json' ? 'bg-blue-400' : 'bg-gray-400'
                        }`} />
                      <span className="text-sm font-medium text-gray-200">
                        {extractionMetadata.strategy.method?.toUpperCase() || 'UNKNOWN'}
                      </span>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Proxy Configuration</div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-200">
                        {extractionMetadata.strategy.proxy === 'web_unblocker' ? 'Web Unblocker' :
                          extractionMetadata.strategy.proxy === 'residential' ? 'Residential Proxy' :
                            extractionMetadata.strategy.proxy || 'Standard'}
                      </span>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Confidence Score</div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 bg-gray-700 rounded-full w-24 overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 rounded-full"
                          style={{ width: `${(extractionMetadata.strategy.confidence || 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400">
                        {Math.round((extractionMetadata.strategy.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Source</div>
                    <span className="text-xs px-2 py-1 rounded bg-gray-800 text-gray-300 border border-gray-700">
                      {extractionMetadata.strategy.source === 'cache' ? 'Cached Strategy' : 'Live Detection'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-2">Browser Configuration</h3>
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div className="flex justify-between py-1 border-b border-gray-800">
                    <span className="text-gray-500">Anti-Detection</span>
                    <span className="text-green-400">Enabled (Camoufox)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-gray-800">
                    <span className="text-gray-500">Geo-IP Check</span>
                    <span className={extractionMetadata.strategy.method === 'json_ld' ? 'text-yellow-400' : 'text-gray-300'}>
                      {extractionMetadata.strategy.method === 'json_ld' ? 'Disabled (Optimization)' : 'Enabled'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <span className="text-4xl mb-3 block opacity-30">🎯</span>
                <p>No strategy detected yet</p>
                <p className="text-sm mt-1">Run an extraction to see the optimized strategy</p>
              </div>
            </div>
          )}
        </div>
      ),
    },
  ];

  return (
    <>
      <UnsavedChangesModal
        isOpen={showUnsavedModal}
        onSave={handleSaveAndNavigate}
        onDiscard={handleDiscardAndNavigate}
        onCancel={handleCancelNavigation}
      />
      <div className="h-[calc(100vh-4rem)]">
        <AgentBuilderLayout
          toolbar={
            <AgentToolbar
              agent={mockAgent}
              isRunning={false}
              isSaving={false}
              onSave={handleSave}
              hasUnsavedChanges={hasUnsavedChanges}
              lastSaved={lastSaved}
              autoSaveEnabled={autoSaveEnabled}
              isDraft={true}
              onRun={() => setTriggerExtraction(prev => prev + 1)}
            />
          }
          canvas={
            <BrowserWorkspace
              onExtract={handleExtract}
              onLog={handleLog}
              cachedPatterns={cachedPatterns}
              triggerExtraction={triggerExtraction}
              className="h-full"
            />
          }
          bottomDock={
            <BottomDock
              tabs={bottomTabs}
              defaultTab="extracted-content"
            />
          }
          bottomDockCollapsed={bottomDockCollapsed}
          onToggleBottomDock={() => setBottomDockCollapsed(!bottomDockCollapsed)}
        />
      </div>
    </>
  );
}
