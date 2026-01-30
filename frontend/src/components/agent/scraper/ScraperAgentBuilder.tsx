import { useState } from 'react';
import { AgentBuilderLayout } from '../layouts/AgentBuilderLayout';
import { AgentToolbar } from '../shared/AgentToolbar';
import { BottomDock } from '../shared/BottomDock';
import { ScraperAgentTree } from './ScraperAgentTree';
import { BrowserWorkspace } from '../../BrowserWorkspace';
import type { ScraperAgent } from '../../../types';
import {
  TableCellsIcon,
  CursorArrowRaysIcon,
  CodeBracketSquareIcon,
  EyeIcon,
  ClockIcon,
  DocumentTextIcon,
} from '@heroicons/react/24/outline';

interface ScraperAgentBuilderProps {
  agent: ScraperAgent;
  onBack?: () => void;
  onUpdate?: (agentId: string) => void;
}

/**
 * ScraperAgentBuilder - Builder interface for scraper agents
 * 
 * Reuses BrowserWorkspace as the main canvas
 * Adds left tree for configuration and bottom tabs for outputs
 */
export function ScraperAgentBuilder({ 
  agent, 
  onBack,
  onUpdate 
}: ScraperAgentBuilderProps) {
  const [bottomDockCollapsed, setBottomDockCollapsed] = useState(false);
  const [extractionResults, setExtractionResults] = useState<any[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  
  const handleRun = async () => {
    setIsRunning(true);
    try {
      // Trigger extraction in BrowserWorkspace
      // The BrowserWorkspace component will handle the actual scraping
      console.log('Running scraper agent:', agent.id);
    } catch (err) {
      console.error('Failed to run agent:', err);
    } finally {
      setIsRunning(false);
    }
  };
  
  const handleSave = async () => {
    setIsSaving(true);
    try {
      // Save agent configuration
      console.log('Saving agent:', agent.id);
      onUpdate?.(agent.id);
    } catch (err) {
      console.error('Failed to save agent:', err);
    } finally {
      setIsSaving(false);
    }
  };
  
  const handleExtract = (results: any[], fields: string[]) => {
    setExtractionResults(results);
    console.log('Extraction complete:', { results, fields });
  };
  
  // Bottom dock tabs for scraper
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
                <p className="text-sm mt-1">Run the scraper to see results here</p>
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
          <p className="text-sm">Field selectors and detection rules will be shown here</p>
        </div>
      ),
    },
    {
      id: 'schema',
      label: 'Schema',
      icon: <CodeBracketSquareIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full text-gray-400">
          <p className="text-sm">Data schema and field types will be shown here</p>
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
        <div className="p-4 overflow-auto h-full text-gray-400">
          <p className="text-sm">Run history and activity log will be shown here</p>
        </div>
      ),
    },
    {
      id: 'logs',
      label: 'Logs',
      icon: <DocumentTextIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full">
          <div className="text-xs text-gray-300 font-mono">
            <div className="mb-2 text-gray-500">[00:00:00] Agent initialized</div>
            <div className="mb-2 text-blue-400">[00:00:01] Ready to run</div>
          </div>
        </div>
      ),
    },
  ];
  
  return (
    <AgentBuilderLayout
      toolbar={
        <AgentToolbar
          agent={agent}
          onBack={onBack}
          onRun={handleRun}
          onSave={handleSave}
          isRunning={isRunning}
          isSaving={isSaving}
        />
      }
      leftTree={
        <ScraperAgentTree agent={agent} />
      }
      canvas={
        <BrowserWorkspace
          onExtract={handleExtract}
          cachedPatterns={[]}
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
  );
}



