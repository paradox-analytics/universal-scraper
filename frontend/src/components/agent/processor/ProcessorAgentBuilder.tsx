import { useState } from 'react';
import { AgentBuilderLayout } from '../layouts/AgentBuilderLayout';
import { AgentToolbar } from '../shared/AgentToolbar';
import { BottomDock } from '../shared/BottomDock';
import { DocumentViewer } from './DocumentViewer';
import type { ProcessorAgent } from '../../../types';
import {
  DocumentTextIcon,
  Squares2X2Icon,
  TableCellsIcon,
  CodeBracketSquareIcon,
  CubeIcon,
  ClockIcon,
  DocumentTextIcon as LogIcon,
} from '@heroicons/react/24/outline';

interface ProcessorAgentBuilderProps {
  agent: ProcessorAgent;
  onBack?: () => void;
  onUpdate?: (agentId: string) => void;
}

/**
 * ProcessorAgentBuilder - Builder interface for document processor agents
 * 
 * TODO: Implement full processor UI with document viewer
 */
export function ProcessorAgentBuilder({ 
  agent, 
  onBack 
}: ProcessorAgentBuilderProps) {
  const [bottomDockCollapsed, setBottomDockCollapsed] = useState(false);
  const [documentUrl] = useState<string | undefined>(
    agent.definition.sources?.[0]?.uri
  ); // Will be updated when user selects different source
  
  const bottomTabs = [
    {
      id: 'document-preview',
      label: 'Document Preview',
      icon: <DocumentTextIcon className="w-4 h-4" />,
      content: (
        <div className="h-full overflow-hidden">
          <DocumentViewer 
            documentUrl={documentUrl}
            documentType="pdf"
          />
        </div>
      ),
    },
    {
      id: 'chunks',
      label: 'Chunks',
      icon: <Squares2X2Icon className="w-4 h-4" />,
      content: (
        <div className="p-4 h-full flex items-center justify-center text-gray-400">
          <div className="text-center">
            <Squares2X2Icon className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Document chunks will be shown here</p>
            <p className="text-xs text-gray-500 mt-1">After chunking is processed</p>
          </div>
        </div>
      ),
    },
    {
      id: 'extracted-fields',
      label: 'Extracted Fields',
      icon: <TableCellsIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 h-full flex items-center justify-center text-gray-400">
          <div className="text-center">
            <TableCellsIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Extracted fields will be shown here</p>
            <p className="text-xs text-gray-500 mt-1">After field extraction completes</p>
          </div>
        </div>
      ),
    },
    {
      id: 'schema',
      label: 'Schema',
      icon: <CodeBracketSquareIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 h-full overflow-auto">
          <div className="bg-gray-900 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-2">Output Schema</h3>
            <pre className="text-xs text-gray-400 font-mono">
{JSON.stringify({
  fields: agent.definition.fields || [],
  chunking: agent.definition.chunking || { enabled: false },
  enrichment: agent.definition.enrichment || {},
  output: agent.definition.output || { format: 'json' }
}, null, 2)}
            </pre>
          </div>
        </div>
      ),
    },
    {
      id: 'artifacts',
      label: 'Artifacts',
      icon: <CubeIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 h-full flex items-center justify-center text-gray-400">
          <div className="text-center">
            <CubeIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Indexes and exports will be shown here</p>
            <p className="text-xs text-gray-500 mt-1">Embeddings, summaries, etc.</p>
          </div>
        </div>
      ),
    },
    {
      id: 'activity',
      label: 'Activity',
      icon: <ClockIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full text-gray-400">
          <p className="text-sm">Processing history will be shown here</p>
        </div>
      ),
    },
    {
      id: 'logs',
      label: 'Logs',
      icon: <LogIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full">
          <div className="text-xs text-gray-300 font-mono">
            <div className="mb-2 text-gray-500">[00:00:00] Processor agent initialized</div>
            <div className="mb-2 text-blue-400">[00:00:01] Ready to process documents</div>
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
          onRun={() => console.log('Run processor')}
          onSave={() => console.log('Save processor')}
        />
      }
      leftTree={
        <div className="p-4 text-sm">
          <div className="mb-4">
            <h3 className="text-gray-400 font-medium mb-2">Document Sources</h3>
            <div className="space-y-2">
              {agent.definition.sources && agent.definition.sources.length > 0 ? (
                agent.definition.sources.map((source) => (
                  <div key={source.id} className="p-2 bg-gray-700/30 rounded text-xs">
                    <div className="font-medium text-gray-300">{source.name}</div>
                    <div className="text-gray-500 mt-1">{source.type}</div>
                  </div>
                ))
              ) : (
                <div className="text-gray-500 text-xs">No sources configured</div>
              )}
            </div>
          </div>
          
          {agent.definition.fields && agent.definition.fields.length > 0 && (
            <div className="mb-4">
              <h3 className="text-gray-400 font-medium mb-2">Fields to Extract</h3>
              <div className="space-y-1">
                {agent.definition.fields.map((field, idx) => (
                  <div key={idx} className="px-2 py-1 bg-gray-700/30 rounded text-xs text-gray-300">
                    {field}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          <div className="mb-4">
            <h3 className="text-gray-400 font-medium mb-2">Processing Options</h3>
            <div className="space-y-2 text-xs text-gray-400">
              <div className="flex justify-between">
                <span>OCR:</span>
                <span className={agent.definition.use_ocr ? 'text-green-400' : 'text-gray-500'}>
                  {agent.definition.use_ocr ? 'Enabled' : 'Disabled'}
                </span>
              </div>
              {agent.definition.max_pages && (
                <div className="flex justify-between">
                  <span>Max Pages:</span>
                  <span className="text-gray-300">{agent.definition.max_pages}</span>
                </div>
              )}
              {agent.definition.chunking?.enabled && (
                <div className="flex justify-between">
                  <span>Chunking:</span>
                  <span className="text-green-400">
                    {agent.definition.chunking.strategy || 'Enabled'}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      }
      canvas={
        <DocumentViewer 
          documentUrl={documentUrl}
          documentType="pdf"
        />
      }
      bottomDock={
        <BottomDock
          tabs={bottomTabs}
          defaultTab="document-preview"
        />
      }
      bottomDockCollapsed={bottomDockCollapsed}
      onToggleBottomDock={() => setBottomDockCollapsed(!bottomDockCollapsed)}
    />
  );
}

