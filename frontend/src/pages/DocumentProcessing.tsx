import { useState, useEffect } from 'react';
import { AgentBuilderLayout } from '../components/agent/layouts/AgentBuilderLayout';
import { AgentToolbar } from '../components/agent/shared/AgentToolbar';
import { BottomDock } from '../components/agent/shared/BottomDock';
import { DocumentViewer as ProcessorDocumentViewer } from '../components/agent/processor/DocumentViewer';
import type { CachedPattern, ProcessorAgent } from '../types';
import {
  DocumentTextIcon,
  Squares2X2Icon,
  TableCellsIcon,
  CodeBracketSquareIcon,
  CubeIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';

export default function DocumentProcessing() {
  const [cachedPatterns] = useState<CachedPattern[]>([]); // Will be used for pattern suggestions
  const [extractionResults, setExtractionResults] = useState<any[]>([]);
  const [bottomDockCollapsed, setBottomDockCollapsed] = useState(false);
  const [selectedFields, setSelectedFields] = useState<string[]>([]);

  useEffect(() => {
    loadCachedPatterns();
  }, []);

  const loadCachedPatterns = async () => {
    // Document patterns loading - will be implemented when backend is ready
    console.log('Document patterns loading...');
  };

  const handleExtract = (results: any[], fields: string[]) => {
    setExtractionResults(results);
    setSelectedFields(fields);
  };

  // Suppress unused variable warnings - these will be used when backend integration is complete
  void cachedPatterns;
  void handleExtract;

  // Create a mock ProcessorAgent for the toolbar
  const mockAgent: ProcessorAgent = {
    id: 'new',
    name: 'New Document Processor',
    type: 'DOC_PROCESSOR',
    status: 'pending',
    tenant_id: 'local',
    created_at: new Date().toISOString(),
    definition: {
      subType: 'document_processing',
      fields: selectedFields,
      use_ocr: false,
    },
  };

  // Bottom dock tabs
  const bottomTabs = [
    {
      id: 'document-preview',
      label: 'Document Preview',
      icon: <DocumentTextIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 h-full flex items-center justify-center text-gray-400">
          <div className="text-center">
            <DocumentTextIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p>Upload a document to preview</p>
            <p className="text-sm mt-1">Supports PDF, DOCX, HTML, and TXT</p>
          </div>
        </div>
      ),
    },
    {
      id: 'chunks',
      label: 'Chunks',
      icon: <Squares2X2Icon className="w-4 h-4" />,
      content: (
        <div className="p-4 h-full flex items-center justify-center text-gray-400">
          <p className="text-sm">Document chunks will appear here after processing</p>
        </div>
      ),
    },
    {
      id: 'extracted-fields',
      label: 'Extracted Fields',
      icon: <TableCellsIcon className="w-4 h-4" />,
      badge: extractionResults.length > 0 ? extractionResults.length : undefined,
      content: (
        <div className="p-4 overflow-auto h-full">
          {extractionResults.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-gray-850 sticky top-0">
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-2 px-3 text-gray-400 font-medium">Field</th>
                    <th className="text-left py-2 px-3 text-gray-400 font-medium">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(extractionResults[0] || {}).map(([key, value]: [string, any]) => (
                    <tr key={key} className="border-b border-gray-800 hover:bg-gray-800/30 transition-colors">
                      <td className="py-2 px-3 text-gray-400 font-medium">{key}</td>
                      <td className="py-2 px-3 text-gray-300">{String(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <TableCellsIcon className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p>No extracted fields yet</p>
                <p className="text-sm mt-1">Upload and process a document to extract fields</p>
              </div>
            </div>
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
          <p className="text-sm">Output schema will be shown here</p>
        </div>
      ),
    },
    {
      id: 'artifacts',
      label: 'Artifacts',
      icon: <CubeIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full text-gray-400">
          <p className="text-sm">Generated artifacts (indexes, embeddings) will appear here</p>
        </div>
      ),
    },
    {
      id: 'activity',
      label: 'Activity',
      icon: <ClockIcon className="w-4 h-4" />,
      content: (
        <div className="p-4 overflow-auto h-full text-gray-400">
          <p className="text-sm">Processing activity will be shown here</p>
        </div>
      ),
    },
  ];

  return (
    <div className="h-[calc(100vh-4rem)]">
      <AgentBuilderLayout
        toolbar={
          <AgentToolbar
            agent={mockAgent}
            isRunning={false}
            isSaving={false}
          />
        }
        canvas={
          <ProcessorDocumentViewer
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
    </div>
  );
}
