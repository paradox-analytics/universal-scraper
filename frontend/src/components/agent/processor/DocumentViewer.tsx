import { useState } from 'react';
import { 
  DocumentTextIcon, 
  ArrowDownTrayIcon,
  MagnifyingGlassIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';

interface DocumentViewerProps {
  documentUrl?: string;
  documentType?: 'pdf' | 'docx' | 'html' | 'txt';
  onPageChange?: (page: number) => void;
}

/**
 * DocumentViewer - Canvas component for document processor agents
 * 
 * Displays PDF, DOCX, HTML, and text documents
 * Supports page navigation, zoom, and download
 */
export function DocumentViewer({ 
  documentUrl, 
  documentType = 'pdf',
  onPageChange 
}: DocumentViewerProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages] = useState(1); // Will be set dynamically when document loads
  const [zoom, setZoom] = useState(100);
  
  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      onPageChange?.(newPage);
    }
  };
  
  const handleZoom = (delta: number) => {
    setZoom(prev => Math.max(50, Math.min(200, prev + delta)));
  };
  
  // Placeholder rendering - will be enhanced with actual document rendering
  if (!documentUrl) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-950 text-gray-400">
        <div className="text-center">
          <DocumentTextIcon className="w-16 h-16 mx-auto mb-4 opacity-30" />
          <p className="text-lg font-medium mb-2">No Document Loaded</p>
          <p className="text-sm text-gray-500">
            Upload or select a document to begin processing
          </p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="h-full flex flex-col bg-gray-950">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-700 flex-shrink-0">
        {/* Left: Page Navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
            className="p-1.5 rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Previous page"
          >
            <ChevronLeftIcon className="w-4 h-4 text-gray-400" />
          </button>
          
          <div className="flex items-center gap-1 text-sm">
            <input
              type="number"
              value={currentPage}
              onChange={(e) => handlePageChange(parseInt(e.target.value) || 1)}
              className="w-12 px-2 py-1 bg-gray-800 border border-gray-700 rounded text-center text-gray-300 focus:outline-none focus:border-indigo-500"
              min={1}
              max={totalPages}
            />
            <span className="text-gray-500">/ {totalPages}</span>
          </div>
          
          <button
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="p-1.5 rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Next page"
          >
            <ChevronRightIcon className="w-4 h-4 text-gray-400" />
          </button>
        </div>
        
        {/* Center: Zoom Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleZoom(-25)}
            disabled={zoom <= 50}
            className="px-2 py-1 text-sm rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-gray-400"
          >
            −
          </button>
          
          <span className="text-sm text-gray-400 w-12 text-center">{zoom}%</span>
          
          <button
            onClick={() => handleZoom(25)}
            disabled={zoom >= 200}
            className="px-2 py-1 text-sm rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-gray-400"
          >
            +
          </button>
        </div>
        
        {/* Right: Actions */}
        <div className="flex items-center gap-2">
          <button
            className="p-1.5 rounded hover:bg-gray-700 transition-colors"
            title="Search document"
          >
            <MagnifyingGlassIcon className="w-4 h-4 text-gray-400" />
          </button>
          
          <button
            className="p-1.5 rounded hover:bg-gray-700 transition-colors"
            title="Download document"
          >
            <ArrowDownTrayIcon className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>
      
      {/* Document Canvas */}
      <div className="flex-1 overflow-auto bg-gray-900 p-4">
        <div 
          className="mx-auto bg-white shadow-lg"
          style={{ 
            width: `${zoom}%`,
            minHeight: '100%',
            transition: 'width 0.2s ease-in-out'
          }}
        >
          {/* Document rendering area */}
          {documentType === 'pdf' && (
            <div className="w-full h-full flex items-center justify-center p-8 text-gray-400">
              <div className="text-center">
                <DocumentTextIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-sm">PDF Viewer</p>
                <p className="text-xs text-gray-500 mt-1">
                  Integration with react-pdf or pdf.js will go here
                </p>
                <p className="text-xs text-gray-600 mt-2">
                  Document URL: {documentUrl}
                </p>
              </div>
            </div>
          )}
          
          {documentType === 'html' && (
            <iframe
              src={documentUrl}
              className="w-full h-full border-0"
              style={{ minHeight: '800px' }}
              title="HTML Document"
              sandbox="allow-same-origin"
            />
          )}
          
          {documentType === 'txt' && (
            <div className="p-8">
              <pre className="text-sm text-gray-800 whitespace-pre-wrap font-mono">
                {/* Text content will be loaded here */}
                Loading document content...
              </pre>
            </div>
          )}
          
          {documentType === 'docx' && (
            <div className="w-full h-full flex items-center justify-center p-8 text-gray-400">
              <div className="text-center">
                <DocumentTextIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-sm">DOCX Viewer</p>
                <p className="text-xs text-gray-500 mt-1">
                  Integration with mammoth.js or similar will go here
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

