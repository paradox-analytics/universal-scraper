import { useState, useEffect, useRef } from 'react';
import {
  DocumentTextIcon,
  ArrowPathIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  MagnifyingGlassPlusIcon,
  MagnifyingGlassMinusIcon,
  ArrowUpTrayIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  PlayIcon,
  CircleStackIcon,
  ArrowDownTrayIcon,
  PlusIcon,
  TableCellsIcon,
  PhotoIcon,
  EyeIcon,
} from '@heroicons/react/24/outline';
import type { CachedPattern } from '../../types';

interface DocumentViewerProps {
  onExtract?: (results: any[], fields: string[]) => void;
  onPatternSave?: (pattern: any) => void;
  cachedPatterns?: CachedPattern[]; // eslint-disable-line @typescript-eslint/no-unused-vars
  className?: string;
}

interface SelectedField {
  name: string;
  type: 'text' | 'table' | 'image' | 'region';
  sample?: string;
  pageNumber?: number;
  bounds?: { x: number; y: number; width: number; height: number };
}

type ViewMode = 'document' | 'text' | 'tables' | 'ocr';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://universal-scraper-api-oo6mrfwkma-uc.a.run.app';

export default function DocumentViewer({
  onExtract,
  onPatternSave,
  // cachedPatterns will be used for pattern selection in future
  cachedPatterns: _cachedPatterns = [],
  className = ''
}: DocumentViewerProps) {
  // Suppress unused variable warning - will be used for pattern loading
  void _cachedPatterns;
  // Document state
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [fileType, setFileType] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // View state
  const [viewMode, setViewMode] = useState<ViewMode>('document');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [zoom, setZoom] = useState(100);
  const [extractedText, setExtractedText] = useState<string>('');
  const [extractedTables, setExtractedTables] = useState<any[]>([]);
  
  // Extraction state
  const [selectedFields, setSelectedFields] = useState<SelectedField[]>([]);
  const [extractionResults, setExtractionResults] = useState<any[]>([]);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractionStatus, setExtractionStatus] = useState<'idle' | 'cached' | 'direct_llm' | 'extracting'>('idle');
  const [extractionTime, setExtractionTime] = useState<number | null>(null);
  
  // Options state
  const [useOcr, setUseOcr] = useState(false);
  const [maxPages] = useState<number | undefined>(undefined);
  const [context, setContext] = useState('');
  
  // Pattern state
  const [patternName, setPatternName] = useState('');
  const [patternVisibility, setPatternVisibility] = useState<'private' | 'public'>('private');
  const [showSavePatternModal, setShowSavePatternModal] = useState(false);
  
  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pdfDocRef = useRef<any>(null);
  
  // Handle file selection
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setError(null);
    setExtractedText('');
    setExtractedTables([]);
    setExtractionResults([]);
    setSelectedFields([]);
    setCurrentPage(1);
    
    // Determine file type
    const ext = selectedFile.name.split('.').pop()?.toLowerCase() || '';
    setFileType(ext);
    
    // Create URL for preview
    const url = URL.createObjectURL(selectedFile);
    setFileUrl(url);
    
    // Handle different file types
    if (ext === 'pdf') {
      await loadPdf(url);
    } else if (['doc', 'docx'].includes(ext)) {
      await loadDocx(selectedFile);
    } else if (['txt', 'md'].includes(ext)) {
      await loadText(selectedFile);
    } else if (['png', 'jpg', 'jpeg', 'gif', 'bmp'].includes(ext)) {
      // Images are displayed directly
      setTotalPages(1);
    }
  };
  
  // Load PDF using PDF.js
  const loadPdf = async (url: string) => {
    setIsLoading(true);
    try {
      // Dynamically import PDF.js
      const pdfjsLib = await import('pdfjs-dist');
      pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;
      
      const loadingTask = pdfjsLib.getDocument(url);
      const pdf = await loadingTask.promise;
      pdfDocRef.current = pdf;
      setTotalPages(pdf.numPages);
      
      // Render first page
      await renderPdfPage(pdf, 1);
      
    } catch (err: any) {
      console.error('PDF loading failed:', err);
      setError('Failed to load PDF: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Render PDF page
  const renderPdfPage = async (pdf: any, pageNum: number) => {
    if (!canvasRef.current) return;
    
    try {
      const page = await pdf.getPage(pageNum);
      const scale = zoom / 100 * 1.5;
      const viewport = page.getViewport({ scale });
      
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      
      await page.render({
        canvasContext: context,
        viewport: viewport
      }).promise;
      
    } catch (err: any) {
      console.error('PDF page render failed:', err);
    }
  };
  
  // Load DOCX using mammoth.js
  const loadDocx = async (docFile: File) => {
    setIsLoading(true);
    try {
      const mammoth = await import('mammoth');
      const arrayBuffer = await docFile.arrayBuffer();
      const result = await mammoth.convertToHtml({ arrayBuffer });
      setExtractedText(result.value);
      setTotalPages(1);
    } catch (err: any) {
      console.error('DOCX loading failed:', err);
      setError('Failed to load DOCX: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Load text file
  const loadText = async (textFile: File) => {
    setIsLoading(true);
    try {
      const text = await textFile.text();
      setExtractedText(text);
      setTotalPages(1);
    } catch (err: any) {
      console.error('Text loading failed:', err);
      setError('Failed to load text file: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Update PDF when page or zoom changes
  useEffect(() => {
    if (pdfDocRef.current && fileType === 'pdf') {
      renderPdfPage(pdfDocRef.current, currentPage);
    }
  }, [currentPage, zoom, fileType]);
  
  // Cleanup URL on unmount
  useEffect(() => {
    return () => {
      if (fileUrl) {
        URL.revokeObjectURL(fileUrl);
      }
    };
  }, [fileUrl]);
  
  // Add field
  const addField = (name: string, type: SelectedField['type'] = 'text') => {
    if (selectedFields.some(f => f.name === name)) return;
    
    setSelectedFields(prev => [...prev, {
      name,
      type,
      pageNumber: currentPage
    }]);
  };
  
  // Remove field
  const removeField = (name: string) => {
    setSelectedFields(prev => prev.filter(f => f.name !== name));
  };
  
  // Add custom field
  const addCustomField = () => {
    const name = prompt('Enter field name:');
    if (!name) return;
    addField(name, 'text');
  };
  
  // Run extraction
  const runExtraction = async () => {
    if (!file) {
      setError('Please upload a document first');
      return;
    }
    
    if (selectedFields.length === 0) {
      setError('Please select at least one field to extract');
      return;
    }
    
    setIsExtracting(true);
    setExtractionStatus('extracting');
    setError(null);
    const startTime = Date.now();
    
    try {
      const token = localStorage.getItem('firebase_token');
      const apiKey = localStorage.getItem('api_key');
      
      const formData = new FormData();
      formData.append('file', file);
      formData.append('fields', JSON.stringify(selectedFields.map(f => f.name)));
      formData.append('use_ocr', String(useOcr));
      if (maxPages) {
        formData.append('max_pages', String(maxPages));
      }
      if (context) {
        formData.append('context', context);
      }
      
      const response = await fetch(`${API_BASE_URL}/document-processing/extract`, {
        method: 'POST',
        headers: {
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          ...(apiKey ? { 'X-API-Key': apiKey } : {}),
        },
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Extraction failed');
      }
      
      const data = await response.json();
      const endTime = Date.now();
      setExtractionTime((endTime - startTime) / 1000);
      
      if (data.success && data.data) {
        setExtractionResults(data.data);
        setExtractionStatus('direct_llm'); // Documents always use LLM currently
        onExtract?.(data.data, selectedFields.map(f => f.name));
      } else {
        throw new Error('No data returned');
      }
      
    } catch (err: any) {
      console.error('Extraction failed:', err);
      setError(err.message);
      setExtractionStatus('idle');
    } finally {
      setIsExtracting(false);
    }
  };
  
  // Save pattern
  const savePattern = async () => {
    if (!patternName || selectedFields.length === 0) {
      setError('Please enter a pattern name and select fields');
      return;
    }
    
    try {
      const token = localStorage.getItem('firebase_token');
      
      const response = await fetch(`${API_BASE_URL}/api/v1/patterns/store`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          domain: `document:${fileType}`,
          fields: selectedFields.map(f => f.name),
          pattern_data: {
            name: patternName,
            fields: selectedFields,
            fileType,
            useOcr,
            context,
            created_at: Date.now()
          },
          visibility: patternVisibility
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to save pattern');
      }
      
      setShowSavePatternModal(false);
      setPatternName('');
      onPatternSave?.({ name: patternName, fields: selectedFields, fileType });
      
    } catch (err: any) {
      console.error('Pattern save failed:', err);
      setError(err.message);
    }
  };
  
  // Export results
  const exportResults = (format: 'json' | 'csv') => {
    if (extractionResults.length === 0) return;
    
    let content: string;
    let filename: string;
    let mimeType: string;
    
    if (format === 'json') {
      content = JSON.stringify(extractionResults, null, 2);
      filename = 'document-extraction.json';
      mimeType = 'application/json';
    } else {
      const headers = Object.keys(extractionResults[0] || {});
      const rows = extractionResults.map(row => 
        headers.map(h => JSON.stringify(row[h] || '')).join(',')
      );
      content = [headers.join(','), ...rows].join('\n');
      filename = 'document-extraction.csv';
      mimeType = 'text/csv';
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  // Common fields for documents
  const commonFields = [
    { name: 'title', type: 'text' as const },
    { name: 'date', type: 'text' as const },
    { name: 'author', type: 'text' as const },
    { name: 'summary', type: 'text' as const },
    { name: 'total', type: 'text' as const },
    { name: 'invoice_number', type: 'text' as const },
    { name: 'line_items', type: 'table' as const },
    { name: 'signatures', type: 'image' as const },
  ];
  
  return (
    <div className={`flex flex-col h-full bg-gray-900 ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center gap-2 p-3 bg-gray-800 border-b border-gray-700">
        {/* Upload button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center gap-2"
        >
          <ArrowUpTrayIcon className="w-4 h-4" />
          Upload Document
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.doc,.docx,.txt,.md,.png,.jpg,.jpeg,.gif,.bmp"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        {file && (
          <>
            <div className="h-6 w-px bg-gray-700" />
            
            {/* File info */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 rounded-lg">
              <DocumentTextIcon className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-white truncate max-w-xs">{file.name}</span>
              <button
                onClick={() => {
                  setFile(null);
                  setFileUrl(null);
                  setExtractedText('');
                  setExtractionResults([]);
                  setSelectedFields([]);
                }}
                className="text-gray-400 hover:text-white"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
            
            {/* Page navigation for PDFs */}
            {fileType === 'pdf' && totalPages > 1 && (
              <>
                <div className="h-6 w-px bg-gray-700" />
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage <= 1}
                    className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded disabled:opacity-50"
                  >
                    <ChevronLeftIcon className="w-4 h-4" />
                  </button>
                  <span className="text-sm text-gray-300">
                    {currentPage} / {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage >= totalPages}
                    className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded disabled:opacity-50"
                  >
                    <ChevronRightIcon className="w-4 h-4" />
                  </button>
                </div>
              </>
            )}
            
            {/* Zoom controls */}
            <div className="h-6 w-px bg-gray-700" />
            <div className="flex items-center gap-1">
              <button
                onClick={() => setZoom(z => Math.max(25, z - 25))}
                className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
              >
                <MagnifyingGlassMinusIcon className="w-4 h-4" />
              </button>
              <span className="text-sm text-gray-300 w-12 text-center">{zoom}%</span>
              <button
                onClick={() => setZoom(z => Math.min(200, z + 25))}
                className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded"
              >
                <MagnifyingGlassPlusIcon className="w-4 h-4" />
              </button>
            </div>
          </>
        )}
        
        <div className="flex-1" />
        
        {/* Options */}
        <label className="flex items-center gap-2 text-sm text-gray-300">
          <input
            type="checkbox"
            checked={useOcr}
            onChange={(e) => setUseOcr(e.target.checked)}
            className="rounded bg-gray-700 border-gray-600"
          />
          Use OCR
        </label>
        
        {/* Status indicator */}
        {extractionStatus !== 'idle' && (
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
            extractionStatus === 'cached' ? 'bg-green-900/50 text-green-400' :
            extractionStatus === 'direct_llm' ? 'bg-amber-900/50 text-amber-400' :
            'bg-indigo-900/50 text-indigo-400'
          }`}>
            {extractionStatus === 'extracting' && <ArrowPathIcon className="w-4 h-4 animate-spin" />}
            {extractionStatus === 'direct_llm' && 'LLM Extraction'}
            {extractionStatus === 'extracting' && 'Extracting...'}
            {extractionTime && ` (${extractionTime.toFixed(1)}s)`}
          </div>
        )}
      </div>
      
      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Document View */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* View Mode Tabs */}
          <div className="flex items-center gap-1 px-3 py-2 bg-gray-800/50 border-b border-gray-700">
            <button
              onClick={() => setViewMode('document')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'document' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <EyeIcon className="w-4 h-4 inline mr-1" />
              Document
            </button>
            <button
              onClick={() => setViewMode('text')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'text' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <DocumentTextIcon className="w-4 h-4 inline mr-1" />
              Text
            </button>
            <button
              onClick={() => setViewMode('tables')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                viewMode === 'tables' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <TableCellsIcon className="w-4 h-4 inline mr-1" />
              Tables
            </button>
          </div>
          
          {/* Document Content */}
          <div className="flex-1 relative overflow-auto bg-gray-950 p-4">
            {!file ? (
              <div 
                className="flex items-center justify-center h-full border-2 border-dashed border-gray-700 rounded-xl cursor-pointer hover:border-indigo-500 transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="text-center">
                  <ArrowUpTrayIcon className="w-16 h-16 mx-auto mb-4 text-gray-500" />
                  <h3 className="text-lg font-medium text-gray-300 mb-2">Drop a document or click to upload</h3>
                  <p className="text-sm text-gray-500">Supports PDF, DOCX, TXT, MD, and images</p>
                </div>
              </div>
            ) : isLoading ? (
              <div className="flex items-center justify-center h-full">
                <ArrowPathIcon className="w-8 h-8 text-indigo-400 animate-spin" />
              </div>
            ) : viewMode === 'document' ? (
              <div className="flex justify-center">
                {fileType === 'pdf' ? (
                  <canvas ref={canvasRef} className="shadow-xl" />
                ) : ['png', 'jpg', 'jpeg', 'gif', 'bmp'].includes(fileType) ? (
                  <img 
                    src={fileUrl || ''} 
                    alt="Document" 
                    style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}
                    className="shadow-xl"
                  />
                ) : ['doc', 'docx'].includes(fileType) && extractedText ? (
                  <div 
                    className="bg-white text-black p-8 rounded shadow-xl max-w-3xl prose"
                    style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}
                    dangerouslySetInnerHTML={{ __html: extractedText }}
                  />
                ) : (
                  <pre 
                    className="bg-white text-black p-8 rounded shadow-xl max-w-3xl whitespace-pre-wrap"
                    style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center' }}
                  >
                    {extractedText}
                  </pre>
                )}
              </div>
            ) : viewMode === 'text' ? (
              <div className="max-w-3xl mx-auto">
                <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono">
                  {extractedText || 'No text extracted yet'}
                </pre>
              </div>
            ) : viewMode === 'tables' ? (
              <div className="max-w-4xl mx-auto">
                {extractedTables.length > 0 ? (
                  extractedTables.map((table, idx) => (
                    <div key={idx} className="mb-6 bg-gray-800 rounded-lg overflow-hidden">
                      <table className="w-full text-sm text-gray-300">
                        <tbody>
                          {table.map((row: any[], rowIdx: number) => (
                            <tr key={rowIdx} className={rowIdx === 0 ? 'bg-gray-700' : ''}>
                              {row.map((cell, cellIdx) => (
                                <td key={cellIdx} className="px-4 py-2 border-b border-gray-700">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ))
                ) : (
                  <p className="text-gray-400 text-center py-8">No tables detected</p>
                )}
              </div>
            ) : null}
            
            {/* Error overlay */}
            {error && (
              <div className="absolute inset-x-0 top-0 p-4 bg-red-900/90 text-red-200">
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
        
        {/* Fields Sidebar */}
        <div className="w-72 bg-gray-800 border-l border-gray-700 flex flex-col overflow-hidden">
          <div className="p-3 border-b border-gray-700">
            <h3 className="text-sm font-medium text-white">Extraction Fields</h3>
            <p className="text-xs text-gray-400 mt-1">Select fields to extract from document</p>
          </div>
          
          <div className="flex-1 overflow-auto p-3 space-y-4">
            {/* Common fields */}
            <div>
              <h4 className="text-xs font-medium text-indigo-400 mb-2">Common Fields</h4>
              <div className="space-y-1">
                {commonFields.map((field) => (
                  <button
                    key={field.name}
                    onClick={() => addField(field.name, field.type)}
                    disabled={selectedFields.some(f => f.name === field.name)}
                    className={`w-full p-2 rounded text-left transition-colors flex items-center gap-2 ${
                      selectedFields.some(f => f.name === field.name)
                        ? 'bg-indigo-600/20 text-indigo-400'
                        : 'bg-gray-700 hover:bg-gray-600 text-white'
                    }`}
                  >
                    {field.type === 'table' && <TableCellsIcon className="w-4 h-4" />}
                    {field.type === 'image' && <PhotoIcon className="w-4 h-4" />}
                    {field.type === 'text' && <DocumentTextIcon className="w-4 h-4" />}
                    <span className="text-sm capitalize">{field.name.replace(/_/g, ' ')}</span>
                    {selectedFields.some(f => f.name === field.name) && (
                      <CheckCircleIcon className="w-4 h-4 ml-auto" />
                    )}
                  </button>
                ))}
              </div>
            </div>
            
            {/* Custom field */}
            <button
              onClick={addCustomField}
              className="w-full p-2 bg-gray-700 hover:bg-gray-600 rounded text-left transition-colors flex items-center gap-2 text-gray-300"
            >
              <PlusIcon className="w-4 h-4" />
              <span className="text-sm">Add Custom Field</span>
            </button>
            
            {/* Context input */}
            <div>
              <h4 className="text-xs font-medium text-gray-400 mb-2">Context (optional)</h4>
              <textarea
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="Add context to help extraction..."
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm text-white placeholder-gray-500 resize-none"
                rows={3}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Extraction Panel */}
      <div className="bg-gray-800 border-t border-gray-700 p-4">
        <div className="flex items-center justify-between">
          {/* Selected fields */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm text-gray-400">Fields:</span>
            {selectedFields.map((field, idx) => (
              <span 
                key={idx}
                className="inline-flex items-center gap-1 px-2 py-1 bg-indigo-600/20 text-indigo-400 text-xs rounded"
              >
                {field.name}
                <button onClick={() => removeField(field.name)} className="hover:text-white">
                  <XMarkIcon className="w-3 h-3" />
                </button>
              </span>
            ))}
            {selectedFields.length === 0 && (
              <span className="text-sm text-gray-500">No fields selected</span>
            )}
          </div>
          
          {/* Action buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={runExtraction}
              disabled={isExtracting || selectedFields.length === 0 || !file}
              className="px-4 py-1.5 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
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
            
            {extractionResults.length > 0 && (
              <>
                <button
                  onClick={() => setShowSavePatternModal(true)}
                  className="px-4 py-1.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors flex items-center gap-2"
                >
                  <CircleStackIcon className="w-4 h-4" />
                  Save Pattern
                </button>
                
                <div className="relative group">
                  <button className="px-4 py-1.5 bg-gray-700 text-white text-sm font-medium rounded-lg hover:bg-gray-600 transition-colors flex items-center gap-2">
                    <ArrowDownTrayIcon className="w-4 h-4" />
                    Export
                  </button>
                  <div className="absolute right-0 bottom-full mb-1 hidden group-hover:block">
                    <div className="bg-gray-700 rounded-lg shadow-xl py-1">
                      <button
                        onClick={() => exportResults('json')}
                        className="block w-full px-4 py-2 text-sm text-white hover:bg-gray-600 text-left"
                      >
                        Export JSON
                      </button>
                      <button
                        onClick={() => exportResults('csv')}
                        className="block w-full px-4 py-2 text-sm text-white hover:bg-gray-600 text-left"
                      >
                        Export CSV
                      </button>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
        
        {/* Results preview */}
        {extractionResults.length > 0 && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-48 overflow-auto">
            {extractionResults.slice(0, 6).map((item, idx) => (
              <div key={idx} className="bg-gray-900 rounded-lg p-3">
                {Object.entries(item).slice(0, 4).map(([key, value]) => (
                  <div key={key} className="mb-1">
                    <span className="text-xs text-gray-500">{key}:</span>
                    <p className="text-sm text-white truncate">{String(value)}</p>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Save Pattern Modal */}
      {showSavePatternModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 w-96 max-w-full mx-4">
            <h3 className="text-lg font-semibold text-white mb-4">Save Document Pattern</h3>
            
            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-400">Pattern Name</label>
                <input
                  type="text"
                  value={patternName}
                  onChange={(e) => setPatternName(e.target.value)}
                  placeholder="e.g., Invoice Extraction"
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
      )}
    </div>
  );
}

