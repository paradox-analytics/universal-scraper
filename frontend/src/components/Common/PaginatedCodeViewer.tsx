import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState, useEffect } from 'react';
import { ClipboardIcon, CheckIcon, ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

interface PaginatedCodeViewerProps {
  data: any;
  language?: 'json' | 'html' | 'javascript' | 'python';
  title?: string;
  itemsPerPage?: number;
}

export function PaginatedCodeViewer({ data, language = 'json', title = 'Raw Data', itemsPerPage: initialItemsPerPage }: PaginatedCodeViewerProps) {
  const [copied, setCopied] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  
  // Calculate items per page based on screen size
  const [itemsPerPage, setItemsPerPage] = useState(() => {
    if (initialItemsPerPage) return initialItemsPerPage;
    if (typeof window !== 'undefined') {
      const width = window.innerWidth;
      if (width >= 1920) return 25;
      if (width >= 1536) return 20;
      if (width >= 1280) return 15;
      if (width >= 1024) return 12;
      if (width >= 768) return 10;
      return 8;
    }
    return 20;
  });

  useEffect(() => {
    const handleResize = () => {
      if (initialItemsPerPage) return;
      const width = window.innerWidth;
      if (width >= 1920) setItemsPerPage(25);
      else if (width >= 1536) setItemsPerPage(20);
      else if (width >= 1280) setItemsPerPage(15);
      else if (width >= 1024) setItemsPerPage(12);
      else if (width >= 768) setItemsPerPage(10);
      else setItemsPerPage(8);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [initialItemsPerPage]);

  // If data is an array, paginate it
  const isArray = Array.isArray(data);
  let paginatedData = data;
  let totalPages = 1;
  let startIndex = 0;
  let endIndex = data.length;

  if (isArray && data.length > itemsPerPage) {
    totalPages = Math.ceil(data.length / itemsPerPage);
    startIndex = (currentPage - 1) * itemsPerPage;
    endIndex = startIndex + itemsPerPage;
    paginatedData = data.slice(startIndex, endIndex);
  }

  const codeString =
    language === 'json'
      ? JSON.stringify(paginatedData, null, 2)
      : typeof paginatedData === 'string'
      ? paginatedData
      : JSON.stringify(paginatedData, null, 2);

  const fullCodeString =
    language === 'json'
      ? JSON.stringify(data, null, 2)
      : typeof data === 'string'
      ? data
      : JSON.stringify(data, null, 2);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(fullCodeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const goToPage = (page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  };

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden bg-gray-900">
      <div className="bg-gray-800 px-4 py-2 flex justify-between items-center border-b border-gray-700">
        <div className="flex items-center gap-4">
          <span className="text-gray-100 text-sm font-medium">{title}</span>
          {isArray && data.length > itemsPerPage && (
            <span className="text-gray-400 text-xs">
              Showing items {startIndex + 1}-{Math.min(endIndex, data.length)} of {data.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {isArray && data.length > itemsPerPage && totalPages > 1 && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="p-1.5 rounded border border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-700 hover:border-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronLeftIcon className="h-4 w-4" />
              </button>
              
              <div className="flex items-center gap-1">
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  let pageNum;
                  if (totalPages <= 5) {
                    pageNum = i + 1;
                  } else if (currentPage <= 3) {
                    pageNum = i + 1;
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i;
                  } else {
                    pageNum = currentPage - 2 + i;
                  }
                  
                  return (
                    <button
                      key={pageNum}
                      onClick={() => goToPage(pageNum)}
                      className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                        currentPage === pageNum
                          ? 'bg-purple-600 text-white'
                          : 'bg-gray-900 text-gray-300 hover:bg-gray-700 border border-gray-700'
                      }`}
                    >
                      {pageNum}
                    </button>
                  );
                })}
              </div>
              
              <button
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="p-1.5 rounded border border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-700 hover:border-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <ChevronRightIcon className="h-4 w-4" />
              </button>
            </div>
          )}
          
          <button
            onClick={handleCopy}
            className="flex items-center gap-2 text-gray-300 text-sm hover:text-gray-100 transition-colors"
          >
            {copied ? (
              <>
                <CheckIcon className="h-4 w-4" />
                Copied!
              </>
            ) : (
              <>
                <ClipboardIcon className="h-4 w-4" />
                Copy
              </>
            )}
          </button>
        </div>
      </div>
      <div className="max-h-[600px] overflow-y-auto">
        <SyntaxHighlighter
          language={language}
          style={vscDarkPlus}
          customStyle={{ margin: 0, borderRadius: 0 }}
          showLineNumbers
        >
          {codeString}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}




