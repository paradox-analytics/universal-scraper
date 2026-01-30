import { useState, useEffect } from 'react';
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

interface PaginatedDataTableProps {
  data: any[];
  columns?: string[];
  itemsPerPage?: number;
}

export function PaginatedDataTable({ data, columns, itemsPerPage: initialItemsPerPage }: PaginatedDataTableProps) {
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [filter, setFilter] = useState('');
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
      if (initialItemsPerPage) return; // Don't auto-adjust if explicitly set
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

  // Auto-detect columns if not provided
  const detectedColumns = columns || (data.length > 0 ? Object.keys(data[0]) : []);
  
  // Filter data
  const filteredData = filter
    ? data.filter((row) =>
        Object.values(row).some((val) =>
          String(val).toLowerCase().includes(filter.toLowerCase())
        )
      )
    : data;

  // Sort data
  const sortedData = [...filteredData].sort((a, b) => {
    if (!sortColumn) return 0;
    const aVal = a[sortColumn];
    const bVal = b[sortColumn];
    
    if (aVal === bVal) return 0;
    const comparison = aVal > bVal ? 1 : -1;
    return sortDirection === 'asc' ? comparison : -comparison;
  });

  // Calculate pagination
  const totalPages = Math.ceil(sortedData.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedData = sortedData.slice(startIndex, endIndex);

  // Reset to page 1 when filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filter]);

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const goToPage = (page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  };

  if (data.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        No data to display
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <div className="mb-4 flex items-center justify-between gap-4">
        <input
          type="text"
          placeholder="Filter data..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="input-field max-w-xs"
        />
        <div className="text-sm text-gray-400">
          Showing {startIndex + 1}-{Math.min(endIndex, sortedData.length)} of {sortedData.length} items
        </div>
      </div>
      
      <div className="relative">
        <table className="min-w-full divide-y divide-gray-700">
          <thead className="bg-gray-800">
            <tr>
              {detectedColumns.map((column) => (
                <th
                  key={column}
                  onClick={() => handleSort(column)}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-700 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    {column}
                    {sortColumn === column && (
                      <span className="text-purple-400">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-gray-900 divide-y divide-gray-700">
            {paginatedData.map((row, idx) => (
              <tr key={idx} className="hover:bg-gray-800 transition-colors">
                {detectedColumns.map((column) => (
                  <td key={column} className="px-6 py-4 whitespace-nowrap text-sm text-gray-200">
                    {row[column] !== null && row[column] !== undefined
                      ? String(row[column])
                      : <span className="text-gray-500">-</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="mt-6 flex items-center justify-between border-t border-gray-700 pt-4">
          <div className="flex items-center gap-2">
            <button
              onClick={() => goToPage(currentPage - 1)}
              disabled={currentPage === 1}
              className="p-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-300 hover:bg-gray-700 hover:border-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeftIcon className="h-5 w-5" />
            </button>
            
            <div className="flex items-center gap-1">
              {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
                let pageNum;
                if (totalPages <= 7) {
                  pageNum = i + 1;
                } else if (currentPage <= 4) {
                  pageNum = i + 1;
                } else if (currentPage >= totalPages - 3) {
                  pageNum = totalPages - 6 + i;
                } else {
                  pageNum = currentPage - 3 + i;
                }
                
                return (
                  <button
                    key={pageNum}
                    onClick={() => goToPage(pageNum)}
                    className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                      currentPage === pageNum
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700'
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
              className="p-2 rounded-lg border border-gray-700 bg-gray-800 text-gray-300 hover:bg-gray-700 hover:border-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRightIcon className="h-5 w-5" />
            </button>
          </div>
          
          <div className="text-sm text-gray-400">
            Page {currentPage} of {totalPages}
          </div>
        </div>
      )}
    </div>
  );
}




