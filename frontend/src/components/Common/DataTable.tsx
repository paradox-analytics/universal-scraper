import { useState } from 'react';

interface DataTableProps {
  data: any[];
  columns?: string[];
}

export function DataTable({ data, columns }: DataTableProps) {
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [filter, setFilter] = useState('');

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

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  if (data.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No data to display
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <div className="mb-4">
        <input
          type="text"
          placeholder="Filter data..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="input-field max-w-xs"
        />
      </div>
      
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {detectedColumns.map((column) => (
              <th
                key={column}
                onClick={() => handleSort(column)}
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
              >
                <div className="flex items-center gap-2">
                  {column}
                  {sortColumn === column && (
                    <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {sortedData.map((row, idx) => (
            <tr key={idx} className="hover:bg-gray-50">
              {detectedColumns.map((column) => (
                <td key={column} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {row[column] !== null && row[column] !== undefined
                    ? String(row[column])
                    : '-'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      
      <div className="mt-4 text-sm text-gray-500">
        Showing {sortedData.length} of {data.length} items
      </div>
    </div>
  );
}

