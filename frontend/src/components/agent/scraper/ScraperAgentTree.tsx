import { useState } from 'react';
import {
  ChevronRightIcon,
  ChevronDownIcon,
  GlobeAltIcon,
  MapIcon,
  AdjustmentsHorizontalIcon,
  TableCellsIcon,
  ArrowDownTrayIcon,
} from '@heroicons/react/24/outline';
import type { ScraperAgent } from '../../../types';

interface ScraperAgentTreeProps {
  agent: ScraperAgent;
}

/**
 * ScraperAgentTree - Left navigation tree for scraper configuration
 * 
 * Nodes:
 * - Agent (root)
 * - URL / Seeds
 * - Navigation / Crawl
 * - Detecting / Selection
 * - Schema
 * - Export (CSV/JSON)
 */
export function ScraperAgentTree({ agent }: ScraperAgentTreeProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(
    new Set(['agent', 'url', 'schema'])
  );
  
  const toggleNode = (nodeId: string) => {
    setExpandedNodes(prev => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };
  
  const TreeNode = ({ 
    id, 
    label, 
    icon, 
    children, 
    level = 0 
  }: { 
    id: string; 
    label: string; 
    icon: React.ReactNode; 
    children?: React.ReactNode;
    level?: number;
  }) => {
    const isExpanded = expandedNodes.has(id);
    const hasChildren = !!children;
    
    return (
      <div>
        <button
          onClick={() => hasChildren && toggleNode(id)}
          className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700/50 transition-colors"
          style={{ paddingLeft: `${12 + level * 16}px` }}
        >
          {hasChildren && (
            isExpanded ? (
              <ChevronDownIcon className="w-4 h-4 text-gray-500 flex-shrink-0" />
            ) : (
              <ChevronRightIcon className="w-4 h-4 text-gray-500 flex-shrink-0" />
            )
          )}
          {!hasChildren && <div className="w-4" />}
          <span className="text-gray-400">{icon}</span>
          <span className="flex-1 text-left">{label}</span>
        </button>
        {isExpanded && children && (
          <div>{children}</div>
        )}
      </div>
    );
  };
  
  return (
    <div className="py-4">
      <TreeNode
        id="agent"
        label="Agent"
        icon={<GlobeAltIcon className="w-4 h-4" />}
      >
        {/* URL / Seeds */}
        <TreeNode
          id="url"
          label="URL / Seeds"
          icon={<GlobeAltIcon className="w-4 h-4" />}
          level={1}
        >
          <div className="px-3 py-2 text-xs text-gray-400" style={{ paddingLeft: '60px' }}>
            {agent.definition.url && (
              <div className="mb-1 truncate">
                <span className="text-gray-500">URL:</span> {agent.definition.url}
              </div>
            )}
            {agent.definition.urls && agent.definition.urls.length > 0 && (
              <div className="mb-1">
                <span className="text-gray-500">URLs:</span> {agent.definition.urls.length} urls
              </div>
            )}
            <div className="mb-1">
              <span className="text-gray-500">Mode:</span> {agent.definition.mode || 'hybrid'}
            </div>
          </div>
        </TreeNode>
        
        {/* Navigation / Crawl */}
        <TreeNode
          id="navigation"
          label="Navigation / Crawl"
          icon={<MapIcon className="w-4 h-4" />}
          level={1}
        >
          <div className="px-3 py-2 text-xs text-gray-400" style={{ paddingLeft: '60px' }}>
            <div className="mb-1">
              <span className="text-gray-500">Pagination:</span>{' '}
              {agent.definition.pagination_config?.enableAutoPagination ? 'Enabled' : 'Disabled'}
            </div>
            {agent.definition.pagination_config?.maxPages && (
              <div className="mb-1">
                <span className="text-gray-500">Max Pages:</span>{' '}
                {agent.definition.pagination_config.maxPages}
              </div>
            )}
          </div>
        </TreeNode>
        
        {/* Detecting / Selection */}
        <TreeNode
          id="selection"
          label="Detecting / Selection"
          icon={<AdjustmentsHorizontalIcon className="w-4 h-4" />}
          level={1}
        >
          <div className="px-3 py-2 text-xs text-gray-400" style={{ paddingLeft: '60px' }}>
            <div className="mb-1">
              <span className="text-gray-500">Fields:</span>{' '}
              {agent.definition.fields?.length || 0} fields
            </div>
            {agent.definition.fields && agent.definition.fields.length > 0 && (
              <div className="mt-2 space-y-1">
                {agent.definition.fields.map((field, idx) => (
                  <div key={idx} className="px-2 py-1 bg-gray-700/30 rounded text-[10px]">
                    {field}
                  </div>
                ))}
              </div>
            )}
          </div>
        </TreeNode>
        
        {/* Schema */}
        <TreeNode
          id="schema"
          label="Schema"
          icon={<TableCellsIcon className="w-4 h-4" />}
          level={1}
        >
          <div className="px-3 py-2 text-xs text-gray-400" style={{ paddingLeft: '60px' }}>
            <p>Schema configuration will be shown here</p>
          </div>
        </TreeNode>
        
        {/* Export */}
        <TreeNode
          id="export"
          label="Export"
          icon={<ArrowDownTrayIcon className="w-4 h-4" />}
          level={1}
        >
          <div className="px-3 py-2 text-xs text-gray-400" style={{ paddingLeft: '60px' }}>
            <div className="mb-1">
              <span className="text-gray-500">Format:</span>{' '}
              {agent.definition.export?.format || 'json'}
            </div>
            {agent.definition.cache_enabled !== false && (
              <div className="mb-1 text-green-400">
                ✓ Cache enabled
              </div>
            )}
          </div>
        </TreeNode>
      </TreeNode>
    </div>
  );
}



