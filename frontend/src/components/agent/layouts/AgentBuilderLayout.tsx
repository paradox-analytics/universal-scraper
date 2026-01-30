import { ReactNode } from 'react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

interface AgentBuilderLayoutProps {
  // Top toolbar
  toolbar: ReactNode;
  
  // Left tree navigation (optional)
  leftTree?: ReactNode;
  leftTreeWidth?: number; // Default 280px
  
  // Main canvas (center)
  canvas: ReactNode;
  
  // Bottom dock with tabs
  bottomDock: ReactNode;
  bottomDockHeight?: number; // Default 400px
  bottomDockCollapsed?: boolean;
  onToggleBottomDock?: () => void;
  
  // Optional right panel (for future use)
  rightPanel?: ReactNode;
  rightPanelWidth?: number;
}

/**
 * AgentBuilderLayout - Sequentum-style 4-region layout
 * 
 * Layout Structure:
 * ┌─────────────────────────────────────┐
 * │          Toolbar (Top)              │
 * ├──────────┬──────────────────────────┤
 * │          │                          │
 * │   Left   │     Canvas (Center)      │
 * │   Tree   │                          │
 * │          │                          │
 * ├──────────┴──────────────────────────┤
 * │      Bottom Dock (Tabbed)           │
 * └─────────────────────────────────────┘
 */
export function AgentBuilderLayout({
  toolbar,
  leftTree,
  leftTreeWidth = 280,
  canvas,
  bottomDock,
  bottomDockHeight = 400,
  bottomDockCollapsed = false,
  onToggleBottomDock,
  rightPanel,
  rightPanelWidth = 0,
}: AgentBuilderLayoutProps) {
  return (
    <div className="flex flex-col h-full bg-gray-900 overflow-hidden">
      {/* Top Toolbar */}
      <div className="flex-shrink-0 border-b border-gray-700 bg-gray-800">
        {toolbar}
      </div>
      
      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Tree (Optional) */}
        {leftTree && (
          <div 
            className="flex-shrink-0 border-r border-gray-700 bg-gray-800 overflow-y-auto"
            style={{ width: `${leftTreeWidth}px` }}
          >
            {leftTree}
          </div>
        )}
        
        {/* Canvas (Center) */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-hidden bg-gray-950">
            {canvas}
          </div>
          
          {/* Bottom Dock */}
          <div 
            className={`border-t border-gray-700 bg-gray-800 transition-all flex-shrink-0 ${
              bottomDockCollapsed ? 'h-12' : ''
            }`}
            style={{ height: bottomDockCollapsed ? '48px' : `${bottomDockHeight}px` }}
          >
            <div className="h-full flex flex-col">
              {/* Dock Header with collapse toggle */}
              <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700 bg-gray-850 flex-shrink-0">
                <div className="flex-1">
                  {/* Tab headers will be rendered here by bottomDock */}
                </div>
                {onToggleBottomDock && (
                  <button
                    onClick={onToggleBottomDock}
                    className="p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
                    title={bottomDockCollapsed ? 'Expand' : 'Collapse'}
                  >
                    {bottomDockCollapsed ? (
                      <ChevronUpIcon className="w-4 h-4" />
                    ) : (
                      <ChevronDownIcon className="w-4 h-4" />
                    )}
                  </button>
                )}
              </div>
              
              {/* Dock Content */}
              {!bottomDockCollapsed && (
                <div className="flex-1 overflow-hidden">
                  {bottomDock}
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Right Panel (Optional - Future) */}
        {rightPanel && rightPanelWidth > 0 && (
          <div 
            className="flex-shrink-0 border-l border-gray-700 bg-gray-800 overflow-y-auto"
            style={{ width: `${rightPanelWidth}px` }}
          >
            {rightPanel}
          </div>
        )}
      </div>
    </div>
  );
}



