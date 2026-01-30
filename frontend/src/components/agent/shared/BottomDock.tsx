import { useState, ReactNode } from 'react';

interface Tab {
  id: string;
  label: string;
  icon?: ReactNode;
  badge?: string | number;
  content: ReactNode;
}

interface BottomDockProps {
  tabs: Tab[];
  defaultTab?: string;
  onTabChange?: (tabId: string) => void;
  actions?: ReactNode; // Additional action buttons (e.g., Export, Re-run)
}

/**
 * BottomDock - Tabbed panel for agent outputs
 * 
 * Used in both Scraper and Processor agents with different tab sets
 */
export function BottomDock({ 
  tabs, 
  defaultTab,
  onTabChange,
  actions 
}: BottomDockProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);
  
  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
    onTabChange?.(tabId);
  };
  
  const activeTabContent = tabs.find(t => t.id === activeTab)?.content;
  
  return (
    <div className="h-full flex flex-col">
      {/* Tab Headers */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-gray-700 bg-gray-850 flex-shrink-0">
        {/* Tabs */}
        <div className="flex items-center gap-1 flex-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`px-4 py-1.5 text-sm font-medium rounded-t-lg transition-colors flex items-center gap-2 ${
                activeTab === tab.id
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {tab.icon && <span className="w-4 h-4">{tab.icon}</span>}
              {tab.label}
              {tab.badge && (
                <span className={`px-1.5 py-0.5 text-xs rounded ${
                  activeTab === tab.id 
                    ? 'bg-indigo-700 text-white' 
                    : 'bg-gray-600 text-gray-300'
                }`}>
                  {tab.badge}
                </span>
              )}
            </button>
          ))}
        </div>
        
        {/* Action Buttons */}
        {actions && (
          <div className="flex items-center gap-2 ml-4">
            {actions}
          </div>
        )}
      </div>
      
      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {activeTabContent}
      </div>
    </div>
  );
}



