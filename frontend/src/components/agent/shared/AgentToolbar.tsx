import { 
  ArrowLeftIcon,
  PlayIcon,
  ArrowPathIcon,
  Cog6ToothIcon,
  CircleStackIcon,
  BookmarkIcon,
} from '@heroicons/react/24/outline';
import type { Agent, AgentStatus } from '../../../types';
import { UnsavedChangesIndicator } from '../../../hooks/useAgentDraft';

interface AgentToolbarProps {
  agent?: Agent;
  onBack?: () => void;
  onRun?: () => void;
  onSave?: () => void;
  isRunning?: boolean;
  isSaving?: boolean;
  hasUnsavedChanges?: boolean;
  lastSaved?: Date | null;
  autoSaveEnabled?: boolean;
  isDraft?: boolean;
}

/**
 * AgentToolbar - Top toolbar for agent builder
 * 
 * Shows agent name, type badge, status, and action buttons
 */
export function AgentToolbar({
  agent,
  onBack,
  onRun,
  onSave,
  isRunning = false,
  isSaving = false,
  hasUnsavedChanges = false,
  lastSaved = null,
  autoSaveEnabled = true,
  isDraft = false,
}: AgentToolbarProps) {
  const getStatusColor = (status?: AgentStatus) => {
    switch (status) {
      case 'completed': return 'bg-green-900/50 text-green-400 border-green-600/50';
      case 'running': return 'bg-blue-900/50 text-blue-400 border-blue-600/50';
      case 'failed': return 'bg-red-900/50 text-red-400 border-red-600/50';
      case 'pending': return 'bg-yellow-900/50 text-yellow-400 border-yellow-600/50';
      case 'queued': return 'bg-purple-900/50 text-purple-400 border-purple-600/50';
      default: return 'bg-gray-700 text-gray-400 border-gray-600';
    }
  };
  
  const getTypeBadge = (type?: string) => {
    switch (type) {
      case 'SCRAPER': 
        return <span className="px-2 py-1 bg-indigo-900/50 text-indigo-400 border border-indigo-600/50 rounded text-xs font-medium">Scraper</span>;
      case 'DOC_PROCESSOR': 
        return <span className="px-2 py-1 bg-purple-900/50 text-purple-400 border border-purple-600/50 rounded text-xs font-medium">Document Processor</span>;
      default: 
        return null;
    }
  };
  
  return (
    <div className="flex items-center gap-4 px-4 py-3">
      {/* Back Button */}
      {onBack && (
        <button
          onClick={onBack}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
          title="Back to agents"
        >
          <ArrowLeftIcon className="w-5 h-5" />
        </button>
      )}
      
      {/* Agent Info */}
      <div className="flex items-center gap-3 flex-1">
        {agent && (
          <>
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold text-white">
                  {agent.name || 'Untitled Agent'}
                </h2>
                {getTypeBadge(agent.type)}
                {isDraft && (
                  <span className="px-2 py-1 bg-gray-700 text-gray-300 border border-gray-600 rounded text-xs font-medium flex items-center gap-1">
                    <BookmarkIcon className="w-3 h-3" />
                    Draft
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3 mt-1">
                {agent.description && (
                  <p className="text-sm text-gray-400">{agent.description}</p>
                )}
                <UnsavedChangesIndicator
                  hasUnsavedChanges={hasUnsavedChanges}
                  lastSaved={lastSaved}
                  autoSaveEnabled={autoSaveEnabled}
                />
              </div>
            </div>
            
            {agent.status && (
              <span className={`px-2 py-1 rounded text-xs font-medium border ${getStatusColor(agent.status)}`}>
                {agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}
              </span>
            )}
          </>
        )}
      </div>
      
      {/* Action Buttons */}
      <div className="flex items-center gap-2">
        {/* Cache/Settings Button */}
        <button
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
          title="Cache settings"
        >
          <CircleStackIcon className="w-5 h-5" />
        </button>
        
        <button
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
          title="Agent settings"
        >
          <Cog6ToothIcon className="w-5 h-5" />
        </button>
        
        {/* Save Button */}
        {onSave && (
          <button
            onClick={onSave}
            disabled={isSaving}
            className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
          >
            {isSaving ? 'Saving...' : 'Save'}
          </button>
        )}
        
        {/* Run Button */}
        {onRun && (
          <button
            onClick={onRun}
            disabled={isRunning}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 text-sm font-medium"
          >
            {isRunning ? (
              <>
                <ArrowPathIcon className="w-4 h-4 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <PlayIcon className="w-4 h-4" />
                Run
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
}

