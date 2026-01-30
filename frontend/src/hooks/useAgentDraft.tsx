import { useState, useEffect } from 'react';
import { Agent } from '../types';

const DRAFT_STORAGE_KEY = 'paradocs_draft_agents';
const UNSAVED_CHANGES_KEY = 'paradocs_unsaved_changes';

interface DraftAgent {
  id: string;
  type: 'SCRAPER' | 'DOC_PROCESSOR';
  name: string;
  definition: any;
  lastModified: number;
  isDraft: boolean;
}

/**
 * Hook to manage agent draft state with auto-save
 */
export function useAgentDraft(agentId?: string) {
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [autoSaveEnabled, setAutoSaveEnabled] = useState(true);

  // Load draft from localStorage
  const loadDraft = (id: string): DraftAgent | null => {
    try {
      const drafts = JSON.parse(localStorage.getItem(DRAFT_STORAGE_KEY) || '{}');
      return drafts[id] || null;
    } catch (e) {
      console.error('Failed to load draft:', e);
      return null;
    }
  };

  // Save draft to localStorage
  const saveDraft = (agent: Partial<Agent>) => {
    try {
      const drafts = JSON.parse(localStorage.getItem(DRAFT_STORAGE_KEY) || '{}');
      const draftAgent: DraftAgent = {
        id: agent.id || `draft-${Date.now()}`,
        type: agent.type || 'SCRAPER',
        name: agent.name || 'Untitled Agent',
        definition: agent.definition || {},
        lastModified: Date.now(),
        isDraft: true,
      };
      drafts[draftAgent.id] = draftAgent;
      localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(drafts));
      setLastSaved(new Date());
      setHasUnsavedChanges(false);
      return draftAgent;
    } catch (e) {
      console.error('Failed to save draft:', e);
      return null;
    }
  };

  // Delete draft from localStorage
  const deleteDraft = (id: string) => {
    try {
      const drafts = JSON.parse(localStorage.getItem(DRAFT_STORAGE_KEY) || '{}');
      delete drafts[id];
      localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(drafts));
      setHasUnsavedChanges(false);
    } catch (e) {
      console.error('Failed to delete draft:', e);
    }
  };

  // List all drafts
  const listDrafts = (): DraftAgent[] => {
    try {
      const drafts = JSON.parse(localStorage.getItem(DRAFT_STORAGE_KEY) || '{}');
      return Object.values(drafts);
    } catch (e) {
      console.error('Failed to list drafts:', e);
      return [];
    }
  };

  // Mark as having unsaved changes
  const markDirty = () => {
    setHasUnsavedChanges(true);
    if (agentId) {
      try {
        const unsaved = JSON.parse(localStorage.getItem(UNSAVED_CHANGES_KEY) || '{}');
        unsaved[agentId] = true;
        localStorage.setItem(UNSAVED_CHANGES_KEY, JSON.stringify(unsaved));
      } catch (e) {
        console.error('Failed to mark dirty:', e);
      }
    }
  };

  // Clear unsaved changes flag
  const clearDirty = () => {
    setHasUnsavedChanges(false);
    if (agentId) {
      try {
        const unsaved = JSON.parse(localStorage.getItem(UNSAVED_CHANGES_KEY) || '{}');
        delete unsaved[agentId];
        localStorage.setItem(UNSAVED_CHANGES_KEY, JSON.stringify(unsaved));
      } catch (e) {
        console.error('Failed to clear dirty:', e);
      }
    }
  };

  // Auto-save effect
  useEffect(() => {
    if (agentId && hasUnsavedChanges && autoSaveEnabled) {
      const timeout = setTimeout(() => {
        const draft = loadDraft(agentId);
        if (draft) {
          saveDraft(draft);
        }
      }, 3000); // Auto-save after 3 seconds of inactivity

      return () => clearTimeout(timeout);
    }
  }, [agentId, hasUnsavedChanges, autoSaveEnabled]);

  return {
    hasUnsavedChanges,
    lastSaved,
    autoSaveEnabled,
    setAutoSaveEnabled,
    loadDraft,
    saveDraft,
    deleteDraft,
    listDrafts,
    markDirty,
    clearDirty,
  };
}

/**
 * Hook to prompt user before leaving with unsaved changes
 */
export function useBeforeUnload(hasUnsavedChanges: boolean) {
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = '';
        return '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [hasUnsavedChanges]);
}

/**
 * Component to show unsaved changes indicator
 */
interface UnsavedChangesIndicatorProps {
  hasUnsavedChanges: boolean;
  lastSaved: Date | null;
  autoSaveEnabled: boolean;
}

export function UnsavedChangesIndicator({ hasUnsavedChanges, lastSaved, autoSaveEnabled }: UnsavedChangesIndicatorProps) {
  if (!hasUnsavedChanges && !lastSaved) return null;

  return (
    <div className="flex items-center gap-2 text-xs">
      {hasUnsavedChanges ? (
        <>
          <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
          <span className="text-yellow-400">
            {autoSaveEnabled ? 'Saving...' : 'Unsaved changes'}
          </span>
        </>
      ) : (
        lastSaved && (
          <>
            <div className="w-2 h-2 bg-green-500 rounded-full" />
            <span className="text-gray-400">
              Saved {formatRelativeTime(lastSaved)}
            </span>
          </>
        )
      )}
    </div>
  );
}

function formatRelativeTime(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 10) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return date.toLocaleDateString();
}

