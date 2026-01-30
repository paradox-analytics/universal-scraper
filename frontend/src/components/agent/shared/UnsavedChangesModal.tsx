import { useNavigate } from 'react-router-dom';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface UnsavedChangesModalProps {
  isOpen: boolean;
  onSave: () => void;
  onDiscard: () => void;
  onCancel: () => void;
}

export function UnsavedChangesModal({ isOpen, onSave, onDiscard, onCancel }: UnsavedChangesModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl p-6 max-w-md w-full mx-4">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 bg-yellow-900/30 rounded-full flex items-center justify-center flex-shrink-0">
            <ExclamationTriangleIcon className="w-6 h-6 text-yellow-400" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white mb-2">Unsaved Changes</h3>
            <p className="text-sm text-gray-400 mb-6">
              You have unsaved changes to this agent. Would you like to save them before leaving?
            </p>
            <div className="flex gap-3">
              <button
                onClick={onSave}
                className="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
              >
                Save
              </button>
              <button
                onClick={onDiscard}
                className="flex-1 px-4 py-2 bg-red-900/30 hover:bg-red-900/50 text-red-400 font-medium rounded-lg border border-red-700 transition-colors"
              >
                Discard
              </button>
              <button
                onClick={onCancel}
                className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 font-medium rounded-lg transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Hook to handle navigation with unsaved changes prompt
 */
export function usePromptBeforeNavigate(hasUnsavedChanges: boolean, _onSave: () => Promise<void>) {
  const navigate = useNavigate();

  const navigateWithPrompt = async (to: string) => {
    if (!hasUnsavedChanges) {
      navigate(to);
      return;
    }

    // Show modal - we'll handle this in the component using this hook
    return new Promise<boolean>((resolve) => {
      // This will be implemented by the component
      resolve(false);
    });
  };

  return { navigateWithPrompt };
}

