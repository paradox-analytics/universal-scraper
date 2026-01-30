import { useState } from 'react';
import { XMarkIcon, ClockIcon } from '@heroicons/react/24/outline';
import { agentApi } from '../../services/api';

interface DeployAsAgentModalProps {
  isOpen: boolean;
  onClose: () => void;
  type: 'web_scraping' | 'document_processing';
  config: {
    url?: string;
    fields?: string[];
    domain?: string;
  };
}

export function DeployAsAgentModal({ isOpen, onClose, type, config }: DeployAsAgentModalProps) {
  const [schedule, setSchedule] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState(false);

  if (!isOpen) return null;

  const handleDeploy = async () => {
    if (!config.url && type === 'web_scraping') {
      setError('URL is required for web scraping');
      return;
    }

    if (type === 'document_processing' && (!config.fields || config.fields.length === 0)) {
      setError('Fields are required for document processing');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const agentConfig: any = {
        fields: config.fields || [],
      };

      if (type === 'web_scraping') {
        // Add web scraping specific config
        agentConfig.url = config.url;
        agentConfig.mode = 'hybrid';
      } else if (type === 'document_processing') {
        // Document processing config
        agentConfig.useOCR = true; // Default to OCR enabled
      }

      // Create agent
      const createResponse = await agentApi.createAgent({
        type,
        config: agentConfig,
        queue_immediately: !schedule, // Only queue immediately if no schedule
      });

      const agentId = createResponse.data.agent.id;

      // If schedule provided, schedule the agent
      if (schedule) {
        await agentApi.scheduleAgent(agentId, {
          schedule,
          timezone: 'UTC',
        });
      }

      setSuccess(true);
      setTimeout(() => {
        onClose();
        // Redirect to agents page
        window.location.href = '/agents';
      }, 2000);
    } catch (err: any) {
      console.error('Failed to deploy agent:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to deploy agent');
    } finally {
      setLoading(false);
    }
  };

  const schedulePresets = [
    { label: 'Every hour', value: '0 * * * *' },
    { label: 'Every 6 hours', value: '0 */6 * * *' },
    { label: 'Daily at midnight', value: '0 0 * * *' },
    { label: 'Weekly (Sunday)', value: '0 0 * * 0' },
    { label: 'Monthly (1st)', value: '0 0 1 * *' },
  ];

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 max-w-md w-full mx-4">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">Deploy as Agent</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <XMarkIcon className="h-6 w-6" />
          </button>
        </div>

        {success ? (
          <div className="text-center py-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-green-900/20 rounded-full mb-4">
              <svg className="w-8 h-8 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <p className="text-green-400 font-semibold mb-2">Agent deployed successfully!</p>
            <p className="text-gray-400 text-sm">Redirecting to agents page...</p>
          </div>
        ) : (
          <>
            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Schedule (Optional)
                </label>
                <p className="text-xs text-gray-500 mb-3">
                  Leave empty to run once immediately, or set a recurring schedule
                </p>
                
                <div className="space-y-2 mb-3">
                  {schedulePresets.map((preset) => (
                    <button
                      key={preset.value}
                      onClick={() => setSchedule(preset.value)}
                      className={`w-full text-left px-3 py-2 rounded-lg border transition-colors ${
                        schedule === preset.value
                          ? 'border-purple-500 bg-purple-900/20 text-purple-300'
                          : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-gray-600'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span>{preset.label}</span>
                        <code className="text-xs text-gray-500">{preset.value}</code>
                      </div>
                    </button>
                  ))}
                </div>

                <input
                  type="text"
                  placeholder="Or enter custom cron expression (e.g., 0 */4 * * *)"
                  value={schedule}
                  onChange={(e) => setSchedule(e.target.value)}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
              </div>

              {config.url && (
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-400 mb-1">URL</p>
                  <p className="text-sm text-gray-300 truncate">{config.url}</p>
                </div>
              )}

              {config.fields && config.fields.length > 0 && (
                <div className="bg-gray-800/50 rounded-lg p-3">
                  <p className="text-xs text-gray-400 mb-1">Fields</p>
                  <p className="text-sm text-gray-300">{config.fields.join(', ')}</p>
                </div>
              )}
            </div>

            {error && (
              <div className="mb-4 p-3 bg-red-900/20 border border-red-700 rounded-lg">
                <p className="text-sm text-red-300">{error}</p>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-300 hover:bg-gray-700 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleDeploy}
                disabled={loading}
                className="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                    <span>Deploying...</span>
                  </>
                ) : (
                  <>
                    <ClockIcon className="h-5 w-5" />
                    <span>Deploy Agent</span>
                  </>
                )}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

