import { useState } from 'react';
import { XMarkIcon, CalendarIcon, ClockIcon } from '@heroicons/react/24/outline';

interface ScheduleModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSchedule: (schedule: string) => void;
  type?: 'web_scraping' | 'document_processing';
  config?: any;
}

const schedulePresets = [
  { label: 'Every Hour', value: '0 * * * *' },
  { label: 'Every 6 Hours', value: '0 */6 * * *' },
  { label: 'Every 12 Hours', value: '0 */12 * * *' },
  { label: 'Daily at Midnight', value: '0 0 * * *' },
  { label: 'Daily at 9 AM', value: '0 9 * * *' },
  { label: 'Weekly (Monday)', value: '0 0 * * 1' },
  { label: 'Monthly (1st)', value: '0 0 1 * *' },
];

export function ScheduleModal({ isOpen, onClose, onSchedule, type: _type, config: _config }: ScheduleModalProps) {
  const [schedule, setSchedule] = useState('');
  const [customCron, setCustomCron] = useState('');
  const [useCustom, setUseCustom] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cronExpression = useCustom ? customCron : schedule;
    if (!cronExpression) {
      alert('Please select or enter a schedule');
      return;
    }
    onSchedule(cronExpression);
    setSchedule('');
    setCustomCron('');
    setUseCustom(false);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-800 rounded-lg border border-gray-700 w-full max-w-md mx-4 shadow-xl">
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-900/20 rounded-lg border border-purple-700">
              <CalendarIcon className="h-5 w-5 text-purple-400" />
            </div>
            <h2 className="text-xl font-semibold text-white">Schedule Agent</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 transition-colors"
          >
            <XMarkIcon className="h-6 w-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Select Schedule Preset
            </label>
            <div className="space-y-2">
              {schedulePresets.map((preset) => (
                <button
                  key={preset.value}
                  type="button"
                  onClick={() => {
                    setSchedule(preset.value);
                    setUseCustom(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg border-2 transition-all ${
                    schedule === preset.value && !useCustom
                      ? 'border-purple-500 bg-purple-900/20'
                      : 'border-gray-700 bg-gray-900/50 hover:border-gray-600'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-white font-medium">{preset.label}</span>
                    <span className="text-xs text-gray-400 font-mono">{preset.value}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="border-t border-gray-700 pt-4">
            <div className="flex items-center gap-2 mb-3">
              <input
                type="checkbox"
                id="use-custom"
                checked={useCustom}
                onChange={(e) => {
                  setUseCustom(e.target.checked);
                  if (e.target.checked) {
                    setSchedule('');
                  }
                }}
                className="w-4 h-4 text-purple-600 bg-gray-800 border-gray-700 rounded focus:ring-purple-500"
              />
              <label htmlFor="use-custom" className="text-sm font-medium text-gray-300">
                Use Custom Cron Expression
              </label>
            </div>
            {useCustom && (
              <div>
                <input
                  type="text"
                  value={customCron}
                  onChange={(e) => setCustomCron(e.target.value)}
                  placeholder="0 0 * * *"
                  className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 font-mono text-sm"
                />
                <p className="text-xs text-gray-400 mt-2">
                  Format: minute hour day month weekday (e.g., "0 9 * * *" = daily at 9 AM)
                </p>
              </div>
            )}
          </div>

          <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <ClockIcon className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-blue-300 mb-1">Schedule Info</h3>
                <p className="text-xs text-blue-200">
                  This agent will run automatically according to the schedule you set. You can view and manage scheduled agents in the History page.
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 pt-4 border-t border-gray-700">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
            >
              Schedule Agent
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

