import { 
  SparklesIcon, 
  CircleStackIcon, 
  BoltIcon,
  CheckCircleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

export interface ExtractionFlowStep {
  id: string;
  label: string;
  status: 'pending' | 'active' | 'completed' | 'cached';
  description?: string;
}

interface ExtractionFlowIndicatorProps {
  currentStep: 'llm' | 'template' | 'cache' | 'deterministic';
  cacheHit?: boolean;
  templateId?: string;
  llmTokensUsed?: number;
}

/**
 * Visual indicator for the extraction flow:
 * LLM → Template Generation → Cache Storage → Deterministic Execution
 */
export function ExtractionFlowIndicator({ 
  currentStep, 
  cacheHit = false,
  templateId,
  llmTokensUsed,
}: ExtractionFlowIndicatorProps) {
  const steps: ExtractionFlowStep[] = [
    {
      id: 'llm',
      label: cacheHit ? 'LLM (Skipped)' : 'LLM Analysis',
      status: cacheHit ? 'cached' : currentStep === 'llm' ? 'active' : 'completed',
      description: cacheHit ? 'Using cached template' : `${llmTokensUsed || 0} tokens`,
    },
    {
      id: 'template',
      label: 'Template Spec',
      status: currentStep === 'template' ? 'active' : currentStep === 'llm' ? 'pending' : 'completed',
      description: templateId ? `ID: ${templateId.slice(0, 8)}...` : undefined,
    },
    {
      id: 'cache',
      label: 'Cache Storage',
      status: currentStep === 'cache' ? 'active' : ['llm', 'template'].includes(currentStep) ? 'pending' : 'completed',
      description: cacheHit ? 'Cache hit' : 'Caching for future runs',
    },
    {
      id: 'deterministic',
      label: 'Deterministic Extract',
      status: currentStep === 'deterministic' ? 'active' : ['llm', 'template', 'cache'].includes(currentStep) ? 'pending' : 'completed',
      description: 'No LLM calls',
    },
  ];

  return (
    <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-4">
        <BoltIcon className="w-5 h-5 text-purple-400" />
        <h3 className="text-sm font-semibold text-white">Extraction Flow</h3>
        {cacheHit && (
          <span className="px-2 py-0.5 bg-green-900/30 text-green-400 border border-green-700 rounded text-xs font-medium">
            Cache Hit
          </span>
        )}
      </div>
      
      <div className="space-y-3">
        {steps.map((step, index) => (
          <div key={step.id} className="flex items-start gap-3">
            {/* Step Icon */}
            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${getStepStyles(step.status).bg}`}>
              {getStepIcon(step.status, step.id)}
            </div>
            
            {/* Step Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className={`text-sm font-medium ${getStepStyles(step.status).text}`}>
                  {step.label}
                </span>
                {step.description && (
                  <span className="text-xs text-gray-500">{step.description}</span>
                )}
              </div>
              
              {/* Progress bar for active step */}
              {step.status === 'active' && (
                <div className="mt-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-purple-600 rounded-full animate-pulse w-3/4"></div>
                </div>
              )}
            </div>
            
            {/* Connector line */}
            {index < steps.length - 1 && (
              <div className="absolute left-[19px] w-0.5 h-6 bg-gray-700 -mb-3 mt-8" style={{ transform: 'translateY(100%)' }} />
            )}
          </div>
        ))}
      </div>
      
      {/* Summary */}
      <div className="mt-4 pt-3 border-t border-gray-700">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">
            {cacheHit ? 'Loaded from cache' : 'New pattern learned'}
          </span>
          <span className="text-purple-400 font-medium">
            {cacheHit ? '⚡ Instant' : `🧠 ${llmTokensUsed || 0} tokens`}
          </span>
        </div>
      </div>
    </div>
  );
}

function getStepIcon(status: string, stepId: string) {
  if (status === 'completed' || status === 'cached') {
    return <CheckCircleIcon className="w-4 h-4 text-green-400" />;
  }
  
  if (status === 'active') {
    return <ArrowPathIcon className="w-4 h-4 text-purple-400 animate-spin" />;
  }
  
  // Pending - show step-specific icon
  const iconMap: Record<string, JSX.Element> = {
    llm: <SparklesIcon className="w-4 h-4 text-gray-500" />,
    template: <BoltIcon className="w-4 h-4 text-gray-500" />,
    cache: <CircleStackIcon className="w-4 h-4 text-gray-500" />,
    deterministic: <BoltIcon className="w-4 h-4 text-gray-500" />,
  };
  
  return iconMap[stepId] || <div className="w-4 h-4 bg-gray-700 rounded-full" />;
}

function getStepStyles(status: string) {
  const styles = {
    pending: {
      bg: 'bg-gray-800 border border-gray-700',
      text: 'text-gray-500',
    },
    active: {
      bg: 'bg-purple-900/30 border border-purple-600',
      text: 'text-purple-300',
    },
    completed: {
      bg: 'bg-green-900/30 border border-green-600',
      text: 'text-green-300',
    },
    cached: {
      bg: 'bg-green-900/30 border border-green-600',
      text: 'text-green-300',
    },
  };
  
  return styles[status as keyof typeof styles] || styles.pending;
}

/**
 * Compact badge version for toolbar
 */
interface CacheStatusBadgeProps {
  cacheHit: boolean;
  templateId?: string;
}

export function CacheStatusBadge({ cacheHit, templateId }: CacheStatusBadgeProps) {
  return (
    <div className="flex items-center gap-2">
      {cacheHit ? (
        <div className="flex items-center gap-1 px-2 py-1 bg-green-900/30 text-green-400 border border-green-700 rounded text-xs font-medium">
          <CircleStackIcon className="w-3 h-3" />
          <span>Cached</span>
          {templateId && <span className="text-green-600">#{templateId.slice(0, 6)}</span>}
        </div>
      ) : (
        <div className="flex items-center gap-1 px-2 py-1 bg-purple-900/30 text-purple-400 border border-purple-700 rounded text-xs font-medium">
          <SparklesIcon className="w-3 h-3" />
          <span>LLM Mode</span>
        </div>
      )}
    </div>
  );
}

