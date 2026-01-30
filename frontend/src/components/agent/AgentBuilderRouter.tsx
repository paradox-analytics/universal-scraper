import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import type { Agent } from '../../types';
import { agentApi } from '../../services/api';
import { ScraperAgentBuilder } from './scraper/ScraperAgentBuilder';
import { ProcessorAgentBuilder } from './processor/ProcessorAgentBuilder';

/**
 * AgentBuilderRouter - Routes to correct builder based on agent type
 * 
 * Supports:
 * - /agents/:id (canonical, determines type)
 * - /scrapers/:id (explicit scraper)
 * - /processors/:id (explicit processor)
 */
export function AgentBuilderRouter() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [agent, setAgent] = useState<Agent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    if (id) {
      loadAgent(id);
    }
  }, [id]);
  
  const loadAgent = async (agentId: string) => {
    try {
      setLoading(true);
      setError(null);
      
      // TODO: Replace with actual API call
      // For now, mock data based on route
      const response = await agentApi.get(agentId);
      
      if (response.data && response.data.success) {
        let agentData = response.data.agent as any;
        
        // Adapter: If backend doesn't have new type field, infer it
        if (!agentData.type || (agentData.type !== 'SCRAPER' && agentData.type !== 'DOC_PROCESSOR')) {
          // Legacy agent - determine type from old fields
          const hasConfig = 'config' in agentData;
          const hasUrl = hasConfig && (agentData.config?.url || agentData.config?.urls);
          const oldType = agentData.type;
          
          if (hasUrl || oldType === 'web_scraping' || oldType === 'batch_scraping') {
            agentData = {
              ...agentData,
              type: 'SCRAPER' as const,
              definition: {
                subType: oldType === 'batch_scraping' ? 'batch_scraping' : 'web_scraping',
                url: hasConfig ? agentData.config?.url : undefined,
                urls: hasConfig ? agentData.config?.urls : undefined,
                fields: hasConfig ? agentData.config?.fields || [] : [],
                mode: hasConfig ? agentData.config?.mode || 'hybrid' : 'hybrid',
                proxy_config: hasConfig ? agentData.config?.proxy_config : undefined,
                pagination_config: hasConfig ? agentData.config?.pagination_config : undefined,
                browser_timeout: hasConfig ? agentData.config?.browser_timeout : undefined,
              },
            };
          } else {
            agentData = {
              ...agentData,
              type: 'DOC_PROCESSOR' as const,
              definition: {
                subType: 'document_processing',
                fields: hasConfig ? agentData.config?.fields || [] : [],
                use_ocr: hasConfig ? agentData.config?.use_ocr : false,
                max_pages: hasConfig ? agentData.config?.max_pages : undefined,
                context: hasConfig ? agentData.config?.context : undefined,
              },
            };
          }
        }
        
        setAgent(agentData);
      } else {
        setError('Agent not found');
      }
    } catch (err: any) {
      console.error('Failed to load agent:', err);
      setError(err.message || 'Failed to load agent');
    } finally {
      setLoading(false);
    }
  };
  
  const handleBack = () => {
    navigate('/agents');
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading agent...</p>
        </div>
      </div>
    );
  }
  
  if (error || !agent) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-red-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-red-400 text-2xl">!</span>
          </div>
          <h2 className="text-xl font-semibold text-white mb-2">Agent Not Found</h2>
          <p className="text-gray-400 mb-4">{error || 'The requested agent could not be loaded.'}</p>
          <button
            onClick={handleBack}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Back to Agents
          </button>
        </div>
      </div>
    );
  }
  
  // Route to correct builder based on agent type
  switch (agent.type) {
    case 'SCRAPER':
      return <ScraperAgentBuilder agent={agent} onBack={handleBack} onUpdate={loadAgent} />;
    case 'DOC_PROCESSOR':
      return <ProcessorAgentBuilder agent={agent} onBack={handleBack} onUpdate={loadAgent} />;
    default:
      return (
        <div className="flex items-center justify-center h-full bg-gray-900">
          <p className="text-gray-400">Unknown agent type: {(agent as any).type}</p>
        </div>
      );
  }
}

