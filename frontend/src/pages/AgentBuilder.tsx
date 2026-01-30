import { AgentBuilderRouter } from '../components/agent/AgentBuilderRouter';

/**
 * AgentBuilder page - Wrapper for AgentBuilderRouter
 * 
 * Supports routes:
 * - /agents/:id
 * - /scrapers/:id
 * - /processors/:id
 */
export default function AgentBuilder() {
  return (
    <div className="h-[calc(100vh-4rem)]">
      <AgentBuilderRouter />
    </div>
  );
}



