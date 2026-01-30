import type { Agent, ScraperRunOutput, ProcessorRunOutput, ScraperAgent } from '../types';

/**
 * Output Adapters - Convert type-specific run outputs to unified table format
 */

export interface TableColumn {
  name: string;
  type: string;
  required?: boolean;
}

export interface TableData {
  columns: TableColumn[];
  rows: Array<Record<string, any>>;
  metadata?: Record<string, any>;
}

/**
 * Adapt ScraperRunOutput to table format
 */
export function adaptScraperOutput(output: ScraperRunOutput, agent: ScraperAgent): TableData {
  return {
    columns: output.schema?.columns || agent.definition.fields?.map(field => ({
      name: field,
      type: 'string',
      required: false,
    })) || [],
    rows: output.rows || [],
    metadata: {
      itemsExtracted: output.rows?.length || 0,
      paginationDetected: output.pagination?.detected || false,
      totalPages: output.pagination?.totalPages,
      htmlSnapshots: output.htmlSnapshots?.length || 0,
      screenshots: output.screenshots?.length || 0,
      selectors: output.selection?.selectors,
    },
  };
}

/**
 * Adapt ProcessorRunOutput to table format
 */
export function adaptProcessorOutput(output: ProcessorRunOutput): TableData {
  // Convert fields array to rows
  const rows = output.fields.length > 0 
    ? [output.fields.reduce((acc, field) => {
        acc[field.name] = field.value;
        return acc;
      }, {} as Record<string, any>)]
    : [];
  
  const columns: TableColumn[] = output.fields.map(field => ({
    name: field.name,
    type: typeof field.value,
    required: false,
  }));
  
  return {
    columns,
    rows,
    metadata: {
      documentsProcessed: output.documents?.length || 0,
      chunksCreated: output.chunks?.length || 0,
      fieldsExtracted: output.fields?.length || 0,
      artifacts: output.artifacts?.length || 0,
      avgConfidence: output.fields.length > 0
        ? output.fields.reduce((sum, f) => sum + (f.confidence || 0), 0) / output.fields.length
        : 0,
    },
  };
}

/**
 * Generic adapter that routes to correct type-specific adapter
 */
export function adaptAgentOutput(agent: Agent, output: any): TableData {
  switch (agent.type) {
    case 'SCRAPER':
      return adaptScraperOutput(output as ScraperRunOutput, agent);
    case 'DOC_PROCESSOR':
      return adaptProcessorOutput(output as ProcessorRunOutput);
    default:
      // Fallback for unknown types
      return {
        columns: [],
        rows: [],
        metadata: { error: 'Unknown agent type' },
      };
  }
}

/**
 * Check if agent run output matches expected schema
 */
export function validateAgentOutput(agent: Agent, output: any): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];
  
  switch (agent.type) {
    case 'SCRAPER':
      if (!output.rows || !Array.isArray(output.rows)) {
        errors.push('Missing or invalid "rows" field');
      }
      if (output.schema && !output.schema.columns) {
        errors.push('Invalid schema structure');
      }
      break;
      
    case 'DOC_PROCESSOR':
      if (!output.documents || !Array.isArray(output.documents)) {
        errors.push('Missing or invalid "documents" field');
      }
      if (!output.fields || !Array.isArray(output.fields)) {
        errors.push('Missing or invalid "fields" field');
      }
      break;
      
    default:
      errors.push(`Unknown agent type: ${(agent as any).type}`);
  }
  
  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Get display name for agent type
 */
export function getAgentTypeDisplay(type: string): string {
  switch (type) {
    case 'SCRAPER':
      return 'Web Scraper';
    case 'DOC_PROCESSOR':
      return 'Document Processor';
    default:
      return type;
  }
}

/**
 * Get icon name for agent type (for UI components)
 */
export function getAgentTypeIcon(type: string): string {
  switch (type) {
    case 'SCRAPER':
      return 'GlobeAltIcon';
    case 'DOC_PROCESSOR':
      return 'DocumentTextIcon';
    default:
      return 'CubeIcon';
  }
}

