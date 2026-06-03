# Agent Type Split Implementation Plan

## Overview
This document outlines the implementation of a discriminated union for agent types: `SCRAPER` and `DOC_PROCESSOR`.

## 1. ✅ Data Model (COMPLETED)

### New Types (`frontend/src/types/index.ts`)
- `AgentType = 'SCRAPER' | 'DOC_PROCESSOR'`
- `ScraperAgent extends BaseAgent` with scraper-specific `definition`
- `ProcessorAgent extends BaseAgent` with processor-specific `definition`
- `Agent = ScraperAgent | ProcessorAgent` (discriminated union)
- `ScraperRun` and `ProcessorRun` with type-specific outputs
- Legacy adapter functions for backward compatibility

## 2. Routing Changes (TODO)

### App.tsx Routes
```tsx
<Route path="/scrapers/:id" element={<AgentBuilder />} />
<Route path="/processors/:id" element={<AgentBuilder />} />
<Route path="/agents/:id" element={<AgentBuilder />} /> {/* Fallback */}
<Route path="/agents" element={<AgentsList />} />
```

## 3. Component Architecture

### Shared Layout
- `layouts/AgentBuilderLayout` - 4-region Sequentum-style shell
- `components/agent/AgentToolbar` - Top toolbar with agent info
- `components/agent/BottomDock` - Tabbed bottom panel

### Agent Builder Router
```
components/agent/AgentBuilderRouter.tsx
├── Loads agent by ID
├── Determines type
└── Renders ScraperAgentBuilder or ProcessorAgentBuilder
```

### Scraper-Specific
```
components/agent/scraper/
├── ScraperAgentBuilder.tsx          # Main builder component
├── ScraperAgentTree.tsx             # Left tree: URL/Navigation/Schema/Export
├── ScraperCanvas.tsx                # Center: Browser preview (BrowserWorkspace)
└── tabs/
    ├── ExtractedContentTab.tsx      # Table view
    ├── SelectionTab.tsx             # Selector management
    ├── SchemaTab.tsx                # Field types
    ├── DataPreviewTab.tsx           # JSON preview
    └── LogsTab.tsx                  # Extraction logs
```

### Processor-Specific
```
components/agent/processor/
├── ProcessorAgentBuilder.tsx        # Main builder component
├── ProcessorAgentTree.tsx           # Left tree: Inputs/Preprocess/Chunking/etc
├── ProcessorCanvas.tsx              # Center: DocumentViewer
├── DocumentViewer.tsx               # PDF/DOCX/HTML viewer
└── tabs/
    ├── DocumentPreviewTab.tsx       # Doc viewer
    ├── ChunksTab.tsx                # Chunk table
    ├── ExtractedFieldsTab.tsx       # Key/value pairs
    ├── SchemaTab.tsx                # Output schema
    ├── ArtifactsTab.tsx             # Indexes/exports
    └── LogsTab.tsx                  # Processing logs
```

## 4. Agents List Page Updates

### Filter Pills
- All Agents
- Scraper Agents (filter by type='SCRAPER')
- Document Processor Agents (filter by type='DOC_PROCESSOR')

### New Agent Dropdown
```tsx
<Dropdown>
  <DropdownItem onClick={createScraper}>New Scraper Agent</DropdownItem>
  <DropdownItem onClick={createProcessor}>New Document Processor Agent</DropdownItem>
</Dropdown>
```

## 5. Output Contracts

### Scraper Output
```typescript
{
  rows: Array<Record<string, any>>,
  schema: { columns: [...] },
  selection: { selectors, detectedFields },
  htmlSnapshots: [...],
  screenshots: [...]
}
```

### Processor Output
```typescript
{
  documents: [{id, name, type, sourceUri}],
  chunks: [{docId, chunkId, text, page, tokens}],
  fields: [{name, value, confidence, sourceChunkId}],
  artifacts: [{type, uri, metadata}]
}
```

## 6. Cache Integration

Cache functionality works across both types:
- Scrapers: Pattern cache (selectors, schemas)
- Processors: Template cache (extraction rules, chunking strategies)
- Both: Share cache infrastructure (Redis/file-based)

## 7. Migration Strategy

### Backward Compatibility
1. Existing agents default to `type: 'SCRAPER'`
2. `adaptLegacyAgent()` function converts old format to new
3. API adapters handle backend responses that don't have new type field

### Phased Rollout
1. ✅ Phase 1: Add types, keep old UI working
2. Phase 2: Build new agent builders alongside old
3. Phase 3: Update routing to use new builders
4. Phase 4: Deprecate old agent creation flow

## Implementation Order

1. ✅ Types & interfaces
2. Create shared layout components
3. Build AgentBuilderRouter
4. Implement ScraperAgentBuilder (reuse existing BrowserWorkspace)
5. Implement ProcessorAgentBuilder
6. Update Agents list page
7. Update routing
8. Test & refine

## File Locations

```
frontend/src/
├── types/
│   └── index.ts (✅ Updated)
├── components/
│   ├── agent/
│   │   ├── AgentBuilderRouter.tsx (NEW)
│   │   ├── layouts/
│   │   │   └── AgentBuilderLayout.tsx (NEW)
│   │   ├── scraper/
│   │   │   └── ... (NEW - wraps BrowserWorkspace)
│   │   └── processor/
│   │       └── ... (NEW)
│   └── BrowserWorkspace/ (EXISTS - reuse for scraper)
├── pages/
│   ├── Agents.tsx (UPDATE - add filters & dropdown)
│   └── AgentBuilder.tsx (NEW - uses AgentBuilderRouter)
└── App.tsx (UPDATE - new routes)
```



