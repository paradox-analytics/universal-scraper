# UI Architecture Clarification

## Current State:

### What Changed (Agent Builder Routes):
- ✅ **NEW**: `/agents/:id` - Opens ScraperAgentBuilder or ProcessorAgentBuilder
- ✅ **NEW**: `/scrapers/:id` - Directly opens ScraperAgentBuilder  
- ✅ **NEW**: `/processors/:id` - Directly opens ProcessorAgentBuilder

These routes now show the **new Sequentum-style 4-region layout** with:
- Top toolbar (type badges, status)
- Left tree (configuration)
- Center canvas (BrowserWorkspace or DocumentViewer)
- Bottom dock (tabbed outputs)

### What Didn't Change (Create Pages):
- ⚠️ **UNCHANGED**: `/web-scraping` - Still uses old `BrowserWorkspace` directly
- ⚠️ **UNCHANGED**: `/document-processing` - Still uses old interface
- ⚠️ **UNCHANGED**: `/agents` list - Stubbed out (placeholder)
- ⚠️ **UNCHANGED**: `/history` - Stubbed out (placeholder)

## Why You're Not Seeing Changes:

If you're on **`https://universal-scaper.web.app/web-scraping`**, you're seeing the **OLD interface** (intentionally preserved for backward compatibility).

## How to See the New UI:

### Option 1: Navigate to an Agent (Recommended)
1. Go to `/web-scraping` and create an agent
2. Note the agent ID from the response
3. Navigate to: `/agents/{YOUR_AGENT_ID}`
4. You'll see the new builder!

### Option 2: Test with Mock Route (If no agents exist)
Navigate directly to:
```
https://universal-scaper.web.app/agents/test-123
```
You'll see an error screen, but it will be the **new builder's error screen**, not the old interface.

### Option 3: Check Another Page
Navigate to:
```
https://universal-scaper.web.app/agents
```
You'll see a placeholder that says "Agents page is being migrated..." - this confirms the new code is deployed.

## Visual Comparison:

### OLD UI (what you're seeing on /web-scraping):
```
┌────────────────────────────────────┐
│  Browser view (full width)         │
│                                    │
├────────────────────────────────────┤
│  Tabbed results panel              │
│  (Extracted Content, Schema, etc)  │
└────────────────────────────────────┘
```

### NEW UI (what you'll see on /agents/:id):
```
┌──────────────────────────────────────────┐
│  Toolbar (Type badge, Status, Actions)   │
├──────────┬───────────────────┬───────────┤
│          │                   │           │
│  Left    │    Canvas         │  (future) │
│  Tree    │  (BrowserWork     │  Right    │
│  (Config)│   space or Doc    │  Panel    │
│          │   Viewer)         │           │
├──────────┴───────────────────┴───────────┤
│  Bottom Dock (Tabbed)                    │
│  - Extracted Content                     │
│  - Selection                             │
│  - Schema                                │
│  - Data Preview                          │
└──────────────────────────────────────────┘
```

## Next Steps for Full UI Migration:

To make the new UI the default, we need to:
1. **Update `/web-scraping` page** to use `ScraperAgentBuilder` instead of `BrowserWorkspace`
2. **Rebuild `/agents` list** with new agent cards
3. **Add "New Agent" button** that routes to `/agents/new` (create mode)

Would you like me to:
A) Update `/web-scraping` to use the new builder UI?
B) First show you the new UI by creating a test agent?
C) Something else?



