# Unified Proxy & Web Unblocker Configuration

## Current Problems
1. **Settings page**: Shows Web Unblocker as separate tab (API key + zone)
2. **Scraper page**: Shows proxy config (username + password)
3. **No global indicator**: Can't see if proxy/Web Unblocker is active
4. **No persistence**: Settings don't carry over between pages
5. **Confusing UX**: Users don't know which is which or how to configure

## Proposed Solution

### 1. Unified Configuration Model
```typescript
interface UnifiedProxyConfig {
  provider: 'none' | 'brightdata' | 'brightdata_webunblocker' | 'oxylabs' | 'scraperapi' | 'custom';
  
  // For residential proxies (Bright Data, Oxylabs, etc.)
  server?: string;
  username?: string;
  password?: string;
  country?: string;
  
  // For Web Unblocker (API-based)
  apiKey?: string;
  zone?: string;
}
```

### 2. Settings Page - Single "Proxy & Anti-Bot" Tab
- **Provider Dropdown**:
  - None
  - Bright Data Residential Proxy
  - Bright Data Web Unblocker (Recommended for Cloudflare)
  - Oxylabs
  - ScraperAPI
  - Custom HTTP/SOCKS
  
- **Dynamic Fields** based on provider:
  - **Bright Data Residential**: Server, Username, Password, Country
  - **Web Unblocker**: API Key, Zone
  - **Oxylabs**: Username, Password, Country
  - **ScraperAPI**: API Key, Country
  - **Custom**: Server, Username, Password

### 3. Global Status Indicator (Header)
Located in the top-right header, next to user profile:

```tsx
<ProxyStatusIndicator>
  {proxyConfig.provider !== 'none' && (
    <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-900/30 border border-green-600">
      <ShieldCheckIcon className="w-4 h-4 text-green-400" />
      <span className="text-xs text-green-400">
        {proxyConfig.provider === 'brightdata_webunblocker' ? 'Web Unblocker' : 'Proxy'} Active
      </span>
    </div>
  )}
</ProxyStatusIndicator>
```

### 4. Remove Per-Page Config
- Remove proxy config from BrowserWorkspace
- Remove proxy config from DocumentViewer
- Always use global settings from Settings page
- Option to "Override for this session" (advanced)

### 5. Connection Test
- Add "Test Connection" button on Settings page
- Shows real-time status: ✅ Connected, ⏳ Testing, ❌ Failed
- Display in header if test passes

## Implementation Plan

### Phase 1: Update Type Definitions
- [ ] Update `ProxyConfiguration` type in `types/index.ts`
- [ ] Remove `WebUnblockerConfig` type (merge into `ProxyConfiguration`)
- [ ] Add provider option for `brightdata_webunblocker`

### Phase 2: Update Settings Page
- [ ] Merge "Proxy" and "Web Unblocker" tabs into single "Proxy & Anti-Bot"
- [ ] Add provider option for Web Unblocker
- [ ] Dynamic fields based on provider
- [ ] Connection test button

### Phase 3: Create Global Status Indicator
- [ ] Create `ProxyStatusIndicator` component
- [ ] Add to `Layout.tsx` or main header
- [ ] Show active provider and status
- [ ] Click to open Settings

### Phase 4: Update BrowserWorkspace
- [ ] Remove local proxy config UI
- [ ] Load proxy config from global settings (AuthContext)
- [ ] Pass to API automatically
- [ ] Option: "Advanced > Override Proxy"

### Phase 5: Update Backend API
- [ ] Already done! API correctly handles both proxy and Web Unblocker
- [ ] Just need frontend to send correct format

## User Flow

### Setup (First Time)
1. User clicks "Settings" in sidebar
2. Goes to "Proxy & Anti-Bot" tab
3. Selects "Bright Data Web Unblocker"
4. Enters API Key and Zone
5. Clicks "Test Connection"
6. ✅ Shows "Connected"
7. Clicks "Save Settings"
8. Green indicator appears in header

### Usage
1. User navigates to Scraper page
2. Sees green "Web Unblocker Active" in header
3. Enters URL and clicks "Navigate"
4. Web Unblocker is automatically used
5. Page loads successfully (bypasses Cloudflare)

### Change Provider
1. User clicks green indicator in header
2. Opens Settings page
3. Changes to "Bright Data Residential Proxy"
4. Enters username/password
5. Clicks "Test Connection"
6. ✅ Shows "Connected"
7. Clicks "Save"
8. Indicator updates to "Proxy Active"

## Benefits
1. **Consistent**: One place to configure proxies
2. **Clear**: User always knows what's active
3. **Persistent**: Settings saved and reused across all pages
4. **Simple**: No duplicate config on every page
5. **Obvious**: Green indicator = protected, no indicator = direct connection

## Files to Modify
1. `frontend/src/types/index.ts` - Update types
2. `frontend/src/pages/Settings.tsx` - Merge tabs, update UI
3. `frontend/src/components/Layout/ProxyStatusIndicator.tsx` - New component
4. `frontend/src/components/Layout/index.tsx` - Add indicator to header
5. `frontend/src/components/BrowserWorkspace/BrowserWorkspace.tsx` - Remove local config
6. `frontend/src/components/DocumentViewer/DocumentViewer.tsx` - Remove local config
7. `frontend/src/contexts/AuthContext.tsx` - Expose global proxy config




