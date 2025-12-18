# Replit Setup Guide
## Frontend Development Environment

---

## Current Status

✅ Replit connected to GitHub repository  
📋 Next steps to set up frontend development

---

## Step 1: Create Frontend Project Structure in Replit

### Option A: Use Replit's React Template (Recommended)

1. **In Replit:**
   - Click "Create Repl"
   - Search for "React" template
   - Select "React + TypeScript" or "React + Tailwind"
   - Name it: `universal-scraper-frontend`
   - Connect to your GitHub repo

### Option B: Initialize Manually

If you want to set it up manually in your existing Replit:

```bash
# In Replit terminal
cd frontend  # or create it: mkdir frontend && cd frontend

# Initialize React + TypeScript project
npx create-react-app . --template typescript
# OR use Vite (faster):
npm create vite@latest . -- --template react-ts
```

---

## Step 2: Install Dependencies

Create `frontend/package.json`:

```json
{
  "name": "universal-scraper-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "@tanstack/react-query": "^5.12.0",
    "axios": "^1.6.2",
    "zustand": "^4.4.7",
    "tailwindcss": "^3.3.6",
    "@headlessui/react": "^1.7.17",
    "react-syntax-highlighter": "^15.5.0",
    "@types/react-syntax-highlighter": "^15.5.0",
    "react-hook-form": "^7.48.2",
    "zod": "^3.22.4",
    "@hookform/resolvers": "^3.3.2",
    "socket.io-client": "^4.5.4",
    "recharts": "^2.10.3",
    "date-fns": "^2.30.0",
    "clsx": "^2.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "typescript": "^5.3.3",
    "vite": "^5.0.8",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "eslint": "^8.55.0",
    "@typescript-eslint/eslint-plugin": "^6.14.0",
    "@typescript-eslint/parser": "^6.14.0"
  },
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0"
  }
}
```

Then install:

```bash
npm install
```

---

## Step 3: Configure Tailwind CSS

Create `frontend/tailwind.config.js`:

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
      },
    },
  },
  plugins: [],
}
```

Create `frontend/postcss.config.js`:

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

Create `frontend/src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-gray-50 text-gray-900;
  }
}
```

---

## Step 4: Set Up Environment Variables

Create `frontend/.env` (or use Replit's Secrets):

```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8080
VITE_WS_URL=ws://localhost:8080

# Feature Flags
VITE_ENABLE_WEBSOCKET=true
VITE_ENABLE_CACHE_INDICATOR=true
```

**In Replit:**
- Go to "Secrets" tab (🔒 icon)
- Add environment variables:
  - `VITE_API_BASE_URL` (will be your Cloud Run URL in production)
  - `VITE_WS_URL` (WebSocket URL)

---

## Step 5: Create Basic Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Layout/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   └── Footer.tsx
│   │   ├── WebScraping/
│   │   │   ├── UrlInput.tsx
│   │   │   ├── FieldSelector.tsx
│   │   │   ├── BrowserPreview.tsx
│   │   │   └── CacheIndicator.tsx
│   │   ├── DocumentProcessing/
│   │   │   ├── FileUpload.tsx
│   │   │   └── DocumentViewer.tsx
│   │   └── Common/
│   │       ├── DataTable.tsx
│   │       ├── CodeViewer.tsx
│   │       └── StatusBadge.tsx
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── WebScraping.tsx
│   │   ├── DocumentProcessing.tsx
│   │   ├── Jobs.tsx
│   │   └── Settings.tsx
│   ├── services/
│   │   ├── api.ts
│   │   ├── websocket.ts
│   │   └── cache.ts
│   ├── hooks/
│   │   ├── useScrape.ts
│   │   ├── useDocument.ts
│   │   └── useWebSocket.ts
│   ├── types/
│   │   └── index.ts
│   ├── utils/
│   │   └── validation.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── public/
├── package.json
├── tailwind.config.js
├── postcss.config.js
├── tsconfig.json
├── vite.config.ts
└── .env
```

---

## Step 6: Create Vite Configuration

Create `frontend/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE_URL || 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
})
```

---

## Step 7: Create TypeScript Configuration

Create `frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

---

## Step 8: Initialize Basic App Structure

Create `frontend/src/App.tsx`:

```typescript
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Sidebar } from './components/Layout/Sidebar';
import { Header } from './components/Layout/Header';
import Dashboard from './pages/Dashboard';
import WebScraping from './pages/WebScraping';
import DocumentProcessing from './pages/DocumentProcessing';
import Jobs from './pages/Jobs';
import Settings from './pages/Settings';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="flex h-screen bg-gray-50">
          <Sidebar />
          <div className="flex-1 flex flex-col overflow-hidden">
            <Header />
            <main className="flex-1 overflow-y-auto p-6">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/web-scraping" element={<WebScraping />} />
                <Route path="/document-processing" element={<DocumentProcessing />} />
                <Route path="/jobs" element={<Jobs />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
```

---

## Step 9: Set Up Replit Run Configuration

Create `.replit` file (if not exists):

```toml
run = "cd frontend && npm run dev"

[nix]
channel = "stable-22_11"

[deploy]
run = ["sh", "-c", "cd frontend && npm run build"]

[env]
VITE_API_BASE_URL = "http://localhost:8080"
```

Or use Replit's GUI:
- Click "Run" button settings (⚙️)
- Set run command: `cd frontend && npm run dev`
- Set port: `3000`

---

## Step 10: Create Basic API Service

Create `frontend/src/services/api.ts`:

```typescript
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token interceptor (when auth is implemented)
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('api_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Scraping endpoints
export const scrapingApi = {
  scrape: (data: any) => api.post('/api/v1/web-scraping/scrape', data),
  getJob: (jobId: string) => api.get(`/api/v1/web-scraping/jobs/${jobId}`),
  getResults: (jobId: string) => api.get(`/api/v1/web-scraping/jobs/${jobId}/results`),
  preview: (url: string) => api.get(`/api/v1/web-scraping/preview`, { params: { url } }),
  checkCache: (url: string) => api.get(`/api/v1/cache/status`, { params: { url } }),
};

// Document processing endpoints
export const documentApi = {
  extract: (data: any) => api.post('/api/v1/document-processing/extract', data),
  getJob: (jobId: string) => api.get(`/api/v1/document-processing/jobs/${jobId}`),
};

// Configuration endpoints
export const configApi = {
  testProxy: (config: any) => api.post('/api/v1/proxy/test', config),
  testWarehouse: (config: any) => api.post('/api/v1/warehouse/test', config),
};
```

---

## Step 11: Quick Start Checklist

- [ ] Replit connected to GitHub ✅
- [ ] Create `frontend/` directory in Replit
- [ ] Initialize React + TypeScript project
- [ ] Install dependencies (`npm install`)
- [ ] Configure Tailwind CSS
- [ ] Set up environment variables (Secrets)
- [ ] Create basic project structure
- [ ] Create `App.tsx` with routing
- [ ] Configure Replit run command
- [ ] Test: Run `npm run dev` in Replit

---

## Step 12: First Component to Build

Start with a simple component to test everything works:

Create `frontend/src/pages/WebScraping.tsx`:

```typescript
import { useState } from 'react';
import { UrlInput } from '../components/WebScraping/UrlInput';
import { FieldSelector } from '../components/WebScraping/FieldSelector';

export default function WebScraping() {
  const [url, setUrl] = useState('');
  const [fields, setFields] = useState<string[]>([]);

  const handleScrape = async () => {
    // TODO: Implement scraping
    console.log('Scraping:', url, fields);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Web Scraping</h1>
      
      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <UrlInput onScrape={handleScrape} />
        <FieldSelector fields={fields} onChange={setFields} />
        
        <button
          onClick={handleScrape}
          className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Start Scraping
        </button>
      </div>
    </div>
  );
}
```

---

## Step 13: Replit-Specific Tips

### Using Replit Secrets
1. Click 🔒 icon (Secrets)
2. Add secrets:
   - `VITE_API_BASE_URL` = Your backend URL
   - `VITE_WS_URL` = WebSocket URL
   - `OPENAI_API_KEY` = (for testing, if needed)

### Running Commands
- Use Replit's terminal (bottom panel)
- Commands run in the workspace root
- Use `cd frontend` before npm commands

### Auto-Reload
- Replit auto-reloads on file changes
- Check browser console for errors
- Use Replit's built-in browser preview

### Git Integration
- Replit auto-syncs with GitHub
- Commit changes: `git add . && git commit -m "message"`
- Push: `git push origin main`

---

## Step 14: Next Steps After Setup

1. **Build First Component**: Start with `UrlInput.tsx`
2. **Connect to Backend**: Update API base URL when backend is ready
3. **Add Routing**: Set up React Router
4. **Add State Management**: Set up Zustand stores
5. **Add WebSocket**: Connect to real-time updates
6. **Style Components**: Use Tailwind classes
7. **Add Forms**: Use React Hook Form + Zod

---

## Troubleshooting

### Issue: Dependencies not installing
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Issue: Tailwind not working
```bash
# Rebuild Tailwind
npx tailwindcss -i ./src/index.css -o ./dist/output.css --watch
```

### Issue: Port already in use
- Change port in `vite.config.ts`
- Or kill process: `lsof -ti:3000 | xargs kill`

### Issue: TypeScript errors
```bash
# Check TypeScript config
npx tsc --noEmit
```

---

## Ready to Code!

Once you've completed these steps, you'll have:
- ✅ React + TypeScript project
- ✅ Tailwind CSS configured
- ✅ Basic routing set up
- ✅ API service ready
- ✅ Development server running

You can now start building components based on `UI_REQUIREMENTS.md`!

---

## Quick Command Reference

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Type check
npx tsc --noEmit

# Lint
npm run lint
```

