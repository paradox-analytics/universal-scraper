# Universal Scraper Frontend

React + TypeScript + Tailwind CSS frontend for the Universal Scraper SaaS platform.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

- `VITE_API_BASE_URL` - Backend API URL (default: http://localhost:8080)
- `VITE_WS_URL` - WebSocket URL for real-time updates

## Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/         # Page components
├── services/      # API clients and WebSocket
├── hooks/         # Custom React hooks
├── types/         # TypeScript type definitions
└── utils/         # Utility functions
```

## Development

The app runs on `http://localhost:3000` by default.

## Features

- ✅ URL input with validation
- ✅ Field selector (natural language + structured)
- ✅ Cache indicator
- ✅ Results table with sorting/filtering
- ✅ Raw data viewer (JSON syntax highlighting)
- ✅ Responsive layout with sidebar navigation

## Next Steps

See `UI_REQUIREMENTS.md` for complete component specifications.

