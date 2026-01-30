import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './contexts/AuthContext';
import { ProtectedRoute } from './components/Auth/ProtectedRoute';
import { Sidebar } from './components/Layout/Sidebar';
import { Header } from './components/Layout/Header';
import Dashboard from './pages/Dashboard';
import WebScraping from './pages/WebScraping';
import DocumentProcessing from './pages/DocumentProcessing';
import Cache from './pages/Cache';
import History from './pages/History';
import Settings from './pages/Settings';
import AgentBuilder from './pages/AgentBuilder';
import Login from './pages/Login';
import Signup from './pages/Signup';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            {/* Public routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />

            {/* Protected routes */}
            <Route
              path="/*"
              element={
                <ProtectedRoute>
                  <div className="flex h-screen bg-dark-950">
                    <Sidebar />
                    <div className="flex-1 flex flex-col overflow-hidden">
                      <Header />
                      <main className="flex-1 overflow-hidden">
                        <Routes>
                          <Route path="/" element={<div className="h-full overflow-y-auto p-6 md:p-8 lg:p-10"><Dashboard /></div>} />
                          <Route path="/web-scraping" element={<WebScraping />} />
                          <Route path="/document-processing" element={<DocumentProcessing />} />
                          <Route path="/history" element={<div className="h-full overflow-y-auto p-6 md:p-8 lg:p-10"><History /></div>} />
                          <Route path="/cache" element={<div className="h-full overflow-y-auto p-6 md:p-8 lg:p-10"><Cache /></div>} />
                          <Route path="/settings" element={<div className="h-full overflow-y-auto p-6 md:p-8 lg:p-10"><Settings /></div>} />

                          {/* New agent builder routes */}
                          <Route path="/agents/:id" element={<AgentBuilder />} />
                          <Route path="/scrapers/:id" element={<AgentBuilder />} />
                          <Route path="/processors/:id" element={<AgentBuilder />} />

                          <Route path="*" element={<Navigate to="/" replace />} />
                        </Routes>
                      </main>
                    </div>
                  </div>
                </ProtectedRoute>
              }
            />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;

