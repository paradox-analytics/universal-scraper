export default function Agents() {
  return (
    <div className="max-w-7xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-white">Agents</h1>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-400">
          Agents page is being migrated to support new agent types (SCRAPER / DOC_PROCESSOR).
        </p>
        <p className="text-gray-500 mt-2 text-sm">
          Temporarily disabled during refactoring. Use /web-scraping or /document-processing to create agents.
        </p>
        <p className="text-gray-500 mt-4 text-sm">
          Direct agent builder access available at:
          <br />
          <code className="text-indigo-400">/agents/:id</code>, <code className="text-indigo-400">/scrapers/:id</code>, <code className="text-indigo-400">/processors/:id</code>
        </p>
      </div>
    </div>
  );
}
