export default function Dashboard() {
  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Total Jobs</h3>
          <p className="text-3xl font-bold text-gray-900">1,234</p>
          <p className="text-sm text-gray-500 mt-1">+12% from last month</p>
        </div>
        
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Success Rate</h3>
          <p className="text-3xl font-bold text-green-600">98.5%</p>
          <p className="text-sm text-gray-500 mt-1">1,216 successful</p>
        </div>
        
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Cache Hit Rate</h3>
          <p className="text-3xl font-bold text-blue-600">87%</p>
          <p className="text-sm text-gray-500 mt-1">Subsecond responses</p>
        </div>
      </div>
      
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Recent Jobs</h2>
        <p className="text-gray-500">Job history will appear here</p>
      </div>
    </div>
  );
}

