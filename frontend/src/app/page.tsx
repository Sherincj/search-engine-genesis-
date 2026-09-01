import { checkHealth } from "../../lib/api";

export default async function Home() {
  let healthStatus = "Checking...";
  let isHealthy = false;
  let backendInfo = "";

  try {
    const health = await checkHealth();
    healthStatus = "Backend Connected";
    isHealthy = true;
    backendInfo = `Index Loaded: ${health.index_loaded} | Documents: ${health.document_count}`;
  } catch (error) {
    healthStatus = "Backend Disconnected";
    isHealthy = false;
    if (error instanceof Error) {
      backendInfo = error.message;
    }
  }

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 row-start-2 items-center text-center">
        <h1 className="text-4xl font-bold tracking-tight">Search Engine</h1>
        
        <div className="max-w-md w-full relative">
          <input 
            type="text" 
            placeholder="Search the web..." 
            className="w-full px-4 py-3 rounded-full border border-gray-300 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-black"
            disabled
          />
          <span className="absolute right-4 top-3 text-gray-400">🔍</span>
        </div>

        <div className="mt-8 p-6 bg-gray-50 rounded-xl border border-gray-100 max-w-lg w-full text-left dark:bg-gray-800 dark:border-gray-700">
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3 dark:text-gray-400">
            System Status
          </h2>
          <div className="flex items-center gap-3 mb-2">
            <div className={`w-3 h-3 rounded-full ${isHealthy ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="font-medium text-gray-900 dark:text-gray-100">{healthStatus}</span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300 font-mono">
            {backendInfo || "Make sure the FastAPI backend is running on port 8000."}
          </p>
        </div>
      </main>
    </div>
  );
}
