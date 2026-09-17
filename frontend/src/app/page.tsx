"use client";

import { useState } from "react";
import { searchDocuments } from "../../lib/api";
import { SearchResponse, SearchResultItem } from "../../lib/types";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (page: number = 1) => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const data = await searchDocuments(query, page, 10);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  const totalPages = results ? Math.ceil(results.total / results.limit) : 0;

  return (
    <div className="min-h-screen p-8 font-[family-name:var(--font-geist-sans)]">
      <main className="max-w-3xl mx-auto">
        {/* Header */}
        <h1 className="text-4xl font-bold tracking-tight text-center mb-8">
          Search Engine
        </h1>

        {/* Search box */}
        <div className="flex gap-2 mb-6">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search documents..."
            className="flex-1 px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 text-black dark:text-white dark:bg-gray-800 dark:border-gray-600"
          />
          <button
            onClick={() => handleSearch()}
            disabled={loading || !query.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            {loading ? "..." : "Search"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="p-4 mb-4 bg-red-50 border border-red-200 rounded-lg text-red-700 dark:bg-red-900/20 dark:border-red-800 dark:text-red-400">
            {error}
          </div>
        )}

        {/* Results */}
        {results && (
          <div>
            {/* Result stats */}
            <div className="text-sm text-gray-500 mb-4 dark:text-gray-400">
              {results.total} result{results.total !== 1 ? "s" : ""} for
              &ldquo;{results.query}&rdquo; ({results.took_ms.toFixed(1)} ms)
            </div>

            {/* Result list */}
            {results.results.length === 0 ? (
              <div className="text-gray-500 dark:text-gray-400">
                No results found.
              </div>
            ) : (
              <div className="space-y-4">
                {results.results.map((item: SearchResultItem) => (
                  <div
                    key={item.document_id}
                    className="p-4 border border-gray-200 rounded-lg hover:border-gray-300 dark:border-gray-700 dark:hover:border-gray-600"
                  >
                    <a
                      href={item.url}
                      className="text-lg font-medium text-blue-600 hover:underline dark:text-blue-400"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {item.title}
                    </a>
                    <div className="text-sm text-green-700 dark:text-green-500 mt-1">
                      {item.url}
                    </div>
                    {item.snippet && (
                      <p
                        className="text-sm text-gray-700 dark:text-gray-300 mt-2 leading-relaxed [&>mark]:bg-yellow-200 [&>mark]:text-black [&>mark]:px-0.5 [&>mark]:rounded dark:[&>mark]:bg-yellow-600 dark:[&>mark]:text-white"
                        dangerouslySetInnerHTML={{ __html: item.snippet }}
                      />
                    )}
                    <div className="text-xs text-gray-400 mt-1">
                      Score: {item.score} &middot; ID: {item.document_id}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-6">
                <button
                  onClick={() => handleSearch(results.page - 1)}
                  disabled={results.page <= 1}
                  className="px-3 py-1 border rounded disabled:opacity-30 dark:border-gray-600"
                >
                  ← Prev
                </button>
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Page {results.page} of {totalPages}
                </span>
                <button
                  onClick={() => handleSearch(results.page + 1)}
                  disabled={results.page >= totalPages}
                  className="px-3 py-1 border rounded disabled:opacity-30 dark:border-gray-600"
                >
                  Next →
                </button>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
