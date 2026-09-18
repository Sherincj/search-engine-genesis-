"use client";

import { useState } from "react";
import { searchDocuments, startCrawl } from "../../lib/api";
import { SearchResponse, SearchResultItem, CrawlResponse } from "../../lib/types";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Crawl state
  const [crawlUrl, setCrawlUrl] = useState("");
  const [crawling, setCrawling] = useState(false);
  const [crawlResult, setCrawlResult] = useState<CrawlResponse | null>(null);
  const [crawlError, setCrawlError] = useState<string | null>(null);

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

  const handleCrawl = async () => {
    if (!crawlUrl.trim()) return;

    setCrawling(true);
    setCrawlError(null);
    setCrawlResult(null);

    try {
      const data = await startCrawl(crawlUrl);
      setCrawlResult(data);
    } catch (err) {
      setCrawlError(err instanceof Error ? err.message : "Crawl failed");
    } finally {
      setCrawling(false);
    }
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

        {/* ── Crawl section ─────────────────────────────────── */}
        <div className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-semibold mb-4">Crawl a Website</h2>

          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={crawlUrl}
              onChange={(e) => setCrawlUrl(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleCrawl(); }}
              placeholder="https://example.com"
              className="flex-1 px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-green-500 text-black dark:text-white dark:bg-gray-800 dark:border-gray-600"
            />
            <button
              onClick={handleCrawl}
              disabled={crawling || !crawlUrl.trim()}
              className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              {crawling ? "Crawling..." : "Start Crawl"}
            </button>
          </div>

          {/* Crawl error */}
          {crawlError && (
            <div className="p-4 mb-4 bg-red-50 border border-red-200 rounded-lg text-red-700 dark:bg-red-900/20 dark:border-red-800 dark:text-red-400">
              {crawlError}
            </div>
          )}

          {/* Crawl result */}
          {crawlResult && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg dark:bg-green-900/20 dark:border-green-800">
              <h3 className="font-semibold text-green-800 dark:text-green-400 mb-2">
                Crawl Completed
              </h3>
              <div className="grid grid-cols-2 gap-2 text-sm text-green-700 dark:text-green-300">
                <div>Pages crawled:</div>
                <div className="font-medium">{crawlResult.pages_crawled}</div>
                <div>Pages stored:</div>
                <div className="font-medium">{crawlResult.pages_stored}</div>
                <div>Pages failed:</div>
                <div className="font-medium">{crawlResult.pages_failed}</div>
                <div>Pages skipped:</div>
                <div className="font-medium">{crawlResult.pages_skipped}</div>
                <div>URLs discovered:</div>
                <div className="font-medium">{crawlResult.discovered_urls}</div>
                <div>Max depth reached:</div>
                <div className="font-medium">{crawlResult.max_depth_reached}</div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
